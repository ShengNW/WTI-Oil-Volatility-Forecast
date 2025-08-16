#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch metrics CSVs by filling/adding ONLY the 'RK' and 'CBV' rows.
- No recomputation of existing rows; only fill NaNs or append if missing.
- Period windows: AUTO tertiles of the merged sample (no fixed/paper dates).
- Unit alignment = match evaluation logic:
    ANN(h>1):   treat cumulative file as mean(ν²) -> sqrt -> RMS σ
    ARFIMA(h>1): if column implies cumVar/variance -> sqrt; if cumVolRMS -> as-is
    GARCH(h>1): try rebuild RMS σ from single-step horizons; else fallback cumulative
    HAR:        already σ (including cumulative)
- Actuals (RK/CBV): prefer '*_sigma_cumulative_h*'; if we accidentally pick '*var*' -> sqrt

Verbose logs for forensic debugging.
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Config -----------------------------

DEFAULT_ROOT = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
BACKUP_DIR   = r"03_results/evaluation_reports_BackUp"

MEASURES_TO_HANDLE = ["RK", "CBV"]   # Row label uses "CBV" (BV≡CBV)

# Forecast roots
DIR_ANN_FINAL = "03_results/final_forecasts/ANN"
DIR_ANN_CUM2  = "03_results/intermediate_results/cumulative_forecasts/ANN"
DIR_ARFIMA    = "03_results/final_forecasts/ARFIMA"
DIR_HAR       = "03_results/final_forecasts/HAR"
DIR_GARCH     = "03_results/final_forecasts/GARCH"

# Labels (actuals) dir
DIR_ACTUAL_CUM = "03_results/intermediate_results/volatility_estimates/cumulative_forecasts"

# Accept asset with underscores; match your filenames
CSV_RE = re.compile(
    r"^(?P<asset>.+?)__metrics__(?P<period>pre|crisis|post|whole)__h(?P<h>\d+)__(?P<freq>\d+min)__(?P<metric>RMSE|MAE|MME_O|MME_U|OverRate|UnderRate|N)\.csv$",
    re.IGNORECASE
)

# ----------------------------- Utils -----------------------------

def log(msg: str):
    print(msg, flush=True)

def sqrt_clip(arr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(arr, 0, None))

def ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    for c in df.columns:
        if str(c).lower() in ("date","datetime","time","timestamp"):
            out = df.copy()
            out["date"] = pd.to_datetime(out[c], errors="coerce")
            return out
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.reset_index().rename(columns={df.index.name or "index": "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out
    try:
        idx = pd.to_datetime(df.index, errors="coerce")
        if idx.notna().mean() > 0.7:
            out = df.copy()
            out.insert(0, "date", idx)
            return out
    except Exception:
        pass
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.7:
            out = df.copy()
            out["date"] = s
            return out
    return df

def read_parquet(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        try:
            df = pd.read_parquet(path)
            return ensure_dt(df)
        except Exception as e:
            log(f"[READ FAIL] {path}: {e}")
    return None

def pick_first_float_col(df: pd.DataFrame, prefer_contains: List[str] = None, avoid: List[str] = None) -> Optional[str]:
    prefer_contains = [s.lower() for s in (prefer_contains or [])]
    avoid = set([s.lower() for s in (avoid or [])] + ["date"])
    # strict preference
    for c in df.columns:
        lc = str(c).lower()
        if lc in avoid: continue
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            if all(tok in lc for tok in prefer_contains):
                return c
    # fallback: first numeric
    for c in df.columns:
        lc = str(c).lower()
        if lc in avoid: continue
        if pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c]):
            return c
    return None

def auto_periods(min_date: pd.Timestamp, max_date: pd.Timestamp) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    if pd.isna(min_date) or pd.isna(max_date) or min_date >= max_date:
        return {"whole": (min_date, max_date)}
    span = max_date - min_date
    if span.days <= 2:
        return {"whole": (min_date, max_date)}
    t1 = min_date + span / 3
    t2 = min_date + 2 * span / 3
    return {
        "pre":    (min_date, pd.Timestamp(t1.floor("D"))),
        "crisis": (pd.Timestamp((t1 + pd.Timedelta(days=1)).floor("D")), pd.Timestamp(t2.floor("D"))),
        "post":   (pd.Timestamp((t2 + pd.Timedelta(days=1)).floor("D")), max_date),
    }

def find_files_by_tokens(dirpath: str, tokens: List[str]) -> List[str]:
    if not os.path.isdir(dirpath): return []
    toks = [t.lower() for t in tokens]
    out = []
    for fn in os.listdir(dirpath):
        l = fn.lower()
        if all(t in l for t in toks):
            out.append(os.path.join(dirpath, fn))
    return sorted(out)

def debug_df_info(tag: str, df: Optional[pd.DataFrame], col: Optional[str] = None):
    if df is None or df.empty:
        log(f"[DF] {tag}: EMPTY")
        return
    non = df[col].notna().sum() if (col and col in df.columns) else df.drop(columns=["date"], errors="ignore").notna().sum().sum()
    d0, d1 = df["date"].min(), df["date"].max()
    log(f"[DF] {tag}: shape={df.shape}, date=[{d0} .. {d1}], non_null={non}, col='{col or 'NA'}'")

# ----------------------------- Load ACTUALS (labels) -----------------------------

def load_actual_RK(root: str, freq: str, h: int) -> Optional[pd.DataFrame]:
    base = os.path.join(root, DIR_ACTUAL_CUM)
    toks = ["RK", freq, f"h{h}"]
    cand = find_files_by_tokens(base, toks)
    log(f"[ACTUAL RK] tokens={toks} -> {len(cand)} hit(s)")
    if not cand:
        return None
    path = cand[0]
    df = read_parquet(path)
    if df is None:
        log(f"[ACTUAL RK] Failed to read {path}")
        return None
    prefer = [ "rk", "sigma", f"h{h}" ]
    col = pick_first_float_col(df, prefer_contains=prefer) or pick_first_float_col(df)
    if col is None:
        log(f"[ACTUAL RK] No numeric column in {path}. Columns={list(df.columns)}")
        return None
    vals = df[col].to_numpy()
    picked = col.lower()
    if ("var" in picked or "variance" in picked) and "sigma" not in picked:
        log(f"[ACTUAL RK] Detected variance column -> sqrt to σ")
        vals = sqrt_clip(vals)
    out = df[["date"]].copy()
    out["actual"] = vals
    log(f"[ACTUAL RK] Using {os.path.basename(path)} :: column='{col}'")
    debug_df_info("actual_RK", out, "actual")
    return out.dropna(subset=["date", "actual"])

def load_actual_CBV(root: str, asset: str, freq: str, h: int) -> Optional[pd.DataFrame]:
    base = os.path.join(root, DIR_ACTUAL_CUM)
    cand = find_files_by_tokens(base, [asset, "bv", freq, f"h{h}"]) or \
           find_files_by_tokens(base, ["bv", freq, f"h{h}"])
    log(f"[ACTUAL CBV] search -> {len(cand)} hit(s)")
    if not cand:
        return None
    path = cand[0]
    df = read_parquet(path)
    if df is None:
        log(f"[ACTUAL CBV] Failed to read {path}")
        return None
    prefer_sets = [
        ["bv", "sigma", f"h{h}"],
        ["cbv", "sigma", f"h{h}"],
        ["jv", "sigma", f"h{h}"],   # your example: JV_sigma_cumulative_h*
    ]
    col = None
    for pref in prefer_sets:
        col = pick_first_float_col(df, prefer_contains=pref)
        if col: break
    col = col or pick_first_float_col(df)
    if col is None:
        log(f"[ACTUAL CBV] No numeric column in {path}. Columns={list(df.columns)}")
        return None
    vals = df[col].to_numpy()
    picked = col.lower()
    if ("var" in picked or "variance" in picked) and "sigma" not in picked:
        log(f"[ACTUAL CBV] Detected variance column -> sqrt to σ")
        vals = sqrt_clip(vals)
    out = df[["date"]].copy()
    out["actual"] = vals
    log(f"[ACTUAL CBV] Using {os.path.basename(path)} :: column='{col}'")
    debug_df_info("actual_CBV", out, "actual")
    return out.dropna(subset=["date", "actual"])

# ----------------------------- Load FORECASTS (with unit alignment) -----------------------------

def load_forecast_ANN(root, asset, measure_token, freq, h) -> Optional[pd.DataFrame]:
    # h=1
    if h == 1:
        fn = f"{asset}_{measure_token}_daily_{freq}_ANN_1step_test.parquet"
        path = os.path.join(root, DIR_ANN_FINAL, fn)
        df = read_parquet(path)
        if df is not None:
            col = pick_first_float_col(df, prefer_contains=[measure_token.lower(), "ann", "1step"]) or pick_first_float_col(df)
            if col:
                out = df[["date", col]].copy(); out.columns = ["date", "ANN"]
                log(f"[ANN h=1] {os.path.basename(path)} -> col='{col}' (σ as-is)")
                debug_df_info("ANN_h1", out, "ANN")
                return out.dropna()
    # h>1 cumulative (mean ν²) -> sqrt to RMS σ
    fn_cum = f"{asset}_{measure_token}_daily_{freq}_ANN_cumulative_h{h}_test.parquet"
    for base in [DIR_ANN_FINAL, DIR_ANN_CUM2]:
        path = os.path.join(root, base, fn_cum)
        df = read_parquet(path)
        if df is not None:
            col = pick_first_float_col(df, prefer_contains=[measure_token.lower(), "cumulative", f"h{h}"]) or pick_first_float_col(df)
            if col:
                s = sqrt_clip(df[col].to_numpy())
                out = df[["date"]].copy(); out["ANN"] = s
                log(f"[ANN h={h}] {os.path.basename(path)} -> col='{col}' treated as mean(ν²) -> sqrt to RMS σ")
                debug_df_info(f"ANN_h{h}", out, "ANN")
                return out.dropna()
    log(f"[ANN] Not found for measure={measure_token}, freq={freq}, h={h}")
    return None

def load_forecast_ARFIMA(root, measure_token, freq, h) -> Optional[pd.DataFrame]:
    base = os.path.join(root, DIR_ARFIMA)
    # h=1
    path1 = os.path.join(base, f"ARFIMA_{measure_token}_{freq}_forecast.parquet")
    if h == 1:
        df = read_parquet(path1)
        if df is None:
            log(f"[ARFIMA h=1] Missing {path1}")
            return None
        col = "forecast" if "forecast" in df.columns else pick_first_float_col(df)
        if col is None:
            log(f"[ARFIMA h=1] No numeric column in {path1}")
            return None
        out = df[["date", col]].copy(); out.columns = ["date", "ARFIMA"]
        log(f"[ARFIMA h=1] {os.path.basename(path1)} -> col='{col}' (σ as-is)")
        debug_df_info("ARFIMA_h1", out, "ARFIMA")
        return out.dropna()
    # h>1: try explicit _cumulative_h{h}, else generic cumulative
    paths = [
        os.path.join(base, f"ARFIMA_{measure_token}_{freq}_cumulative_h{h}.parquet"),
        os.path.join(base, f"ARFIMA_{measure_token}_{freq}_cumulative.parquet"),
    ]
    df = None; used = None
    for p in paths:
        df = read_parquet(p)
        if df is not None:
            used = p; break
    if df is None:
        log(f"[ARFIMA h={h}] Missing both specific and generic cumulative files.")
        return None
    # choose column and unit alignment
    candidates = [
        f"{measure_token}_cumVolRMS_h{h}",  # σ
        f"{measure_token}_cumulative_h{h}",
        f"{measure_token}_cumVar_h{h}",     # ν² -> sqrt
        f"cumVolRMS_h{h}",
        f"cumVar_h{h}",
    ]
    col = None
    for c in candidates:
        if c in df.columns:
            col = c; break
    col = col or pick_first_float_col(df, prefer_contains=[f"h{h}"]) or pick_first_float_col(df)
    if col is None:
        log(f"[ARFIMA h={h}] No suitable column in {used}. Columns={list(df.columns)}")
        return None
    picked = col.lower()
    s = df[col].to_numpy()
    if ("cumvar" in picked or "variance" in picked) and "rms" not in picked:
        log(f"[ARFIMA h={h}] {os.path.basename(used)} :: col='{col}' indicates variance -> sqrt to σ")
        s = sqrt_clip(s)
    else:
        log(f"[ARFIMA h={h}] {os.path.basename(used)} :: col='{col}' treated as σ (RMS already)")
    out = df[["date"]].copy(); out["ARFIMA"] = s
    debug_df_info(f"ARFIMA_h{h}", out, "ARFIMA")
    return out.dropna()

def load_forecast_HAR(root, measure_token, freq, h) -> Optional[pd.DataFrame]:
    base = os.path.join(root, DIR_HAR)
    if h == 1:
        path = os.path.join(base, f"HAR_{measure_token}_{freq}_forecast.parquet")
        df = read_parquet(path)
        if df is None:
            log(f"[HAR h=1] Missing {path}")
            return None
        col = "forecast" if "forecast" in df.columns else pick_first_float_col(df)
        if col is None:
            log(f"[HAR h=1] No numeric column in {path}")
            return None
        out = df[["date", col]].copy(); out.columns = ["date", "HAR"]
        log(f"[HAR h=1] {os.path.basename(path)} -> col='{col}' (σ as-is)")
        debug_df_info("HAR_h1", out, "HAR")
        return out.dropna()
    else:
        path = os.path.join(base, f"HAR_{measure_token}_{freq}_cumulative_h{h}.parquet")
        df = read_parquet(path)
        if df is None:
            log(f"[HAR h={h}] Missing {path}")
            return None
        col = f"cumulative_h{h}" if f"cumulative_h{h}" in df.columns else pick_first_float_col(df)
        if col is None:
            log(f"[HAR h={h}] No suitable column in {path}")
            return None
        out = df[["date", col]].copy(); out.columns = ["date", "HAR"]
        log(f"[HAR h={h}] {os.path.basename(path)} -> col='{col}' (σ as-is)")
        debug_df_info(f"HAR_h{h}", out, "HAR")
        return out.dropna()

def load_forecast_GARCH(root, h) -> Optional[pd.DataFrame]:
    base = os.path.join(root, DIR_GARCH)
    # h=1: cumulative_h1 file has σ
    if h == 1:
        p1 = os.path.join(base, "garch_11_test_cumulative_h1.parquet")
        df = read_parquet(p1)
        if df is not None:
            # e.g., 'garch_11' or any numeric
            col = None
            for c in df.columns:
                if "garch_11" in str(c).lower():
                    col = c; break
            col = col or pick_first_float_col(df)
            if col:
                out = df[["date", col]].copy(); out.columns = ["date", "GARCH"]
                log(f"[GARCH h=1] {os.path.basename(p1)} -> col='{col}' (σ as-is)")
                debug_df_info("GARCH_h1", out, "GARCH")
                return out.dropna()
        log(f"[GARCH h=1] Missing or unreadable: {p1}")
        return None

    # h>1: try to rebuild RMS σ from single-step horizons
    single_path = os.path.join(base, "garch_11_test.parquet")
    df_single = read_parquet(single_path)
    if df_single is not None:
        cols = [c for c in df_single.columns if re.match(r"garch_forecast_h\d+$", str(c))]
        want = [f"garch_forecast_h{i}" for i in range(1, h + 1)]
        have = [c for c in want if c in cols]
        if len(have) == h:
            stack = df_single[have].to_numpy()
            rms = np.sqrt(np.mean(np.square(stack), axis=1))
            out = df_single[["date"]].copy(); out["GARCH"] = rms
            log(f"[GARCH h={h}] Rebuilt RMS σ from {os.path.basename(single_path)} using columns {have}")
            debug_df_info(f"GARCH_h{h}_RMS", out, "GARCH")
            return out.dropna()
        else:
            log(f"[WARN][GARCH h={h}] Single-step file found but missing some horizons {want}; fallback to cumulative.")
    else:
        log(f"[WARN][GARCH h={h}] Single-step file not found; fallback to cumulative.")

    # Fallback: cumulative file (σ,但口径可能为mean-σ，作为次优)
    p = os.path.join(base, f"garch_11_test_cumulative_h{h}.parquet")
    df = read_parquet(p)
    if df is None:
        log(f"[GARCH h={h}] Missing {p}")
        return None
    # examples: 'garch_forecast_h1_cumulative_h5' or any numeric
    prefer = [f"cumulative_h{h}"]
    col = pick_first_float_col(df, prefer_contains=prefer)
    if col is None:
        for c in df.columns:
            lc = str(c).lower()
            if "garch" in lc and (pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])):
                col = c; break
    col = col or pick_first_float_col(df)
    if col is None:
        log(f"[GARCH h={h}] No suitable column in {p}. Columns={list(df.columns)}")
        return None
    out = df[["date", col]].copy(); out.columns = ["date", "GARCH"]
    log(f"[GARCH h={h}] Using {os.path.basename(p)} -> col='{col}' (fallback cumulative σ)")
    debug_df_info(f"GARCH_h{h}_fallback", out, "GARCH")
    return out.dropna()

def build_forecast_panel(root, asset, measure, freq, h, model_cols: List[str]) -> pd.DataFrame:
    meas_token = "BV" if measure == "CBV" else measure
    pieces = []
    if "ANN" in model_cols:
        df = load_forecast_ANN(root, asset, meas_token, freq, h)
        if df is not None: pieces.append(df)
    if "ARFIMA" in model_cols:
        df = load_forecast_ARFIMA(root, meas_token, freq, h)
        if df is not None: pieces.append(df)
    if "HAR" in model_cols:
        df = load_forecast_HAR(root, meas_token, freq, h)
        if df is not None: pieces.append(df)
    if "GARCH" in model_cols:
        df = load_forecast_GARCH(root, h)
        if df is not None: pieces.append(df)

    if not pieces:
        log(f"[PANEL] No forecasts found (measure={measure}, freq={freq}, h={h})")
        return pd.DataFrame(columns=["date"] + model_cols)

    panel = pieces[0]
    for df in pieces[1:]:
        panel = panel.merge(df, on="date", how="outer")
    panel = panel.sort_values("date").reset_index(drop=True)

    if "HAR-ANN" in model_cols:
        if "HAR" in panel.columns and "ANN" in panel.columns:
            panel["HAR-ANN"] = 0.5 * (panel["HAR"] + panel["ANN"])
            log(f"[HAR-ANN] Built (σ-aligned) for measure={measure}, freq={freq}, h={h}")
        else:
            panel["HAR-ANN"] = np.nan
            log(f"[HAR-ANN] Missing HAR or ANN -> NaN")

    keep = ["date"] + [c for c in model_cols if c in panel.columns]
    for c in model_cols:
        if c not in panel.columns:
            panel[c] = np.nan
            keep.append(c)
    debug_df_info(f"panel_{measure}_h{h}_{freq}", panel)
    return panel[keep]

# ----------------------------- Metrics -----------------------------

def loss_components(pred: np.ndarray, actual: np.ndarray):
    err = pred - actual
    rmse_obs = err ** 2
    mae_obs = np.abs(err)
    under = (err < 0).astype(int)
    over  = (err > 0).astype(int)
    mme_o_obs = np.zeros_like(err)
    mme_u_obs = np.zeros_like(err)
    mme_o_obs[err < 0] = np.abs(err[err < 0])
    mme_o_obs[err > 0] = np.sqrt(np.abs(err[err > 0]))
    mme_u_obs[err < 0] = np.sqrt(np.abs(err[err < 0]))
    mme_u_obs[err > 0] = np.abs(err[err > 0])
    return {
        "RMSE": float(np.sqrt(np.mean(rmse_obs))) if rmse_obs.size else np.nan,
        "MAE": float(np.mean(mae_obs)) if mae_obs.size else np.nan,
        "MME_O": float(np.mean(mme_o_obs)) if mme_o_obs.size else np.nan,
        "MME_U": float(np.mean(mme_u_obs)) if mme_u_obs.size else np.nan,
        "OverRate": float(np.mean(over)) if over.size else np.nan,
        "UnderRate": float(np.mean(under)) if under.size else np.nan,
        "N": int(actual.size),
    }

def compute_metrics_row(root, asset, measure, freq, h, period, models: List[str]) -> Dict[str, Dict[str, float]]:
    if measure == "RK":
        df_a = load_actual_RK(root, freq, h)
    else:
        df_a = load_actual_CBV(root, asset, freq, h)
    panel = build_forecast_panel(root, asset, measure, freq, h, models)

    if df_a is None or panel.empty:
        log(f"[METRICS] Skip measure={measure}, no actuals/panel.")
        return {m: {k: np.nan for k in ["RMSE","MAE","MME_O","MME_U","OverRate","UnderRate","N"]} for m in models}

    df = df_a.merge(panel, on="date", how="inner").dropna(subset=["actual"])
    if df.empty:
        log(f"[METRICS] Empty after merge (measure={measure}).")
        return {m: {k: np.nan for k in ["RMSE","MAE","MME_O","MME_U","OverRate","UnderRate","N"]} for m in models}

    sample_min, sample_max = df["date"].min(), df["date"].max()
    per = auto_periods(sample_min, sample_max)
    if period not in per:
        start, end = sample_min, sample_max
    else:
        start, end = per[period]
    sub = df[(df["date"] >= start) & (df["date"] <= end)].copy()

    log(f"[METRICS] measure={measure}, period={period}, "
        f"sample=[{sample_min.date()} .. {sample_max.date()}], "
        f"use=[{start.date()} .. {end.date()}], N={sub.shape[0]}")

    out: Dict[str, Dict[str, float]] = {}
    for model in models:
        if model not in sub.columns:
            out[model] = {k: np.nan for k in ["RMSE","MAE","MME_O","MME_U","OverRate","UnderRate","N"]}
            continue
        tmp = sub[["actual", model]].dropna()
        if tmp.empty:
            out[model] = {k: np.nan for k in ["RMSE","MAE","MME_O","MME_U","OverRate","UnderRate","N"]}
        else:
            res = loss_components(tmp[model].to_numpy(), tmp["actual"].to_numpy())
            out[model] = res
            log("  [VAL] {model}: " +
                ", ".join([f"{k}={res[k]:.6g}" if k!='N' else f"N={res[k]}" for k in ["RMSE","MAE","MME_O","MME_U","OverRate","UnderRate","N"]]).format(model=model))
    return out

# ----------------------------- CSV patching -----------------------------

def sniff_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        head = f.readline()
    return "\t" if head.count("\t") >= head.count(",") else ","

def parse_csv_name(fname: str):
    m = CSV_RE.match(fname)
    if not m: return None
    d = m.groupdict()
    d["h"] = int(d["h"])
    return d  # asset, period, h, freq, metric

def fill_or_append_row(df_csv: pd.DataFrame, new_vals: Dict[str, float], models: List[str], measure_name: str) -> Tuple[pd.DataFrame, bool]:
    changed = False
    mask = df_csv["measure"].astype(str).str.strip().str.upper() == measure_name.upper()
    if mask.any():
        idx = df_csv.index[mask][0]
        for m in models:
            if m not in df_csv.columns:
                continue
            if pd.isna(df_csv.at[idx, m]):
                df_csv.at[idx, m] = new_vals.get(m, np.nan)
                changed = True
        return df_csv, changed
    else:
        row = {"measure": measure_name}
        for m in models:
            row[m] = new_vals.get(m, np.nan)
        df_csv = pd.concat([df_csv, pd.DataFrame([row])], ignore_index=True)
        changed = True
        return df_csv, changed

def patch_one_csv(root: str, csv_path: str) -> bool:
    fname = os.path.basename(csv_path)
    info = parse_csv_name(fname)
    if not info:
        log(f"[SKIP] Not a metrics csv name (regex mismatch): {fname}")
        return False
    log(f"\n=== Patch: {fname} ===")
    log(f"[PARSE] asset={info['asset']} period={info['period']} h={info['h']} freq={info['freq']} metric={info['metric']}")

    sep = sniff_sep(csv_path)
    df_csv = pd.read_csv(csv_path, sep=sep)
    if "measure" not in df_csv.columns:
        mcol = [c for c in df_csv.columns if c.lower()=="measure"]
        if mcol: df_csv = df_csv.rename(columns={mcol[0]:"measure"})
        else:
            log(f"[SKIP] No 'measure' column in {fname}")
            return False

    model_cols = [c for c in df_csv.columns if c != "measure"]
    log(f"[CSV] models={model_cols}")

    changed_any = False
    for meas in MEASURES_TO_HANDLE:
        stats_dict = compute_metrics_row(
            root=root,
            asset=info["asset"],
            measure=meas,
            freq=info["freq"],
            h=info["h"],
            period=info["period"],
            models=model_cols
        )
        metric_name = info["metric"]
        vals = {m: stats_dict.get(m, {}).get(metric_name, np.nan) for m in model_cols}
        log(f"[WRITE] measure={meas}, metric={metric_name}, values={vals}")
        df_csv, changed = fill_or_append_row(df_csv, vals, model_cols, meas)
        changed_any = changed_any or changed

    if changed_any:
        df_csv.to_csv(csv_path, index=False, sep=sep, float_format="%.6g")
        log(f"[UPDATED] {csv_path}")
    else:
        log(f"[NO CHANGE] {fname}: RK/CBV already populated.")

    return changed_any

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Fill/append RK & CBV rows in metrics CSVs (backup folder only).")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args()

    backup_dir = os.path.join(args.root, BACKUP_DIR)
    if not os.path.isdir(backup_dir):
        log(f"[ERR] Backup folder not found: {backup_dir}")
        return

    files = [f for f in os.listdir(backup_dir) if f.endswith(".csv") and "__metrics__" in f]
    files.sort()
    if not files:
        log("[ERR] No metrics CSVs in backup folder.")
        return

    log(f"[INFO] Found {len(files)} metrics CSV(s) in backup folder.")
    touched = 0
    for f in files:
        path = os.path.join(backup_dir, f)
        try:
            if patch_one_csv(args.root, path):
                touched += 1
        except Exception as e:
            log(f"[ERR] Failed to patch {f}: {e}")

    if touched == 0:
        log("Done. No files required changes (RK/CBV rows already filled).")
    else:
        log(f"Done. Patched {touched} file(s).")

if __name__ == "__main__":
    main()
