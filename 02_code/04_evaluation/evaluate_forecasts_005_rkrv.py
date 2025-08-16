#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add missing RK & CBV rows into previously generated metrics CSVs
under 03_results/evaluation_reports_BackUp/, without touching other rows.

- Reuses the same data sources and alignment logic as evaluate_forecasts_002_alingment.py
- Only computes rows for measures in {"RK","CBV"} and appends them if missing
- Keeps existing model columns exactly as in each CSV (fills NaN when forecast not found)
- Safe defaults for PyCharm on Windows; no CLI args required
"""

from __future__ import annotations
import os, re
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# -----------------------------
# Project defaults (match your repo)
# -----------------------------
ROOT = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
BACKUP_DIR = os.path.join(ROOT, r"03_results/evaluation_reports_BackUp")

ASSETS = ["CL_WTI"]          # 只补你目前的资产；需要更多可扩展
FREQS = ["1min", "5min"]     # 文件名中会解析
MEASURES_TO_ADD = ["RK", "CBV"]
MODELS_ALL = ["GARCH", "ARFIMA", "HAR", "ANN", "HAR-ANN"]
H_LIST = [1, 5, 10]          # 文件名中会解析

# -----------------------------
# Paper windows (fixed dates)
# -----------------------------
PRE_START = pd.Timestamp("2006-07-06")
PRE_END   = pd.Timestamp("2008-08-31")
CRI_START = pd.Timestamp("2008-09-01")
CRI_END   = pd.Timestamp("2010-10-31")
POST_START= pd.Timestamp("2010-11-01")
POST_END  = pd.Timestamp("2012-12-31")
PERIODS_PAPER = {
    "pre":    (PRE_START, PRE_END),
    "crisis": (CRI_START, CRI_END),
    "post":   (POST_START, POST_END),
}

# -----------------------------
# Measure columns for actuals & filename tokens
# -----------------------------
MEASURE_COLS: Dict[str, List[str]] = {
    "TSRV":   ["TSRV"],
    "RV":     ["RV"],
    "RK":     ["RK"],
    "JWTSRV": ["JWTSRV"],
    "CBV":    ["BV", "CBV"],   # 代码里 CBV，有些文件名/列名是 BV
    "MedRV":  ["MedRV"],
}
MEASURE_FILE_TOKENS: Dict[str, List[str]] = {
    "CBV":    ["BV", "CBV"],
    "TSRV":   ["TSRV"],
    "RV":     ["RV"],
    "RK":     ["RK"],
    "JWTSRV": ["JWTSRV"],
    "MedRV":  ["MedRV"],
}

# -----------------------------
# File location patterns (复制并精简自你的脚本)
# -----------------------------
MEAS_FILE_1H = "03_results/intermediate_results/volatility_estimates/test_set/{asset}_{measure}_daily_{freq}_test.parquet"
#MEAS_FILE_CUM = "03_results/intermediate_results/volatility_estimates/cumulative_forecasts/{asset}_{measure}_daily_{freq}_test_test_cumulative_h{h}.parquet"
MEAS_FILE_CUM = "03_results/intermediate_results/volatility_estimates/cumulative_forecasts/{asset}_{measure}_daily_{freq}_test_cumulative_h{h}.parquet"

ANN_CUM_FALLBACK_DIR = r"03_results/intermediate_results/cumulative_forecasts/ANN"

PATTERNS = {
    "ANN": [
        r"{asset}_{measure}_daily_{freq}_ANN_1step_test.parquet",
        r"{asset}_{measure}_daily_{freq}_ANN_cumulative_h{h}_test.parquet",
        r"CL_WTI_{measure}_daily_{freq}_ANN_1step_test.parquet",
    ],
    "HAR": [
        r"HAR_{measure}_{freq}_forecast.parquet",
        r"HAR_{measure}_{freq}_cumulative_h{h}.parquet",
        r"{asset}_{measure}_daily_{freq}_HAR_1step_test.parquet",
    ],
    "ARFIMA": [
        r"ARFIMA_{measure}_{freq}_forecast.parquet",
        r"ARFIMA_{measure}_{freq}_cumulative_h{h}.parquet",
        r"{asset}_{measure}_daily_{freq}_ARFIMA_1step_test.parquet",
    ],
    "GARCH": [
        r"garch_11_test_cumulative_h{h}.parquet",
        r"garch_11_1step_test.parquet",
        r"garch_11_test.parquet",
    ],
    "HAR-ANN": [],  # 组合，运行时构造
}
MODEL_DIR = {
    "ANN": "03_results/final_forecasts/ANN",
    "HAR": "03_results/final_forecasts/HAR",
    "ARFIMA": "03_results/final_forecasts/ARFIMA",
    "GARCH": "03_results/final_forecasts/GARCH",
    "HAR-ANN": "03_results/final_forecasts/ANN",  # 不读取，仅占位
}

# -----------------------------
# Utils
# -----------------------------
def try_load_parquet(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        try:
            df = pd.read_parquet(path)
            return parse_dates(df)
        except Exception as e:
            print(f"[READ FAIL] {path}: {e}")
            return None
    return None

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    # 与原脚本一致的尽量鲁棒的日期识别
    direct = [c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")]
    if direct:
        col = direct[0]
        out = df.copy()
        out[col] = pd.to_datetime(out[col], errors="coerce")
        return out.rename(columns={col: "date"}).sort_values("date").reset_index(drop=True)
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.reset_index().rename(columns={df.index.name or "index": "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values("date").reset_index(drop=True)
    try:
        idx_parsed = pd.to_datetime(df.index, errors="coerce")
        if idx_parsed.notna().mean() > 0.8:
            out = df.copy(); out.insert(0, "date", idx_parsed)
            return out.sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        if s.notna().mean() > 0.8:
            out = df.copy(); out["date"] = s
            return out.sort_values("date").reset_index(drop=True)
    if "Unnamed: 0" in df.columns:
        s = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
        if s.notna().any():
            out = df.rename(columns={"Unnamed: 0": "date"}).copy(); out["date"] = s
            return out.sort_values("date").reset_index(drop=True)
    raise ValueError("No usable date column/index found.")

def pick_forecast_column(df: pd.DataFrame) -> str:
    candidates = [
        "forecast", "yhat", "pred", "prediction", "value", "vol_forecast",
        "nu_hat", "nu_pred", "vhat", "fcast",
        "cumVar", "cumVolRMS"
    ]
    for c in candidates:
        if c in df.columns: return c
    # 特例：ANN 一步列名如 RK_ann_1step / BV_ann_1step
    one_float = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if one_float:
        return one_float[0]
    raise ValueError("Could not identify forecast column.")

def sqrt_clip(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(x, 0, None))

# -----------------------------
# Load actuals (already σ in your latest)
# -----------------------------
def load_actual(root: str, asset: str, measure: str, freq: str, h: int) -> Optional[pd.DataFrame]:
    if h == 1:
        path = os.path.join(root, MEAS_FILE_1H.format(asset=asset, measure=measure, freq=freq))
    else:
        path = os.path.join(root, MEAS_FILE_CUM.format(asset=asset, measure=measure, freq=freq, h=h))
    df = try_load_parquet(path)
    if df is None: return None
    candidates = MEASURE_COLS.get(measure, [measure])
    hit = [c for c in candidates if c in df.columns]
    if hit:
        actual_col = hit[0]
    else:
        # 兜底：取首个浮点列
        float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        actual_col = float_cols[0] if float_cols else df.columns[1]
    out = df[["date", actual_col]].rename(columns={actual_col: "actual"}).copy()
    return parse_dates(out)

# -----------------------------
# Load forecasts (convert to σ by model rules)
# -----------------------------
def glob_candidates(dirpath: str, regex: str) -> List[str]:
    pat = re.compile(regex)
    if not os.path.isdir(dirpath):
        return []
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath) if pat.match(f)]

def load_forecast_generic(base: str, regex: str, model: str) -> Optional[pd.DataFrame]:
    for path in glob_candidates(base, regex):
        df = try_load_parquet(path)
        if df is not None:
            col = pick_forecast_column(df)
            return df[["date", col]].rename(columns={col: model})
    return None

def load_forecast(root: str, model: str, asset: str, measure: str, freq: str, h: int) -> Optional[pd.DataFrame]:
    if model == "HAR-ANN":
        return None  # 稍后组合
    base = os.path.join(root, MODEL_DIR[model])

    def try_patterns(tokens: List[str]) -> Optional[pd.DataFrame]:
        for token in tokens:
            for pat in PATTERNS.get(model, []):
                fname = pat.format(asset=asset, measure=token, freq=freq, h=h)
                path = os.path.join(base, fname)
                df = try_load_parquet(path)
                if df is not None:
                    col = pick_forecast_column(df)
                    return df[["date", col]].rename(columns={col: model})
        return None

    tokens = MEASURE_FILE_TOKENS.get(measure, [measure])
    df = try_patterns(tokens)
    if df is None:
        # 宽松匹配
        for token in tokens:
            regex = rf".*{token}.*{freq}.*(h{h}|1step).*\.parquet$"
            df = load_forecast_generic(base, regex, model)
            if df is not None: break
    if df is None and model == "GARCH":
        df = load_forecast_generic(base, r"garch_11_test\.parquet$", model)
    if df is None:
        return None
    # 2) 若 ANN 且累计、主目录没找到，则去备选目录再搜一遍
    if df is None and (model == "ANN") and (h > 1):
        alt_base = os.path.join(root, ANN_CUM_FALLBACK_DIR)
        for token in MEASURE_FILE_TOKENS.get(measure, [measure]):
            # 文件名固定模式：{asset}_{measure}_daily_{freq}_ANN_cumulative_h{h}_test.parquet
            alt_name = f"{asset}_{token}_daily_{freq}_ANN_cumulative_h{h}_test.parquet"
            alt_path = os.path.join(alt_base, alt_name)
            tmp = try_load_parquet(alt_path)
            if tmp is not None:
                col = pick_forecast_column(tmp)
                df = tmp[["date", col]].rename(columns={col: model})
                break
    # 单位对齐到 σ（标准差）
    s = df[model].to_numpy()
    if model == "ANN":
        if h > 1:  # 累计文件为 mean(ν²)，取 sqrt 为 RMS σ
            s = sqrt_clip(s)
    elif model == "ARFIMA":
        if h > 1:
            colname = df.columns[1].lower()
            if ("cumvar" in colname) or ("variance" in colname) or re.search(r"\bcum.*var\b", colname):
                s = sqrt_clip(s)
    elif model == "GARCH":
        if h == 1:
            pass
        else:
            src = try_load_parquet(os.path.join(base, "garch_11_test.parquet"))
            if src is not None:
                cols = [c for c in src.columns if re.match(r"garch_forecast_h\d+", c)]
                if cols:
                    want = [f"garch_forecast_h{i}" for i in range(1, h + 1)]
                    have = [c for c in want if c in cols]
                    if len(have) == h:
                        src = parse_dates(src)
                        rms = np.sqrt(np.mean(np.square(src[have].to_numpy()), axis=1))
                        df = pd.DataFrame({"date": src["date"], model: rms})
                        s = df[model].to_numpy()
                    else:
                        print(f"[WARN] GARCH single-step missing horizons for h={h}, fallback to provided cumulative.")
                else:
                    print("[WARN] GARCH single-step lacks horizon columns; fallback to provided cumulative.")
    elif model == "HAR":
        pass  # HAR 累计/一步均已是 σ

    out = df[["date"]].copy()
    out[model] = s
    return out

# -----------------------------
# Losses aggregation (与原脚本一致)
# -----------------------------
def loss_components(pred: np.ndarray, actual: np.ndarray) -> Dict[str, np.ndarray]:
    err = pred - actual
    rmse = (err ** 2)
    mae = np.abs(err)
    under = err < 0
    over = err > 0
    mme_o = np.empty_like(err); mme_u = np.empty_like(err)
    mme_o[under] = np.abs(err[under]);  mme_o[over] = np.sqrt(np.abs(err[over]));  mme_o[~(under|over)] = 0.0
    mme_u[under] = np.sqrt(np.abs(err[under])); mme_u[over] = np.abs(err[over]);   mme_u[~(under|over)] = 0.0
    return {"rmse_obs": rmse, "mae_obs": mae, "mme_o_obs": mme_o, "mme_u_obs": mme_u,
            "over": over.astype(int), "under": under.astype(int)}

def aggregate_losses(loss_obs: Dict[str, np.ndarray]) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean(loss_obs["rmse_obs"])))
    mae = float(np.mean(loss_obs["mae_obs"]))
    mme_o = float(np.mean(loss_obs["mme_o_obs"]))
    mme_u = float(np.mean(loss_obs["mme_u_obs"]))
    over_rate = float(np.mean(loss_obs["over"]))
    under_rate = float(np.mean(loss_obs["under"]))
    return {"RMSE": rmse, "MAE": mae, "MME_O": mme_o, "MME_U": mme_u,
            "OverRate": over_rate, "UnderRate": under_rate, "N": int(loss_obs["mae_obs"].shape[0])}

# -----------------------------
# Period resolver
# -----------------------------
def get_period_bounds(period_name: str,
                      sample_min: Optional[pd.Timestamp],
                      sample_max: Optional[pd.Timestamp]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    pname = period_name.lower()
    if pname in PERIODS_PAPER:
        return PERIODS_PAPER[pname]
    # whole 或其它：用样本范围兜底
    if sample_min is None or sample_max is None:
        raise ValueError("Cannot infer whole-period bounds without sample range.")
    return sample_min, sample_max

# -----------------------------
# Core: compute one measure row for one CSV
# -----------------------------
def compute_measure_row(asset: str, freq: str, h: int,
                        period_name: str, metric: str, measure: str,
                        root: str, model_cols: List[str]) -> pd.Series:
    # 1) actual
    df_a = load_actual(root, asset, measure, freq, h)
    if df_a is None or df_a.empty:
        return pd.Series(dtype=float, name=measure)

    # 2) forecasts per model
    panel = df_a.copy()
    found_any = False
    for model in model_cols:
        if model not in MODELS_ALL:  # 忽略未知列
            continue
        df_f = load_forecast(root, model, asset, measure, freq, h)
        if df_f is not None:
            found_any = True
            panel = panel.merge(df_f, on="date", how="left")
    if not found_any:
        return pd.Series(dtype=float, name=measure)

    # 3) HAR-ANN 组合（若两者都有且列存在）
    if ("HAR" in panel.columns) and ("ANN" in panel.columns) and ("HAR-ANN" in model_cols):
        panel["HAR-ANN"] = 0.5 * (panel["HAR"] + panel["ANN"])

    # 4) 时段裁剪
    sample_min = panel["date"].min()
    sample_max = panel["date"].max()
    start, end = get_period_bounds(period_name, sample_min, sample_max)
    sub = panel[(panel["date"] >= start) & (panel["date"] <= end)].copy()

    # 5) 逐模型计算指标
    results: Dict[str, float] = {}
    for model in model_cols:
        if model not in sub.columns or model in ("date", "actual"):
            results[model] = np.nan
            continue
        tmp = sub[["actual", model]].dropna()
        if tmp.empty:
            results[model] = np.nan
            continue
        comp = loss_components(tmp[model].to_numpy(), tmp["actual"].to_numpy())
        agg = aggregate_losses(comp)
        results[model] = float(agg.get(metric, np.nan))
    s = pd.Series(results, name=measure)
    return s

# -----------------------------
# Parse CSV filename
# -----------------------------
FNAME_RE = re.compile(
    r"^(?P<asset>.+?)__metrics__"
    r"(?P<period>[^_]+)__h(?P<h>\d+)__"
    r"(?P<freq>[^_]+)__"
    r"(?P<metric>RMSE|MAE|MME_O|MME_U|OverRate|UnderRate|N)\.csv$"
)



def parse_filename(fn: str):
    m = FNAME_RE.match(fn)
    if not m:
        return None
    g = m.groupdict()
    return g["asset"], g["period"], int(g["h"]), g["freq"], g["metric"]

# -----------------------------
# Main walker
# -----------------------------
def main():
    if not os.path.isdir(BACKUP_DIR):
        raise FileNotFoundError(f"Backup folder not found: {BACKUP_DIR}")

    files = [f for f in os.listdir(BACKUP_DIR) if f.endswith(".csv")]
    target_files = []
    for f in files:
        parsed = parse_filename(f)
        if parsed is None:
            continue
        # 只处理我们关心的资产/频率/步长
        asset, period, h, freq, metric = parsed
        if (asset in ASSETS) and (freq in FREQS) and (h in H_LIST):
            target_files.append((f, parsed))
    print(f"[INFO] Found {len(target_files)} metrics CSV(s) to check in backup folder.")

    if not target_files:
        print("No matching metrics CSVs found in backup folder.")
        return

    for fname, (asset, period, h, freq, metric) in sorted(target_files):
        path = os.path.join(BACKUP_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[SKIP] Cannot read {fname}: {e}")
            continue

        # 规范化列名（第一列应为 measure）
        if df.columns[0] != "measure":
            df = df.rename(columns={df.columns[0]: "measure"})
        model_cols = [c for c in df.columns if c != "measure"]

        # 已有行集合
        existing_measures = set(str(x).strip() for x in df["measure"].astype(str).tolist())

        need_add = [m for m in MEASURES_TO_ADD if m not in existing_measures]
        if not need_add:
            # 已都有，略过
            continue

        print(f"[{fname}] add rows for: {need_add}")

        new_rows = []
        for measure in need_add:
            s = compute_measure_row(asset, freq, h, period, metric, measure, ROOT, model_cols)
            # 确保列对齐
            s = s.reindex(model_cols)
            s = s.rename(measure)
            new_rows.append(s)

        if new_rows:
            add_df = pd.DataFrame(new_rows).reset_index().rename(columns={"index": "measure"})
            out = pd.concat([df, add_df], axis=0, ignore_index=True)
            # 按原顺序保存（覆盖原文件）
            out = out[["measure"] + model_cols]
            out.to_csv(path, index=False, float_format="%.6g")
            print(f"  -> updated: {path}")

    print("Done. RK & CBV rows have been appended where missing.")

if __name__ == "__main__":
    main()
