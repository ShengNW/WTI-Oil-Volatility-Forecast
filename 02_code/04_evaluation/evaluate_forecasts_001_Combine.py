#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate volatility forecasts exactly following Baruník & Křehlič (2016)
for the "WTI_volatility_forecast_replication" project structure.

Goal:
- 保留你“后者”的 **算法选择**（严格按论文划分 pre/crisis/post 且默认只用论文窗口），
- 同时引入“前者”里那些能让脚本跑通的 **工程性改进**（更鲁棒的文件匹配、CBV/BV 别名、可选自动分段、日志）。

Implements
---------
- Loss metrics: RMSE, MAE, MME(O), MME(U)
- Period splits: 默认严格使用论文窗口（可用 --periods 控制为 paper/auto/whole）
- Horizons: h = 1, 5, 10（1 步 & 累积多步）
- MCS (Hansen–Lunde–Nason 2011) via stationary bootstrap
- SPA (Hansen 2005) via stationary bootstrap
- Mincer–Zarnowitz 回归及 α=0, β=1 的联合 Wald 检验

Key robustness borrowed from your “前者”
--------------------------------------
- 允许度量名别名：CBV ↔ BV（你的 HAR/ARFIMA 目录通常写 BV）。
- load_forecast() 同时尝试多种文件名模式 + 宽松正则模糊匹配。
- parse_dates() 更强健。
- 可选 --periods，以 paper 为默认；若样本与论文窗口 **完全不重叠** 且仍选择 paper，将自动降级为 auto（仅为避免“全空”；会给出明显日志）。

Outputs
-------
存放于 03_results/evaluation_reports/：
- <asset>__metrics__<period>__h<h>__<freq>__<metric>.csv
- <asset>__mcs_rowwise__<period>__h<h>__<freq>.csv
- <asset>__mcs_colwise__<period>__h<h>__<freq>.csv
- <asset>__spa__<period>__h<h>__<freq>.csv
- <asset>__mz_efficiency__<period>__h<h>__<freq>.csv

How to run
----------
python 04_evaluation/evaluate_forecasts_paper_periods_robust.py \
  --root F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication \
  --assets CL_WTI \
  --freq 1min 5min \
  --measures TSRV RV RK JWTSRV CBV MedRV \
  --models GARCH ARFIMA HAR ANN HAR-ANN \
  --horizons 1 5 10 \
  --bootstrap 2000 --avg_block 20 --alpha 0.10 --periods paper

Tip: 先用 --bootstrap 500 试跑，再提到 2000。
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# -----------------------------
# Configuration defaults
# -----------------------------
DEFAULT_ROOT = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
DEFAULT_ASSETS = ["CL_WTI"]
DEFAULT_FREQS = ["1min", "5min"]
DEFAULT_MEASURES = ["TSRV", "RV", "RK", "JWTSRV", "CBV", "MedRV"]
DEFAULT_MODELS = ["GARCH", "ARFIMA", "HAR", "ANN", "HAR-ANN"]
DEFAULT_HORIZONS = [1, 5, 10]
DEFAULT_ALPHA = 0.10
DEFAULT_BOOTSTRAP = 1000
DEFAULT_AVG_BLOCK = 20
DEFAULT_PERIOD_MODE = "auto"  # 保持“后者”算法：默认严格按论文窗口

# 论文窗口（含首含尾）
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

# 各度量对应常见列名（读取 actual 时使用）
MEASURE_COLS: Dict[str, List[str]] = {
    "TSRV":   ["TSRV"],
    "RV":     ["RV"],
    "RK":     ["RK"],
    "JWTSRV": ["JWTSRV"],
    # 论文叫 CBV，你的文件多写 BV；两者都接受
    "CBV":    ["BV", "CBV"],
    "MedRV":  ["MedRV"],
}

# 用于文件名 token 替换（匹配 forecast 文件名）
MEASURE_FILE_TOKENS: Dict[str, List[str]] = {
    "CBV":    ["BV", "CBV"],
    "TSRV":   ["TSRV"],
    "RV":     ["RV"],
    "RK":     ["RK"],
    "JWTSRV": ["JWTSRV"],
    "MedRV":  ["MedRV"],
}

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def date_range_of_frames(frames: Iterable[pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    mins, maxs = [], []
    for df in frames:
        if df is None or "date" not in df.columns:
            continue
        s = pd.to_datetime(df["date"], errors="coerce")
        if s.notna().any():
            mins.append(s.min()); maxs.append(s.max())
    if not mins:
        raise ValueError("No dates found to build sample range.")
    return min(mins), max(maxs)

def resolve_periods(sample_min: pd.Timestamp, sample_max: pd.Timestamp, mode: str = DEFAULT_PERIOD_MODE) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    if mode == "whole":
        return {"whole": (sample_min, sample_max)}
    if mode == "paper":
        # 如样本与论文窗口完全不重叠，则退回 auto（避免全空）
        any_overlap = any(not (sample_max < s or sample_min > e) for s, e in PERIODS_PAPER.values())
        if any_overlap:
            return PERIODS_PAPER
        # 明确提示降级
        print("[WARN] Sample window does not overlap paper windows. Falling back to auto split.")
        mode = "auto"
    if mode == "auto":
        span = (sample_max - sample_min)
        if span.days <= 2:
            return {"whole": (sample_min, sample_max)}
        t1 = sample_min + span / 3
        t2 = sample_min + 2 * span / 3
        return {
            "pre":    (sample_min, pd.Timestamp(t1.floor("D"))),
            "crisis": (pd.Timestamp((t1 + pd.Timedelta(days=1)).floor("D")), pd.Timestamp(t2.floor("D"))),
            "post":   (pd.Timestamp((t2 + pd.Timedelta(days=1)).floor("D")), sample_max),
        }
    return PERIODS_PAPER


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """尽力抽取名为 'date' 的时间列，容错各种导出格式。"""
    direct = [c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")]
    if direct:
        col = direct[0]
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().all():
            pass
        else:
            return df.rename(columns={col: "date"}).sort_values("date").reset_index(drop=True)

    if isinstance(df.index, pd.DatetimeIndex):
        out = df.reset_index().rename(columns={df.index.name or "index": "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values("date").reset_index(drop=True)
    try:
        idx_parsed = pd.to_datetime(df.index, errors="coerce")
        if idx_parsed.notna().mean() > 0.8:
            out = df.copy()
            out.insert(0, "date", idx_parsed)
            return out.sort_values("date").reset_index(drop=True)
    except Exception:
        pass

    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().mean() > 0.8:
                out = df.copy(); out["date"] = s
                return out.sort_values("date").reset_index(drop=True)
        except Exception:
            continue

    if "Unnamed: 0" in df.columns:
        s = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
        if s.notna().any():
            out = df.rename(columns={"Unnamed: 0": "date"}).copy()
            out["date"] = s
            return out.sort_values("date").reset_index(drop=True)

    cols = ", ".join([f"{c}:{str(df[c].dtype)}" for c in df.columns])
    raise ValueError(f"No usable date found. Columns= [{cols}] | index={type(df.index)}")


def pick_forecast_column(df: pd.DataFrame) -> str:
    candidates = [
        "forecast", "yhat", "pred", "prediction", "value", "vol_forecast",
        "nu_hat", "nu_pred", "vhat", "fcast"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    for c in float_cols:
        if c.lower() not in ("actual", "target", "truth", "y"):
            return c
    raise ValueError("Could not identify forecast column.")


def pick_actual_column(df: pd.DataFrame) -> str:
    candidates = ["actual", "target", "truth", "y", "volatility", "nu", "value"]
    for c in candidates:
        if c in df.columns:
            return c
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if float_cols:
        return float_cols[0]  # 选第一个浮点列以避免误选末尾辅助列
    raise ValueError("Could not identify actual column.")


# -----------------------------
# Stationary bootstrap
# -----------------------------

def stationary_bootstrap_indices(T: int, B: int, avg_block: int, rng: np.random.Generator) -> np.ndarray:
    p = 1.0 / float(avg_block)
    starts = rng.integers(0, T, size=(B, T), endpoint=False)
    geom = rng.random((B, T)) > p
    idx = np.empty((B, T), dtype=int)
    idx[:, 0] = starts[:, 0]
    for b in range(B):
        for t in range(1, T):
            if geom[b, t - 1]:
                idx[b, t] = (idx[b, t - 1] + 1) % T
            else:
                idx[b, t] = starts[b, t]
    return idx


# -----------------------------
# Loss functions
# -----------------------------

def loss_components(pred: np.ndarray, actual: np.ndarray) -> Dict[str, np.ndarray]:
    err = pred - actual
    rmse = (err ** 2)
    mae = np.abs(err)
    under = err < 0
    over = err > 0
    mme_o = np.empty_like(err)
    mme_u = np.empty_like(err)
    mme_o[under] = np.abs(err[under])
    mme_o[over] = np.sqrt(np.abs(err[over]))
    mme_o[~(under | over)] = 0.0
    mme_u[under] = np.sqrt(np.abs(err[under]))
    mme_u[over] = np.abs(err[over])
    mme_u[~(under | over)] = 0.0
    return {
        "rmse_obs": rmse,
        "mae_obs": mae,
        "mme_o_obs": mme_o,
        "mme_u_obs": mme_u,
        "err": err,
        "over": over.astype(int),
        "under": under.astype(int),
    }


def aggregate_losses(loss_obs: Dict[str, np.ndarray]) -> Dict[str, float]:
    rmse = float(np.sqrt(np.mean(loss_obs["rmse_obs"])) )
    mae = float(np.mean(loss_obs["mae_obs"]))
    mme_o = float(np.mean(loss_obs["mme_o_obs"]))
    mme_u = float(np.mean(loss_obs["mme_u_obs"]))
    over_rate = float(np.mean(loss_obs["over"]))
    under_rate = float(np.mean(loss_obs["under"]))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MME_O": mme_o,
        "MME_U": mme_u,
        "OverRate": over_rate,
        "UnderRate": under_rate,
    }


# -----------------------------
# Filenames & loaders
# -----------------------------
MEAS_FILE_1H = "03_results/intermediate_results/volatility_estimates/test_set/{asset}_{measure}_daily_{freq}_test.parquet"
MEAS_FILE_CUM = "03_results/intermediate_results/volatility_estimates/cumulative_forecasts/{asset}_{measure}_daily_{freq}_test_test_cumulative_h{h}.parquet"

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
    ],
    "HAR-ANN": [
        r"{asset}_{measure}_daily_{freq}_HAR_ANN_1step_test.parquet",
        r"{asset}_{measure}_daily_{freq}_HAR_ANN_cumulative_h{h}_test.parquet",
    ],
}

MODEL_DIR = {
    "ANN": "03_results/final_forecasts/ANN",
    "HAR": "03_results/final_forecasts/HAR",
    "ARFIMA": "03_results/final_forecasts/ARFIMA",
    "GARCH": "03_results/final_forecasts/GARCH",
    "HAR-ANN": "03_results/final_forecasts/ANN",  # 你的组合通常放 ANN 下
}


def try_load_parquet(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        try:
            df = pd.read_parquet(path)
            return parse_dates(df)
        except Exception as e:
            print(f"[READ FAIL] {path}: {e}")
            return None
    return None


def load_actual(root: str, asset: str, measure: str, freq: str, h: int) -> Optional[pd.DataFrame]:
    if h == 1:
        path = os.path.join(root, MEAS_FILE_1H.format(asset=asset, measure=measure, freq=freq))
    else:
        path = os.path.join(root, MEAS_FILE_CUM.format(asset=asset, measure=measure, freq=freq, h=h))
    df = try_load_parquet(path)
    if df is None:
        return None
    candidates = MEASURE_COLS.get(measure, [measure])
    hit = [c for c in candidates if c in df.columns]
    if hit:
        actual_col = hit[0]
    else:
        float_cols = [c for c in df.columns if c != "date" and pd.api.types.is_float_dtype(df[c])]
        actual_col = float_cols[0] if float_cols else pick_actual_column(df)
    return df[["date", actual_col]].rename(columns={actual_col: "actual"})


def glob_candidates(dirpath: str, regex: str) -> List[str]:
    pat = re.compile(regex)
    if not os.path.isdir(dirpath):
        return []
    return [os.path.join(dirpath, f) for f in os.listdir(dirpath) if pat.match(f)]


def load_forecast(root: str, model: str, asset: str, measure: str, freq: str, h: int) -> Optional[pd.DataFrame]:
    base = os.path.join(root, MODEL_DIR[model])
    tokens = MEASURE_FILE_TOKENS.get(measure, [measure])

    # 先按多模式精确尝试（含 BV/CBV 别名）
    for token in tokens:
        for pat in PATTERNS.get(model, []):
            fname = pat.format(asset=asset, measure=token, freq=freq, h=h)
            path = os.path.join(base, fname)
            df = try_load_parquet(path)
            if df is not None:
                col = pick_forecast_column(df)
                return df[["date", col]].rename(columns={col: model})

    # 再做宽松正则
    for token in tokens:
        regex = rf".*{token}.*{freq}.*(h{h}|1step).*\\.parquet$"
        for path in glob_candidates(base, regex):
            df = try_load_parquet(path)
            if df is not None:
                try:
                    col = pick_forecast_column(df)
                    return df[["date", col]].rename(columns={col: model})
                except Exception:
                    continue
    return None


# -----------------------------
# MCS & SPA
# -----------------------------

def mcs_pvalues(loss_matrix: pd.DataFrame, alpha: float, B: int, avg_block: int, rng: np.random.Generator) -> Tuple[pd.Series, List[str]]:
    L = loss_matrix.copy().dropna(axis=1, how='any')
    names = list(L.columns)
    T = L.shape[0]
    if L.shape[1] < 2:
        return pd.Series(1.0, index=L.columns), names

    centered = L - L.mean(axis=0)

    def tmax(cols: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        sub = centered[cols].values
        d_bar = sub.mean(axis=0)
        idx = stationary_bootstrap_indices(T, B, avg_block, rng)
        boot_means = np.mean(sub[idx, :], axis=1)
        v = boot_means.var(axis=0, ddof=1)
        t = np.where(v > 0, d_bar / np.sqrt(v), 0.0)
        return d_bar, v, float(np.max(t))

    survivors = names.copy()
    pvals = {n: np.nan for n in names}

    while True:
        _, _, t_obs = tmax(survivors)
        sub = centered[survivors].values
        idx = stationary_bootstrap_indices(T, B, avg_block, rng)
        boot_means = np.mean(sub[idx, :], axis=1)
        v = boot_means.var(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_boot = boot_means / np.sqrt(v)
            t_boot = np.where(np.isfinite(t_boot), t_boot, 0.0)
        t_boot_max = np.max(t_boot, axis=1)
        pval = float(np.mean(t_boot_max >= t_obs))
        if pval >= alpha:
            for n in survivors:
                pvals[n] = max(pvals.get(n, np.nan), pval)
            break
        d_bar, vbar, _ = tmax(survivors)
        worst_idx = int(np.argmax(d_bar / np.sqrt(vbar + 1e-12)))
        worst_name = survivors[worst_idx]
        pvals[worst_name] = pval
        survivors.pop(worst_idx)
        if len(survivors) <= 1:
            for n in survivors:
                pvals[n] = max(pvals.get(n, np.nan), pval)
            break

    return pd.Series(pvals).reindex(names), survivors


def spa_pvalue(benchmark: str, loss_matrix: pd.DataFrame, B: int, avg_block: int, rng: np.random.Generator) -> float:
    if benchmark not in loss_matrix.columns:
        return np.nan
    L = loss_matrix.dropna(axis=1, how='any')
    if benchmark not in L.columns or L.shape[1] < 2:
        return np.nan
    T = L.shape[0]
    bench = L[benchmark].values
    diffs = bench[:, None] - L.drop(columns=[benchmark]).values
    d_bar = diffs.mean(axis=0)
    idx = stationary_bootstrap_indices(T, B, avg_block, rng)
    boot_means = np.mean(diffs[idx, :], axis=1)
    v = boot_means.var(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_obs = float(np.max(np.where(v > 0, d_bar / np.sqrt(v), 0.0)))
        t_boot = np.where(v > 0, boot_means / np.sqrt(v), 0.0)
    t_boot_max = np.max(t_boot, axis=1)
    return float(np.mean(t_boot_max >= t_obs))


# -----------------------------
# MZ regression
# -----------------------------

def mz_regression(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, float, float]:
    X = sm.add_constant(x)
    res = sm.OLS(y, X, missing='drop').fit()
    alpha, beta = res.params
    cov = res.cov_params()
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.0, 1.0])
    diff = R @ np.array([alpha, beta]) - q
    W = float(diff.T @ np.linalg.inv(R @ cov @ R.T) @ diff)
    p_joint = 1.0 - stats.chi2.cdf(W, df=2)
    t_alpha = alpha / np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.inf
    p_alpha0 = 2.0 * (1.0 - stats.t.cdf(np.abs(t_alpha), df=res.df_resid))
    t_beta = (beta - 1.0) / np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.inf
    p_beta1 = 2.0 * (1.0 - stats.t.cdf(np.abs(t_beta), df=res.df_resid))
    return alpha, beta, p_alpha0, p_beta1, p_joint


# -----------------------------
# Core evaluation
# -----------------------------

def evaluate(root: str,
             assets: List[str],
             freqs: List[str],
             measures: List[str],
             models: List[str],
             horizons: List[int],
             alpha: float,
             B: int,
             avg_block: int,
             seed: int = 12345,
             period_mode: str = DEFAULT_PERIOD_MODE) -> None:
    out_dir = os.path.join(root, "03_results", "evaluation_reports")
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    for asset in assets:
        print(f"\n=== Asset: {asset} ===")
        for h in horizons:
            for freq in freqs:
                print(f"-- freq={freq}, h={h}")
                # 1) 读 actual
                actual_by_measure: Dict[str, pd.DataFrame] = {}
                for m in measures:
                    df_a = load_actual(root, asset, m, freq, h)
                    if df_a is not None:
                        actual_by_measure[m] = df_a
                if not actual_by_measure:
                    print(f"[WARN] No actuals found: asset={asset}, freq={freq}, h={h}. Skipping.")
                    continue

                # 2) 决定分段（默认 paper，如完全不重叠则自动降级 auto）
                sample_min = min(df["date"].min() for df in actual_by_measure.values())
                sample_max = max(df["date"].max() for df in actual_by_measure.values())
                periods_cur = resolve_periods(sample_min, sample_max, period_mode)
                print("   sample:", sample_min.date(), "→", sample_max.date(), "| periods:",
                      {k: (v[0].date(), v[1].date()) for k, v in periods_cur.items()})

                rows = []
                per_period_frames: Dict[str, List[pd.DataFrame]] = {k: [] for k in periods_cur}

                # 3) 拼接面板 + 计算损失
                for m, df_a in actual_by_measure.items():
                    panel = df_a.copy()
                    found_any = False
                    for model in models:
                        df_f = load_forecast(root, model, asset, m, freq, h)
                        if df_f is not None:
                            found_any = True
                            panel = panel.merge(df_f, on="date", how="left")
                    if not found_any:
                        print(f"  [WARN] No forecasts found for measure={m}. Skip this measure.")
                        continue

                    for model in [c for c in panel.columns if c not in ("date", "actual")]:
                        for per_name, (start, end) in periods_cur.items():
                            sub = panel[(panel["date"] >= start) & (panel["date"] <= end)][["date", "actual", model]].dropna()
                            if sub.empty:
                                continue
                            comp = loss_components(sub[model].to_numpy(), sub["actual"].to_numpy())
                            agg = aggregate_losses(comp)
                            rows.append({
                                "asset": asset,
                                "freq": freq,
                                "h": h,
                                "period": per_name,
                                "measure": m,
                                "model": model,
                                **agg,
                                "N": sub.shape[0],
                            })
                            per_period_frames[per_name].append(sub.assign(model=model, measure=m).set_index("date"))

                if not rows:
                    print("[WARN] No rows computed for this (asset,freq,h). Continue.")
                    continue

                metrics_df = pd.DataFrame(rows)
                for per_name in periods_cur:
                    sub = metrics_df[metrics_df["period"] == per_name]
                    if sub.empty:
                        continue
                    for metric in ["RMSE", "MAE", "MME_O", "MME_U", "OverRate", "UnderRate", "N"]:
                        wide = sub.pivot_table(index=["measure"], columns="model", values=metric)
                        out_path = os.path.join(out_dir, f"{asset}__metrics__{per_name}__h{h}__{freq}__{metric}.csv")
                        wide.to_csv(out_path, float_format="%.6g")

                # 4) MCS & SPA（以 MAE 为基准损失）
                for per_name in periods_cur:
                    per_frames = [f for f in per_period_frames[per_name] if f.shape[0] > 0]
                    if not per_frames:
                        continue
                    group: Dict[str, List[pd.Series]] = {}
                    for df in per_frames:
                        m = df["measure"].iloc[0]
                        model = df["model"].iloc[0]
                        loss = (df[model] - df["actual"]).abs().rename(model)
                        group.setdefault(m, []).append(loss)

                    rowwise_records = []
                    spa_records = []
                    for m, loss_list in group.items():
                        loss_mat = pd.concat(loss_list, axis=1).dropna(how='any')
                        if loss_mat.shape[1] < 2 or loss_mat.shape[0] < 20:
                            continue
                        pvals, survivors = mcs_pvalues(loss_mat, alpha, B, avg_block, rng)
                        for name, pv in pvals.items():
                            rowwise_records.append({
                                "asset": asset, "freq": freq, "h": h, "period": per_name,
                                "measure": m, "model": name, "mcs_p": pv, "in_mcs": name in survivors
                            })
                        for bench in loss_mat.columns:
                            pv = spa_pvalue(bench, loss_mat, B, avg_block, rng)
                            spa_records.append({
                                "asset": asset, "freq": freq, "h": h, "period": per_name,
                                "measure": m, "benchmark": bench, "spa_p": pv
                            })
                    if rowwise_records:
                        pd.DataFrame(rowwise_records).to_csv(
                            os.path.join(out_dir, f"{asset}__mcs_rowwise__{per_name}__h{h}__{freq}.csv"), index=False)
                    if spa_records:
                        pd.DataFrame(spa_records).to_csv(
                            os.path.join(out_dir, f"{asset}__spa__{per_name}__h{h}__{freq}.csv"), index=False)

                    # 列方向 MCS：固定某个 model，在不同 measure 间 MCS
                    by_model: Dict[str, List[pd.Series]] = {}
                    for df in per_frames:
                        m = df["measure"].iloc[0]
                        model = df["model"].iloc[0]
                        loss = (df[model] - df["actual"]).abs().rename(m)
                        by_model.setdefault(model, []).append(loss)

                    colwise_records = []
                    for model, losses in by_model.items():
                        loss_mat = pd.concat(losses, axis=1).dropna(how='any')
                        if loss_mat.shape[1] < 2 or loss_mat.shape[0] < 20:
                            continue
                        pvals, survivors = mcs_pvalues(loss_mat, alpha, B, avg_block, rng)
                        for mname, pv in pvals.items():
                            colwise_records.append({
                                "asset": asset, "freq": freq, "h": h, "period": per_name,
                                "model": model, "measure": mname, "mcs_p": pv, "in_mcs": mname in survivors
                            })
                    if colwise_records:
                        pd.DataFrame(colwise_records).to_csv(
                            os.path.join(out_dir, f"{asset}__mcs_colwise__{per_name}__h{h}__{freq}.csv"), index=False)

                    # 5) MZ 效率回归
                    mz_rows = []
                    actual_aligned = {m: s.set_index("date")["actual"] for m, s in actual_by_measure.items()}
                    fcasts: Dict[Tuple[str, str], pd.Series] = {}
                    for m, df_a in actual_by_measure.items():
                        for model in models:
                            df_f = load_forecast(root, model, asset, m, freq, h)
                            if df_f is None:
                                continue
                            fcasts[(m, model)] = df_f.set_index("date")[model]

                    for rm1, y in actual_aligned.items():
                        y_per = y[(y.index >= periods_cur[per_name][0]) & (y.index <= periods_cur[per_name][1])]
                        for (rm, model), x in fcasts.items():
                            df_xy = pd.concat([y_per, x], axis=1, join='inner').dropna()
                            if df_xy.shape[0] < 30:
                                continue
                            alpha_hat, beta_hat, p_a0, p_b1, p_joint = mz_regression(df_xy.iloc[:, 0].values, df_xy.iloc[:, 1].values)
                            mz_rows.append({
                                "asset": asset, "freq": freq, "h": h, "period": per_name,
                                "RM1_dep": rm1, "RM_pred": rm, "model": model,
                                "alpha": alpha_hat, "beta": beta_hat,
                                "p_alpha_eq_0": p_a0, "p_beta_eq_1": p_b1, "p_joint_alpha0_beta1": p_joint,
                                "N": int(df_xy.shape[0])
                            })
                    if mz_rows:
                        pd.DataFrame(mz_rows).to_csv(
                            os.path.join(out_dir, f"{asset}__mz_efficiency__{per_name}__h{h}__{freq}.csv"), index=False)

    print("\nDone. Reports saved under 03_results/evaluation_reports/.")


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate volatility forecasts (Baruník & Křehlič 2016 replication)")
    p.add_argument("--periods", choices=["paper", "auto", "whole"], default=DEFAULT_PERIOD_MODE,
                   help="paper=按论文窗口；若样本不重叠自动退回auto；auto=按样本三等分；whole=不分段")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--assets", nargs="*", default=DEFAULT_ASSETS)
    p.add_argument("--freq", nargs="*", default=DEFAULT_FREQS)
    p.add_argument("--measures", nargs="*", default=DEFAULT_MEASURES)
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    p.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    p.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    p.add_argument("--avg_block", type=int, default=DEFAULT_AVG_BLOCK)
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    evaluate(
        root=args.root,
        assets=args.assets,
        freqs=args.freq,
        measures=args.measures,
        models=args.models,
        horizons=args.horizons,
        alpha=args.alpha,
        B=args.bootstrap,
        avg_block=args.avg_block,
        seed=args.seed,
        period_mode=args.periods,
    )


if __name__ == "__main__":
    main()
