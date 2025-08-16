#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate volatility forecasts exactly following Baruník & Křehlič (2016)
for the "WTI_volatility_forecast_replication" project structure you shared.

Implements:
- Loss metrics: RMSE, MAE, MME(O), MME(U)
- Period splits: pre‑crisis, crisis, post‑crisis (with exact dates)
- Horizons: h = 1, 5, 10 (1‑step and cumulative multi‑step)
- Model Confidence Set (MCS, Hansen–Lunde–Nason 2011) via stationary bootstrap
- Superior Predictive Ability (SPA, Hansen 2005) via stationary bootstrap
- Mincer–Zarnowitz (1969) efficiency regressions with joint α=0, β=1 Wald tests
- Optional export of row‑wise (across models) and column‑wise (across measures) MC S p‑values

Assumptions about your on‑disk data
-----------------------------------
Project root = F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/
(Works cross‑platform; adjust ROOT below.)

We rely on files already present in your tree (based on your listing):
- Actual (realized) 1‑step volatility for each measure in
  03_results/intermediate_results/volatility_estimates/test_set/
  e.g., CL_WTI_MedRV_daily_1min_test.parquet
- Actual cumulative h‑step volatility for each measure in
  03_results/intermediate_results/volatility_estimates/cumulative_forecasts/
  e.g., CL_WTI_MedRV_daily_1min_test_test_cumulative_h5.parquet
- Forecasts per model in 03_results/final_forecasts/<MODEL>/ (and/or
  intermediate_results/cumulative_forecasts/<MODEL>/ for some ANN files),
  with filenames like:
    CL_WTI_MedRV_daily_1min_ANN_1step_test.parquet
    HAR_BV_1min_cumulative_h10.parquet  (your HAR folder shows patterns)
    ARFIMA_BV_1min_forecast.parquet / _cumulative.parquet (ARFIMA folder shows both)
    garch_11_test_cumulative_h5.parquet (GARCH cumulative)

Because filenames vary slightly across models, the loader uses robust
regex patterns and a small set of fallbacks. If a file cannot be found,
that (asset, measure, horizon, model) cell will be skipped gracefully.

Outputs
-------
Creates CSV and Markdown/Latex‑ready tables in:
03_results/evaluation_reports/
- <asset>__metrics__<period>.csv (RMSE/MAE/MME(O)/MME(U))
- <asset>__mcs_rowwise__<period>__h<h>.csv (across models, per measure)
- <asset>__mcs_colwise__<period>__h<h>.csv (across measures, per model)
- <asset>__spa__<period>__h<h>.csv (SPA p‑values for each benchmark model)
- <asset>__mz_efficiency__<period>__h<h>.csv (MZ α,β, Wald p‑value)

How to run
----------
python 04_evaluation/evaluate_forecasts.py \
    --root F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication \
    --assets CL_WTI \
    --freq 1min 5min \
    --measures TSRV RV RK JWTSRV CBV MedRV \
    --models GARCH ARFIMA HAR ANN HAR-ANN \
    --horizons 1 5 10 \
    --bootstrap 2000 --avg_block 20 --alpha 0.10

Tip: start with --bootstrap 500 for speed, then increase to 2000 for the paper.

Notes
-----
- Stationary bootstrap is implemented with expected block length L = avg_block,
  using geometric block lengths (Politis–Romano). The paper used avg block ≈ 20.
- MCS uses the T_max statistic on average loss differentials and sequentially
  eliminates the worst model until no rejection at level α.
- SPA tests each candidate as a benchmark; p‑value is bootstrap max statistic.
- MZ regressions follow the variant described in your prompt: regress the
  realized volatility of one measure RM1 on forecasts produced by (RM,f).

Author: you + ChatGPT
License: MIT
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# -----------------------------
# Configuration defaults
# -----------------------------
DEFAULT_ROOT = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
DEFAULT_ASSETS = ["CL_WTI"]  # extend to Heating/NatGas if you add data later
DEFAULT_FREQS = ["1min", "5min"]
DEFAULT_MEASURES = ["TSRV", "RV", "RK", "JWTSRV", "CBV", "MedRV"]
# Map each measure to likely column names found in your files
MEASURE_COLS: Dict[str, List[str]] = {
    "TSRV": ["TSRV"],
    "RV": ["RV"],
    "RK": ["RK"],
    "JWTSRV": ["JWTSRV"],
    # Paper uses CBV, your files show BV; accept both with BV preferred when present
    "CBV": ["BV", "CBV"],
    "MedRV": ["MedRV"],
}
DEFAULT_MODELS = ["GARCH", "ARFIMA", "HAR", "ANN", "HAR-ANN"]
DEFAULT_HORIZONS = [1, 5, 10]
DEFAULT_ALPHA = 0.10
DEFAULT_BOOTSTRAP = 1000
DEFAULT_AVG_BLOCK = 20
DEFAULT_PERIOD_MODE = "paper"  # 'paper' | 'auto' | 'whole'

# Exact period boundaries from the paper (all inclusive on start, inclusive on end)
PRE_START = pd.Timestamp("2006-07-06")
PRE_END   = pd.Timestamp("2008-08-31")
CRI_START = pd.Timestamp("2008-09-01")
CRI_END   = pd.Timestamp("2010-10-31")
POST_START= pd.Timestamp("2010-11-01")
POST_END  = pd.Timestamp("2012-12-31")
PERIODS = {
    "pre":  (PRE_START, PRE_END),
    "crisis": (CRI_START, CRI_END),
    "post": (POST_START, POST_END),
}
MEASURE_FILE_TOKENS: Dict[str, List[str]] = {
    "CBV": ["BV", "CBV"],
    "TSRV": ["TSRV"],
    "RV": ["RV"],
    "RK": ["RK"],
    "JWTSRV": ["JWTSRV"],
    "MedRV": ["MedRV"],
}

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
from typing import Iterable

def date_range_of_frames(frames: Iterable[pd.DataFrame]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    mins, maxs = [], []
    for df in frames:
        if "date" not in df.columns:
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
        any_overlap = any(not (sample_max < s or sample_min > e) for s, e in PERIODS.values())
        if any_overlap:
            return PERIODS
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
    return PERIODS


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Make a best effort to extract a datetime column named 'date'.
    Handles cases where the datetime is stored as the index, under different
    column names (e.g., 'Unnamed: 0'), or as strings/ints convertible to datetime.
    """
    # 1) Direct hits by common names (case-insensitive)
    direct = [c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")]
    if direct:
        col = direct[0]
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().all():
            # fall through to heuristics
            pass
        else:
            return df.rename(columns={col: "date"}).sort_values("date").reset_index(drop=True)

    # 2) If the index looks like dates, use it
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

    # 3) Heuristic scan: pick the first column that converts to datetime well
    for c in df.columns:
        try:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().mean() > 0.8:
                out = df.copy()
                out["date"] = s
                if c != "date":
                    # keep original column as well; caller can ignore it
                    pass
                return out.sort_values("date").reset_index(drop=True)
        except Exception:
            continue

    # 4) Common export artifact
    if "Unnamed: 0" in df.columns:
        s = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
        if s.notna().any():
            out = df.rename(columns={"Unnamed: 0": "date"}).copy()
            out["date"] = s
            return out.sort_values("date").reset_index(drop=True)

    # 5) Give a detailed error for debugging
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
    # fallback: first float column that isn't 'actual'/'target'
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
    # fallback: last float column
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if float_cols:
        return float_cols[-1]
    raise ValueError("Could not identify actual column.")


# -----------------------------
# Stationary bootstrap
# -----------------------------

def stationary_bootstrap_indices(T: int, B: int, avg_block: int, rng: np.random.Generator) -> np.ndarray:
    """Generate stationary bootstrap indices (Politis & Romano) of length T for B replicates.
    Returns an array of shape (B, T) with integer indices in [0, T-1].
    Expected block length = avg_block, geom. distribution for block continuation.
    """
    p = 1.0 / float(avg_block)
    starts = rng.integers(0, T, size=(B, T), endpoint=False)
    geom = rng.random((B, T)) > p  # True => continue block; False => start new block
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
    under = err < 0  # prediction < actual
    over = err > 0   # prediction > actual
    # MME per observation (we'll average later)
    mme_o = np.empty_like(err)
    mme_u = np.empty_like(err)
    # MME(O): |.| for under, sqrt(|.|) for over
    mme_o[under] = np.abs(err[under])
    mme_o[over] = np.sqrt(np.abs(err[over]))
    mme_o[~(under | over)] = 0.0
    # MME(U): sqrt(|.|) for under, |.| for over
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
    T = loss_obs["rmse_obs"].shape[0]
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
        "T": T,
    }


# -----------------------------
# MCS (rowwise across models OR columnwise across measures)
# -----------------------------

def mcs_pvalues(loss_matrix: pd.DataFrame, alpha: float, B: int, avg_block: int, rng: np.random.Generator) -> Tuple[pd.Series, List[str]]:
    """Compute MCS p-values (Hansen–Lunde–Nason) with the T_max statistic.
    loss_matrix: shape (T, K) for K competitors; rows aligned in time; lower is better.

    Returns (pvals, survivors) where pvals is a Series of final p-values per model
    and survivors is the final set (model names) that form the MCS at level alpha.
    """
    # Drop columns with any NaNs
    L = loss_matrix.copy().dropna(axis=1, how='any')
    names = list(L.columns)
    T = L.shape[0]
    if L.shape[1] < 2:
        return pd.Series(1.0, index=L.columns), names

    centered = L - L.mean(axis=0)  # center per model

    # Helper to compute T_max statistic for a given subset of columns
    def tmax(cols: List[str]) -> Tuple[np.ndarray, np.ndarray, float]:
        sub = centered[cols].values  # T x k
        # average loss differentials per model vs. cross-section mean
        d_bar = sub.mean(axis=0)  # k
        # Use stationary bootstrap to estimate variance of d_bar
        idx = stationary_bootstrap_indices(T, B, avg_block, rng)
        boot_means = np.mean(sub[idx, :], axis=1)  # (B, k)
        # variance of mean under bootstrap
        v = boot_means.var(axis=0, ddof=1)
        # t-stats; protect against zero variance
        t = np.where(v > 0, d_bar / np.sqrt(v), 0.0)
        t_max = float(np.max(t))
        return d_bar, v, t_max

    # Step-down elimination
    survivors = names.copy()
    pvals = {n: np.nan for n in names}

    while True:
        _, _, t_obs = tmax(survivors)
        # Bootstrap distribution of t_max for current set
        sub = centered[survivors].values
        idx = stationary_bootstrap_indices(T, B, avg_block, rng)
        boot_means = np.mean(sub[idx, :], axis=1)  # (B, k)
        v = boot_means.var(axis=0, ddof=1)
        # compute bootstrap t per replicate and model, then max
        with np.errstate(divide='ignore', invalid='ignore'):
            t_boot = boot_means / np.sqrt(v)
            t_boot = np.where(np.isfinite(t_boot), t_boot, 0.0)
        t_boot_max = np.max(t_boot, axis=1)
        pval = float(np.mean(t_boot_max >= t_obs))
        if pval >= alpha:
            # cannot reject equal predictive ability: stop
            for n in survivors:
                pvals[n] = max(pvals.get(n, np.nan), pval)
            break
        # else eliminate worst model (largest t)
        d_bar, vbar, t_now = tmax(survivors)
        worst_idx = int(np.argmax(d_bar / np.sqrt(vbar + 1e-12)))
        worst_name = survivors[worst_idx]
        pvals[worst_name] = pval
        survivors.pop(worst_idx)
        if len(survivors) <= 1:
            for n in survivors:
                pvals[n] = max(pvals.get(n, np.nan), pval)
            break

    return pd.Series(pvals).reindex(names), survivors


# -----------------------------
# SPA test (Hansen 2005)
# -----------------------------

def spa_pvalue(benchmark: str, loss_matrix: pd.DataFrame, B: int, avg_block: int, rng: np.random.Generator) -> float:
    """SPA p-value for H0: benchmark is the best (minimum average loss).
    loss_matrix columns are competitor names; rows are time.
    """
    if benchmark not in loss_matrix.columns:
        return np.nan
    L = loss_matrix.dropna(axis=1, how='any')
    if benchmark not in L.columns or L.shape[1] < 2:
        return np.nan
    T = L.shape[0]
    # Loss differentials vs. benchmark: d_jt = L_bench - L_j
    bench = L[benchmark].values
    diffs = bench[:, None] - L.drop(columns=[benchmark]).values  # T x (K-1)
    # Test statistic: max over j of sqrt(T) * mean(d_j) / std_boot(d_j)
    d_bar = diffs.mean(axis=0)  # (K-1)
    idx = stationary_bootstrap_indices(T, B, avg_block, rng)
    boot_means = np.mean(diffs[idx, :], axis=1)  # (B, K-1)
    v = boot_means.var(axis=0, ddof=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_obs = np.max(np.where(v > 0, d_bar / np.sqrt(v), 0.0))
        t_boot = np.where(v > 0, boot_means / np.sqrt(v), 0.0)
    t_boot_max = np.max(t_boot, axis=1)
    pval = float(np.mean(t_boot_max >= t_obs))
    return pval


# -----------------------------
# Mincer–Zarnowitz efficiency regression
# -----------------------------

def mz_regression(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, float, float]:
    """OLS y = α + β x + ε. Returns (alpha, beta, p_alpha0, p_beta1, p_joint).
    p_joint is Wald test p-value for H0: α=0 and β=1.
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X, missing='drop')
    res = model.fit()
    alpha, beta = res.params
    cov = res.cov_params()
    # H0: α=0, β=1
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.0, 1.0])
    diff = R @ np.array([alpha, beta]) - q
    W = float(diff.T @ np.linalg.inv(R @ cov @ R.T) @ diff)
    p_joint = 1.0 - stats.chi2.cdf(W, df=2)
    # single-parameter p-values
    t_alpha = alpha / np.sqrt(cov[0, 0]) if cov[0, 0] > 0 else np.inf
    p_alpha0 = 2.0 * (1.0 - stats.t.cdf(np.abs(t_alpha), df=res.df_resid))
    t_beta = (beta - 1.0) / np.sqrt(cov[1, 1]) if cov[1, 1] > 0 else np.inf
    p_beta1 = 2.0 * (1.0 - stats.t.cdf(np.abs(t_beta), df=res.df_resid))
    return alpha, beta, p_alpha0, p_beta1, p_joint


# -----------------------------
# Data loading helpers
# -----------------------------
MEAS_FILE_1H = "03_results/intermediate_results/volatility_estimates/test_set/{asset}_{measure}_daily_{freq}_test.parquet"
MEAS_FILE_CUM = "03_results/intermediate_results/volatility_estimates/cumulative_forecasts/{asset}_{measure}_daily_{freq}_test_test_cumulative_h{h}.parquet"

# Forecast filename patterns per model (these are robust fallbacks; we also glob)
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
        r"garch_11_test_cumulative_h{h}.parquet",  # cumulative
        r"garch_11_1step_test.parquet",             # if present
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
    "HAR-ANN": "03_results/final_forecasts/ANN",  # if you saved combos under ANN; adjust if needed
}


def try_load_parquet(path: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path):
        try:
            df = pd.read_parquet(path)
            return parse_dates(df)
        except Exception as e:
            # Extra debug when date parse fails
            try:
                df = pd.read_parquet(path)
                print("[DEBUG schema]", path)
                print("index:", type(df.index), getattr(df.index, "dtype", None))
                print("columns:", df.columns.tolist())
                print(df.dtypes)
                print(df.head(3))
            except Exception:
                pass
            print(f"Failed to read {path}: {e}")
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
    # Prefer the column that matches the measure (accept aliases)
    candidates = MEASURE_COLS.get(measure, [measure])
    hit = [c for c in candidates if c in df.columns]
    if hit:
        actual_col = hit[0]
    else:
        # Fall back to the FIRST float column (not last), to avoid accidentally picking helper cols like Avg_n
        float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c]) and c != "date"]
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

    # 先试精确模式
    for token in tokens:
        for pat in PATTERNS.get(model, []):
            fname = pat.format(asset=asset, measure=token, freq=freq, h=h)
            path = os.path.join(base, fname)
            df = try_load_parquet(path)
            if df is not None:
                col = pick_forecast_column(df)
                return df[["date", col]].rename(columns={col: model})

    # 再试软正则
    for token in tokens:
        regex = rf".*{token}.*{freq}.*(h{h}|1step).*\.parquet$"
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
# Core evaluation pipeline
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
             seed: int = 12345, period_mode: str = DEFAULT_PERIOD_MODE) -> None:
    out_dir = os.path.join(root, "03_results", "evaluation_reports")
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    for asset in assets:
        print(f"\n=== Asset: {asset} ===")
        # Collect metrics across periods
        for h in horizons:
            for freq in freqs:
                print(f"-- freq={freq}, h={h}")
                # Load actual series for each measure (some may be missing)
                actual_by_measure: Dict[str, pd.DataFrame] = {}
                for m in measures:
                    df_a = load_actual(root, asset, m, freq, h)
                    if df_a is not None:
                        actual_by_measure[m] = df_a
                if not actual_by_measure:
                    sample_min, sample_max = date_range_of_frames(actual_by_measure.values())
                    periods_cur = resolve_periods(sample_min, sample_max, period_mode)
                    print("   sample:", sample_min.date(), "→", sample_max.date(), "| periods:",
                          {k: (v[0].date(), v[1].date()) for k, v in periods_cur.items()})
                    print(f"No actuals found for asset={asset}, freq={freq}, h={h}. Skipping.")
                    continue
                # --- dynamic periods from sample dates ---
                sample_min = min(df["date"].min() for df in actual_by_measure.values())
                sample_max = max(df["date"].max() for df in actual_by_measure.values())
                span = sample_max - sample_min
                t1 = sample_min + span / 3
                t2 = sample_min + 2 * span / 3
                periods_cur = {
                    "pre": (sample_min, t1.floor("D")),
                    "crisis": ((t1 + pd.Timedelta(days=1)).floor("D"), t2.floor("D")),
                    "post": ((t2 + pd.Timedelta(days=1)).floor("D"), sample_max),
                }
                print("   sample:", sample_min.date(), "→", sample_max.date(), "| periods:",
                      {k: (v[0].date(), v[1].date()) for k, v in periods_cur.items()})



                def _resolve_periods(sample_min, sample_max):
                    # If your sample overlaps the paper windows, use them; otherwise split by tertiles ("auto")
                    overlaps = not (sample_max < PRE_START or sample_min > POST_END)
                    if overlaps:
                        return {"pre": (PRE_START, PRE_END),
                                "crisis": (CRI_START, CRI_END),
                                "post": (POST_START, POST_END)}
                    span = sample_max - sample_min
                    t1 = sample_min + span / 3
                    t2 = sample_min + 2 * span / 3
                    return {
                        "pre": (sample_min, t1.floor("D")),
                        "crisis": ((t1 + pd.Timedelta(days=1)).floor("D"), t2.floor("D")),
                        "post": ((t2 + pd.Timedelta(days=1)).floor("D"), sample_max),
                    }

                periods_cur = _resolve_periods(sample_min, sample_max)
                print("   sample:", sample_min.date(), "→", sample_max.date(), "| periods:",
                      {k: (v[0].date(), v[1].date()) for k, v in periods_cur.items()})

                # Load forecasts per model per measure, align, compute metrics

                # Load forecasts per model per measure, align, compute metrics
                rows = []
                per_period_frames: Dict[str, List[pd.DataFrame]] = {k: [] for k in periods_cur}


                for m, df_a in actual_by_measure.items():
                    # Build a panel with actual + all model forecasts
                    panel = df_a.copy()
                    found_any = False
                    for model in models:
                        df_f = load_forecast(root, model, asset, m, freq, h)
                        if df_f is not None:
                            found_any = True
                            panel = panel.merge(df_f, on="date", how="left")
                    if not found_any:
                        print(f"  No forecasts found for measure={m}. Skipping measure.")
                        continue
                    # For each model present, compute losses
                    for model in [c for c in panel.columns if c not in ("date", "actual")]:
                        # slice by periods
                        for per_name, (start, end) in periods_cur.items():
                            sub = panel[(panel["date"] >= start) & (panel["date"] <= end)][["date", "actual", model]].dropna()
                            if sub.empty:
                                continue
                            comp = loss_components(sub[model].to_numpy(), sub["actual"].to_numpy())
                            agg = aggregate_losses(comp)
                            row = {
                                "asset": asset,
                                "freq": freq,
                                "h": h,
                                "period": per_name,
                                "measure": m,
                                "model": model,
                                **agg,
                                "N": sub.shape[0],
                            }
                            rows.append(row)
                            # preserve loss time series for MCS/SPA later
                            per_period_frames[per_name].append(
                                sub.assign(model=model, measure=m).set_index("date")
                            )

                if not rows:
                    print("No rows computed. Continue.")
                    continue

                metrics_df = pd.DataFrame(rows)
                # Save metrics pivot tables per period
                for per_name in periods_cur:
                    sub = metrics_df[metrics_df["period"] == per_name]
                    if sub.empty:
                        continue
                    # wide tables by model or measure
                    for metric in ["RMSE", "MAE", "MME_O", "MME_U", "OverRate", "UnderRate", "N"]:
                        wide = sub.pivot_table(index=["measure"], columns="model", values=metric)
                        out_path = os.path.join(out_dir, f"{asset}__metrics__{per_name}__h{h}__{freq}__{metric}.csv")
                        wide.to_csv(out_path, float_format="%.6g")

                # ---- MCS & SPA ----
                for per_name in periods_cur:

                    # reconstruct per-measure loss matrices across models (row-wise MCS)
                    # We'll use MAE as the base loss for testing (paper does both RMSE+MAE; adapt as needed)
                    per_frames = [f for f in per_period_frames[per_name] if f.shape[0] > 0]
                    if not per_frames:
                        continue
                    # group by measure
                    group = {}
                    for df in per_frames:
                        m = df["measure"].iloc[0]
                        model = df["model"].iloc[0]
                        # Compute per-time absolute error for MAE loss
                        loss = (df[model] - df["actual"]).abs().rename(model)
                        group.setdefault(m, []).append(loss)

                    # Row-wise MCS (across models) for each measure separately
                    rowwise_records = []
                    spa_records = []
                    for m, loss_list in group.items():
                        loss_mat = pd.concat(loss_list, axis=1).dropna(how='any')
                        if loss_mat.shape[1] < 2 or loss_mat.shape[0] < 20:
                            continue
                        pvals, survivors = mcs_pvalues(loss_mat, alpha, B, avg_block, rng)
                        # Save pvals
                        for name, pv in pvals.items():
                            rowwise_records.append({
                                "asset": asset, "freq": freq, "h": h, "period": per_name,
                                "measure": m, "model": name, "mcs_p": pv, "in_mcs": name in survivors
                            })
                        # SPA for each candidate as benchmark
                        for bench in loss_mat.columns:
                            pv = spa_pvalue(bench, loss_mat, B, avg_block, rng)
                            spa_records.append({
                                "asset": asset, "freq": freq, "h": h, "period": per_name,
                                "measure": m, "benchmark": bench, "spa_p": pv
                            })

                    if rowwise_records:
                        df_row = pd.DataFrame(rowwise_records)
                        path_row = os.path.join(out_dir, f"{asset}__mcs_rowwise__{per_name}__h{h}__{freq}.csv")
                        df_row.to_csv(path_row, index=False)
                    if spa_records:
                        df_spa = pd.DataFrame(spa_records)
                        path_spa = os.path.join(out_dir, f"{asset}__spa__{per_name}__h{h}__{freq}.csv")
                        df_spa.to_csv(path_spa, index=False)

                    # Column-wise MCS (across measures) for each model separately
                    # Build loss matrices keyed by model
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
                        df_col = pd.DataFrame(colwise_records)
                        path_col = os.path.join(out_dir, f"{asset}__mcs_colwise__{per_name}__h{h}__{freq}.csv")
                        df_col.to_csv(path_col, index=False)

                    # ---- Mincer–Zarnowitz regressions (efficiency/info content) ----
                    # For each baseline RM1 (dependent), regress on each (RM, model) forecast
                    mz_rows = []
                    # Build dict: measure -> actual series (aligned)
                    actual_aligned = {m: s.set_index("date")["actual"] for m, s in actual_by_measure.items()}
                    # Build forecasts per (RM, model)
                    fcasts: Dict[Tuple[str, str], pd.Series] = {}
                    for m, df_a in actual_by_measure.items():
                        for model in models:
                            df_f = load_forecast(root, model, asset, m, freq, h)
                            if df_f is None:
                                continue
                            s = df_f.set_index("date")[model]
                            fcasts[(m, model)] = s

                    for rm1, y in actual_aligned.items():
                        # period slice
                        y_per = y[(y.index >= periods_cur[per_name][0]) & (y.index <= periods_cur[per_name][1])]

                        for (rm, model), x in fcasts.items():
                            # align and run OLS
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
                        df_mz = pd.DataFrame(mz_rows)
                        path_mz = os.path.join(out_dir, f"{asset}__mz_efficiency__{per_name}__h{h}__{freq}.csv")
                        df_mz.to_csv(path_mz, index=False)

    print("\nDone. Reports saved under 03_results/evaluation_reports/.")


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate volatility forecasts (Baruník & Křehlič 2016 replication)")
    p.add_argument("--periods", choices=["paper", "auto", "whole"], default=DEFAULT_PERIOD_MODE,
                   help="paper=2006–2012（若与样本不重叠自动退回auto），auto=按样本三等分，whole=不分段")
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
        seed=args.seed, period_mode=args.periods,
    )


if __name__ == "__main__":
    main()
