# -*- coding: utf-8 -*-
"""
严格 MZ 评估（h=5），解耦版 v2
- 读取: ./05_outputs/tables/tmp_h5_merged.csv
- 每个 (rm, freq, model) 同时做两种回归并给出对比：
  A) 比值式 + OLS（HC3）：   y/ŷ = α*(1/ŷ) + β
     -> R2_ratio_ols, alpha_hat_ratio, beta_hat_ratio, wald_p_ratio (H0: β=1, α=0)
  B) 原尺度 + WLS：         y     = α + β*ŷ       ，权重 w = 1/(ŷ^2 + eps)
     -> R2_level_wls, alpha_hat_level, beta_hat_level, wald_p_level (H0: α=0, β=1)
- 输出：
  1) ./05_outputs/tables/mz_h5_gls_strict.csv     （主结果：两种口径的系数/R²/联合检验 + n/n_used）
  2) ./05_outputs/tables/mz_h5_gls_diagnostics.csv（体检指标）
  3) ./05_outputs/logs/mz_h5_gls_strict.log       （详细日志）
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime

MERGED_CSV = "./05_outputs/tables/tmp_h5_merged.csv"
OUT_DIR    = "./05_outputs/tables"
LOG_DIR    = "./05_outputs/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH   = os.path.join(LOG_DIR, "mz_h5_gls_strict.log")

def log(msg: str, reset: bool=False):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "w" if reset else "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)

def panel_diagnostics(df):
    y = df["y_true"].to_numpy()
    x = df["y_pred"].to_numpy()
    n = len(df)
    fx, fy = np.isfinite(x), np.isfinite(y)
    n_nonfinite_x = int((~fx).sum())
    n_nonfinite_y = int((~fy).sum())
    n_le0 = int((x <= 0).sum())
    if fx.any():
        q = np.percentile(x[fx], [0,5,50,95,100])
        var_pred = np.var(x[fx], ddof=1)
    else:
        q = [np.nan]*5; var_pred = np.nan
    var_true = np.var(y[fy], ddof=1) if fy.any() else np.nan
    var_ratio = (var_true/var_pred) if (np.isfinite(var_true) and np.isfinite(var_pred) and var_pred>0) else np.nan
    corr = np.corrcoef(y[fx & fy], x[fx & fy])[0,1] if (fx & fy).sum()>2 else np.nan
    return {
        "n": n,
        "n_nonfinite_pred": n_nonfinite_x,
        "n_nonfinite_true": n_nonfinite_y,
        "pct_nonfinite_pred": n_nonfinite_x/n if n else np.nan,
        "n_pred_le0": n_le0,
        "pct_pred_le0": n_le0/n if n else np.nan,
        "pred_min": q[0], "pred_p05": q[1], "pred_p50": q[2], "pred_p95": q[3], "pred_max": q[4],
        "var_ratio_true_over_pred": var_ratio,
        "corr_true_pred": corr
    }

def fit_ratio_ols(y_true, y_pred, eps=1e-12):
    """A) 比值式 + OLS（HC3 协方差），H0: const=1, inv_pred=0"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred > eps)
    y = (y_true[mask] / y_pred[mask]).astype(float)
    if len(y) < 10:
        return dict(R2=np.nan, alpha=np.nan, beta=np.nan, p=np.nan, n_used=int(len(y)))
    X = pd.DataFrame({"inv_pred": 1.0 / y_pred[mask]})
    X = sm.add_constant(X, has_constant='add')
    # OLS + HC3
    res = sm.OLS(y, X).fit(cov_type="HC3")
    beta = float(res.params.get("const", np.nan))
    alpha = float(res.params.get("inv_pred", np.nan))
    R2   = float(res.rsquared)
    # Wald: const=1, inv_pred=0
    idx = list(res.params.index)
    p = len(idx); R = np.zeros((2,p)); q = np.array([1.0, 0.0])
    if "const" in idx:    R[0, idx.index("const")] = 1.0
    if "inv_pred" in idx: R[1, idx.index("inv_pred")] = 1.0
    try:
        w = res.wald_test((R, q))
        pval = float(w.pvalue)
    except Exception:
        pval = np.nan
    return dict(R2=R2, alpha=alpha, beta=beta, p=pval, n_used=int(len(y)))

def fit_level_wls(y_true, y_pred, eps=1e-12):
    """B) 原尺度 + WLS（w=1/(ŷ^2+eps)），H0: const=0, y_pred=1"""
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_pred > eps)
    y = y_true[mask].astype(float)
    X = pd.DataFrame({"y_pred": y_pred[mask]})
    X = sm.add_constant(X, has_constant='add')
    w = 1.0 / np.maximum(y_pred[mask]**2, eps)
    if len(y) < 10:
        return dict(R2=np.nan, alpha=np.nan, beta=np.nan, p=np.nan, n_used=int(len(y)))
    res = sm.WLS(y, X, weights=w).fit()
    alpha = float(res.params.get("const", np.nan))
    beta  = float(res.params.get("y_pred", np.nan))
    R2    = float(res.rsquared)
    # Wald: const=0, y_pred=1
    idx = list(res.params.index)
    p = len(idx); R = np.zeros((2,p)); q = np.array([0.0, 1.0])
    if "const" in idx:   R[0, idx.index("const")]  = 1.0
    if "y_pred" in idx:  R[1, idx.index("y_pred")] = 1.0
    try:
        wald = res.wald_test((R, q))
        pval = float(wald.pvalue)
    except Exception:
        pval = np.nan
    return dict(R2=R2, alpha=alpha, beta=beta, p=pval, n_used=int(len(y)))

def main():
    log("=== main_mz_h5_gls_strict v2 启动 ===", reset=True)
    if not os.path.exists(MERGED_CSV):
        log(f"[ERROR] 找不到合并文件: {MERGED_CSV}")
        return
    df = pd.read_csv(MERGED_CSV, parse_dates=["date"])
    need = ["date","freq","rm","model","y_true","y_pred"]
    if any(c not in df.columns for c in need):
        log(f"[ERROR] 列缺失，期望 {need}，实际 {list(df.columns)}")
        return

    base_n = len(df)
    df = df[np.isfinite(df["y_true"]) & np.isfinite(df["y_pred"])]
    log(f"[加载] {MERGED_CSV}，原始 {base_n} 行，基础过滤后 {len(df)} 行")

    rows, diags = [], []
    for (rm, freq, model), g in df.groupby(["rm","freq","model"], sort=True):
        # 体检（过滤前）
        d0 = panel_diagnostics(g); d0.update({"rm":rm,"freq":freq,"model":model})

        y = g["y_true"].to_numpy()
        x = g["y_pred"].to_numpy()

        a = fit_ratio_ols(y, x)
        b = fit_level_wls(y, x)

        rows.append({
            "rm":rm,"freq":freq,"model":model,
            "n":int(len(g)),
            # 方案A：比值-OLS
            "n_used_ratio": a["n_used"],
            "alpha_hat_ratio": a["alpha"],
            "beta_hat_ratio":  a["beta"],
            "R2_ratio_ols":    a["R2"],
            "wald_p_ratio_H0_beta1_alpha0": a["p"],
            # 方案B：原尺度-WLS
            "n_used_level": b["n_used"],
            "alpha_hat_level": b["alpha"],
            "beta_hat_level":  b["beta"],
            "R2_level_wls":    b["R2"],
            "wald_p_level_H0_alpha0_beta1": b["p"],
        })

        d0.update({"n_used_ratio":a["n_used"], "n_used_level":b["n_used"]})
        diags.append(d0)

        log(f"[PANEL] (rm={rm}, freq={freq}, model={model}) "
            f"R2_ratio_ols={a['R2']:.6f}, R2_level_wls={b['R2']:.6f}, "
            f"beta_ratio={a['beta']}, beta_level={b['beta']}")

    res = pd.DataFrame(rows).sort_values(["rm","freq","model"]).reset_index(drop=True)
    diag= pd.DataFrame(diags).sort_values(["rm","freq","model"]).reset_index(drop=True)

    p_main = os.path.join(OUT_DIR, "mz_h5_gls_strict.csv")
    p_diag = os.path.join(OUT_DIR, "mz_h5_gls_diagnostics.csv")
    res.to_csv(p_main, index=False)
    diag.to_csv(p_diag, index=False)

    log(f"[落地] 主结果 -> {p_main}, shape={res.shape}")
    log(f"[落地] 体检表 -> {p_diag}, shape={diag.shape}")
    log("=== 完成 ===")

if __name__ == "__main__":
    main()
