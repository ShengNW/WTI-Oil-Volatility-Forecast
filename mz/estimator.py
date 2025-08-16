# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def ols_fit(y: np.ndarray, x: np.ndarray):
    """
    OLS 拟合 y = a + b x，返回 a,b,R2。先给出最小实现，后续再加 HAC/GLS。
    """
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    a, b = float(beta[0]), float(beta[1])
    return a, b, r2

def run_mz_grouped(df_design: pd.DataFrame) -> pd.DataFrame:
    """
    对 (RM, model) 分组做 OLS，返回 R2 与系数。后续可扩展 WLS/HAC。
    """
    rows = []
    for (rm, model), g in df_design.groupby(["RM","model"]):
        y = g["y"].to_numpy(float)
        x = g["x"].to_numpy(float)
        if len(g) < 5 or np.allclose(x.std(), 0):
            rows.append({"RM": rm, "model": model, "alpha": np.nan, "beta": np.nan, "R2": np.nan, "n": len(g)})
            continue
        a, b, r2 = ols_fit(y, x)
        rows.append({"RM": rm, "model": model, "alpha": a, "beta": b, "R2": r2, "n": len(g)})
    return pd.DataFrame(rows)
