# -*- coding: utf-8 -*-
"""
main_mz_h10.py
- 独立于 h1/h5，不修改任何已有源码
- 流程：读 YAML -> adapters/reader_h10.py -> 内连接 -> OLS 版 MZ -> 导出表 & 画图
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 仅依赖我们新建的 reader_h10
try:
    from adapters.reader_h10 import load_h10_tables
except Exception as e:
    print("[ERROR] 请先把 adapters/reader_h10.py 放到项目中。")
    raise

# ----------- OLS MZ: y = α + β x -----------
def _ols_mz(y: np.ndarray, x: np.ndarray):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = y.size
    if n < 5:
        return np.nan, np.nan, np.nan, n
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        beta_hat = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    y_hat = X @ beta_hat
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 0.0 if ss_tot <= 1e-30 else max(0.0, 1.0 - ss_res / ss_tot)
    return alpha, beta, R2, n

def _safe_makedirs(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _join_true_pred(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    on_cols = ["date","freq","rm"]
    merged = pd.merge(y_true, y_pred, on=on_cols, how="inner")
    return merged.sort_values(on_cols + ["model","date"]).reset_index(drop=True)

def _summarize_r2(df_joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rm, model, freq), g in df_joined.groupby(["rm","model","freq"], dropna=False):
        a, b, r2, n = _ols_mz(g["y_true_var"].values, g["y_pred_var"].values)
        rows.append({"RM": rm, "Model": model, "Freq": freq, "alpha": a, "beta": b, "R2": r2, "n": n})
    return pd.DataFrame(rows).sort_values(["Freq","RM","Model"]).reset_index(drop=True)

def _plot_r2_bar(df_r2: pd.DataFrame, cfg_plot: dict, out_path: str):
    order_models = cfg_plot.get("order_models", sorted(df_r2["Model"].unique()))
    order_rms    = cfg_plot.get("order_rms",    sorted(df_r2["RM"].unique()))
    freqs        = sorted(df_r2["Freq"].unique())
    n_rm, n_freq = len(order_rms), len(freqs)
    fig_h = 3.2 * n_freq
    fig_w = 1.8 * max(6, len(order_models))
    fig, axes = plt.subplots(n_freq, n_rm, figsize=(fig_w, fig_h), squeeze=False)
    for r, freq in enumerate(freqs):
        for c, rm in enumerate(order_rms):
            ax = axes[r][c]
            sub = df_r2[(df_r2["Freq"]==freq) & (df_r2["RM"]==rm)].set_index("Model").reindex(order_models)
            vals = sub["R2"].values
            x = np.arange(len(order_models))
            ax.bar(x, vals)
            ymax = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
            ax.set_ylim(0, max(0.01, ymax) * 1.1)
            ax.set_xticks(x); ax.set_xticklabels(order_models, rotation=45, ha="right")
            ax.set_title(f"{rm} ({freq})");
            if c == 0: ax.set_ylabel("R²")
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _safe_makedirs(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    root = os.path.dirname(__file__)
    cfg_path = os.path.join(root, "04_configs", "mz_h10_config.yaml")
    cfg = _load_yaml(cfg_path)

    # 1) 读取（h10 累积；单位已在 reader 中统一为 VAR）
    tables = load_h10_tables(cfg)
    y_true, y_pred = tables["y_true"], tables["y_pred"]

    # 2) 对齐
    joined = _join_true_pred(y_true, y_pred)

    # 3) 分组 MZ (OLS)
    df_r2 = _summarize_r2(joined)

    # 4) 落盘
    out_csv = cfg.get("output_table", "05_outputs/tables/mz_h10_r2.csv")
    out_png = cfg.get("plot", {}).get("out_path", "05_outputs/figures/Fig3_MZ_R2_h10.png")
    _safe_makedirs(os.path.join(root, out_csv))
    _safe_makedirs(os.path.join(root, out_png))
    df_r2.to_csv(os.path.join(root, out_csv), index=False)

    # 5) 画图
    _plot_r2_bar(df_r2, cfg.get("plot", {}), os.path.join(root, out_png))

    print(f"[OK] MZ(h=10) R² 表 -> {os.path.join(root, out_csv)}  (shape={df_r2.shape})")
    print(f"[OK] 图 -> {os.path.join(root, out_png)}")

if __name__ == "__main__":
    main()
