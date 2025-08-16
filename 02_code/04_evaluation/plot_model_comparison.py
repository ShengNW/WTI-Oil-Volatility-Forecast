#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Defaults ----------
DEFAULT_ASSETS   = ["CL_WTI"]
DEFAULT_FREQS    = ["1min", "5min"]
DEFAULT_PERIODS  = ["pre", "crisis", "post"]
DEFAULT_HORIZONS = [1, 5, 10]
DEFAULT_METRICS  = ["RMSE", "MAE"]
DEFAULT_PLOTS    = ["heatmap", "bar"]

# 若未传 --root，则按以下顺序猜测：
# 1) 环境变量 WTI_REPL_ROOT
# 2) 以当前脚本位置为基准，回退两级到项目根（.../02_code/04_evaluation -> 根）
# 3) 固定回退路径（你之前用的 F 盘）；如果不存在，就退回到 2)
def guess_root() -> str:
    env = os.environ.get("WTI_REPL_ROOT")
    if env and os.path.isdir(env):
        return env
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_from_here = os.path.abspath(os.path.join(script_dir, "..", ".."))
    if os.path.isdir(os.path.join(root_from_here, "03_results")):
        return root_from_here
    fallback = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
    if os.path.isdir(os.path.join(fallback, "03_results")):
        return fallback
    return root_from_here  # 最后兜底

def load_metric_table(out_dir, asset, period, h, freq, metric):
    path = os.path.join(out_dir, f"{asset}__metrics__{period}__h{h}__{freq}__{metric}.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, index_col=0)
    return df  # index: measure, columns: model

def heatmap(df, title, savepath):
    plt.figure(figsize=(10, 5))
    im = plt.imshow(df.values, aspect='auto', interpolation='nearest')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(df.shape[1]), df.columns, rotation=45, ha='right')
    plt.yticks(range(df.shape[0]), df.index)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200)
    plt.close()

def bargrid(df, title, saveprefix):
    # 每个度量单独一张柱状图
    for m in df.index:
        plt.figure(figsize=(8, 4))
        vals = df.loc[m]
        vals.plot(kind='bar')
        plt.ylabel('Value')
        plt.title(f"{title} | {m}")
        plt.tight_layout()
        plt.savefig(f"{saveprefix}__{m}.png", dpi=200)
        plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=guess_root())
    ap.add_argument("--assets",  nargs="*", default=DEFAULT_ASSETS)
    ap.add_argument("--freqs",   nargs="*", default=DEFAULT_FREQS)
    ap.add_argument("--periods", nargs="*", default=DEFAULT_PERIODS)
    ap.add_argument("--horizons", nargs="*", type=int, default=DEFAULT_HORIZONS)
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--plots",   nargs="*", default=DEFAULT_PLOTS)
    args = ap.parse_args()

    root = args.root
    out_dir = os.path.join(root, "03_results", "evaluation_reports")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"[plot] project root = {root}")
    print(f"[plot] figures -> {fig_dir}")

    for asset in args.assets:
        for freq in args.freqs:
            for period in args.periods:
                for h in args.horizons:
                    for metric in args.metrics:
                        df = load_metric_table(out_dir, asset, period, h, freq, metric)
                        if df is None or df.empty:
                            continue
                        # 统一模型顺序（存在的才保留）
                        prefer = ["GARCH","ARFIMA","HAR","ANN","HAR-ANN"]
                        cols = [c for c in prefer if c in df.columns] + [c for c in df.columns if c not in prefer]
                        df = df.loc[:, cols]
                        title = f"{asset} | {freq} | {period} | h={h} | {metric}"
                        base = os.path.join(fig_dir, f"{asset}__{freq}__{period}__h{h}__{metric}")
                        if "heatmap" in args.plots:
                            heatmap(df, title, base + "__heatmap.png")
                        if "bar" in args.plots:
                            bargrid(df, title, base + "__bar")

    print("[plot] done.")

if __name__ == "__main__":
    main()
