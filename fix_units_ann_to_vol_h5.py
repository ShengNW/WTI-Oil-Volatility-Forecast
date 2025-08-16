# -*- coding: utf-8 -*-
# 文件名：fix_units_ann_to_vol_h5.py
"""
把 ANN 的累积预测（σ²）统一到 σ（开方），不改其他模型，保持现有 main 流程不变。
运行时机：main_read_h5.py 之后、main_mz_h5.py 之前。
"""

import os
import sys
import time
import math
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLE_DIR = ROOT / "05_outputs" / "tables"
PRED_PATH = TABLE_DIR / "tmp_h5_y_pred.csv"

def main():
    if not PRED_PATH.exists():
        print(f"[ERROR] 找不到 {PRED_PATH}")
        sys.exit(1)

    df = pd.read_csv(PRED_PATH)
    need_cols = {"date", "freq", "rm", "model", "y_pred"}
    if not need_cols.issubset(df.columns):
        print(f"[ERROR] {PRED_PATH} 缺少列：{need_cols - set(df.columns)}")
        sys.exit(2)

    # 只处理 ANN
    mask_ann = (df["model"] == "ANN")
    n_ann = int(mask_ann.sum())
    if n_ann == 0:
        print("[WARN] y_pred 中没有 ANN 行，无需处理。")
        return

    # 处理前的诊断
    ann_before = df.loc[mask_ann, "y_pred"].astype(float)
    print("=== [ANN before] 摘要 ===")
    print(ann_before.describe(percentiles=[0.05, 0.5, 0.95]).to_string())

    # clip 到 >=0，然后开方
    ann_fixed = ann_before.clip(lower=0).apply(math.sqrt)

    # 写回
    df.loc[mask_ann, "y_pred"] = ann_fixed.values

    # 处理后的诊断
    print("=== [ANN after] 摘要 ===")
    print(df.loc[mask_ann, "y_pred"].astype(float).describe(percentiles=[0.05, 0.5, 0.95]).to_string())

    # 备份 + 覆盖
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = TABLE_DIR / f"tmp_h5_y_pred.bak_{ts}.csv"
    PRED_PATH.replace(bak)  # 原文件先改名成备份
    df.to_csv(PRED_PATH, index=False)
    print(f"[OK] 已把 ANN 的 σ²→σ（开方），备份到 {bak.name}，覆盖写回 {PRED_PATH.name}")

if __name__ == "__main__":
    main()
