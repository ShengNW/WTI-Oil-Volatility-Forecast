#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
from pathlib import Path

def main():
    # 路径保持不变
    input_dir = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates")
    train_dir = input_dir / "train_set"
    test_dir = input_dir / "test_set"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 仅处理“TSRV”文件（不改原目录与文件名规则）
    files = [p for p in input_dir.glob("*.parquet") if "TSRV" in p.stem.upper()]
    print(f"Found {len(files)} TSRV files to process")

    for file in files:
        print(f"\nProcessing {file.name}...")

        # 读取
        df = pd.read_parquet(file)

        # 确保有时间索引（兼容存在 'DateTime' 列的情况）
        if not isinstance(df.index, pd.DatetimeIndex):
            if "DateTime" in df.columns:
                df["DateTime"] = pd.to_datetime(df["DateTime"])
                df = df.set_index("DateTime")
            else:
                raise ValueError(f"{file.name}: no DatetimeIndex and no 'DateTime' column found.")

        # 排序
        df = df.sort_index()
        print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Total rows: {len(df)}")

        # 划分（前600行为训练，之后为测试；长度不足也能安全运行）
        cut = min(600, len(df))
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()

        print(f"Train set: {len(train)} rows ({train.index.min().date()} to {train.index.max().date()})")
        if len(test) > 0:
            print(f"Test set:  {len(test)} rows ({test.index.min().date()} to {test.index.max().date()})")
        else:
            print("Test set:  0 rows (dataset shorter than 600)")

        # 保存（文件名规则不变）
        train_output = train_dir / f"{file.stem}_train.parquet"
        test_output = test_dir / f"{file.stem}_test.parquet"

        train.to_parquet(train_output)
        test.to_parquet(test_output)

        print(f"Saved train set to {train_output}")
        print(f"Saved test set  to {test_output}")

    print("\nSample splitting for TSRV completed successfully!")

if __name__ == "__main__":
    main()
