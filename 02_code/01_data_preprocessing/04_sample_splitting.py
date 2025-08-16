import os
import pandas as pd
from pathlib import Path


def main():
    # 设置路径
    input_dir = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates")
    train_dir = input_dir / "train_set"
    test_dir = input_dir / "test_set"

    # 创建输出目录
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有parquet文件
    files = list(input_dir.glob("*.parquet"))
    print(f"Found {len(files)} volatility files to process")

    # 处理每个文件
    for file in files:
        print(f"\nProcessing {file.name}...")

        # 读取数据
        df = pd.read_parquet(file)

        # 确保按时间排序
        df = df.sort_index()
        print(f"Data range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"Total rows: {len(df)}")

        # 划分训练集和测试集
        train = df.iloc[:600].copy()
        test = df.iloc[600:].copy()

        print(f"Train set: {len(train)} rows ({train.index.min().date()} to {train.index.max().date()})")
        print(f"Test set: {len(test)} rows ({test.index.min().date()} to {test.index.max().date()})")

        # 保存结果
        train_output = train_dir / f"{file.stem}_train.parquet"
        test_output = test_dir / f"{file.stem}_test.parquet"

        train.to_parquet(train_output)
        test.to_parquet(test_output)

        print(f"Saved train set to {train_output}")
        print(f"Saved test set to {test_output}")

    print("\nSample splitting completed successfully!")


if __name__ == "__main__":
    main()