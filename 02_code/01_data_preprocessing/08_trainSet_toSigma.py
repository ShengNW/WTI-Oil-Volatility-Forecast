import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(project_dir: str):
    """
    将训练集中的方差列统一转换为标准差列（sigma），并另存为 *_sigma.parquet。
    """
    # === 路径 ===
    train_set_dir = os.path.join(project_dir, "03_results", "intermediate_results",
                                 "volatility_estimates", "train_set")
    output_dir = os.path.join(project_dir, "03_results", "intermediate_results",
                              "volatility_estimates", "train_set_sigma")
    os.makedirs(output_dir, exist_ok=True)

    # 收集 parquet 文件
    train_files = glob.glob(os.path.join(train_set_dir, "*.parquet"))
    print(f"[INFO] 在训练集目录找到 {len(train_files)} 个 parquet 文件")
    print(f"[INFO] 输入目录: {train_set_dir}")
    print(f"[INFO] 输出目录: {output_dir}")

    # 已知的“方差度量”列
    known_variance_like = {"BV", "JV", "JWTSRV", "MedRV", "RK", "TSRV", "RV"}

    for file_path in tqdm(train_files, desc="处理文件", ncols=100):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"[WARN] 无法读取 {file_path}: {e}")
            continue

        # 找出需要转换的列（数值型且可能是方差）
        exclude_cols = {'N_actual', 'N', 'G', 'Avg_n'}
        num_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
        var_cols = [c for c in num_cols if c in known_variance_like or any(k in c for k in known_variance_like)]
        if not var_cols:
            print(f"[WARN] {os.path.basename(file_path)} 未找到方差列，跳过")
            continue

        # 创建新 DataFrame 保存 sigma 列
        sigma_df = pd.DataFrame(index=df.index)

        for v_col in var_cols:
            v = pd.to_numeric(df[v_col], errors='coerce').astype(float).clip(lower=0)
            sigma_df[f"{v_col}_sigma"] = np.sqrt(v)

        # 输出文件名
        base_name = os.path.basename(file_path).replace(".parquet", "_sigma.parquet")
        output_path = os.path.join(output_dir, base_name)

        try:
            sigma_df.to_parquet(output_path, index=True)
        except Exception as e:
            alt = output_path.replace(".parquet", ".csv")
            print(f"[WARN] 写入 parquet 失败（{e}），改为 csv：{alt}")
            sigma_df.to_csv(alt, index=True)

    print(f"[DONE] 训练集 sigma 数据已全部输出到：{output_dir}")


if __name__ == "__main__":
    project_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    main(project_dir)
