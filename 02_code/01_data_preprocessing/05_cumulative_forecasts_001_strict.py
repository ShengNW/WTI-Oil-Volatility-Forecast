import pandas as pd
import os
import sys
import glob
from tqdm import tqdm
import numpy as np


def main(project_dir):
    # 设置输入输出路径（保持原样）
    test_set_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates", "test_set")
    output_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates",
                              "cumulative_forecasts")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有测试集文件
    test_files = glob.glob(os.path.join(test_set_dir, "*.parquet"))

    print(f"找到 {len(test_files)} 个测试集文件，开始处理累计预测数据...")

    # 处理每个测试集文件
    for file_path in tqdm(test_files, desc="处理文件"):
        try:
            # 读取数据
            df = pd.read_parquet(file_path)

            # 获取文件名（不含路径和后缀）
            file_name = os.path.basename(file_path)
            base_name = file_name.replace(".parquet", "")

            # 识别波动率指标列（排除N_actual等辅助列）
            exclude_cols = {'N_actual', 'N', 'G', 'Avg_n'}
            vol_columns = [col for col in df.columns if col not in exclude_cols]

            # 处理每个波动率指标
            for vol_col in vol_columns:
                # ---- h=1：直接使用原始σ（保持原样）----
                df_h1 = df[[vol_col]].copy()
                save_cumulative_data(df_h1, base_name, vol_col, "h1", output_dir)

                # ---- h=5 累计（√(平均 σ^2)）----
                df_h5 = calculate_cumulative_forecast_sigma(df, vol_col, h=5)
                save_cumulative_data(df_h5, base_name, vol_col, "h5", output_dir)

                # ---- h=10 累计（√(平均 σ^2)）----
                df_h10 = calculate_cumulative_forecast_sigma(df, vol_col, h=10)
                save_cumulative_data(df_h10, base_name, vol_col, "h10", output_dir)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    print(f"所有累计预测数据已保存至: {output_dir}")


def calculate_cumulative_forecast_sigma(df: pd.DataFrame, vol_col: str, h: int) -> pd.DataFrame:
    """
    按论文口径计算σ的h步“累计预测标签”：
        sigma_cum(t, h) = sqrt( (1/h) * sum_{j=1..h} sigma_{t+j}^2 )

    实现：对 sigma^2 做 rolling(h).sum() 后整体 shift(-h)，使得行 t 对应未来 [t+1, ..., t+h] 的窗口。
    末尾不足 h 行会是 NaN，按对齐逻辑删除尾部 h 行。
    """
    s = pd.to_numeric(df[vol_col], errors='coerce').astype(float)

    # 未来窗口的方差和：在索引 t 得到 sum_{t+1..t+h} sigma^2
    sum_future = (s.pow(2)
                    .rolling(window=h, min_periods=h)
                    .sum()
                    .shift(-h))

    # 平均并开根号，得到累计σ
    sigma_cum = (sum_future / h).apply(np.sqrt)

    # 生成结果列名与对齐（去除末尾无法形成完整窗口的 h 行）
    cum_col = f"{vol_col}_cumulative_h{h}"
    out = pd.DataFrame({cum_col: sigma_cum})

    # 严格按未来窗口对齐：删除尾部 h 行（这些行为 NaN）
    if len(out) >= h:
        out = out.iloc[:-h]
    else:
        out = out.iloc[0:0]  # 数据太短则返回空

    return out


def save_cumulative_data(df, base_name, vol_col, horizon, output_dir):
    """
    保存累计预测数据到Parquet文件（保持你的命名与 RK 特判）
    """
    # 生成文件名
    if "realized_kernel" in base_name:
        # 从列名中提取频率标识 (RK_5min 或 RK_1min)
        freq_suffix = vol_col.split("_")[-1]
        output_name = f"{base_name}_{freq_suffix}_test_cumulative_{horizon}.parquet"
    else:
        output_name = f"{base_name}_test_cumulative_{horizon}.parquet"

    # 保存文件
    output_path = os.path.join(output_dir, output_name)
    df.to_parquet(output_path)


if __name__ == "__main__":
    # 你原来的默认路径（保持原样）
    project_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    main(project_dir)
