import pandas as pd
import os
import glob
from tqdm import tqdm


def main(project_dir):
    # 输入输出路径（保持不变）
    test_set_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates", "test_set")
    output_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates",
                              "cumulative_forecasts")

    os.makedirs(output_dir, exist_ok=True)

    # 只获取文件名中包含 TSRV 的测试集文件
    test_files = [f for f in glob.glob(os.path.join(test_set_dir, "*.parquet")) if "TSRV" in os.path.basename(f).upper()]
    print(f"找到 {len(test_files)} 个 TSRV 测试集文件，开始处理累计预测数据...")

    # 遍历处理
    for file_path in tqdm(test_files, desc="处理文件"):
        try:
            df = pd.read_parquet(file_path)

            file_name = os.path.basename(file_path)
            base_name = file_name.replace(".parquet", "")

            # 波动率列（排除辅助列）
            vol_columns = [col for col in df.columns if col not in ['N_actual', 'N', 'G', 'Avg_n']]

            for vol_col in vol_columns:
                # 1步
                df_h1 = df[[vol_col]].copy()
                save_cumulative_data(df_h1, base_name, vol_col, "h1", output_dir)

                # 5步
                df_h5 = calculate_cumulative_forecast(df, vol_col, window=5)
                save_cumulative_data(df_h5, base_name, vol_col, "h5", output_dir)

                # 10步
                df_h10 = calculate_cumulative_forecast(df, vol_col, window=10)
                save_cumulative_data(df_h10, base_name, vol_col, "h10", output_dir)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    print(f"所有 TSRV 累计预测数据已保存至: {output_dir}")


def calculate_cumulative_forecast(df, vol_col, window):
    """计算累计预测值"""
    result = df[[vol_col]].copy()
    cum_col = f"{vol_col}_cumulative_h{window}"
    shifted = result[vol_col].shift(-1)
    rolling_result = shifted.rolling(window=window, min_periods=window).mean()
    result[cum_col] = rolling_result
    result = result.iloc[:-(window - 1)]  # 去掉末尾不足窗口的行
    return result[[cum_col]]


def save_cumulative_data(df, base_name, vol_col, horizon, output_dir):
    """保存累计预测数据"""
    if "realized_kernel" in base_name:
        freq_suffix = vol_col.split("_")[-1]
        output_name = f"{base_name}_{freq_suffix}_test_cumulative_{horizon}.parquet"
    else:
        output_name = f"{base_name}_test_cumulative_{horizon}.parquet"

    output_path = os.path.join(output_dir, output_name)
    df.to_parquet(output_path)


if __name__ == "__main__":
    project_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    main(project_dir)
