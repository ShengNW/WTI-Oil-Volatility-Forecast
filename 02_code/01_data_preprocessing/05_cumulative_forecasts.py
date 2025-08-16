import pandas as pd
import os
import sys
import glob
from tqdm import tqdm


def main(project_dir):
    # 设置输入输出路径
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
            vol_columns = [col for col in df.columns if col not in ['N_actual', 'N', 'G', 'Avg_n']]

            # 处理每个波动率指标
            for vol_col in vol_columns:
                # 1步预测（直接使用原始数据）
                df_h1 = df[[vol_col]].copy()
                save_cumulative_data(df_h1, base_name, vol_col, "h1", output_dir)

                # 5步累计预测
                df_h5 = calculate_cumulative_forecast(df, vol_col, window=5)
                save_cumulative_data(df_h5, base_name, vol_col, "h5", output_dir)

                # 10步累计预测
                df_h10 = calculate_cumulative_forecast(df, vol_col, window=10)
                save_cumulative_data(df_h10, base_name, vol_col, "h10", output_dir)

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

    print(f"所有累计预测数据已保存至: {output_dir}")


def calculate_cumulative_forecast(df, vol_col, window):
    """
    计算累计预测值
    :param df: 包含波动率数据的DataFrame
    :param vol_col: 波动率列名
    :param window: 预测步长 (5或10)
    :return: 包含累计预测值的DataFrame
    """
    # 复制数据避免修改原始DataFrame
    result = df[[vol_col]].copy()

    # 创建新列存储累计预测值
    cum_col = f"{vol_col}_cumulative_h{window}"

    # 使用rolling窗口计算未来window天的平均值
    # shift(-1)确保从下一天开始计算
    # result[cum_col] = (
    #     result[vol_col]
    #     .shift(-1)
    #     .rolling(window=window, min_periods=window)
    #     .mean()
    # )
    # 拆分后
    shifted = result[vol_col].shift(-1)  # 第一步：移位
    rolled = shifted.rolling(window=window, min_periods=window)  # 第二步：创建滚动窗口对象
    # result[cum_col] = rolled.mean()  # 第三步：计算均值
    # 替换原第三步代码，添加打印逻辑
    def debug_mean(window):
        # 打印当前窗口的索引和值（窗口是一个Series）
        print(f"窗口索引: {window.index.tolist()}")
        print(f"窗口值: {window.values.tolist()}")
        print(f"非NaN数量: {window.count()}")  # 检查是否满足min_periods
        mean_val = window.mean()
        print(f"计算结果: {mean_val}\n")
        return mean_val

    # 用自定义函数替代mean()，同时打印窗口细节
    #result[cum_col] = shifted.rolling(window=window, min_periods=window).apply(debug_mean)
    # 在第三步前添加打印，检查索引
    rolling_result = shifted.rolling(window=window, min_periods=window).mean()

    # 打印滚动结果的索引和前几行值
    a= rolling_result.index.tolist()
    b= rolling_result.head().tolist()

    # 打印result的索引
    c= result.index.tolist()

    # 再赋值
    result[cum_col] = rolling_result
    # 移除无法形成完整窗口的行（最后window-1行）
    result = result.iloc[:-(window - 1)]

    # 只保留累计预测列
    result = result[[cum_col]]

    return result


def save_cumulative_data(df, base_name, vol_col, horizon, output_dir):
    """
    保存累计预测数据到Parquet文件
    :param df: 包含数据的DataFrame
    :param base_name: 基础文件名
    :param vol_col: 波动率列名
    :param horizon: 预测范围 (h1, h5, h10)
    :param output_dir: 输出目录
    """
    # 生成文件名
    # 对于realized_kernel文件特殊处理
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
    # 检查是否提供了项目目录参数
    # if len(sys.argv) != 2:
    #     print("请提供项目目录作为参数")
    #     print(
    #         "示例: python 05_cumulative_forecasts.py F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication")
    #     sys.exit(1)
    #
    # project_dir = sys.argv[1]
    # print(f"项目目录: {project_dir}")
    project_dir="F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    main(project_dir)