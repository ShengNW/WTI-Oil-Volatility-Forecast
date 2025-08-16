import pandas as pd
import numpy as np
import os
import sys
from arch import arch_model
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


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
    shifted = result[vol_col].shift(-1)  # 移位
    rolling_result = shifted.rolling(window=window, min_periods=window).mean()
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
    output_name = f"{base_name}_test_cumulative_{horizon}.parquet"

    # 保存文件
    output_path = os.path.join(output_dir, output_name)
    df.to_parquet(output_path)


def garch_rolling_forecast(returns, window_size=600, forecast_horizon=10):
    """
    执行GARCH(1,1)滚动预测
    :param returns: 对数收益率序列
    :param window_size: 滚动窗口大小
    :param forecast_horizon: 最大预测步长
    :return: 包含预测结果的DataFrame
    """
    forecasts = pd.DataFrame(index=returns.index,
                             columns=[f'garch_forecast_h{i}' for i in range(1, forecast_horizon + 1)])

    # 使用滚动窗口进行预测
    for i in range(window_size, len(returns)):
        # 获取当前窗口数据
        train_data = returns.iloc[i - window_size:i]

        # 拟合GARCH(1,1)模型
        model = arch_model(train_data, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')

        # 生成多步预测
        forecast = res.forecast(horizon=forecast_horizon, reindex=False)
        variance_forecasts = forecast.variance.values[-1]

        # 存储预测结果
        for h in range(1, forecast_horizon + 1):
            col_name = f'garch_forecast_h{h}'
            if h <= len(variance_forecasts):
                #forecasts.loc[returns.index[i], col_name] = variance_forecasts[h - 1]
                forecasts.loc[returns.index[i], col_name] = np.sqrt(variance_forecasts[h - 1])

    # 只保留有预测结果的行（去掉前window_size行）
    forecasts = forecasts.iloc[window_size:]

    return forecasts


def main(project_dir):
    # 设置路径
    daily_test_path = os.path.join(project_dir, "01_data", "processed", "daily_returns", "daily_test.parquet")
    test_set_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates", "test_set")
    # cumulative_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates",
    #                               "cumulative_forecasts")
    cumulative_dir = os.path.join(project_dir, "03_results", "final_forecasts", "GARCH")
    # 确保目录存在
    os.makedirs(test_set_dir, exist_ok=True)
    os.makedirs(cumulative_dir, exist_ok=True)

    print("加载日线对数收益率数据...")
    daily_test = pd.read_parquet(daily_test_path)

    # 设置日期索引
    if 'DateTime' in daily_test.columns:
        daily_test = daily_test.set_index('DateTime')
    daily_test.index = pd.to_datetime(daily_test.index)

    # 提取对数收益率
    returns = daily_test['LogReturn'].dropna()

    print("执行GARCH(1,1)滚动预测...")
    # 生成预测（最大预测步长为10）
    garch_forecasts = garch_rolling_forecast(returns, window_size=600, forecast_horizon=10)

    # 保存单步预测结果
    garch_h1 = garch_forecasts[['garch_forecast_h1']].rename(columns={'garch_forecast_h1': 'garch_11'})
    garch_h1_path = os.path.join(test_set_dir, "garch_11_test.parquet")
    garch_h1.to_parquet(garch_h1_path)
    print(f"单步预测结果保存至: {garch_h1_path}")

    # 计算累积预测
    print("计算累积预测...")
    base_name = "garch_11"

    # 1步预测（直接使用原始数据）
    save_cumulative_data(garch_h1, base_name, 'garch_11', "h1", cumulative_dir)

    # 5步累计预测
    garch_h5 = calculate_cumulative_forecast(garch_forecasts, 'garch_forecast_h1', window=5)
    save_cumulative_data(garch_h5, base_name, 'garch_forecast_h1_cumulative_h5', "h5", cumulative_dir)

    # 10步累计预测
    garch_h10 = calculate_cumulative_forecast(garch_forecasts, 'garch_forecast_h1', window=10)
    save_cumulative_data(garch_h10, base_name, 'garch_forecast_h1_cumulative_h10', "h10", cumulative_dir)

    print(f"累积预测结果保存至: {cumulative_dir}")


if __name__ == "__main__":
    # # 检查是否提供了项目目录参数
    # if len(sys.argv) != 2:
    #     print("请提供项目目录作为参数")
    #     print(
    #         "示例: python garch_cumulative_forecasts.py F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication")
    #     sys.exit(1)
    #
    # project_dir = sys.argv[1]
    project_dir = f"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
    print(f"项目目录: {project_dir}")
    main(project_dir)