import pandas as pd
import numpy as np
import os
import sys
from arch import arch_model
import warnings

warnings.filterwarnings('ignore')


def calculate_cumulative_forecast(df, vol_col, window):
    """计算累计预测值"""
    result = df[[vol_col]].copy()
    cum_col = f"{vol_col}_cumulative_h{window}"
    shifted = result[vol_col].shift(-1)
    rolling_result = shifted.rolling(window=window, min_periods=window).mean()
    result[cum_col] = rolling_result
    result = result.iloc[:-(window - 1)]
    result = result[[cum_col]]
    return result


def save_cumulative_data(df, base_name, vol_col, horizon, output_dir):
    """保存累计预测数据到Parquet文件"""
    output_name = f"{base_name}_test_cumulative_{horizon}.parquet"
    output_path = os.path.join(output_dir, output_name)
    df.to_parquet(output_path)


def garch_rolling_forecast(returns, train_size=600, forecast_horizon=10):
    """
    执行GARCH(1,1)滚动预测
    :param returns: 合并后的训练集+测试集收益率序列
    :param train_size: 训练集大小（600）
    :param forecast_horizon: 最大预测步长
    :return: 包含预测结果的DataFrame
    """
    # 预测结果的索引为测试集的索引（从训练集结束后开始）
    test_returns = returns.iloc[train_size:]
    forecasts = pd.DataFrame(
        index=test_returns.index,
        columns=[f'garch_forecast_h{i}' for i in range(1, forecast_horizon + 1)]
    )

    # 滚动预测：每次用前i条数据（包含训练集和已预测的测试集数据）训练模型
    for i in range(train_size, len(returns)):
        # 训练数据：从开始到第i条（包含训练集和部分测试集数据）
        train_data = returns.iloc[:i]
        # 拟合GARCH(1,1)模型
        model = arch_model(train_data, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')
        # 生成多步预测
        forecast = res.forecast(horizon=forecast_horizon, reindex=False)
        variance_forecasts = forecast.variance.values[-1]
        # 存储预测结果（对应测试集的第i - train_size条数据）
        for h in range(1, forecast_horizon + 1):
            col_name = f'garch_forecast_h{h}'
            if h <= len(variance_forecasts):
                #forecasts.loc[returns.index[i], col_name] = variance_forecasts[h - 1]
                forecasts.loc[returns.index[i], col_name] = np.sqrt(variance_forecasts[h - 1])

    return forecasts


def main(project_dir):
    # 设置路径
    daily_train_path = os.path.join(project_dir, "01_data", "processed", "daily_returns",
                                    "daily_train.parquet")  # 新增训练集路径
    daily_test_path = os.path.join(project_dir, "01_data", "processed", "daily_returns", "daily_test.parquet")
    test_set_dir = os.path.join(project_dir, "03_results", "intermediate_results", "volatility_estimates", "test_set")
    cumulative_dir = os.path.join(project_dir, "03_results", "final_forecasts", "GARCH")

    # 确保目录存在
    os.makedirs(test_set_dir, exist_ok=True)
    os.makedirs(cumulative_dir, exist_ok=True)

    print("加载训练集和测试集对数收益率数据...")
    # 加载训练集和测试集
    daily_train = pd.read_parquet(daily_train_path)
    daily_test = pd.read_parquet(daily_test_path)

    # 统一设置日期索引
    for df in [daily_train, daily_test]:
        if 'DateTime' in df.columns:
            df.set_index('DateTime', inplace=True)
        df.index = pd.to_datetime(df.index)

    # 提取收益率并合并（训练集在前，测试集在后）
    train_returns = daily_train['LogReturn'].dropna()
    test_returns = daily_test['LogReturn'].dropna()
    all_returns = pd.concat([train_returns, test_returns], axis=0)  # 合并为完整序列

    print("执行GARCH(1,1)滚动预测...")
    # 生成预测（使用训练集大小600，测试集从索引600开始）
    garch_forecasts = garch_rolling_forecast(
        returns=all_returns,
        train_size=len(train_returns),  # 使用训练集的实际长度（600）
        forecast_horizon=10
    )

    # 保存单步预测结果
    garch_h1 = garch_forecasts[['garch_forecast_h1']].rename(columns={'garch_forecast_h1': 'garch_11'})
    garch_h1_path = os.path.join(test_set_dir, "garch_11_test.parquet")
    garch_h1.to_parquet(garch_h1_path)
    print(f"单步预测结果保存至: {garch_h1_path}")

    # 计算累积预测
    print("计算累积预测...")
    base_name = "garch_11"

    # 1步预测
    save_cumulative_data(garch_h1, base_name, 'garch_11', "h1", cumulative_dir)

    # 5步累计预测
    garch_h5 = calculate_cumulative_forecast(garch_forecasts, 'garch_forecast_h1', window=5)
    save_cumulative_data(garch_h5, base_name, 'garch_forecast_h1_cumulative_h5', "h5", cumulative_dir)

    # 10步累计预测
    garch_h10 = calculate_cumulative_forecast(garch_forecasts, 'garch_forecast_h1', window=10)
    save_cumulative_data(garch_h10, base_name, 'garch_forecast_h1_cumulative_h10', "h10", cumulative_dir)

    print(f"累积预测结果保存至: {cumulative_dir}")


if __name__ == "__main__":
    project_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
    print(f"项目目录: {project_dir}")
    main(project_dir)
