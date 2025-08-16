import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path


def parzen_kernel(x):
    """计算Parzen核函数值"""
    if x <= 0.5:
        return 1 - 6 * x ** 2 + 6 * x ** 3
    else:
        return 2 * (1 - x) ** 3


def realized_kernel_estimator(returns, H=None):
    """
    计算实现核方差估计量(RK)

    参数:
    returns -- 一维数组，包含某个交易日的日内收益率
    H -- 整数，计算自协方差的滞后阶数(可选)

    返回:
    rk_estimate -- 实现核方差估计值
    """
    n = len(returns)

    # 如果序列太短，返回NaN
    if n < 2:
        return np.nan

    # 设置默认的H值 (N^{2/3})
    if H is None:
        H = max(1, min(int(round(n ** (2 / 3))), n - 1))

    # 计算γ0 (已实现波动率RV)
    gamma0 = np.sum(returns ** 2)

    # 初始化RK估计量
    rk_estimate = gamma0

    # 计算η从1到H的自协方差并加权求和
    for eta in range(1, H + 1):
        # 计算正向自协方差γη
        gamma_eta = np.sum(returns[:n - eta] * returns[eta:])

        # 计算反向自协方差γ-η
        gamma_neg_eta = np.sum(returns[eta:] * returns[:n - eta])

        # 计算核权重
        x = (eta - 1) / H
        k_weight = parzen_kernel(x)

        # 添加到RK估计量
        rk_estimate += k_weight * (gamma_eta + gamma_neg_eta)

    return rk_estimate


def calculate_daily_rk(data_path, freq):
    """
    计算每日的实现核方差估计量

    参数:
    data_path -- Parquet文件路径
    freq -- 数据频率('5min'或'1min')

    返回:
    rk_results -- 包含日期和RK估计值的DataFrame
    """
    # 读取预处理数据
    df = pd.read_parquet(data_path)

    # 确保索引是DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 提取日期列
    df['Date'] = df.index.date

    # 按日期分组计算每日RK
    results = []
    for date, group in df.groupby('Date'):
        returns = group['LogReturn'].values

        # 根据数据频率设置H
        if freq == '5min':
            H = 5  # 5分钟数据建议H=10
        else:  # 1min
            H = 11  # 1分钟数据建议H=25

        rk = realized_kernel_estimator(returns, H=H)
        results.append({'Date': date, f'RK_{freq}': rk})

    return pd.DataFrame(results)


def main():
    # 设置项目根目录
    project_root = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication")

    # 输入路径
    intraday_5min_path = project_root / "01_data/processed/intraday_returns/CL_WTI_5min_processed.parquet"
    intraday_1min_path = project_root / "01_data/processed/intraday_returns/CL_WTI_1min_processed.parquet"

    # 输出目录
    output_dir = project_root / "03_results/intermediate_results/volatility_estimates"
    os.makedirs(output_dir, exist_ok=True)

    # 计算5分钟数据的RK
    print("计算5分钟数据的实现核方差估计量...")
    rk_5min_df = calculate_daily_rk(intraday_5min_path, '5min')

    # 计算1分钟数据的RK
    print("计算1分钟数据的实现核方差估计量...")
    rk_1min_df = calculate_daily_rk(intraday_1min_path, '1min')

    # 合并结果
    rk_results = pd.merge(rk_5min_df, rk_1min_df, on='Date', how='outer')

    # 保存结果
    output_path = output_dir / "realized_kernel_estimates.parquet"
    rk_results.to_parquet(output_path)
    print(f"结果已保存至: {output_path}")

    return rk_results


if __name__ == "__main__":
    rk_estimates = main()