import numpy as np
import pandas as pd
import os
from math import sqrt, pi

# 常量定义 (论文公式中的系数)
MEDRV_CONSTANT = pi / (6 - 4 * sqrt(3) + pi)  # ≈1.139


def calculate_medrv_daily(returns):
    """
    计算单日的MedRV波动率
    :param returns: 单个交易日的收益率序列 (np.array)
    :return: (MedRV值, 实际观测数)
    """
    n = len(returns)
    if n < 3:
        return np.nan, n

    abs_returns = np.abs(returns)

    # 计算连续三个收益率绝对值的中位数平方和
    med_sq_sum = 0.0
    for i in range(2, n):  # 从第3个观测值开始 (索引2)
        window = abs_returns[i - 2:i + 1]  # 取三个连续观测值
        med = np.median(window)
        med_sq_sum += med ** 2

    # 应用MedRV公式
    scaling_factor = MEDRV_CONSTANT * (n / (n - 2))
    medrv = scaling_factor * med_sq_sum

    return medrv, n


def generate_medrv_estimates(frequency='5min'):
    """
    生成MedRV波动率估计值并保存结果
    :param frequency: 数据频率 ('5min' 或 '1min')
    """
    # 路径设置
    input_path = f'F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/01_data/processed/intraday_returns/CL_WTI_{frequency}_processed.parquet'
    output_dir = 'F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates/'
    output_path = f'{output_dir}CL_WTI_MedRV_daily_{frequency}.parquet'

    os.makedirs(output_dir, exist_ok=True)

    # 读取预处理数据
    df = pd.read_parquet(input_path)

    # 提取日期（不带时间）
    df = df.reset_index()
    df['Date'] = df['DateTime'].dt.date

    # 按日期分组计算MedRV
    results = []
    for date, group in df.groupby('Date'):
        returns = group['LogReturn'].values
        medrv, n_actual = calculate_medrv_daily(returns)
        results.append({
            'DateTime': pd.Timestamp(date),
            'MedRV': medrv,
            'N_actual': n_actual
        })

    # 创建结果DataFrame并保存
    result_df = pd.DataFrame(results)
    result_df.to_parquet(output_path, index=False)

    print(f"MedRV计算完成! {frequency}数据保存至: {output_path}")
    print(f"日期范围: {result_df['DateTime'].min().date()} 到 {result_df['DateTime'].max().date()}")
    print(f"有效交易日数: {len(result_df) - result_df['MedRV'].isna().sum()}/{len(result_df)}")


if __name__ == '__main__':
    # 生成5分钟和1分钟的MedRV估计
    generate_medrv_estimates(frequency='5min')
    generate_medrv_estimates(frequency='1min')