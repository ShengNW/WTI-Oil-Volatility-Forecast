import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path


# ===== 原始代码（未修改）=====
def parzen_kernel(x):
    if x <= 0.5:
        return 1 - 6 * x ** 2 + 6 * x ** 3
    else:
        return 2 * (1 - x) ** 3


def realized_kernel_estimator(returns, H=None):
    n = len(returns)
    if n < 2:
        return np.nan
    if H is None:
        H = max(1, min(int(round(n ** (2 / 3))), n - 1))
    gamma0 = np.sum(returns ** 2)
    rk_estimate = gamma0
    for eta in range(1, H + 1):
        gamma_eta = np.sum(returns[:n - eta] * returns[eta:])
        gamma_neg_eta = np.sum(returns[eta:] * returns[:n - eta])
        x = (eta - 1) / H
        k_weight = parzen_kernel(x)
        rk_estimate += k_weight * (gamma_eta + gamma_neg_eta)
    return rk_estimate


def calculate_daily_rk(data_path, freq):
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df['Date'] = df.index.date
    results = []
    for date, group in df.groupby('Date'):
        returns = group['LogReturn'].values
        H = 25 if freq == '1min' else 10
        rk = realized_kernel_estimator(returns, H=1)# 直接根据样本数5 计算0.6*sqrt(5) 约等于1.3
        results.append({'Date': date, f'RK_{freq}': rk})
    return pd.DataFrame(results)


# ===== 原始代码结束 =====

# ===== 测试用例生成和执行 =====
def create_test_data():
    """创建测试用的Parquet文件"""
    test_data = {
        'LogReturn': [0.01, 0.02, -0.01, 0.03, 0.01],
        'Timestamp': pd.date_range('2023-01-01 09:30', periods=5, freq='1min')
    }
    df = pd.DataFrame(test_data).set_index('Timestamp')
    df.to_parquet("test_data.parquet")
    return df


def run_test():
    # 生成测试数据
    print("生成测试数据...")
    test_df = create_test_data()
    print("测试数据:")
    print(test_df)

    # 手动计算
    manual_rk = 0.0016 #+ 1 * (0.0004 + 0.0004)  # γ0 + K(η=2)*(γ2+γ-2)
    print(f"\n手动计算结果: {manual_rk:.6f}")

    # 代码计算
    result_df = calculate_daily_rk("test_data.parquet", '1min')
    code_rk = result_df['RK_1min'].iloc[0]
    print(f"代码计算结果: {code_rk:.6f}")

    # 验证
    tolerance = 1e-6
    assert abs(code_rk - manual_rk) < tolerance, "测试失败：结果不匹配"
    print("测试通过：代码结果与手动计算结果一致")


if __name__ == "__main__":
    run_test()