# test_rv_estimation.py
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# 创建临时目录结构
temp_dir = Path("temp_test")
processed_dir = temp_dir / "01_data/processed/intraday_returns"
results_dir = temp_dir / "03_results/intermediate_results/volatility_estimates"
processed_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# 生成测试数据（与上述测试数据相同）
test_data = pd.DataFrame({
    'LogReturn': [0.015, -0.008, 0.022, -0.017, 0.009]
}, index=pd.DatetimeIndex([
    '2023-01-01 09:00:00',
    '2023-01-01 09:05:00',
    '2023-01-01 09:10:00',
    '2023-01-01 09:15:00',
    '2023-01-01 09:20:00'
]))
test_data.to_parquet(processed_dir / "CL_WTI_5min_processed.parquet")


# --- 原始代码（完全未修改）---
def rv(day_data, N):
    log_returns = day_data['LogReturn'].values
    return np.sum(log_returns ** 2)


def calculate_and_save_rv(freq="5min"):
    data = pd.read_parquet(
        processed_dir / f"CL_WTI_{freq}_processed.parquet",
        columns=['LogReturn']
    )
    dates = data.index.date
    rv_results = []
    for date, group in data.groupby(dates):
        actual_N = len(group)
        rv_val = rv(group, actual_N)
        rv_results.append({'DateTime': date, 'RV': rv_val, 'N_actual': actual_N})
    rv_df = pd.DataFrame(rv_results).set_index('DateTime')
    rv_df.to_parquet(results_dir / f"CL_WTI_RV_daily_{freq}.parquet")
    return rv_df  # 添加返回用于验证


# 执行测试
if __name__ == "__main__":
    # 运行原始代码
    result_df = calculate_and_save_rv(freq="5min")

    # 验证结果
    expected_rv = 0.015 ** 2 + (-0.008) ** 2 + 0.022 ** 2 + (-0.017) ** 2 + 0.009 ** 2
    calculated_rv = result_df.iloc[0]['RV']

    print("\n测试结果:")
    print(f"手动计算结果: {expected_rv:.6f}")
    print(f"代码计算结果: {calculated_rv:.6f}")
    print(f"实际观测数: {result_df.iloc[0]['N_actual']}")
    print(f"结果匹配: {np.isclose(expected_rv, calculated_rv, atol=1e-8)}")
    print("临时目录位置：", temp_dir.resolve())  # 会输出绝对路径
    # 替换为你的 .parquet 文件路径
    file_path = r"F:\CODE\social\JianZhi\future\WTI_volatility_forecast_replication\04_docs\TestFormula\RV\temp_test\01_data\processed\intraday_returns\CL_WTI_5min_processed.parquet"
    # 读取文件
    df = pd.read_parquet(file_path)

    # 查看前 5 行数据
    print(df.head(5))
    file_path1 = r"F:\CODE\social\JianZhi\future\WTI_volatility_forecast_replication\04_docs\TestFormula\RV\temp_test\03_results\intermediate_results\volatility_estimates\CL_WTI_RV_daily_5min.parquet"
    # 读取文件
    df1 = pd.read_parquet(file_path1)

    # 查看前 5 行数据
    print(df1.head(5))
    # 清理临时文件
    shutil.rmtree(temp_dir)