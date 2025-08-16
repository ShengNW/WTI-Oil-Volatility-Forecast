# WTI_volatility_forecast_replication/02_code/02_volatility_estimation/rv_estimation.py
import numpy as np
import pandas as pd
from pathlib import Path

# 设置路径
project_dir = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication")
processed_dir = project_dir / "01_data/processed/intraday_returns"
results_dir = project_dir / "03_results/intermediate_results/volatility_estimates"


def rv(day_data, N):
    """
    计算单日已实现波动率(RV)

    参数:
        day_data (pd.DataFrame): 单日预处理数据，必须包含'LogReturn'列
        N (int): 理论观测次数（仅用于接口兼容，实际计算不使用）

    返回:
        float: 当日RV值
    """
    # 直接使用预处理计算好的LogReturn列
    log_returns = day_data['LogReturn'].values
    return np.sum(log_returns ** 2)


# 主处理流程
def calculate_and_save_rv(freq="5min"):
    """计算并保存RV结果"""
    # 加载预处理数据
    data = pd.read_parquet(
        processed_dir / f"CL_WTI_{freq}_processed.parquet",
        columns=['LogReturn']  # 只加载必要列
    )

    # 提取日期部分用于分组
    dates = data.index.date

    # 按日分组计算RV
    rv_results = []
    for date, group in data.groupby(dates):
        # 获取当日实际观测次数
        actual_N = len(group)
        # 计算RV（实际N值仅用于记录）
        rv_val = rv(group, actual_N)
        rv_results.append({'DateTime': date, 'RV': rv_val, 'N_actual': actual_N})

    # 转换为DataFrame并保存
    rv_df = pd.DataFrame(rv_results).set_index('DateTime')
    rv_df.to_parquet(results_dir / f"CL_WTI_RV_daily_{freq}.parquet")
    print(f"RV计算完成！{freq}数据共处理{len(rv_df)}个交易日")


# 执行计算（可按需选择频率）
if __name__ == "__main__":
    calculate_and_save_rv(freq="1min")  # 切换为"1min"处理1分钟数据