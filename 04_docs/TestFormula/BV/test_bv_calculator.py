import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
# 生成测试数据 (1个交易日，6个观测值)
test_dates = pd.date_range("2023-01-01 09:35", periods=6, freq="5min")
test_returns = [0.0012, -0.0008, 0.0025, -0.0015, 0.0030, -0.0040]  # 故意包含一个跳跃

# 创建DataFrame
intraday_df = pd.DataFrame(
    {"LogReturn": test_returns},
    index=test_dates
)
intraday_df.index.name = "DateTime"

# 生成对应的RV数据
rv_df = pd.DataFrame(
    {"RV": [sum(np.array(test_returns)**2)]},  # RV = Σr²
    index=[datetime(2023,1,1)]
)
rv_df.index.name = "DateTime"

# 保存测试数据
intraday_df.to_parquet("test_intraday.parquet")
rv_df.to_parquet("test_rv.parquet")


# ... [其余代码保持不变] ...



def calculate_bv(returns_series: pd.Series) -> float:
    """
    计算单日双幂变差(BV)

    参数:
        returns_series: 单交易日LogReturn序列 (已排序)

    返回:
        当日BV值 (float)
    """
    N = len(returns_series)
    if N < 3:
        return np.nan

    # 常数定义 (μ1 = E|Z|, Z~N(0,1))
    mu1 = np.sqrt(2 / np.pi)  # 标准正态绝对值的期望

    # 核心计算: |r_{k-2}| * |r_k| 的滑动乘积和
    abs_returns = np.abs(returns_series.values)
    product_sum = np.sum(abs_returns[:-2] * abs_returns[2:])

    # BV公式实现 (Andersen et al. 2011调整)
    bv = (1 / mu1 ** 2) * (N / (N - 2)) * product_sum
    return bv


def process_bv(data_path: str, freq: str) -> pd.DataFrame:
    """
    处理全样本BV计算

    参数:
        data_path: 日内收益率文件路径
        freq: 数据频率标识 ('5min'/'1min')

    返回:
        BV结果DataFrame
    """
    # 读取数据
    df = pd.read_parquet(data_path)
    df.index = pd.to_datetime(df.index)

    # 按交易日分组计算BV
    results = []
    for date, group in df.groupby(df.index.date):
        if len(group) >= 3:  # 至少3个观测值
            bv = calculate_bv(group['LogReturn'])
            n_obs = len(group)
            results.append({'DateTime': date, 'BV': bv, 'N_actual': n_obs})

    bv_df = pd.DataFrame(results).set_index('DateTime')
    bv_df.index = pd.to_datetime(bv_df.index)
    return bv_df


def calculate_jv(bv_df: pd.DataFrame, rv_path: str) -> pd.DataFrame:
    """
    计算跳跃变异率(JV)

    参数:
        bv_df: BV计算结果DataFrame
        rv_path: RV文件路径

    返回:
        包含BV/JV的完整结果
    """
    rv_df = pd.read_parquet(rv_path)
    merged_df = bv_df.join(rv_df[['RV']], how='inner')

    # JV = RV - BV (限制最小值为0)
    merged_df['JV'] = merged_df['RV'] - merged_df['BV']
    merged_df['JV'] = merged_df['JV'].clip(lower=0)
    return merged_df[['BV', 'JV', 'N_actual']]


# 主执行流程
if __name__ == "__main__":
    # # 5分钟数据
    # intraday_5min_path = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/01_data/processed/intraday_returns/CL_WTI_5min_processed.parquet"
    #
    # # 1分钟数据
    # intraday_1min_path = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/01_data/processed/intraday_returns/CL_WTI_1min_processed.parquet"
    #
    # # 已实现波动率(RV)
    # rv_path = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates/CL_WTI_RV_daily_5min.parquet"
    # # 创建输出目录
    # output_dir = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/03_results/intermediate_results/volatility_estimates")
    # 修改主代码路径后执行
    intraday_5min_path = "test_intraday.parquet"
    rv_path = "test_rv.parquet"
    output_dir = Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理5分钟数据
    bv_5min = process_bv(intraday_5min_path, "5min")
    result_5min = calculate_jv(bv_5min, rv_path)
    result_5min.to_parquet(output_dir / "CL_WTI_BV_daily_5min.parquet")

    # # 处理1分钟数据 (可选)
    # bv_1min = process_bv(intraday_1min_path, "1min")
    # result_1min = calculate_jv(bv_1min, rv_path.replace("5min", "1min"))
    # result_1min.to_parquet(output_dir / "CL_WTI_BV_daily_1min.parquet")
# 验证输出
    print(result_5min)
# 预期输出:
#                   BV       JV  N_actual
# DateTime
# 2023-01-01  0.000042      0.0         6