import pandas as pd
import numpy as np
import os
from datetime import datetime


def calculate_tsrv(returns, c=1.0):
    """
    计算单日的TSRV估计值

    参数:
    returns (pd.Series): 当日的对数收益率序列
    c (float): 计算最优网格数的常数，默认为1.0

    返回:
    tuple: (TSRV值, 原始观测数N, 使用的网格数G, 平均子网格观测数avg_n)
    """
    # 计算观测总数N
    N = len(returns)
    if N < 5:  # 确保有足够的数据点
        return (np.nan, N, np.nan, np.nan)

    # 计算全样本RV (公式2)
    all_rv = np.sum(returns ** 2)

    # 计算最优网格数G (公式3)
    G = max(2, int(round(c * N ** (2 / 3))))  # 确保G至少为2

    # 创建子网格
    subgrids = []
    for g in range(G):
        # 创建等间隔的子网格
        indices = np.arange(g, N, G)
        subgrid = returns.iloc[indices]
        subgrids.append(subgrid)

    # 计算平均子网格观测数
    avg_n = np.mean([len(subgrid) for subgrid in subgrids])

    # 计算每个子网格的RV并取平均
    sub_rvs = [np.sum(subgrid ** 2) for subgrid in subgrids]
    avg_rv = np.mean(sub_rvs)

    # 计算TSRV (核心公式)
    tsrv = avg_rv - (avg_n / N) * all_rv

    # 确保TSRV非负
    return (max(0, tsrv), N, G, avg_n)


def calculate_daily_tsrv(data_path, freq, output_dir, c=1.0):
    """
    计算每日的TSRV估计值并保存结果

    参数:
    data_path (str): 预处理数据的路径
    freq (str): 数据频率 ('1min' 或 '5min')
    output_dir (str): 输出目录
    c (float): 计算最优网格数的常数，默认为1.0
    """
    # 读取数据
    print(f"正在读取{freq}数据...")
    df = pd.read_parquet(data_path)

    # 提取日期部分用于分组
    df['Date'] = df.index.date

    # 按日期分组计算TSRV
    results = []
    for date, group in df.groupby('Date'):
        returns = group['LogReturn']
        tsrv, N, G, avg_n = calculate_tsrv(returns, c)

        results.append({
            'DateTime': date,
            'TSRV': tsrv,
            'N': N,
            'G': G,
            'Avg_n': avg_n
        })

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('DateTime', inplace=True)

    # 保存结果
    output_path = os.path.join(output_dir, f'CL_WTI_TSRV_daily_{freq}.parquet')
    results_df.to_parquet(output_path)
    print(f"已保存{freq} TSRV结果至: {output_path}")
    print(f"数据样例:\n{results_df.head()}")


def main():
    # 设置路径
    base_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
    intraday_dir = os.path.join(base_dir, "01_data/processed/intraday_returns/")
    output_dir = os.path.join(base_dir, "03_results/intermediate_results/volatility_estimates/")

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 计算并保存不同频率的TSRV
    calculate_daily_tsrv(
        data_path=os.path.join(intraday_dir, "CL_WTI_1min_processed.parquet"),
        freq="1min",
        output_dir=output_dir
    )

    calculate_daily_tsrv(
        data_path=os.path.join(intraday_dir, "CL_WTI_5min_processed.parquet"),
        freq="5min",
        output_dir=output_dir
    )


if __name__ == "__main__":
    print("开始计算TSRV估计量...")
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"计算完成! 总耗时: {end_time - start_time}")