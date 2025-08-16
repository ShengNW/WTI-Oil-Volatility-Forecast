#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from datetime import datetime


def calculate_tsrv(returns: pd.Series, c: float = 1.0, small_sample_adjust: bool = False):
    """
    两尺度 TSRV（日频，规则采样）:
    - 先做 K 步（G 步）聚合收益，再平方求和
    - 偏差校正项使用 \bar n / N * RV_all，其中 \bar n 是每个子网格的“区间数”平均值

    参数
    ----
    returns : pd.Series
        当日按时间升序的 1 分钟对数收益
    c : float
        选择 K 的常数系数（K ≈ c * N^(2/3)）
    small_sample_adjust : bool
        是否使用小样本调整因子 1 / (1 - \bar n / N)

    返回
    ----
    (tsrv, N, K, n_bar)
      tsrv : 当日 TSRV（方差量纲）
      N    : 当日 1 分钟收益样本数
      K    : 选取的子网格步长
      n_bar: 每个子网格的区间数平均值（用于偏差校正）
    """
    # 清理并转为 numpy
    r = pd.Series(returns).dropna().values.astype(float)
    N = int(len(r))
    if N < 5:
        return (np.nan, N, np.nan, np.nan)

    # 全样本 RV（1 分钟收益平方和）
    all_rv = float(np.sum(r * r))

    # 选择 K（至少 2；不超过 N-1，保证至少一个区间）
    K = int(round(c * (N ** (2.0 / 3.0))))
    K = max(2, min(K, max(N - 1, 2)))

    rv_sub = []
    n_list = []  # 每个子网格的“区间数”（m）

    # 对每个偏移 g=0..K-1，做 K 步聚合收益
    for g in range(K):
        # 该偏移下能形成的完整 K 块数（也是区间数）
        m = (N - g) // K
        if m <= 0:
            continue
        # 取从 g 开始的前 m*K 个点，reshape 成 (m, K)，每行求和得到 K 步收益
        blocks = r[g: g + m * K].reshape(m, K).sum(axis=1)  # shape: (m,)
        # 子网格的 realized variance：聚合收益平方后求和
        rv_sub.append(float(np.sum(blocks * blocks)))
        n_list.append(float(m))  # 这个子网格的区间数

    if not rv_sub:
        return (np.nan, N, K, np.nan)

    RV_bar = float(np.mean(rv_sub))     # 子网格 RV 的平均
    n_bar  = float(np.mean(n_list))     # 区间数的平均值（用于偏差校正）

    # 两尺度偏差校正：RV_bar - (n_bar / N) * all_rv
    tsrv = RV_bar - (n_bar / N) * all_rv

    # 可选：小样本调整（论文式(41)）：乘以 1 / (1 - n_bar / N)
    if small_sample_adjust and (1.0 - n_bar / N) > 1e-12:
        tsrv = tsrv / (1.0 - n_bar / N)

    # 不做截断；保留可能出现的极小负值（有限样本下正常）
    return (float(tsrv), N, K, n_bar)


def calculate_daily_tsrv(data_path: str, freq: str, output_dir: str, c: float = 1.0,
                         small_sample_adjust: bool = False):
    """
    计算每日的 TSRV 并保存 .parquet（路径与文件名保持不变）
    """
    print(f"正在读取 {freq} 数据: {data_path}")
    df = pd.read_parquet(data_path)

    # 要求索引为时间；若不是，尝试转换
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
            df = df.set_index('DateTime')
        else:
            raise ValueError("输入数据没有 DatetimeIndex，也没有 'DateTime' 列。")

    # 按日期分组（确保时间升序）
    df = df.sort_index()
    df['Date'] = df.index.date

    results = []
    for date, group in df.groupby('Date', sort=True):
        # 约定收益列名为 LogReturn（与你现有流程一致）
        returns = group['LogReturn'].astype(float)
        tsrv, N, K, n_bar = calculate_tsrv(returns, c=c, small_sample_adjust=small_sample_adjust)

        results.append({
            'DateTime': pd.Timestamp(date),
            'TSRV': tsrv,
            'N': N,
            'G': K,         # 与你原字段名保持一致：这里的 G 实为 K（网格步长）
            'Avg_n': n_bar  # 这里用“区间数”平均值，更严格
        })

    results_df = pd.DataFrame(results).set_index('DateTime').sort_index()

    # 保存到原路径结构
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'CL_WTI_TSRV_daily_{freq}.parquet')
    results_df.to_parquet(output_path)
    print(f"已保存 {freq} TSRV 结果至: {output_path}")
    print("数据样例:\n", results_df.head())


def main():
    # === 保持你的原始输入/输出路径不变 ===
    base_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
    intraday_dir = os.path.join(base_dir, "01_data/processed/intraday_returns/")
    output_dir = os.path.join(base_dir, "03_results/intermediate_results/volatility_estimates/")

    os.makedirs(output_dir, exist_ok=True)

    # 1min
    calculate_daily_tsrv(
        data_path=os.path.join(intraday_dir, "CL_WTI_1min_processed.parquet"),
        freq="1min",
        output_dir=output_dir,
        c=1.0,
        small_sample_adjust=False  # 需要的话可改为 True
    )

    # 5min
    calculate_daily_tsrv(
        data_path=os.path.join(intraday_dir, "CL_WTI_5min_processed.parquet"),
        freq="5min",
        output_dir=output_dir,
        c=1.0,
        small_sample_adjust=False
    )


if __name__ == "__main__":
    print("开始计算 TSRV 估计量...")
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"计算完成! 总耗时: {end_time - start_time}")
