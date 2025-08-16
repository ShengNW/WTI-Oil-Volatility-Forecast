import pandas as pd
import numpy as np
import pywt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)  # 忽略PyWavelets的FutureWarning


def compute_jwtsrv(project_root, sampling='1min', wavelet='haar', G=15):
    """
    计算JWTSRV波动率指标

    参数:
    project_root -- 项目根目录路径
    sampling -- 采样频率 ('1min' 或 '5min')
    wavelet -- 使用的小波基
    G -- 双尺度参数
    """
    # ============== 路径配置 ==============
    data_path = Path(project_root) / '01_data'
    processed_path = data_path / 'processed' / 'intraday_returns'
    raw_path = data_path / 'raw'
    output_path = Path(project_root) / '03_results' / 'intermediate_results' / 'volatility_estimates'
    output_path.mkdir(parents=True, exist_ok=True)

    # ============== 加载数据 ==============
    intraday_file = f'CL_WTI_{sampling}_processed.parquet'
    daily_file = 'CL_WTI_daily.csv'

    # 加载日内收益率数据
    intraday_df = pd.read_parquet(processed_path / intraday_file)
    intraday_df = intraday_df[['LogReturn']].copy()

    # 加载日级数据划分交易日
    daily_df = pd.read_csv(raw_path / daily_file, parse_dates=['DateTime'])
    daily_dates = daily_df['DateTime'].dt.normalize().unique()

    # ============== 准备结果容器 ==============
    jwtsrv_results = []

    # ============== 按日处理数据 ==============
    for trade_date in daily_dates:
        #date_str = trade_date.strftime('%Y-%m-%d')
        date_str = pd.Timestamp(trade_date).strftime('%Y-%m-%d')

        # 提取当日数据
        day_data = intraday_df.loc[intraday_df.index.normalize() == trade_date]
        if len(day_data) < 30:  # 跳过数据不足的交易日
            continue

        returns = day_data['LogReturn'].values
        N = len(returns)  # 当日观测点数

        # ===== 1. 跳跃检测 =====
        # MODWT第一层分解
        coeffs = pywt.swt(returns, wavelet, level=1, trim_approx=True)
        w1k = coeffs[0][1]  # 第一层细节系数

        # 计算阈值d (MAD估计)
        median_abs_dev = np.median(np.abs(w1k - np.median(w1k)))
        d = np.sqrt(2) * median_abs_dev / 0.6745
        threshold = d * np.sqrt(2 * np.log(N))

        # 识别跳跃点
        jump_indices = np.where(np.abs(w1k) > threshold)[0]
        delta_J = np.zeros_like(returns)
        delta_J[jump_indices] = returns[jump_indices]

        # ===== 2. 跳跃调整收益率 =====
        returns_adj = returns - delta_J

        # ===== 3. JWTSRV计算 =====
        # 确定最大分解层数 (基于数据长度)
        J_max = min(8, pywt.swt_max_level(N))

        total_iv = 0.0
        for j in range(1, J_max + 1):
            # 执行MODWT分解
            coeffs_j = pywt.swt(returns_adj, wavelet, level=j, start_level=j - 1, trim_approx=True)
            w_jk = coeffs_j[0][1] if j < J_max else coeffs_j[0][0]  # 细节系数或最终层近似系数

            # 计算全样本IV
            iv_all = np.sum(w_jk ** 2)

            # 双尺度计算
            N_tilde = N // G  # 子样本大小
            if N_tilde < 1:
                continue

            # 创建子网格
            sub_grids = [w_jk[i:i + N_tilde] for i in range(0, N - N_tilde + 1, N_tilde)]
            if not sub_grids:
                continue

            # 计算每个子网格的IV
            iv_sub = [np.sum(sub ** 2) for sub in sub_grids]
            iv_avg = np.mean(iv_sub) * (N / N_tilde)  # 缩放因子调整

            # 双尺度修正项
            iv_j = iv_avg - (N_tilde / N) * iv_all
            total_iv += iv_j

        # 保存当日结果
        jwtsrv_results.append({
            'DateTime': trade_date,
            'JWTSRV': total_iv
        })

    # ============== 保存结果 ==============
    result_df = pd.DataFrame(jwtsrv_results)
    result_df.set_index('DateTime', inplace=True)
    output_file = output_path / f'CL_WTI_JWTSRV_daily_{sampling}.parquet'
    result_df.to_parquet(output_file)

    print(f"JWTSRV计算完成! 结果保存至: {output_file}")
    print(f"共处理 {len(jwtsrv_results)} 个交易日数据")
    return result_df


if __name__ == "__main__":
    # 配置项目根目录 (使用时替换为实际路径)
    PROJECT_ROOT = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"

    # 计算1分钟数据的JWTSRV
    jwtsrv_df = compute_jwtsrv(
        project_root=PROJECT_ROOT,
        sampling='1min',
        wavelet='haar',
        G=15
    )

    # 显示前5个结果
    print("\nJWTSRV结果示例:")
    print(jwtsrv_df.head())