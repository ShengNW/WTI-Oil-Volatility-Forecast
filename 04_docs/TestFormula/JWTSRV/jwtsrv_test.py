import pandas as pd
import numpy as np
import pywt
from pathlib import Path
import warnings
import multiprocessing
from functools import partial

warnings.filterwarnings('ignore', category=FutureWarning)

def process_day(trade_date, intraday_df, wavelet, G):
    day_data = intraday_df.loc[intraday_df.index.normalize() == trade_date]

    returns = day_data['LogReturn'].values
    N = len(returns)

    # ===== 新增：调整数据长度为偶数（核心修复）=====
    if N % 2 != 0:
        returns = returns[:-1]  # 删除最后一个元素使长度为偶数
        N = len(returns)

    # ===== 1. 跳跃检测 =====
    coeffs = pywt.swt(returns, wavelet, level=1, trim_approx=True)
    #w1k = coeffs[0][1]
    if len(coeffs) >= 2:
        #approx = coeffs[0]  # 近似系数（cA1）
        w1k = coeffs[1]  # 细节系数（cD1）
    else:
        print(f"警告：交易日{trade_date}的SWT系数不足")
        return None

    # median_abs_dev = np.median(np.abs(w1k - np.median(w1k)))
    # d = np.sqrt(2) * median_abs_dev / 0.6745
    # 计算小波系数绝对值的中位数，复现论文中的med{|W_{1,k}|}
    abs_w1k = np.abs(w1k)
    median_abs_w1k = np.median(abs_w1k)  # 直接取绝对值的中位数，而非MAD

    # 计算d，严格遵循论文公式d = sqrt(2) * med{|W_{1,k}|} / 0.6745
    d = np.sqrt(2) * median_abs_w1k / 0.6745
    threshold = d * np.sqrt(2 * np.log(N))
    jump_indices = np.where(np.abs(w1k) > threshold)[0]
    delta_J = np.zeros_like(returns)
    delta_J[jump_indices] = returns[jump_indices]

    # ===== 2. 跳跃调整收益率 =====
    returns_adj = returns - delta_J

    # ===== 3. JWTSRV计算 =====
    J_max = min(8, pywt.swt_max_level(N))
    total_iv = 0.0

    for j in range(1, J_max + 1):
        try:
            coeffs_j = pywt.swt(returns_adj, wavelet, level=j, start_level=j - 1, trim_approx=True)

            if j < J_max:
                # 非最高级别时，取当前级别的细节系数cDj（列表中第1个元素）
                w_jk = coeffs_j[1]
            else:
                # 最高级别时，取最终级别的近似系数cAj（列表中第0个元素）
                w_jk = coeffs_j[0]

            # 处理可能的标量情况
            if not isinstance(w_jk, np.ndarray):
                if np.isscalar(w_jk):
                    w_jk = np.array([w_jk])
                else:
                    print(f"警告：j={j}时，w_jk类型异常: {type(w_jk)}")
                    continue

            w_len = len(w_jk)
            if w_len == 0:
                print(f"警告：j={j}时，w_jk为空数组")
                continue

            iv_all = np.sum(w_jk ** 2)
            N_tilde = w_len // G

            if N_tilde < 1:
                continue

            # 确保切片操作有效
            sub_grids = []
            for i in range(0, w_len - N_tilde + 1, N_tilde):
                sub_grid = w_jk[i:i + N_tilde]
                if len(sub_grid) > 0:  # 确保子网格非空
                    sub_grids.append(sub_grid)

            if not sub_grids:
                continue

            iv_sub = [np.sum(sub ** 2) for sub in sub_grids]
            iv_avg = np.mean(iv_sub) * (w_len / N_tilde)
            iv_j = iv_avg - (N_tilde / w_len) * iv_all
            total_iv += iv_j

        except Exception as e:
            print(f"处理交易日 {trade_date} 级别 j={j} 时出错: {str(e)}")
            continue

    return {'DateTime': pd.Timestamp(trade_date), 'JWTSRV': total_iv}


def compute_jwtsrv(project_root, sampling='1min', wavelet='haar', G=15):
    # ============== 路径配置 ==============
    data_path = Path(project_root) / '01_data'
    processed_path = data_path / 'processed' / 'intraday_returns'
    output_path = Path(project_root) / '03_results' / 'intermediate_results' / 'volatility_estimates'
    output_path.mkdir(parents=True, exist_ok=True)

    # ============== 加载数据 ==============
    intraday_file = f'CL_WTI_{sampling}_processed.parquet'


    intraday_df = pd.read_parquet(processed_path / intraday_file)
    intraday_df = intraday_df[['LogReturn']].copy()

    daily_dates = intraday_df.index.normalize().unique()  # 从高频数据索引提取交易日
    daily_dates = np.sort(daily_dates)  # 按日期排序

    # # 多进程处理函数



    num_cores = multiprocessing.cpu_count() - 1 or 1
    # ============== 并行处理 ==============
    print(f"开始并行处理{len(daily_dates)}个交易日，使用{num_cores}个核心...")
    with multiprocessing.Pool(processes=num_cores) as pool:
        worker = partial(process_day,
                         intraday_df=intraday_df,
                         wavelet=wavelet,
                         G=G)
        results = pool.map(worker, daily_dates)

    # 过滤并收集结果
    jwtsrv_results = [res for res in results if res is not None]
    print(f"成功处理{len(jwtsrv_results)}个交易日数据")

    # ============== 保存结果 ==============
    if not jwtsrv_results:
        print("警告：没有有效结果，创建空DataFrame")
        result_df = pd.DataFrame(columns=['DateTime', 'JWTSRV'])
    else:
        result_df = pd.DataFrame(jwtsrv_results)

    # 调试信息
    print("结果DataFrame列名:", result_df.columns.tolist())

    # 确保设置正确的索引
    if 'DateTime' in result_df.columns:
        result_df.set_index('DateTime', inplace=True)
    else:
        # 尝试可能的列名变体
        for col in ['datetime', 'date', 'time']:
            if col in result_df.columns:
                result_df.rename(columns={col: 'DateTime'}, inplace=True)
                result_df.set_index('DateTime', inplace=True)
                break
        else:
            # 创建新的日期索引作为后备方案
            print("警告：未找到DateTime列，使用默认日期索引")
            result_df.index = pd.date_range(start='2020-01-01', periods=len(result_df), freq='D')
            result_df.index.name = 'DateTime'

    output_file = output_path / f'CL_WTI_JWTSRV_daily_{sampling}.parquet'
    result_df.to_parquet(output_file)
    print(f"结果保存至: {output_file}")
    return result_df


if __name__ == "__main__":
    #PROJECT_ROOT = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    PROJECT_ROOT = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/04_docs/TestFormula/JWTSRV"
    jwtsrv_df = compute_jwtsrv(project_root=PROJECT_ROOT, sampling='1min')
    print("\nJWTSRV结果示例:")
    print(jwtsrv_df.head())