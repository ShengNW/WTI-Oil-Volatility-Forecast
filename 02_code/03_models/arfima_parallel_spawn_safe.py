import os
# ---- 防止 BLAS 在线程层面过度并行，避免与多进程叠加导致过度占用 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

warnings.filterwarnings("ignore")


class ARFIMA_Model:
    def __init__(self, root_dir):
        """
        初始化ARFIMA模型

        参数:
        root_dir (str): 项目根目录路径
        """
        self.root_dir = root_dir
        # 定义路径映射
        self.paths = {
            'train_data': os.path.join(root_dir, "03_results", "intermediate_results", "volatility_estimates",
                                       "train_set"),
            'test_data': os.path.join(root_dir, "03_results", "intermediate_results", "volatility_estimates",
                                      "test_set"),
            'subperiods': os.path.join(root_dir, "03_results", "intermediate_results", "volatility_estimates",
                                       "subperiods"),
            'output_forecasts': os.path.join(root_dir, "03_results", "final_forecasts"),
            'output_params': os.path.join(root_dir, "03_results", "intermediate_results", "model_parameters", "ARFIMA"),
            'temp_cache': os.path.join(root_dir, "01_data", "temp", "rolling_window_cache")
        }

        # 确保输出目录存在
        os.makedirs(self.paths['output_forecasts'], exist_ok=True)
        os.makedirs(self.paths['output_params'], exist_ok=True)
        os.makedirs(self.paths['temp_cache'], exist_ok=True)

        # 定义波动率类型和频率
        self.volatility_types = ['RV', 'TSRV', 'RK', 'BV', 'MedRV', 'JWTSRV']
        self.frequencies = ['1min', '5min']

        # 存储模型参数（仅在单任务上下文使用）
        self.model_params = {}

    def load_data(self, vol_type, freq, dataset='train'):
        """
        加载波动率数据（不改变原始数据结构与含义）
        """
        if dataset == 'train':
            data_dir = self.paths['train_data']
        elif dataset == 'test':
            data_dir = self.paths['test_data']
        else:
            raise ValueError("dataset 参数必须是 'train' 或 'test'")

        file_name = f"CL_WTI_{vol_type}_daily_{freq}_{dataset}.parquet"
        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'DateTime' in df.columns:
                    df = df.set_index('DateTime')
                else:
                    df.index = pd.to_datetime(df.index)
            return df
        else:
            raise FileNotFoundError(f"文件不存在: {file_path}")

    def get_volatility_series(self, vol_type, freq, dataset='train'):
        """
        获取波动率序列（与原逻辑一致）
        """
        df = self.load_data(vol_type, freq, dataset)

        if vol_type == 'RV':
            return df['RV']
        elif vol_type == 'TSRV':
            return df['TSRV']
        elif vol_type == 'RK':
            return df['RK']
        elif vol_type == 'BV':
            return df['BV']
        elif vol_type == 'MedRV':
            return df['MedRV']
        elif vol_type == 'JWTSRV':
            return df['JWTSRV']
        else:
            raise ValueError(f"未知的波动率类型: {vol_type}")

    def fractionally_difference(self, series, d):
        """
        应用分数差分 (1-L)^d 到时间序列
        —— 完全保留原算法逻辑（逐项二项式展开），仅做实现层优化（局部变量、预分配）。
        """
        n = len(series)
        diff_series = np.zeros(n, dtype=float)

        # 二项式展开系数
        binom_coeffs = [1.0]
        for k in range(1, n):
            binom_coeffs.append(binom_coeffs[-1] * (d - k + 1) / k)

        values = series.to_numpy()
        # 应用分数差分
        for t in range(n):
            # 逐项累加（与原逻辑一致）
            diff_value = 0.0
            for k in range(t + 1):
                diff_value += binom_coeffs[k] * values[t - k]
            diff_series[t] = diff_value

        return pd.Series(diff_series, index=series.index)

    def arfima_log_likelihood(self, params, series):
        """
        计算ARFIMA(1,d,0)的对数似然（与原逻辑一致）
        """
        d, alpha1, mu = params
        if not (-0.5 < d < 0.5) or not (-1 < alpha1 < 1):
            return 1e10

        try:
            diff_series = self.fractionally_difference(series - mu, d)
        except Exception:
            return 1e10

        diff_series = diff_series.dropna()

        residuals = np.zeros(len(diff_series), dtype=float)
        residuals[0] = diff_series.iloc[0]
        # AR(1) 残差（不改变逻辑）
        for t in range(1, len(diff_series)):
            residuals[t] = diff_series.iloc[t] - alpha1 * diff_series.iloc[t - 1]

        n = len(residuals)
        ssr = np.sum(residuals ** 2)
        log_lik = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ssr / n) - n / 2
        return -log_lik  # 最小化负对数似然

    def fit_arfima(self, series):
        """
        拟合ARFIMA(1,d,0)模型（与原逻辑一致）
        """
        initial_params = [0.4, 0.2, float(series.mean())]
        bounds = [(-0.499, 0.499), (-0.999, 0.999), (float(series.min()) * 0.5, float(series.max()) * 1.5)]

        result = minimize(self.arfima_log_likelihood, initial_params, args=(series,),
                          method='L-BFGS-B', bounds=bounds)

        if result.success:
            d_hat, alpha1_hat, mu_hat = result.x
            return float(d_hat), float(alpha1_hat), float(mu_hat)
        else:
            return self.fit_arfima_fallback(series)

    def fit_arfima_fallback(self, series):
        """
        备选ARFIMA拟合方法（当MLE失败时）——严格保留原方法
        """
        acf_vals = acf(series, nlags=100, fft=True)
        lags = np.arange(1, 101)
        log_acf = np.log(np.abs(acf_vals[1:101]))
        log_lags = np.log(lags)

        slope = np.polyfit(log_lags[:20], log_acf[:20], 1)[0]
        d_hat = -slope / 2
        d_hat = max(min(d_hat, 0.499), -0.499)

        diff_series = self.fractionally_difference(series, d_hat)

        model = ARIMA(diff_series.dropna(), order=(1, 0, 0))
        result = model.fit()
        alpha1_hat = result.params[0]
        mu_hat = series.mean()

        return float(d_hat), float(alpha1_hat), float(mu_hat)

    def arfima_forecast(self, series, params, steps=1):
        """
        ARFIMA模型预测（与原逻辑一致）
        """
        d, alpha1, mu = params
        diff_series = self.fractionally_difference(series - mu, d).dropna()

        ar_model = ARIMA(diff_series, order=(1, 0, 0))
        ar_fit = ar_model.fit()
        ar_forecast = ar_fit.forecast(steps=steps)

        # 简化逆变换：与原注释及处理保持一致
        forecast_values = ar_forecast + mu
        return forecast_values

    # ---------------------- 细粒度保存（I/O尽早落盘） ----------------------
    def save_forecast_single(self, vol_type, freq, forecast_series):
        """将单个(vol_type, freq)的预测结果立即保存到文件"""
        forecast_df = pd.DataFrame({
            'DateTime': forecast_series.index,
            'forecast': forecast_series.values
        }).set_index('DateTime')

        output_file = os.path.join(
            self.paths['output_forecasts'],
            f"ARFIMA_{vol_type}_{freq}_forecast.parquet"
        )
        forecast_df.to_parquet(output_file)

    def save_model_parameters_single(self, vol_type, freq, params):
        """将单个(vol_type, freq)的参数保存为独立CSV，避免并行写入冲突"""
        d, alpha1, mu = params
        single_df = pd.DataFrame([{
            'volatility_type': vol_type,
            'frequency': freq,
            'd_hat': d,
            'alpha1_hat': alpha1,
            'mu_hat': mu
        }])
        output_file = os.path.join(self.paths['output_params'], f"ARFIMA_params_{vol_type}_{freq}.csv")
        single_df.to_csv(output_file, index=False)

    # ---------------------- 与原一致：累积预测，调用时机前移到循环内部 ----------------------
    def generate_cumulative_forecasts(self, vol_type, freq, test_series):
        """
        生成累积预测（原逻辑不变），但在单任务内部调用并立即保存。
        """
        subperiod_file = os.path.join(
            self.paths['subperiods'],
            f"CL_WTI_{vol_type}_daily_{freq}_cumulative.parquet"
        )

        if os.path.exists(subperiod_file):
            subperiod_df = pd.read_parquet(subperiod_file)

            params = self.model_params.get((vol_type, freq))
            if params is None:
                print(f"找不到 {vol_type} ({freq}) 的模型参数")
                return

            train_series = self.get_volatility_series(vol_type, freq, 'train')

            cumulative_forecasts = {}
            for h in [5, 10]:
                cum_forecasts = []
                for idx, _row in subperiod_df.iterrows():
                    start_date = idx
                    # 使用 concat 替代 append（不改变数据/结果）
                    history = pd.concat([train_series, test_series[:start_date]])
                    forecast_steps = self.arfima_forecast(history, params, steps=h)
                    # 与原“均方再开方”一致
                    cum_vol = float(np.sqrt(np.mean(np.square(forecast_steps))))
                    cum_forecasts.append(cum_vol)

                col_name = f"{vol_type}_cumulative_h{h}"
                cumulative_forecasts[col_name] = cum_forecasts

            cum_df = pd.DataFrame(cumulative_forecasts, index=subperiod_df.index)
            output_file = os.path.join(
                self.paths['output_forecasts'],
                f"ARFIMA_{vol_type}_{freq}_cumulative.parquet"
            )
            cum_df.to_parquet(output_file)
            print(f"已保存累积预测: {output_file}")

    def run_one_combination(self, vol_type, freq):
        """
        运行单个 (vol_type, freq) 组合（算法保持不变）
        """
        print(f"\n[PID {os.getpid()}] 训练和预测: {vol_type} ({freq})")

        # 加载训练数据与拟合
        train_series = self.get_volatility_series(vol_type, freq, 'train')
        d_hat, alpha1_hat, mu_hat = self.fit_arfima(train_series)
        self.model_params[(vol_type, freq)] = (d_hat, alpha1_hat, mu_hat)

        # 加载测试数据
        test_series = self.get_volatility_series(vol_type, freq, 'test')

        # 生成单步预测（滚动）
        forecasts = []
        history = train_series.copy()

        for i in range(len(test_series)):
            forecast_step = self.arfima_forecast(history, (d_hat, alpha1_hat, mu_hat), steps=1)
            forecasts.append(float(forecast_step.iloc[0]))
            # 用 concat 追加真实值
            history = pd.concat([history, pd.Series([test_series.iloc[i]], index=[test_series.index[i]])])

        forecast_series = pd.Series(forecasts, index=test_series.index, name=f'ARFIMA_{vol_type}_{freq}')

        # —— I/O：立即保存该组合的预测结果 ——
        self.save_forecast_single(vol_type, freq, forecast_series)

        # —— I/O：立即保存该组合的参数 ——
        self.save_model_parameters_single(vol_type, freq, (d_hat, alpha1_hat, mu_hat))

        # —— I/O：立即生成并保存该组合的累积预测 ——
        self.generate_cumulative_forecasts(vol_type, freq, test_series)

        print(f"[PID {os.getpid()}] 完成: {vol_type} ({freq})")
        return (vol_type, freq, d_hat, alpha1_hat, mu_hat)

    def train_and_predict_parallel(self, max_workers=None):
        """
        使用多进程并行：不同(vol_type, freq)组合独立运行。
        - 算法逻辑与计算步骤保持一致
        - 仅改变执行方式与I/O时机（尽早保存）
        """
        combinations = list(product(self.volatility_types, self.frequencies))
        results = []

        if max_workers is None:
            max_workers = os.cpu_count() or 1

        print("=" * 70)
        print(f"并行启动 {len(combinations)} 个任务，max_workers={max_workers}")
        print("=" * 70)

        task_args = [(vt, fq, self.root_dir) for vt, fq in combinations]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_args = {ex.submit(_worker_entry, args): args for args in task_args}
            for fut in as_completed(future_to_args):
                args = future_to_args[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    vt, fq, _ = args
                    print(f"处理 {vt} ({fq}) 时出错: {str(e)}")

        # 可选：汇总参数到一个总CSV（顺序不保证，但不影响内容）
        if results:
            params_df = pd.DataFrame([{
                'volatility_type': vt,
                'frequency': fq,
                'd_hat': d,
                'alpha1_hat': a1,
                'mu_hat': mu
            } for (vt, fq, d, a1, mu) in results])

            combined_file = os.path.join(self.paths['output_params'], "ARFIMA_model_parameters_combined.csv")
            params_df.to_csv(combined_file, index=False)
            print(f"\n已保存模型参数汇总: {combined_file}")

        print("\nARFIMA模型训练和预测完成!")


# ---- Windows/spawn-safe：把 worker 放在模块顶层，确保可被 pickle ----
def _worker_entry(args):
    vol_type, freq, root_dir = args
    model = ARFIMA_Model(root_dir)
    return model.run_one_combination(vol_type, freq)


def main(root_dir, max_workers=None):
    """
    主函数：执行ARFIMA模型训练和预测（并行 + 循环内部保存）
    """
    print("=" * 70)
    print("开始ARFIMA模型训练和预测（并行版）")
    print(f"项目根目录: {root_dir}")
    print("=" * 70)

    arfima_model = ARFIMA_Model(root_dir)
    arfima_model.train_and_predict_parallel(max_workers=max_workers)


if __name__ == "__main__":
    # 支持：python arfima_parallel_spawn_safe.py ROOT_DIR [MAX_WORKERS]
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        root_dir = "D:/SNW/WTI_volatility_forecast_replication/"

    if len(sys.argv) >= 3:
        try:
            max_workers = int(sys.argv[2])
        except Exception:
            max_workers = None
    else:
        max_workers = None

    main(root_dir, max_workers=max_workers)
