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


def _detect_parquet_engine():
    """
    自动探测可用的 parquet 引擎。
    优先 pyarrow，其次 fastparquet；都不可用则返回 None。
    """
    try:
        import pyarrow  # noqa: F401
        return "pyarrow"
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401
        return "fastparquet"
    except Exception:
        pass
    return None


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
                                       "cumulative_forecasts"),
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

        # parquet 引擎
        self.parquet_engine = _detect_parquet_engine()
        if self.parquet_engine is None:
            # 读/写 parquet 会失败，这里不立刻中止，在真正读写时抛出更清晰的报错
            print("警告：未检测到可用的 Parquet 引擎（pyarrow / fastparquet）。读写 parquet 将失败。", flush=True)

        # 存储模型参数（仅在单任务上下文使用）
        self.model_params = {}

    # ---------------------- 基础 I/O ----------------------
    def _read_parquet(self, path):
        if self.parquet_engine is None:
            raise RuntimeError(
                "Unable to find a usable parquet engine. Please install:\n"
                "  - conda: conda install -n WTI_volatility_forecast_replication -c conda-forge pyarrow fastparquet\n"
                "  - pip:   pip install pyarrow fastparquet"
            )
        return pd.read_parquet(path, engine=self.parquet_engine)

    def _to_parquet(self, df, path):
        if self.parquet_engine is None:
            raise RuntimeError(
                "Unable to find a usable parquet engine for writing. Please install:\n"
                "  - conda: conda install -n WTI_volatility_forecast_replication -c conda-forge pyarrow fastparquet\n"
                "  - pip:   pip install pyarrow fastparquet"
            )
        df.to_parquet(path, engine=self.parquet_engine)

    # ---------------------- 数据加载（含 RK 特例） ----------------------
    def load_data(self, vol_type, freq, dataset='train'):
        """
        加载波动率数据（不改变原始数据结构与含义）
        - RK 特例：文件名不是 CL_WTI_RK_...，而是 realized_kernel_estimates_[train|test].parquet
        """
        if dataset == 'train':
            data_dir = self.paths['train_data']
        elif dataset == 'test':
            data_dir = self.paths['test_data']
        else:
            raise ValueError("dataset 参数必须是 'train' 或 'test'")

        if vol_type == 'RK':
            file_name = f"realized_kernel_estimates_{dataset}.parquet"
        else:
            file_name = f"CL_WTI_{vol_type}_daily_{freq}_{dataset}.parquet"

        file_path = os.path.join(data_dir, file_name)

        if os.path.exists(file_path):
            df = self._read_parquet(file_path)
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
        获取波动率序列（与原逻辑一致；仅 RK 做列名分流）
        """
        df = self.load_data(vol_type, freq, dataset)

        if vol_type == 'RV':
            col = 'RV'
        elif vol_type == 'TSRV':
            col = 'TSRV'
        elif vol_type == 'RK':
            col = 'RK_1min' if freq == '1min' else 'RK_5min'
        elif vol_type == 'BV':
            col = 'BV'
        elif vol_type == 'MedRV':
            col = 'MedRV'
        elif vol_type == 'JWTSRV':
            col = 'JWTSRV'
        else:
            raise ValueError(f"未知的波动率类型: {vol_type}")

        if col not in df.columns:
            raise KeyError(f"{vol_type} 的列名 {col} 不存在。实际可用列: {list(df.columns)}")
        return df[col]

    # ---------------------- 模型算法：一行不改，仅做微小实现级优化 ----------------------
    def fractionally_difference(self, series, d):
        """
        应用分数差分 (1-L)^d 到时间序列 —— 完全保留原算法逻辑（逐项二项式展开）
        """
        n = len(series)
        diff_series = np.zeros(n, dtype=float)

        # 二项式展开系数
        binom_coeffs = [1.0]
        for k in range(1, n):
            binom_coeffs.append(binom_coeffs[-1] * (d - k + 1) / k)

        values = series.to_numpy()
        for t in range(n):
            diff_value = 0.0
            for k in range(t + 1):
                diff_value += binom_coeffs[k] * values[t - k]
            diff_series[t] = diff_value

        return pd.Series(diff_series, index=series.index)

    def arfima_log_likelihood(self, params, series):
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
        for t in range(1, len(diff_series)):
            residuals[t] = diff_series.iloc[t] - alpha1 * diff_series.iloc[t - 1]

        n = len(residuals)
        ssr = np.sum(residuals ** 2)
        log_lik = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ssr / n) - n / 2
        return -log_lik  # 最小化负对数似然

    def fit_arfima(self, series):
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
        d, alpha1, mu = params
        diff_series = self.fractionally_difference(series - mu, d).dropna()

        ar_model = ARIMA(diff_series, order=(1, 0, 0))
        ar_fit = ar_model.fit()
        ar_forecast = ar_fit.forecast(steps=steps)

        # 逆变换按原注释的近似处理
        forecast_values = ar_forecast + mu
        return forecast_values

    # ---------------------- I/O：每个组合算完就落盘 ----------------------
    def save_forecast_single(self, vol_type, freq, forecast_series):
        forecast_df = pd.DataFrame({
            'DateTime': forecast_series.index,
            'forecast': forecast_series.values
        }).set_index('DateTime')

        output_file = os.path.join(
            self.paths['output_forecasts'],
            f"ARFIMA_{vol_type}_{freq}_forecast.parquet"
        )
        self._to_parquet(forecast_df, output_file)

    def save_model_parameters_single(self, vol_type, freq, params):
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

    def generate_cumulative_forecasts(self, vol_type, freq, test_series):
        """
        生成累积预测（原逻辑不变），但在单任务内部调用并立即保存。
        对 RK：先尝试默认命名 CL_WTI_RK_daily_{freq}_cumulative.parquet；
              如不存在，再尝试 realized_kernel_estimates_cumulative.parquet（若你有这个文件）。
        """
        # —— 新命名规则：用 h1 文件当“子周期锚”，只取索引用来滚动起点 ——
        subdir = self.paths['subperiods']
        if vol_type == 'RK':
            base = f"realized_kernel_estimates_test_{('1min' if freq == '1min' else '5min')}_test"
        else:
            base = f"CL_WTI_{vol_type}_daily_{freq}_test_test"

        subperiod_path = os.path.join(subdir, base + "_cumulative_h1.parquet")
        if not os.path.exists(subperiod_path):
            print(f"未找到子周期锚文件: {subperiod_path}", flush=True)
            return

        subperiod_df = self._read_parquet(subperiod_path)
        if not isinstance(subperiod_df.index, pd.DatetimeIndex):
            if 'DateTime' in subperiod_df.columns:
                subperiod_df = subperiod_df.set_index('DateTime')
            else:
                subperiod_df.index = pd.to_datetime(subperiod_df.index)



        subperiod_df = self._read_parquet(subperiod_path)

        params = self.model_params.get((vol_type, freq))
        if params is None:
            print(f"找不到 {vol_type} ({freq}) 的模型参数")
            return

        train_series = self.get_volatility_series(vol_type, freq, 'train')

        cumulative_forecasts = {}
        for h in [5, 10]:
            cum_forecasts = []
            # 逐行滚动：与原算法一致（v_{t+h} = sqrt(mean(v^2)))
            for idx, _row in subperiod_df.iterrows():
                start_date = idx
                history = pd.concat([train_series, test_series[:start_date]])
                forecast_steps = self.arfima_forecast(history, params, steps=h)
                cum_vol = float(np.sqrt(np.mean(np.square(forecast_steps))))
                cum_forecasts.append(cum_vol)

            col_name = f"{vol_type}_cumulative_h{h}"
            cumulative_forecasts[col_name] = cum_forecasts

        cum_df = pd.DataFrame(cumulative_forecasts, index=subperiod_df.index)
        output_file = os.path.join(
            self.paths['output_forecasts'],
            f"ARFIMA_{vol_type}_{freq}_cumulative.parquet"
        )
        self._to_parquet(cum_df, output_file)
        print(f"已保存累积预测: {output_file}")

    # ---------------------- 单组合执行 ----------------------
    def run_one_combination(self, vol_type, freq):
        print(f"\n[PID {os.getpid()}] 训练和预测: {vol_type} ({freq})")

        # 1) 训练
        train_series = self.get_volatility_series(vol_type, freq, 'train')
        d_hat, alpha1_hat, mu_hat = self.fit_arfima(train_series)
        self.model_params[(vol_type, freq)] = (d_hat, alpha1_hat, mu_hat)

        # 2) 测试 + 滚动预测
        test_series = self.get_volatility_series(vol_type, freq, 'test')

        forecasts = []
        history = train_series.copy()
        for i in range(len(test_series)):
            forecast_step = self.arfima_forecast(history, (d_hat, alpha1_hat, mu_hat), steps=1)
            forecasts.append(float(forecast_step.iloc[0]))
            history = pd.concat([history, pd.Series([test_series.iloc[i]], index=[test_series.index[i]])])

        forecast_series = pd.Series(forecasts, index=test_series.index, name=f'ARFIMA_{vol_type}_{freq}')

        # 3) 就地落盘
        self.save_forecast_single(vol_type, freq, forecast_series)
        self.save_model_parameters_single(vol_type, freq, (d_hat, alpha1_hat, mu_hat))
        self.generate_cumulative_forecasts(vol_type, freq, test_series)

        print(f"[PID {os.getpid()}] 完成: {vol_type} ({freq})")
        return (vol_type, freq, d_hat, alpha1_hat, mu_hat)

    # ---------------------- 并行调度 ----------------------
    def train_and_predict_parallel(self, max_workers=None):
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

        # 汇总参数（不影响已即时保存的单文件）
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
    print("=" * 70)
    print("开始ARFIMA模型训练和预测（并行版）")
    print(f"项目根目录: {root_dir}")
    print("=" * 70)

    arfima_model = ARFIMA_Model(root_dir)
    arfima_model.train_and_predict_parallel(max_workers=max_workers)


if __name__ == "__main__":
    # 用法：python arfima_parallel_spawn_safe_rkfix.py ROOT_DIR [MAX_WORKERS]
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
