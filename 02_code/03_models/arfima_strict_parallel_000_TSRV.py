#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# ---- 防止 BLAS 在线程层面过度并行，避免与多进程叠加导致过度占用 ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import warnings
import numpy as np
import pandas as pd
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")


# ---------------------- Parquet 引擎探测 ----------------------
def _detect_parquet_engine():
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


# ---------------------- 频域工具：Periodogram ----------------------
def _periodogram(x):
    """
    简单 periodogram（无窗、无去趋势），返回 (freqs, I_lambda)
    - 仅正频率（排除 0，保留到 Nyquist 前）
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    x = x - x.mean()
    fft = np.fft.rfft(x)
    I = (1.0 / (2.0 * np.pi * n)) * np.abs(fft)**2
    freqs = np.fft.rfftfreq(n, d=1.0)

    if n % 2 == 0:
        freqs = freqs[1:-1]
        I = I[1:-1]
    else:
        freqs = freqs[1:]
        I = I[1:]
    lambdas = 2.0 * np.pi * freqs
    return lambdas, I


def _arfima10_spectral_shape(lmbd, d, alpha):
    """
    ARFIMA(1,d,0) 的谱密度形状函数（不含 σ²/2π 常数）
    """
    sin_term = 2.0 * np.sin(lmbd / 2.0)
    sin_term = np.maximum(sin_term, 1e-12)
    part_d = sin_term ** (-2.0 * d)

    denom_ar = (1.0 + alpha * alpha - 2.0 * alpha * np.cos(lmbd))
    denom_ar = np.maximum(denom_ar, 1e-12)
    part_ar = 1.0 / denom_ar

    return part_d * part_ar


def _whittle_profile_objective(params, x):
    d, alpha = params
    if not (-0.499 < d < 0.499) or not (-0.999 < alpha < 0.999):
        return 1e12

    lambdas, I = _periodogram(x)
    g = _arfima10_spectral_shape(lambdas, d, alpha)

    g = np.maximum(g, 1e-18)
    ratio = I / g
    mean_ratio = np.mean(ratio)
    if not np.isfinite(mean_ratio) or mean_ratio <= 0:
        return 1e12

    obj = np.sum(np.log(g)) + I.size * np.log(mean_ratio)
    return obj


# ---------------------- 分数差分与积分系数 ----------------------
def fractional_diff_binom_coeffs(d, n):
    coeffs = np.empty(n, dtype=float)
    coeffs[0] = 1.0
    for k in range(1, n):
        coeffs[k] = coeffs[k - 1] * (d - k + 1.0) / k * (-1.0)
    return coeffs


def fractional_int_coeffs(d, K):
    coeffs = np.empty(K+1, dtype=float)
    coeffs[0] = 1.0
    for k in range(1, K+1):
        coeffs[k] = coeffs[k-1] * (d + k - 1.0) / k
    return coeffs


def fractional_difference(series, d):
    values = series.to_numpy(dtype=float)
    n = values.size
    coeffs = fractional_diff_binom_coeffs(d, n)
    out = np.zeros(n, dtype=float)
    for t in range(n):
        out[t] = np.dot(coeffs[:t+1], values[t::-1])
    return pd.Series(out, index=series.index)


# ---------------------- 更严格的 ARFIMA(1,d,0) 估计与预测 ----------------------
class ARFIMA_Strict:
    def __init__(self, method="whittle", K_inverse=1000):
        self.method = method
        self.K_inverse = int(K_inverse)

        self.d_ = None
        self.alpha_ = None
        self.mu_ = None
        self.sigma2_ = None

    def fit(self, series):
        x = series.to_numpy(dtype=float)
        mu0 = float(np.mean(x))
        self.mu_ = mu0

        if self.method == "whittle":
            init = np.array([0.2, 0.2], dtype=float)
            bounds = [(-0.499, 0.499), (-0.999, 0.999)]
            res = minimize(_whittle_profile_objective, init, args=(x - mu0,),
                           method="L-BFGS-B", bounds=bounds)
            if not res.success:
                d_hat = self._fallback_d_local_whittle(x - mu0)
                d_hat = float(np.clip(d_hat, -0.499, 0.499))
                alpha_hat = self._fit_ar1_on_fractional_diff(x - mu0, d_hat)
            else:
                d_hat, alpha_hat = res.x

            self.d_ = float(d_hat)
            self.alpha_ = float(alpha_hat)

            lambdas, I = _periodogram(x - mu0)
            g = _arfima10_spectral_shape(lambdas, self.d_, self.alpha_)
            g = np.maximum(g, 1e-18)
            self.sigma2_ = float(np.mean(I / g))

        else:
            d_hat = self._fallback_d_local_whittle(x - mu0)
            d_hat = float(np.clip(d_hat, -0.499, 0.499))
            alpha_hat = self._fit_ar1_on_fractional_diff(x - mu0, d_hat)
            self.d_ = float(d_hat)
            self.alpha_ = float(alpha_hat)
            w = fractional_difference(pd.Series(x - mu0), self.d_).to_numpy()
            eps = w[1:] - self.alpha_ * w[:-1]
            self.sigma2_ = float(np.var(eps, ddof=1))
        return self

    @staticmethod
    def _fallback_d_local_whittle(x):
        lambdas, I = _periodogram(x)
        n = x.size
        m = int(np.floor(n ** 0.7))
        m = np.clip(m, 10, I.size)
        lam = lambdas[:m]
        I_m = I[:m]
        log_lam = np.log(lam)

        def obj(d):
            if not (-0.499 < d < 0.499):
                return 1e12
            w = (lam ** (2.0 * d)) * I_m
            mean_w = np.mean(w)
            if mean_w <= 0 or not np.isfinite(mean_w):
                return 1e12
            return np.log(mean_w) - (2.0 * d) * np.mean(log_lam)

        res = minimize(lambda dd: obj(dd[0]), x0=np.array([0.2]), bounds=[(-0.499, 0.499)], method="L-BFGS-B")
        if res.success:
            return float(res.x[0])
        return 0.0

    @staticmethod
    def _fit_ar1_on_fractional_diff(x, d):
        w = fractional_difference(pd.Series(x), d).to_numpy()
        w0 = w[:-1]
        w1 = w[1:]
        denom = np.dot(w0, w0)
        if denom <= 1e-18:
            return 0.0
        alpha = float(np.dot(w1, w0) / denom)
        alpha = float(np.clip(alpha, -0.999, 0.999))
        return alpha

    def forecast(self, series, steps=1):
        assert self.d_ is not None and self.alpha_ is not None and self.mu_ is not None, "Model not fitted."

        x = series.to_numpy(dtype=float)
        w = fractional_difference(pd.Series(x - self.mu_), self.d_).to_numpy()
        w_t = w[-1]

        h = int(steps)
        w_future = np.empty(h, dtype=float)
        pow_alpha = 1.0
        for ℓ in range(1, h + 1):
            pow_alpha *= self.alpha_
            w_future[ℓ - 1] = pow_alpha * w_t

        K = int(self.K_inverse)
        pi = fractional_int_coeffs(self.d_, K)

        v_fore = np.empty(h, dtype=float)
        w_ext = np.concatenate([w, w_future])

        T = w.size - 1
        for idx in range(1, h + 1):
            tplus = T + idx
            k_max = min(K, tplus)
            w_slice = w_ext[tplus - k_max: tplus + 1][::-1]
            pi_slice = pi[:k_max + 1]
            v_fore[idx - 1] = self.mu_ + float(np.dot(pi_slice, w_slice))

        return pd.Series(v_fore, index=pd.RangeIndex(start=1, stop=h + 1, name="h"))


# ---------------------- 与项目结构的并行封装 ----------------------
class ARFIMA_Model:
    def __init__(self, root_dir, method="whittle", K_inverse=1000):
        self.root_dir = root_dir
        self.method = method
        self.K_inverse = int(K_inverse)

        self.paths = {
            'train_data': os.path.join(root_dir, "03_results", "intermediate_results", "volatility_estimates", "train_set"),
            'test_data': os.path.join(root_dir, "03_results", "intermediate_results", "volatility_estimates", "test_set"),
            'subperiods': os.path.join(root_dir, "03_results", "intermediate_results", "volatility_estimates","cumulative_forecasts"),
            'output_forecasts': os.path.join(root_dir, "03_results", "final_forecasts","ARFIMA"),
            'output_params': os.path.join(root_dir, "03_results", "intermediate_results", "model_parameters", "ARFIMA"),
            'temp_cache': os.path.join(root_dir, "01_data", "temp", "rolling_window_cache")
        }
        os.makedirs(self.paths['output_forecasts'], exist_ok=True)
        os.makedirs(self.paths['output_params'], exist_ok=True)
        os.makedirs(self.paths['temp_cache'], exist_ok=True)

        # ★★★ 仅处理 TSRV，其余不动；频率仍为 1min、5min ★★★
        self.volatility_types = ['TSRV']
        self.frequencies = ['1min', '5min']

        self.parquet_engine = _detect_parquet_engine()
        if self.parquet_engine is None:
            print("警告：未检测到可用的 Parquet 引擎（pyarrow / fastparquet）。读写 parquet 将失败。", flush=True)

        self.model_params = {}

    # ---------- I/O ----------
    def _read_parquet(self, path):
        if self.parquet_engine is None:
            raise RuntimeError(
                "Unable to find a usable parquet engine. Please install:\n"
                "  - conda: conda install -c conda-forge pyarrow fastparquet\n"
                "  - pip:   pip install pyarrow fastparquet"
            )
        return pd.read_parquet(path, engine=self.parquet_engine)

    def _to_parquet(self, df, path):
        if self.parquet_engine is None:
            raise RuntimeError(
                "Unable to find a usable parquet engine for writing. Please install:\n"
                "  - conda: conda install -c conda-forge pyarrow fastparquet\n"
                "  - pip:   pip install pyarrow fastparquet"
            )
        df.to_parquet(path, engine=self.parquet_engine)

    # ---------- 数据加载 ----------
    def load_data(self, vol_type, freq, dataset='train'):
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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        df = self._read_parquet(file_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'DateTime' in df.columns:
                df = df.set_index('DateTime')
            else:
                df.index = pd.to_datetime(df.index)
        return df

    def get_volatility_series(self, vol_type, freq, dataset='train'):
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

    # ---------- 严格估计与预测 ----------
    def fit_arfima_strict(self, series):
        model = ARFIMA_Strict(method=self.method, K_inverse=self.K_inverse).fit(series)
        return model

    def strict_forecast(self, strict_model, series, steps=1):
        return strict_model.forecast(series, steps=steps)

    # ---------- I/O：单组合保存 ----------
    def save_forecast_single(self, vol_type, freq, forecast_series):
        forecast_df = pd.DataFrame({'DateTime': forecast_series.index, 'forecast': forecast_series.values}).set_index('DateTime')
        output_file = os.path.join(self.paths['output_forecasts'], f"ARFIMA_{vol_type}_{freq}_forecast.parquet")
        self._to_parquet(forecast_df, output_file)

    def save_model_parameters_single(self, vol_type, freq, strict_model):
        single_df = pd.DataFrame([{
            'volatility_type': vol_type,
            'frequency': freq,
            'd_hat': strict_model.d_,
            'alpha1_hat': strict_model.alpha_,
            'mu_hat': strict_model.mu_,
            'sigma2_hat_profile': strict_model.sigma2_
        }])
        output_file = os.path.join(self.paths['output_params'], f"ARFIMA_params_{vol_type}_{freq}.csv")
        single_df.to_csv(output_file, index=False)

    def generate_cumulative_forecasts(self, vol_type, freq, strict_model, test_series):

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

        train_series = self.get_volatility_series(vol_type, freq, 'train').clip(lower=0).pow(0.5)
        cumulative_forecasts = {}

        for h in [5, 10]:
            cum_var_list = []
            cum_rms_list = []
            for idx in subperiod_df.index:
                history = pd.concat([train_series, test_series[:idx]])
                steps = self.strict_forecast(strict_model, history, steps=h).to_numpy()
                cum_var = float(np.mean(steps ** 2))
                cum_rms = float(np.sqrt(cum_var))
                cum_var_list.append(cum_var)
                cum_rms_list.append(cum_rms)

            cumulative_forecasts[f"{vol_type}_cumVar_h{h}"] = cum_var_list
            cumulative_forecasts[f"{vol_type}_cumVolRMS_h{h}"] = cum_rms_list

        cum_df = pd.DataFrame(cumulative_forecasts, index=subperiod_df.index)
        output_file = os.path.join(self.paths['output_forecasts'], f"ARFIMA_{vol_type}_{freq}_cumulative.parquet")
        self._to_parquet(cum_df, output_file)
        print(f"已保存累积预测: {output_file}")

    # ---------- 单组合执行 ----------
    def run_one_combination(self, vol_type, freq):
        print(f"\n[PID {os.getpid()}] 训练和预测: {vol_type} ({freq})")

        test_series = self.get_volatility_series(vol_type, freq, 'test')
        train_series = self.get_volatility_series(vol_type, freq, 'train').clip(lower=0).pow(0.5)
        strict_model = self.fit_arfima_strict(train_series)
        self.model_params[(vol_type, freq)] = (strict_model.d_, strict_model.alpha_, strict_model.mu_)
        test_series = self.get_volatility_series(vol_type, freq, 'test').clip(lower=0).pow(0.5)

        forecasts = []
        history = train_series.copy()
        for i in range(len(test_series)):
            forecast_step = self.strict_forecast(strict_model, history, steps=1)
            forecasts.append(float(forecast_step.iloc[0]))
            history = pd.concat([history, pd.Series([test_series.iloc[i]], index=[test_series.index[i]])])

        forecast_series = pd.Series(forecasts, index=test_series.index, name=f'ARFIMA_{vol_type}_{freq}')

        self.save_forecast_single(vol_type, freq, forecast_series)
        self.save_model_parameters_single(vol_type, freq, strict_model)
        self.generate_cumulative_forecasts(vol_type, freq, strict_model, test_series)

        print(f"[PID {os.getpid()}] 完成: {vol_type} ({freq})")
        return (vol_type, freq, strict_model.d_, strict_model.alpha_, strict_model.mu_, strict_model.sigma2_)

    # ---------- 并行调度 ----------
    def train_and_predict_parallel(self, max_workers=None):
        combinations = list(product(self.volatility_types, self.frequencies))
        results = []

        if max_workers is None:
            max_workers = os.cpu_count() or 1

        print("=" * 70)
        print(f"并行启动 {len(combinations)} 个任务，max_workers={max_workers}")
        print("=" * 70)

        task_args = [(vt, fq, self.root_dir, self.method, self.K_inverse) for vt, fq in combinations]

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_args = {ex.submit(_worker_entry, args): args for args in task_args}
            for fut in as_completed(future_to_args):
                args = future_to_args[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    vt, fq, *_ = args
                    print(f"处理 {vt} ({fq}) 时出错: {str(e)}")

        if results:
            params_df = pd.DataFrame([{
                'volatility_type': vt,
                'frequency': fq,
                'd_hat': d,
                'alpha1_hat': a1,
                'mu_hat': mu,
                'sigma2_hat_profile': s2
            } for (vt, fq, d, a1, mu, s2) in results])

            combined_file = os.path.join(self.paths['output_params'], "ARFIMA_model_parameters_combined.csv")
            params_df.to_csv(combined_file, index=False)
            print(f"\n已保存模型参数汇总: {combined_file}")

        print("\nARFIMA模型训练和预测完成!")


# ---- Windows/spawn-safe：把 worker 放在模块顶层，确保可被 pickle ----
def _worker_entry(args):
    vol_type, freq, root_dir, method, K_inverse = args
    model = ARFIMA_Model(root_dir, method=method, K_inverse=K_inverse)
    return model.run_one_combination(vol_type, freq)


def main(root_dir, max_workers=None, method="whittle", K_inverse=1000):
    print("=" * 70)
    print("开始ARFIMA模型训练和预测（严格 + 并行版）")
    print(f"项目根目录: {root_dir}")
    print(f"估计方法: {method}（推荐 whittle）; 逆分差阶数 K={K_inverse}")
    print("=" * 70)

    arfima_model = ARFIMA_Model(root_dir, method=method, K_inverse=K_inverse)
    arfima_model.train_and_predict_parallel(max_workers=max_workers)


if __name__ == "__main__":
    # 用法示例：python arfima_strict_parallel.py ROOT_DIR [MAX_WORKERS] [METHOD] [K]
    # METHOD in {"whittle", "conditional"}
    if len(sys.argv) >= 2:
        root_dir = sys.argv[1]
    else:
        root_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"

    if len(sys.argv) >= 3:
        try:
            max_workers = int(sys.argv[2])
        except Exception:
            max_workers = None
    else:
        max_workers = None

    method = "whittle"
    if len(sys.argv) >= 4:
        if sys.argv[3].lower() in ("whittle", "conditional"):
            method = sys.argv[3].lower()

    K_inverse = 1000
    if len(sys.argv) >= 5:
        try:
            K_inverse = int(sys.argv[4])
        except Exception:
            K_inverse = 1000

    main(root_dir, max_workers=max_workers, method=method, K_inverse=K_inverse)
