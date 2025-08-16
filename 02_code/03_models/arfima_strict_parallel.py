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
    # Demean: Whittle 通常基于去均值数据
    x = x - x.mean()
    fft = np.fft.rfft(x)  # 包含 0 和 Nyquist（若 n 偶数）
    I = (1.0 / (2.0 * np.pi * n)) * np.abs(fft)**2  # 常用 periodogram 归一化之一
    freqs = np.fft.rfftfreq(n, d=1.0)

    # 排除 0 频（以及可选排除 Nyquist）
    # 这里保留 0<lambda<pi 的频率点
    if n % 2 == 0:
        # 偶数：rfft 包含 Nyquist 点 freqs[-1] = 0.5 周期/采样
        freqs = freqs[1:-1]
        I = I[1:-1]
    else:
        freqs = freqs[1:]
        I = I[1:]
    # 将频率换算到 [0, pi] 上的角频率
    lambdas = 2.0 * np.pi * freqs  # in (0, pi)
    return lambdas, I


def _arfima10_spectral_shape(lmbd, d, alpha):
    """
    ARFIMA(1,d,0) 的谱密度形状函数（不含 σ²/2π 常数），返回 g(λ; d, α)。
    f(λ) = (σ² / 2π) * g(λ; d, α)

    g(λ; d, α) = |1 - e^{-iλ}|^{-2d} * |1 - α e^{-iλ}|^{-2}
               = [ 2 * sin(λ/2) ]^{-2d} * [ 1 + α^2 - 2α cos λ ]^{-1}
    """
    sin_term = 2.0 * np.sin(lmbd / 2.0)
    sin_term = np.maximum(sin_term, 1e-12)  # 避免 log(0)
    part_d = sin_term ** (-2.0 * d)

    denom_ar = (1.0 + alpha * alpha - 2.0 * alpha * np.cos(lmbd))
    denom_ar = np.maximum(denom_ar, 1e-12)
    part_ar = 1.0 / denom_ar

    return part_d * part_ar


def _whittle_profile_objective(params, x):
    """
    Whittle 轮廓似然（剖面似然）目标函数：给定 (d, α)，剖出 σ² 后的目标。
    Lw(d, α) ≈ sum_j [ log g_j(d, α) ] + m * log( mean_j I_j / g_j(d, α) )

    其中 f_j = (σ²/2π) g_j，最优 σ²_hat = mean(I_j / g_j) * 2π。
    由于常数 2π 不影响最小化，这里省略。
    """
    d, alpha = params
    # 参数约束
    if not (-0.499 < d < 0.499) or not (-0.999 < alpha < 0.999):
        return 1e12

    lambdas, I = _periodogram(x)
    g = _arfima10_spectral_shape(lambdas, d, alpha)

    # 数值稳定
    g = np.maximum(g, 1e-18)
    ratio = I / g
    mean_ratio = np.mean(ratio)
    if not np.isfinite(mean_ratio) or mean_ratio <= 0:
        return 1e12

    obj = np.sum(np.log(g)) + I.size * np.log(mean_ratio)
    return obj


# ---------------------- 分数差分与积分系数 ----------------------
def fractional_diff_binom_coeffs(d, n):
    """
    计算 (1 - L)^d 的二项式系数（长度 n）：
    c_0 = 1; c_k = c_{k-1} * (d - k + 1) / k
    """
    coeffs = np.empty(n, dtype=float)
    coeffs[0] = 1.0
    for k in range(1, n):
        #coeffs[k] = coeffs[k-1] * (d - k + 1.0) / k
        coeffs[k] = coeffs[k - 1] * (d - k + 1.0) / k * (-1.0)  # ← 加这个 -1.0
    return coeffs


def fractional_int_coeffs(d, K):
    """
    计算 (1 - L)^{-d} 的系数 {π_k(d)}_{k=0..K}：
    π_0 = 1; π_k = π_{k-1} * (d + k - 1) / k, k>=1
    相当于 π_k = Γ(k + d) / (Γ(k+1) Γ(d))
    """
    coeffs = np.empty(K+1, dtype=float)
    coeffs[0] = 1.0
    for k in range(1, K+1):
        coeffs[k] = coeffs[k-1] * (d + k - 1.0) / k
    return coeffs


def fractional_difference(series, d):
    """
    O(n^2) 朴素分数差分： (1 - L)^d 应用于 series。
    与原实现保持一致（无截断），返回与输入等长的 Series。
    """
    values = series.to_numpy(dtype=float)
    n = values.size
    coeffs = fractional_diff_binom_coeffs(d, n)
    out = np.zeros(n, dtype=float)
    # 卷积：out[t] = Σ_{k=0..t} coeffs[k] * values[t-k]
    for t in range(n):
        out[t] = np.dot(coeffs[:t+1], values[t::-1])
    return pd.Series(out, index=series.index)


# ---------------------- 更严格的 ARFIMA(1,d,0) 估计与预测 ----------------------
class ARFIMA_Strict:
    def __init__(self, method="whittle", K_inverse=1000):
        """
        method: "whittle" | "conditional"
            - "whittle": 频域 Whittle 似然，剖面化 σ²，优化 (d, α)，μ 取样本均值；
            - "conditional": 与原似然近似类似，用 SSR 近似（保留回退方案时可用）。
        K_inverse: 逆分差（分数积分）截断阶数 K，建议 500–2000 之间。
        """
        self.method = method
        self.K_inverse = int(K_inverse)

        self.d_ = None
        self.alpha_ = None
        self.mu_ = None
        self.sigma2_ = None  # 便于记录（Whittle 可剖面估计）

    def fit(self, series):
        x = series.to_numpy(dtype=float)
        mu0 = float(np.mean(x))
        self.mu_ = mu0

        # 选择估计方法
        if self.method == "whittle":
            # 优化 (d, α) 的 Whittle 剖面似然
            init = np.array([0.2, 0.2], dtype=float)
            bounds = [(-0.499, 0.499), (-0.999, 0.999)]
            res = minimize(_whittle_profile_objective, init, args=(x - mu0,),
                           method="L-BFGS-B", bounds=bounds)
            if not res.success:
                # 回退到 local Whittle 估 d + AR(1) on diff
                d_hat = self._fallback_d_local_whittle(x - mu0)
                d_hat = float(np.clip(d_hat, -0.499, 0.499))
                alpha_hat = self._fit_ar1_on_fractional_diff(x - mu0, d_hat)
            else:
                d_hat, alpha_hat = res.x

            self.d_ = float(d_hat)
            self.alpha_ = float(alpha_hat)

            # 剖面 σ² 估计（用于记录）
            lambdas, I = _periodogram(x - mu0)
            g = _arfima10_spectral_shape(lambdas, self.d_, self.alpha_)
            g = np.maximum(g, 1e-18)
            self.sigma2_ = float(np.mean(I / g))

        else:
            # conditional 近似：与原始代码思路相近（不再详细实现，以 whittle 为主）
            # 这里做一个简单调用：先用 local Whittle 估 d，再在分差域拟合 AR(1)
            d_hat = self._fallback_d_local_whittle(x - mu0)
            d_hat = float(np.clip(d_hat, -0.499, 0.499))
            alpha_hat = self._fit_ar1_on_fractional_diff(x - mu0, d_hat)
            self.d_ = float(d_hat)
            self.alpha_ = float(alpha_hat)
            # σ²（分差域残差方差）
            w = fractional_difference(pd.Series(x - mu0), self.d_).to_numpy()
            eps = w[1:] - self.alpha_ * w[:-1]
            self.sigma2_ = float(np.var(eps, ddof=1))

        return self

    @staticmethod
    def _fallback_d_local_whittle(x):
        """
        简化版 local Whittle 估计 d：
        选择 m = floor(n^0.7) 个最低频率点，最小化 Robinson (1995) 目标。
        """
        lambdas, I = _periodogram(x)
        n = x.size
        m = int(np.floor(n ** 0.7))
        m = np.clip(m, 10, I.size)  # 基本防护
        lam = lambdas[:m]
        I_m = I[:m]

        # 目标：Q(d) = log( (1/m) Σ λ_j^{2d} I_j ) - (2d/m) Σ log λ_j
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
        # 极端失败：回退到 0
        return 0.0

    @staticmethod
    def _fit_ar1_on_fractional_diff(x, d):
        """
        在分差域 w_t 上拟合 AR(1)：w_t = α w_{t-1} + ε_t
        解析解 α_hat = Σ w_t w_{t-1} / Σ w_{t-1}^2
        """
        w = fractional_difference(pd.Series(x), d).to_numpy()
        w0 = w[:-1]
        w1 = w[1:]
        denom = np.dot(w0, w0)
        if denom <= 1e-18:
            return 0.0
        alpha = float(np.dot(w1, w0) / denom)
        # 约束到稳定区间
        alpha = float(np.clip(alpha, -0.999, 0.999))
        return alpha

    def forecast(self, series, steps=1):
        """
        更严格的 ARFIMA(1,d,0) 预测：
        1) w_t = (1-L)^d (v_t - μ)
        2) 未来分差域预测：w_{t+h|t} = α^h * w_t
        3) 逆分差：v_{t+h|t} - μ = Σ_{k=0}^{K} π_k(d) * w_{t+h-k|t}
           其中 π_k(d) 为 (1-L)^{-d} 的系数，K=K_inverse。
        """
        assert self.d_ is not None and self.alpha_ is not None and self.mu_ is not None, \
            "Model not fitted."

        x = series.to_numpy(dtype=float)
        # 历史分差序列（等长，无截断 NaN）
        w = fractional_difference(pd.Series(x - self.mu_), self.d_).to_numpy()
        w_t = w[-1]

        # 未来分差域路径
        h = int(steps)
        w_future = np.empty(h, dtype=float)
        # 递推：w_{t+ℓ|t} = α^ℓ w_t
        pow_alpha = 1.0
        for ℓ in range(1, h + 1):
            pow_alpha *= self.alpha_
            w_future[ℓ - 1] = pow_alpha * w_t

        # 逆分差系数
        K = int(self.K_inverse)
        pi = fractional_int_coeffs(self.d_, K)

        # 为了计算 v_{t+h|t}，需要 w_{t+h-k|t}，当 k>h 时使用历史 w_{t-(k-h)}
        v_fore = np.empty(h, dtype=float)
        # 扩展 w 序列：先历史，再未来
        w_ext = np.concatenate([w, w_future])

        # 目标是对每个 h：y_{T+h} = μ + Σ_{k=0}^K π_k * w_{T+h-k}
        T = w.size - 1  # w 的最后一个索引
        for idx in range(1, h + 1):
            tplus = T + idx  # 对应 w_ext 的索引
            # 卷积窗口 w_{tplus - k}, k=0..K
            k_max = min(K, tplus)  # 不能越过最早的索引 0
            # 注意：Python 切片右开
            w_slice = w_ext[tplus - k_max: tplus + 1][::-1]  # 倒序成 k=0..k_max
            pi_slice = pi[:k_max + 1]
            v_fore[idx - 1] = self.mu_ + float(np.dot(pi_slice, w_slice))

        #return pd.Series(v_fore, index=series.index[-1:] + pd.to_timedelta(np.arange(1, h + 1), unit="D"))

        return pd.Series(v_fore, index=pd.RangeIndex(start=1, stop=h + 1, name="h"))


# ---------------------- 与项目结构的并行封装 ----------------------
class ARFIMA_Model:
    def __init__(self, root_dir, method="whittle", K_inverse=1000):
        """
        method: "whittle"（推荐）或 "conditional"
        K_inverse: 逆分差截断阶数（建议 500–2000）
        """
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

        self.volatility_types = ['RV', 'TSRV', 'RK', 'BV', 'MedRV', 'JWTSRV']
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



        #train_series = self.get_volatility_series(vol_type, freq, 'train')
        train_series = self.get_volatility_series(vol_type, freq, 'train').clip(lower=0).pow(0.5)
        cumulative_forecasts = {}
        # for h in [5, 10]:
        #     cum_forecasts = []
        #     for idx, _row in subperiod_df.iterrows():
        #         start_date = idx
        #         history = pd.concat([train_series, test_series[:start_date]])
        #         forecast_steps = self.strict_forecast(strict_model, history, steps=h).to_numpy()
        #         #cum_vol = float(np.sqrt(np.mean(np.square(forecast_steps))))
        #         #cum_forecasts.append(cum_vol)
        #         cum_var = float(np.mean(forecast_steps ** 2))  # 论文用：未来 h 天“方差均值”
        #         cum_rms = float(np.sqrt(cum_var))  # 你的展示用：RMS（标准差量纲）
        #
        #     col_name = f"{vol_type}_cumulative_h{h}"
        #     cumulative_forecasts[col_name] = cum_forecasts
        for h in [5, 10]:
            # 新增：两条列表分别存“方差均值”和“RMS”
            cum_var_list = []
            cum_rms_list = []

            # 用 index 遍历即可
            for idx in subperiod_df.index:
                history = pd.concat([train_series, test_series[:idx]])
                steps = self.strict_forecast(strict_model, history, steps=h).to_numpy()

                cum_var = float(np.mean(steps ** 2))  # 论文口径：未来 h 天的方差均值
                cum_rms = float(np.sqrt(cum_var))  # 展示用：RMS（标准差量纲）

                # 追加到列表（你之前少了这两行）
                cum_var_list.append(cum_var)
                cum_rms_list.append(cum_rms)

            # 保存两列（别再用原来的 col_name/cum_forecasts 了）
            cumulative_forecasts[f"{vol_type}_cumVar_h{h}"] = cum_var_list
            cumulative_forecasts[f"{vol_type}_cumVolRMS_h{h}"] = cum_rms_list

        cum_df = pd.DataFrame(cumulative_forecasts, index=subperiod_df.index)
        output_file = os.path.join(self.paths['output_forecasts'], f"ARFIMA_{vol_type}_{freq}_cumulative.parquet")
        self._to_parquet(cum_df, output_file)
        print(f"已保存累积预测: {output_file}")

    # ---------- 单组合执行 ----------
    def run_one_combination(self, vol_type, freq):
        print(f"\n[PID {os.getpid()}] 训练和预测: {vol_type} ({freq})")

        # 1) 训练
        # train_series = self.get_volatility_series(vol_type, freq, 'train')
        # strict_model = self.fit_arfima_strict(train_series)
        #self.model_params[(vol_type, freq)] = (strict_model.d_, strict_model.alpha_, strict_model.mu_)

        # 2) 测试 + 滚动预测（严格外推，不再分差域重拟合）
        test_series = self.get_volatility_series(vol_type, freq, 'test')
        # run_one_combination() 里
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

        # 3) 就地落盘
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

        # 汇总参数（不影响已即时保存的单文件）
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
        root_dir = "D:/SNW/WTI_volatility_forecast_replication/"

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
