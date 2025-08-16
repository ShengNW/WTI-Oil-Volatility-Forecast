import pandas as pd
import numpy as np
import os
import sys
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from statsmodels.tsa.stattools import acf
from arch.unitroot import ADF
import warnings

warnings.filterwarnings('ignore')


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

        # 存储模型参数
        self.model_params = {}

    def load_data(self, vol_type, freq, dataset='train'):
        """
        加载波动率数据

        参数:
        vol_type (str): 波动率类型 (RV, TSRV, RK, BV, MedRV, JWTSRV)
        freq (str): 数据频率 (1min 或 5min)
        dataset (str): 数据集类型 (train 或 test)

        返回:
        pd.DataFrame: 加载的数据
        """
        if dataset == 'train':
            data_dir = self.paths['train_data']
        elif dataset == 'test':
            data_dir = self.paths['test_data']
        else:
            raise ValueError("dataset 参数必须是 'train' 或 'test'")

        # 构建文件名
        file_name = f"CL_WTI_{vol_type}_daily_{freq}_{dataset}.parquet"
        file_path = os.path.join(data_dir, file_name)

        # 加载数据
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            # 确保索引是DatetimeIndex
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
        获取波动率序列

        参数:
        vol_type (str): 波动率类型
        freq (str): 数据频率
        dataset (str): 数据集类型

        返回:
        pd.Series: 波动率序列
        """
        df = self.load_data(vol_type, freq, dataset)

        # 根据波动率类型选择正确的列
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

        参数:
        series (pd.Series): 输入时间序列
        d (float): 分数差分阶数 (-0.5 < d < 0.5)

        返回:
        pd.Series: 分数差分后的序列
        """
        n = len(series)
        diff_series = np.zeros(n)

        # 二项式展开系数
        binom_coeffs = [1]
        for k in range(1, n):
            binom_coeffs.append(binom_coeffs[-1] * (d - k + 1) / k)

        # 应用分数差分
        for t in range(n):
            diff_value = 0
            for k in range(t + 1):
                diff_value += binom_coeffs[k] * series.iloc[t - k]
            diff_series[t] = diff_value

        return pd.Series(diff_series, index=series.index)

    def arfima_log_likelihood(self, params, series):
        """
        计算ARFIMA(1,d,0)的对数似然

        参数:
        params (list): 参数列表 [d, alpha1, mu]
        series (pd.Series): 时间序列数据

        返回:
        float: 对数似然值
        """
        d, alpha1, mu = params

        # 参数约束
        if not (-0.5 < d < 0.5) or not (-1 < alpha1 < 1):
            return 1e10  # 返回大值表示无效参数

        # 应用分数差分
        try:
            diff_series = self.fractionally_difference(series - mu, d)
        except:
            return 1e10

        # 移除初始值
        diff_series = diff_series.dropna()

        # 计算AR(1)残差
        residuals = np.zeros(len(diff_series))
        residuals[0] = diff_series.iloc[0]  # 初始值

        for t in range(1, len(diff_series)):
            residuals[t] = diff_series.iloc[t] - alpha1 * diff_series.iloc[t - 1]

        # 计算对数似然 (假设正态分布)
        n = len(residuals)
        ssr = np.sum(residuals ** 2)
        log_lik = -n / 2 * np.log(2 * np.pi) - n / 2 * np.log(ssr / n) - n / 2

        return -log_lik  # 最小化负对数似然

    def fit_arfima(self, series):
        """
        拟合ARFIMA(1,d,0)模型

        参数:
        series (pd.Series): 时间序列数据

        返回:
        tuple: (d, alpha1, mu) 估计参数
        """
        # 初始参数猜测
        initial_params = [0.4, 0.2, series.mean()]

        # 参数边界
        bounds = [(-0.499, 0.499), (-0.999, 0.999), (series.min() * 0.5, series.max() * 1.5)]

        # 最小化负对数似然
        result = minimize(self.arfima_log_likelihood, initial_params, args=(series,),
                          method='L-BFGS-B', bounds=bounds)

        if result.success:
            d_hat, alpha1_hat, mu_hat = result.x
            return d_hat, alpha1_hat, mu_hat
        else:
            # 如果优化失败，使用备选方法
            return self.fit_arfima_fallback(series)

    def fit_arfima_fallback(self, series):
        """
        备选ARFIMA拟合方法（当MLE失败时）

        参数:
        series (pd.Series): 时间序列数据

        返回:
        tuple: (d, alpha1, mu) 估计参数
        """
        # 1. 使用GPH方法估计d
        # 基于自相关函数(ACF)的衰减率
        acf_vals = acf(series, nlags=100, fft=True)
        lags = np.arange(1, 101)
        log_acf = np.log(np.abs(acf_vals[1:101]))
        log_lags = np.log(lags)

        # 使用前20个lags进行回归
        slope = np.polyfit(log_lags[:20], log_acf[:20], 1)[0]
        d_hat = -slope / 2

        # 确保d在合理范围内
        d_hat = max(min(d_hat, 0.499), -0.499)

        # 2. 应用分数差分
        diff_series = self.fractionally_difference(series, d_hat)

        # 3. 拟合AR(1)模型
        model = ARIMA(diff_series.dropna(), order=(1, 0, 0))
        result = model.fit()
        alpha1_hat = result.params[0]
        mu_hat = series.mean()  # 使用原始序列的均值

        return d_hat, alpha1_hat, mu_hat

    def arfima_forecast(self, series, params, steps=1):
        """
        ARFIMA模型预测

        参数:
        series (pd.Series): 历史数据
        params (tuple): 模型参数 (d, alpha1, mu)
        steps (int): 预测步长

        返回:
        np.array: 预测值
        """
        d, alpha1, mu = params

        # 应用分数差分
        diff_series = self.fractionally_difference(series - mu, d)
        diff_series = diff_series.dropna()

        # 使用AR(1)进行预测
        ar_model = ARIMA(diff_series, order=(1, 0, 0))
        ar_fit = ar_model.fit()
        ar_forecast = ar_fit.forecast(steps=steps)

        # 逆分数差分变换
        # 这里简化处理，实际应用中需要更精确的逆变换
        # 对于预测目的，我们近似处理
        forecast_values = ar_forecast + mu

        return forecast_values

    def train_and_predict(self):
        """
        训练ARFIMA模型并生成预测
        """
        # 存储所有预测结果
        all_forecasts = {}

        for vol_type in self.volatility_types:
            for freq in self.frequencies:
                print(f"\n训练和预测: {vol_type} ({freq})")

                try:
                    # 加载训练数据
                    train_series = self.get_volatility_series(vol_type, freq, 'train')

                    # 拟合ARFIMA模型
                    d_hat, alpha1_hat, mu_hat = self.fit_arfima(train_series)

                    # 保存模型参数
                    self.model_params[(vol_type, freq)] = (d_hat, alpha1_hat, mu_hat)

                    # 加载测试数据
                    test_series = self.get_volatility_series(vol_type, freq, 'test')

                    # 生成单步预测
                    forecasts = []
                    history = train_series.copy()

                    # 使用滚动窗口预测
                    for i in range(len(test_series)):
                        # 使用历史数据预测下一步
                        forecast_step = self.arfima_forecast(
                            history,
                            (d_hat, alpha1_hat, mu_hat),
                            steps=1
                        )
                        forecasts.append(forecast_step.iloc[0])#[0])

                        # 添加真实值到历史数据
                        history = history.append(pd.Series([test_series.iloc[i]], index=[test_series.index[i]]))

                    # 保存预测结果
                    forecast_series = pd.Series(forecasts, index=test_series.index, name=f'ARFIMA_{vol_type}_{freq}')
                    all_forecasts[(vol_type, freq)] = forecast_series

                    # 生成并保存累积预测
                    self.generate_cumulative_forecasts(vol_type, freq, test_series)

                except Exception as e:
                    print(f"处理 {vol_type} ({freq}) 时出错: {str(e)}")

        # 保存模型参数
        self.save_model_parameters()

        # 保存预测结果
        self.save_forecasts(all_forecasts)

        print("\nARFIMA模型训练和预测完成!")

    def generate_cumulative_forecasts(self, vol_type, freq, test_series):
        """
        生成累积预测

        参数:
        vol_type (str): 波动率类型
        freq (str): 数据频率
        test_series (pd.Series): 测试数据序列
        """
        # 加载子周期数据
        subperiod_file = os.path.join(
            self.paths['subperiods'],
            f"CL_WTI_{vol_type}_daily_{freq}_cumulative.parquet"
        )

        if os.path.exists(subperiod_file):
            subperiod_df = pd.read_parquet(subperiod_file)

            # 获取模型参数
            params = self.model_params.get((vol_type, freq))
            if params is None:
                print(f"找不到 {vol_type} ({freq}) 的模型参数")
                return

            # 加载完整训练数据
            train_series = self.get_volatility_series(vol_type, freq, 'train')

            # 为每个子周期生成预测
            cumulative_forecasts = {}

            for h in [5, 10]:  # 5步和10步累积预测
                cum_forecasts = []

                for idx, row in subperiod_df.iterrows():
                    start_date = idx  # 子周期开始日期

                    # 获取到开始日期的所有历史数据
                    history = train_series.append(test_series[:start_date])

                    # 预测未来h步
                    forecast_steps = self.arfima_forecast(history, params, steps=h)

                    # 计算累积波动率: v_{t+h} = sqrt((1/h) * sum_{j=1}^h v_{t+j}^2)
                    cum_vol = np.sqrt(np.mean(forecast_steps ** 2))
                    cum_forecasts.append(cum_vol)

                # 保存累积预测
                col_name = f"{vol_type}_cumulative_h{h}"
                cumulative_forecasts[col_name] = cum_forecasts

            # 创建DataFrame并保存
            cum_df = pd.DataFrame(cumulative_forecasts, index=subperiod_df.index)
            output_file = os.path.join(
                self.paths['output_forecasts'],
                f"ARFIMA_{vol_type}_{freq}_cumulative.parquet"
            )
            cum_df.to_parquet(output_file)
            print(f"已保存累积预测: {output_file}")

    def save_model_parameters(self):
        """保存模型参数到文件"""
        params_list = []

        for (vol_type, freq), params in self.model_params.items():
            d, alpha1, mu = params
            params_list.append({
                'volatility_type': vol_type,
                'frequency': freq,
                'd_hat': d,
                'alpha1_hat': alpha1,
                'mu_hat': mu
            })

        params_df = pd.DataFrame(params_list)
        output_file = os.path.join(self.paths['output_params'], "ARFIMA_model_parameters.csv")
        params_df.to_csv(output_file, index=False)
        print(f"\n已保存模型参数: {output_file}")

    def save_forecasts(self, all_forecasts):
        """保存预测结果到文件"""
        for (vol_type, freq), forecast_series in all_forecasts.items():
            # 创建DataFrame
            forecast_df = pd.DataFrame({
                'DateTime': forecast_series.index,
                'forecast': forecast_series.values
            })
            forecast_df.set_index('DateTime', inplace=True)

            # 保存文件
            output_file = os.path.join(
                self.paths['output_forecasts'],
                f"ARFIMA_{vol_type}_{freq}_forecast.parquet"
            )
            forecast_df.to_parquet(output_file)
            print(f"已保存预测结果: {output_file}")


def main(root_dir):
    """
    主函数：执行ARFIMA模型训练和预测

    参数:
    root_dir (str): 项目根目录路径
    """
    print("=" * 70)
    print("开始ARFIMA模型训练和预测")
    print(f"项目根目录: {root_dir}")
    print("=" * 70)

    # 初始化并执行
    arfima_model = ARFIMA_Model(root_dir)
    arfima_model.train_and_predict()


if __name__ == "__main__":
    # 从命令行参数获取根目录或使用默认值
    if len(sys.argv) > 1:
        root_dir = sys.argv[1]
    else:
        root_dir = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"

    main(root_dir)