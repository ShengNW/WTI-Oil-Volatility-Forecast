import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import warnings

warnings.filterwarnings('ignore')


class HARModel:
    def __init__(self, root_dir):
        """
        初始化HAR模型路径配置

        参数:
        root_dir (str): 项目根目录路径
        """
        self.root_dir = root_dir
        self.paths = {
            'train': os.path.join(root_dir, "03_results/intermediate_results/volatility_estimates/train_set/"),
            'test': os.path.join(root_dir, "03_results/intermediate_results/volatility_estimates/test_set/"),
            'subperiods': os.path.join(root_dir, "03_results/intermediate_results/volatility_estimates/subperiods/"),
            'params': os.path.join(root_dir, "03_results/intermediate_results/model_parameters/HAR/"),
            'forecasts': os.path.join(root_dir, "03_results/final_forecasts/HAR/")
        }

        # 确保输出目录存在
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

        # 波动率类型与列名映射
        self.vol_columns = {
            'RV': 'RV',
            'TSRV': 'TSRV',
            #'RK': 'RK',  # 假设RK文件中列名为RK
            'RK_1min': 'RK_1min',  # 新增1min频率的RK映射
            'RK_5min': 'RK_5min',  # 新增5min频率的RK映射
            'BV': 'BV',
            'MedRV': 'MedRV',
            'JWTSRV': 'JWTSRV'
        }

    def load_data(self, vol_type, freq):
        """
        加载训练集和测试集数据

        参数:
        vol_type (str): 波动率类型 (RV, TSRV, RK, BV, MedRV, JWTSRV)
        freq (str): 数据频率 (1min 或 5min)

        返回:
        tuple: (训练集DataFrame, 测试集DataFrame)
        """
        # 构建文件名
        train_file = f"CL_WTI_{vol_type}_daily_{freq}_train.parquet"
        test_file = f"CL_WTI_{vol_type}_daily_{freq}_test.parquet"

        # RK文件特殊处理
        if vol_type == 'RK':
            train_file = "realized_kernel_estimates_train.parquet"
            test_file = "realized_kernel_estimates_test.parquet"

        # 加载数据
        train_path = os.path.join(self.paths['train'], train_file)
        test_path = os.path.join(self.paths['test'], test_file)

        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        # if vol_type == 'RK':
        #     vol_type = f"RK_{freq}"
        # # 提取波动率列
        # vol_col = self.vol_columns[vol_type]
        processed_vol_type = f"RK_{freq}" if vol_type == 'RK' else vol_type
        vol_col = self.vol_columns[processed_vol_type]

        return train_df[[vol_col]], test_df[[vol_col]]

    def create_har_features(self, df):
        """
        创建HAR模型特征

        参数:
        df (DataFrame): 包含波动率数据的DataFrame

        返回:
        DataFrame: 包含特征和目标变量的DataFrame
        """
        df = df.copy()
        vol_col = df.columns[0]

        # 计算滞后项
        df['v_daily'] = df[vol_col].shift(1)  # v_t (滞后1天)
        df['v_weekly'] = df[vol_col].rolling(5).mean().shift(1)  # 过去5天平均
        df['v_monthly'] = df[vol_col].rolling(22).mean().shift(1)  # 过去22天平均

        # 目标变量 (下一日波动率)
        df['target'] = df[vol_col].shift(-1)

        # 删除缺失值 (前22行)
        return df.dropna()

    def train_model(self, train_features):
        """
        训练HAR模型

        参数:
        train_features (DataFrame): 包含特征和目标变量的训练数据

        返回:
        LinearRegression: 训练好的模型
        """
        model = LinearRegression()
        features = train_features[['v_daily', 'v_weekly', 'v_monthly']]
        target = train_features['target']
        model.fit(features, target)
        return model

    def save_model_params(self, model, vol_type, freq):
        """
        保存模型参数

        参数:
        model (LinearRegression): 训练好的模型
        vol_type (str): 波动率类型
        freq (str): 数据频率
        """
        params = {
            'intercept': model.intercept_,
            'beta_D': model.coef_[0],
            'beta_W': model.coef_[1],
            'beta_M': model.coef_[2]
        }
        params_df = pd.DataFrame([params])
        file_name = f"HAR_{vol_type}_{freq}_params.csv"
        params_df.to_csv(os.path.join(self.paths['params'], file_name), index=False)

    def generate_forecasts(self, model, test_df, vol_type,freq):
        """
        生成测试集预测结果

        参数:
        model (LinearRegression): 训练好的模型
        test_df (DataFrame): 测试集数据

        返回:
        tuple: (单步预测结果, 多步累积预测结果)
        """
        # 创建测试集特征
        test_features = self.create_har_features(test_df)

        # 单步预测
        X_test = test_features[['v_daily', 'v_weekly', 'v_monthly']]
        single_step = model.predict(X_test)
        single_step_df = pd.DataFrame(single_step,
                                      index=test_features.index,
                                      columns=['forecast'])

        # 多步累积预测 (h=5, 10)
        cumulative = {}
        for h in [5, 10]:
            # 递归预测未来h步
            predictions = []
            current_data = test_features.copy()

            for _ in range(h):
                # 预测下一步
                X_current = current_data[['v_daily', 'v_weekly', 'v_monthly']].iloc[[-1]]
                next_pred = model.predict(X_current)[0]
                predictions.append(next_pred)
                processed_vol_type = f"RK_{freq}" if vol_type == 'RK' else vol_type
                #vol_col = self.vol_columns[processed_vol_type]
                # 更新特征vol_type
                new_row = pd.DataFrame({
                    self.vol_columns[processed_vol_type]: [next_pred],
                    'v_daily': next_pred,
                    'v_weekly': current_data['v_weekly'].iloc[-4:].mean(),
                    'v_monthly': current_data['v_monthly'].iloc[-21:].mean()
                }, index=[current_data.index[-1] + pd.Timedelta(days=1)])

                current_data = pd.concat([current_data, new_row])

            # 计算累积波动率
            cum_vol = np.mean(predictions)
            cumulative[h] = cum_vol

        return single_step_df, cumulative

    def save_forecasts(self, forecasts, vol_type, freq):
        """
        保存预测结果

        参数:
        forecasts (tuple): (单步预测结果, 多步累积预测结果)
        vol_type (str): 波动率类型
        freq (str): 数据频率
        """
        single_step, cumulative = forecasts

        # 保存单步预测
        single_file = f"HAR_{vol_type}_{freq}_forecast.parquet"
        single_step.to_parquet(os.path.join(self.paths['forecasts'], single_file))

        # 保存多步累积预测
        for h, value in cumulative.items():
            # 创建累积预测DataFrame
            cum_df = pd.DataFrame([value],
                                  columns=[f'{vol_type}_cumulative_h{h}'],
                                  index=[single_step.index[0]])

            file_name = f"HAR_{vol_type}_{freq}_cumulative_h{h}.parquet"
            cum_df.to_parquet(os.path.join(self.paths['forecasts'], file_name))

    def process_all_combinations(self):
        """
        只处理 TSRV 的 1min 和 5min 组合
        """
        vol_types = ['TSRV']  # 只跑 TSRV
        freqs = ['1min', '5min']

        for vol_type in vol_types:
            for freq in freqs:
                print(f"处理 {vol_type} ({freq})...")

                try:
                    # 加载数据
                    train_df, test_df = self.load_data(vol_type, freq)

                    # 创建特征
                    train_features = self.create_har_features(train_df)

                    # 训练模型
                    model = self.train_model(train_features)

                    # 保存参数
                    self.save_model_params(model, vol_type, freq)

                    # 生成预测
                    forecasts = self.generate_forecasts(model, test_df, vol_type, freq)

                    # 保存预测结果
                    self.save_forecasts(forecasts, vol_type, freq)

                except Exception as e:
                    print(f"处理 {vol_type} ({freq}) 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 通过参数传入根目录路径
    import sys

    root_dir = sys.argv[1] if len(
        sys.argv) > 1 else "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"

    har_model = HARModel(root_dir)
    har_model.process_all_combinations()
    print("HAR模型复现完成！")