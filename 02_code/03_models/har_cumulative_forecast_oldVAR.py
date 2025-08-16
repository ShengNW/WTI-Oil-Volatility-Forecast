import pandas as pd
import numpy as np
import os
from har_model import HARModel  # 从原文件中导入HARModel类


class HARCumulativeForecaster:
    def __init__(self, root_dir):
        """
        初始化HAR累积预测器

        参数:
        root_dir (str): 项目根目录路径
        """
        self.har_model = HARModel(root_dir)
        self.root_dir = root_dir

    def load_model_params(self, vol_type, freq):
        """
        加载HAR模型参数

        参数:
        vol_type (str): 波动率类型
        freq (str): 数据频率

        返回:
        tuple: (截距, beta_D, beta_W, beta_M)
        """
        # 构建参数文件名
        file_name = f"HAR_{vol_type}_{freq}_params.csv"
        params_path = os.path.join(self.har_model.paths['params'], file_name)

        # 加载参数
        params_df = pd.read_csv(params_path)
        return (
            params_df['intercept'].iloc[0],
            params_df['beta_D'].iloc[0],
            params_df['beta_W'].iloc[0],
            params_df['beta_M'].iloc[0]
        )

    def generate_cumulative_forecasts(self, vol_type, freq):
        """
        为测试集生成完整的累积预测结果(h=5,10)

        参数:
        vol_type (str): 波动率类型
        freq (str): 数据频率
        """
        print(f"生成累积预测: {vol_type} ({freq})...")

        # 加载数据
        train_df, test_df = self.har_model.load_data(vol_type, freq)
        processed_vol_type = f"RK_{freq}" if vol_type == 'RK' else vol_type
        vol_col = self.har_model.vol_columns[processed_vol_type]

        # 合并训练集和测试集以创建完整的历史序列
        full_df = pd.concat([train_df, test_df])
        full_features = self.har_model.create_har_features(full_df)

        # 获取模型参数
        intercept, beta_D, beta_W, beta_M = self.load_model_params(vol_type, freq)

        # 分离训练集和测试集特征
        train_len = len(train_df) - 22  # 减去特征创建时删除的行数
        test_features = full_features.iloc[train_len:]

        # 初始化存储累积预测结果的DataFrame
        cumulative_5 = pd.DataFrame(index=test_features.index, columns=[f'cumulative_h5'])
        cumulative_10 = pd.DataFrame(index=test_features.index, columns=[f'cumulative_h10'])

        # 对测试集的每一天生成累积预测
        for i in range(len(test_features)):
            # 获取当前特征
            current_row = test_features.iloc[i]
            v_daily = current_row['v_daily']
            v_weekly = current_row['v_weekly']
            v_monthly = current_row['v_monthly']

            # 创建临时序列用于递归预测
            temp_vol = full_df.iloc[:train_len + i].copy()

            # 递归预测h=10步
            predictions = []
            for step in range(10):
                # 预测下一步
                next_vol = intercept + beta_D * v_daily + beta_W * v_weekly + beta_M * v_monthly
                predictions.append(next_vol)

                # 更新序列
                new_index = temp_vol.index[-1] + pd.Timedelta(days=1)
                temp_vol.loc[new_index] = [next_vol]

                # 更新特征
                v_daily = next_vol
                v_weekly = temp_vol.iloc[-5:].mean().values[0]  # 过去5天平均
                v_monthly = temp_vol.iloc[-22:].mean().values[0]  # 过去22天平均

            # 计算累积波动率
            current_date = test_features.index[i]
            cumulative_5.loc[current_date] = np.mean(predictions[:5])
            cumulative_10.loc[current_date] = np.mean(predictions)

        return cumulative_5, cumulative_10

    def save_cumulative_forecasts(self, cumulative_5, cumulative_10, vol_type, freq):
        """
        保存累积预测结果

        参数:
        cumulative_5 (DataFrame): h=5累积预测结果
        cumulative_10 (DataFrame): h=10累积预测结果
        vol_type (str): 波动率类型
        freq (str): 数据频率
        """
        # 保存h=5预测
        file_name_5 = f"HAR_{vol_type}_{freq}_cumulative_h5.parquet"
        path_5 = os.path.join(self.har_model.paths['forecasts'], file_name_5)
        cumulative_5.to_parquet(path_5)

        # 保存h=10预测
        file_name_10 = f"HAR_{vol_type}_{freq}_cumulative_h10.parquet"
        path_10 = os.path.join(self.har_model.paths['forecasts'], file_name_10)
        cumulative_10.to_parquet(path_10)

        print(f"保存的累积预测: {path_5} 和 {path_10}")

    def process_all_combinations(self):
        """
        处理所有波动率类型和频率组合
        """
        vol_types = ['RV', 'TSRV', 'RK', 'BV', 'MedRV', 'JWTSRV']
        freqs = ['1min', '5min']

        for vol_type in vol_types:
            for freq in freqs:
                try:
                    # 生成累积预测
                    cum5, cum10 = self.generate_cumulative_forecasts(vol_type, freq)

                    # 保存结果
                    self.save_cumulative_forecasts(cum5, cum10, vol_type, freq)

                except Exception as e:
                    print(f"处理 {vol_type} ({freq}) 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    import sys

    # 获取根目录路径
    root_dir = sys.argv[1] if len(
        sys.argv) > 1 else "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"

    # 初始化并运行
    forecaster = HARCumulativeForecaster(root_dir)
    forecaster.process_all_combinations()
    print("累积预测生成完成！")