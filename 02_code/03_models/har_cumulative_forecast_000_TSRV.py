import pandas as pd
import numpy as np
import os
from har_model import HARModel  # 从原文件中导入HARModel类


class HARCumulativeForecaster:
    def __init__(self, root_dir):
        """
        初始化HAR累积预测器
        """
        self.har_model = HARModel(root_dir)
        self.root_dir = root_dir

    def load_model_params(self, vol_type, freq):
        """
        加载HAR模型参数
        """
        file_name = f"HAR_{vol_type}_{freq}_params.csv"
        params_path = os.path.join(self.har_model.paths['params'], file_name)

        params_df = pd.read_csv(params_path)
        return (
            float(params_df['intercept'].iloc[0]),
            float(params_df['beta_D'].iloc[0]),
            float(params_df['beta_W'].iloc[0]),
            float(params_df['beta_M'].iloc[0])
        )

    def generate_cumulative_forecasts(self, vol_type, freq):
        """
        为测试集生成完整的累积预测结果(h=5,10)
        ——保留原来的“递归+均值”算法，只修正数据切分与特征对齐
        """
        print(f"生成累积预测: {vol_type} ({freq})...")

        # 1) 加载训练/测试数据（只取该波动率的单列）
        train_df, test_df = self.har_model.load_data(vol_type, freq)
        processed_vol_type = f"RK_{freq}" if vol_type == 'RK' else vol_type
        vol_col = self.har_model.vol_columns[processed_vol_type]

        # 2) 合并为完整历史序列，并创建特征（用完整历史确保周/月均值正确）
        full_df = pd.concat([train_df, test_df]).sort_index()
        full_features = self.har_model.create_har_features(full_df)

        # 3) 计算训练特征长度，用它来“按特征长度”切分测试段，避免只得到一步
        train_features = self.har_model.create_har_features(train_df)
        train_feat_len = len(train_features)
        test_features = full_features.iloc[train_feat_len:]  # 对齐到测试段的特征索引

        # 4) 读取参数（不用 sklearn 预测器，与你给的逻辑一致）
        intercept, beta_D, beta_W, beta_M = self.load_model_params(vol_type, freq)

        # 5) 为测试期每一天生成 h=5、h=10 的递归均值（逐日序列）
        cumulative_5 = pd.DataFrame(index=test_features.index, columns=['cumulative_h5'], dtype=float)
        cumulative_10 = pd.DataFrame(index=test_features.index, columns=['cumulative_h10'], dtype=float)

        for current_date in test_features.index:
            # 起点的滞后特征，用 create_har_features 已经算好，直接取
            v_daily = float(test_features.at[current_date, 'v_daily'])
            v_weekly = float(test_features.at[current_date, 'v_weekly'])
            v_monthly = float(test_features.at[current_date, 'v_monthly'])

            # 从“当前日”为边界切出历史序列（含当前日）用于后续递归更新均值
            temp_series = full_df.loc[:current_date, vol_col].copy()

            preds = []
            for _ in range(10):
                # 下一步预测：线性模型 y = b0 + bD*v_d + bW*v_w + bM*v_m
                next_vol = intercept + beta_D * v_daily + beta_W * v_weekly + beta_M * v_monthly
                preds.append(float(next_vol))

                # 把预测值接到序列末尾，时间索引 +1 天
                new_index = temp_series.index[-1] + pd.Timedelta(days=1)
                # 只更新单列，避免列名错乱
                temp_series.loc[new_index] = float(next_vol)

                # 用最新序列更新滞后特征（周=近5日均，月=近22日均；下一步的 v_daily 就是 next_vol）
                v_daily = float(next_vol)
                v_weekly = float(temp_series.tail(5).mean())
                v_monthly = float(temp_series.tail(22).mean())

            # 写入当前测试日的 h=5 与 h=10 均值
            cumulative_5.at[current_date, 'cumulative_h5'] = float(np.mean(preds[:5]))
            cumulative_10.at[current_date, 'cumulative_h10'] = float(np.mean(preds))

        return cumulative_5, cumulative_10

    def save_cumulative_forecasts(self, cumulative_5, cumulative_10, vol_type, freq):
        """
        保存累积预测结果（路径与命名保持不变）
        """
        file_name_5 = f"HAR_{vol_type}_{freq}_cumulative_h5.parquet"
        path_5 = os.path.join(self.har_model.paths['forecasts'], file_name_5)
        cumulative_5.to_parquet(path_5)

        file_name_10 = f"HAR_{vol_type}_{freq}_cumulative_h10.parquet"
        path_10 = os.path.join(self.har_model.paths['forecasts'], file_name_10)
        cumulative_10.to_parquet(path_10)

        print(f"保存的累积预测: {path_5} 和 {path_10}")

    def process_all_combinations(self):
        """
        只处理 TSRV（1min, 5min），其余类型跳过；不改任何路径/文件名规则
        """
        vol_types = ['TSRV']        # 只跑 TSRV
        freqs = ['1min', '5min']    # 两个频率

        for vol_type in vol_types:
            for freq in freqs:
                try:
                    cum5, cum10 = self.generate_cumulative_forecasts(vol_type, freq)
                    self.save_cumulative_forecasts(cum5, cum10, vol_type, freq)
                except Exception as e:
                    print(f"处理 {vol_type} ({freq}) 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    forecaster = HARCumulativeForecaster(root_dir)
    forecaster.process_all_combinations()
    print("累积预测生成完成！")
