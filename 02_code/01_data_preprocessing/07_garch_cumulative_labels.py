import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


def calculate_garch_cumulative_labels(project_dir):
    """
    计算GARCH模型的日线累积波动率测试集标签

    参数:
    project_dir: 项目根目录路径

    返回:
    无，结果直接保存到文件
    """
    # 1. 设置文件路径
    daily_test_path = Path(project_dir) / "01_data/processed/daily_returns/daily_test.parquet"
    output_dir = Path(project_dir) / "03_results/intermediate_results/volatility_estimates/cumulative_forecasts"

    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取日度测试数据: {daily_test_path}")
    daily_df = pd.read_parquet(daily_test_path)

    # 2. 检查并统一日期索引格式
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        if "DateTime" in daily_df.columns:
            daily_df = daily_df.set_index("DateTime")
        elif "Date" in daily_df.columns:
            daily_df = daily_df.set_index("Date")
            daily_df.index.name = "DateTime"
        else:
            # 尝试将索引转换为日期时间
            try:
                daily_df.index = pd.to_datetime(daily_df.index)
                daily_df.index.name = "DateTime"
            except:
                raise ValueError("无法识别日期索引或日期列")

    # 确保索引是DateTimeIndex
    if not isinstance(daily_df.index, pd.DatetimeIndex):
        raise ValueError("日期索引格式不正确")

    print(f"数据日期范围: {daily_df.index.min()} 至 {daily_df.index.max()}")
    print(f"总数据点数: {len(daily_df)}")

    # 3. 计算波动率代理 - 使用日收益率的平方
    daily_df['squared_return'] = daily_df['LogReturn'] ** 2

    # 4. 计算累积波动率标签
    def calculate_cumulative_labels(series, window):
        """
        计算累积波动率标签

        参数:
        series: 波动率序列
        window: 累积窗口大小

        返回:
        累积波动率标签Series
        """
        # 使用未来窗口计算平均值（shift(-1)确保从下一天开始）
        shifted = series.shift(-1)
        cum_labels = shifted.rolling(window=window, min_periods=window).mean()

        # 移除无法形成完整窗口的行
        # cum_labels = cum_labels.iloc[:-(window - 1)]
        if window > 1:
            cum_labels = cum_labels.iloc[:-(window - 1)]
        return cum_labels

    # 新增：转换为波动率（方差的平方根）
    daily_df['volatility_proxy'] = np.sqrt(daily_df['squared_return'])  # 这一行是关键

    # 后续计算累积标签时，使用volatility_proxy而非squared_return
    h1_labels = calculate_cumulative_labels(daily_df['volatility_proxy'], 1)
    h5_labels = calculate_cumulative_labels(daily_df['volatility_proxy'], 5)
    h10_labels = calculate_cumulative_labels(daily_df['volatility_proxy'], 10)
    # 计算不同步长的累积标签
    # h1_labels = calculate_cumulative_labels(daily_df['squared_return'], 1)
    # h5_labels = calculate_cumulative_labels(daily_df['squared_return'], 5)
    # h10_labels = calculate_cumulative_labels(daily_df['squared_return'], 10)

    # 5. 创建并保存结果DataFrame
    def save_cumulative_labels(labels, horizon, output_dir):
        """保存累积波动率标签到Parquet文件"""
        # 创建结果DataFrame
        result_df = pd.DataFrame({
            f'cumulative_h{horizon}': labels
        })

        # 确保索引是DateTimeIndex
        result_df.index = labels.index
        result_df.index.name = "DateTime"

        # 设置输出路径
        output_path = output_dir / f"garch_daily_test_cumulative_h{horizon}.parquet"
        result_df.to_parquet(output_path)
        print(f"已保存 {horizon} 步累积波动率标签至: {output_path}")
        print(f"标签数量: {len(result_df)}")

    # 保存不同步长的结果
    save_cumulative_labels(h1_labels, 1, output_dir)
    save_cumulative_labels(h5_labels, 5, output_dir)
    save_cumulative_labels(h10_labels, 10, output_dir)

    print("所有GARCH累积波动率标签已生成!")


if __name__ == "__main__":
    # 检查是否提供了项目目录参数
    # if len(sys.argv) != 2:
    #     print("请提供项目目录作为参数")
    #     print(
    #         "示例: python 07_garch_cumulative_labels.py F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication")
    #     sys.exit(1)
    #
    # project_dir = sys.argv[1]
    project_dir = f"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
    print(f"项目目录: {project_dir}")
    calculate_garch_cumulative_labels(project_dir)
