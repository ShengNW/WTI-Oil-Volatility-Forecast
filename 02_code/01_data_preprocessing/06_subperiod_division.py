import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def main():
    # 设置参数解析器
    # parser = argparse.ArgumentParser(description='划分波动率预测数据为子时期并生成验证箱线图')
    # parser.add_argument('base_dir', type=str, help='项目主目录路径')
    # args = parser.parse_args()
    #
    # # 设置路径
    # BASE_DIR = args.base_dir
    BASE_DIR = "F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication/"
    INPUT_DIR = os.path.join(BASE_DIR, '03_results', 'intermediate_results', 'volatility_estimates',
                             'cumulative_forecasts')
    OUTPUT_DIR = os.path.join(BASE_DIR, '03_results', 'intermediate_results', 'volatility_estimates', 'subperiods')
    FIGURE_DIR = os.path.join(BASE_DIR, '03_results', 'figures', 'subperiod_validation')

    # 创建输出目录
    os.makedirs(os.path.join(OUTPUT_DIR, '2018-2019'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '2020-2022'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '2023-2024'), exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # 定义子时期范围
    subperiods = {
        '2018-2019': ('2018-01-01', '2019-12-31'),
        '2020-2022': ('2020-01-01', '2022-12-31'),
        '2023-2024': ('2023-01-01', '2024-12-31')
    }

    # 获取所有累计预测文件
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')]
    print(f"找到 {len(all_files)} 个累计预测文件")

    # 存储用于验证的数据
    validation_data = {}

    # 处理每个文件
    for file in all_files:
        file_path = os.path.join(INPUT_DIR, file)
        df = pd.read_parquet(file_path)

        # 确保索引是DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 筛选2018年及以后的数据
        df = df[df.index >= '2018-01-01']

        # 将文件保存到各个子时期目录
        for period, (start, end) in subperiods.items():
            period_df = df[(df.index >= start) & (df.index <= end)]

            # 跳过空数据集
            if period_df.empty:
                print(f"警告: {file} 在 {period} 子时期没有数据")
                continue

            # 保存文件
            output_path = os.path.join(OUTPUT_DIR, period, file)
            period_df.to_parquet(output_path)

        # 收集5分钟频率的h1数据用于验证
        if '5min' in file and '_h1.' in file:
            col_name = df.columns[0]
            indicator = col_name.split('_')[0]  # 提取指标名称

            # 存储指标数据用于后续验证
            for period, (start, end) in subperiods.items():
                period_data = df[(df.index >= start) & (df.index <= end)][col_name]
                if not period_data.empty:
                    if indicator not in validation_data:
                        validation_data[indicator] = {}
                    validation_data[indicator][period] = period_data.dropna()

    print("子时期划分完成")

    # 生成验证箱线图
    if validation_data:
        print("生成验证箱线图...")

        # 设置图形风格
        sns.set(style="whitegrid", palette="pastel")
        plt.figure(figsize=(14, 10))

        # 为每个指标创建子图
        num_indicators = len(validation_data)
        fig, axes = plt.subplots(num_indicators, 1, figsize=(14, 6 * num_indicators))

        if num_indicators == 1:
            axes = [axes]

        for ax, (indicator, period_data) in zip(axes, validation_data.items()):
            # 准备箱线图数据
            plot_data = []
            labels = []
            for period, data in period_data.items():
                plot_data.append(data.values)
                labels.append(period)

            # 绘制箱线图
            ax.boxplot(plot_data, labels=labels, patch_artist=True)
            ax.set_title(f'{indicator} 波动率分布 - 子时期比较', fontsize=14)
            ax.set_ylabel('波动率值', fontsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.tick_params(axis='y', labelsize=10)

            # 添加统计信息注释
            for i, data in enumerate(plot_data):
                median = np.median(data)
                mean = np.mean(data)
                ax.text(i + 1, median * 1.05, f'中位数: {median:.6f}',
                        ha='center', va='bottom', fontsize=10)
                ax.text(i + 1, mean * 0.95, f'均值: {mean:.6f}',
                        ha='center', va='top', fontsize=10)

        plt.tight_layout()

        # 保存图形
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = os.path.join(FIGURE_DIR, f'subperiod_validation_{timestamp}.png')
        plt.savefig(fig_path, dpi=300)
        print(f"验证箱线图已保存至: {fig_path}")
    else:
        print("未找到合适的5分钟h1数据用于验证")

    print("子时期划分和验证完成")


if __name__ == "__main__":
    main()