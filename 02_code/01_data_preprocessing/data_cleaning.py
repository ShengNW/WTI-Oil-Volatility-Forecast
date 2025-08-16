import pandas as pd
import numpy as np
import os
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, '01_data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, '01_data', 'processed', 'intraday_returns')
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 定义核心交易时段 (9:00-14:30 EST)
EST_TRADING_HOURS = ('09:00', '14:30')

# 定义特殊清淡日期 (即使交易所开市也剔除)
SPECIAL_LIGHT_DAYS = [
    '12-24',  # 平安夜
    '12-26',  # 圣诞节次日
    '12-31',  # 新年前夜
    '01-02',  # 元旦次日
]


def load_and_preprocess_data(freq):
    """加载并预处理1分钟或5分钟数据"""
    # 1. 加载原始数据
    file_path = os.path.join(RAW_DATA_DIR, f'CL_WTI_{freq}min.csv')
    print(f"正在加载{freq}分钟数据: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['DateTime'], index_col='DateTime')

    # 2. 转换时区: UTC → EST (美国东部时间)
    #print("转换时区: UTC → EST...")
    #df = df.tz_localize('UTC').tz_convert('US/Eastern')# 正确代码：直接将原始数据识别为US/Eastern时区
    df = df.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')


    # 3. 过滤非交易时段 (9:00-14:30 EST)
    print(f"过滤交易时段 ({EST_TRADING_HOURS[0]}-{EST_TRADING_HOURS[1]} EST)...")
    time_mask = (df.index.time >= pd.to_datetime(EST_TRADING_HOURS[0]).time()) & \
                (df.index.time <= pd.to_datetime(EST_TRADING_HOURS[1]).time())
    df = df[time_mask]

    # 4. 创建日期列用于后续过滤
    df['TradeDate'] = df.index.date

    # 5. 剔除周末
    print("剔除周末数据...")
    df = df[df.index.dayofweek < 5]  # 0-4为周一到周五

    # 6. 创建美国交易日历 (剔除联邦假日)
    print("创建交易日历并剔除节假日...")
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df.index.min(), end=df.index.max())
    valid_dates = pd.Series(df['TradeDate'].unique())
    valid_dates = valid_dates[~valid_dates.isin(holidays.date)]

    # 7. 剔除特殊清淡日期
    print("剔除特殊清淡日期...")
    special_dates = valid_dates[valid_dates.astype(str).str[5:].isin(SPECIAL_LIGHT_DAYS)]
    valid_dates = valid_dates[~valid_dates.isin(special_dates)]

    # 8. 应用日期过滤
    df = df[df['TradeDate'].isin(valid_dates)]

    # 9. 计算对数收益率
    print("计算对数收益率...")
    df['LogReturn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))

    # 10. 处理缺失值
    print("处理缺失值...")
    # 10.1 标记数据稀疏的日期
    date_counts = df.groupby('TradeDate').size()
    sparse_dates = date_counts[date_counts < (0.8 * date_counts.median())].index
    df = df[~df['TradeDate'].isin(sparse_dates)]

    # 10.2 插值处理少量缺失值
    df['LogReturn'] = df['LogReturn'].interpolate()

    # 11. 清理并返回结果
    df.drop(columns=['TradeDate'], inplace=True)
    df.dropna(subset=['LogReturn'], inplace=True)

    return df


def main():
    # 处理1分钟数据
    df_1min = load_and_preprocess_data('1')
    output_path_1min = os.path.join(PROCESSED_DIR, 'CL_WTI_1min_processed.parquet')
    df_1min.to_parquet(output_path_1min)
    print(f"1分钟数据预处理完成! 保存至: {output_path_1min}")
    print(f"处理后数据量: {len(df_1min)}行")

    # 处理5分钟数据
    df_5min = load_and_preprocess_data('5')
    output_path_5min = os.path.join(PROCESSED_DIR, 'CL_WTI_5min_processed.parquet')
    df_5min.to_parquet(output_path_5min)
    print(f"5分钟数据预处理完成! 保存至: {output_path_5min}")
    print(f"处理后数据量: {len(df_5min)}行")

    # 生成处理报告
    report = {
        'original_1min_rows': 3518810,
        'processed_1min_rows': len(df_1min),
        'original_5min_rows': 709256,
        'processed_5min_rows': len(df_5min),
        'retention_rate_1min': f"{len(df_1min) / 3518810:.2%}",
        'retention_rate_5min': f"{len(df_5min) / 709256:.2%}",
        'output_path': PROCESSED_DIR
    }

    print("\n===== 数据处理报告 =====")
    for k, v in report.items():
        print(f"{k.replace('_', ' ').title()}: {v}")


if __name__ == "__main__":
    main()