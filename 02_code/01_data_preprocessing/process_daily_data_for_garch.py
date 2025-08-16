import os
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path

# 配置路径
PROJECT_ROOT = Path(__file__).parents[2]  # 向上两级到项目根目录
RAW_DATA_DIR = PROJECT_ROOT / '01_data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / '01_data' / 'processed' / 'daily_returns'

# 定义特殊清淡日期 (与高频数据处理一致)
SPECIAL_LIGHT_DAYS = ['12-24', '12-26', '12-31', '01-02']


def create_trading_calendar(start_date, end_date):
    """创建有效的交易日历（剔除周末、假日和特殊日期）"""
    # 生成所有日期范围
    all_dates = pd.date_range(start_date, end_date, freq='D')

    # 移除周末
    trading_dates = all_dates[all_dates.dayofweek < 5]

    # 移除联邦假日
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date, end=end_date)
    trading_dates = trading_dates[~trading_dates.isin(holidays)]

    # 移除特殊清淡日
    month_day = trading_dates.strftime('%m-%d')
    trading_dates = trading_dates[~month_day.isin(SPECIAL_LIGHT_DAYS)]

    return trading_dates


def process_daily_data(project_root):
    """处理日线数据用于GARCH模型"""
    # 1. 配置路径
    raw_data_dir = project_root / '01_data' / 'raw'
    processed_dir = project_root / '01_data' / 'processed' / 'daily_returns'
    os.makedirs(processed_dir, exist_ok=True)

    # 2. 加载原始日线数据
    daily_path = raw_data_dir / 'CL_WTI_daily.csv'
    print(f"加载日线数据: {daily_path}")
    df = pd.read_csv(daily_path, parse_dates=['DateTime'])

    # 3. 设置高频数据日期范围 (2015-01-05 到 2024-12-30)
    start_date = pd.Timestamp('2015-01-05')
    end_date = pd.Timestamp('2024-12-30')
    df = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]

    # 4. 创建有效交易日历
    trading_dates = create_trading_calendar(start_date, end_date)

    # 5. 过滤有效交易日
    df = df[df['DateTime'].isin(trading_dates)]
    df.sort_values('DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 6. 计算日对数收益率
    df['LogReturn'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    df.dropna(subset=['LogReturn'], inplace=True)

    # 7. 按研究要求划分数据集
    train = df.iloc[:600].copy()
    test = df.iloc[600:600 + 1875].copy()

    # 8. 保存处理结果
    train.to_parquet(processed_dir / 'daily_train.parquet')
    test.to_parquet(processed_dir / 'daily_test.parquet')

    print(f"训练集: {len(train)}个观测值 ({train['DateTime'].min().date()} 至 {train['DateTime'].max().date()})")
    print(f"测试集: {len(test)}个观测值 ({test['DateTime'].min().date()} 至 {test['DateTime'].max().date()})")

    return train, test


if __name__ == "__main__":
    # 从环境变量获取项目路径或使用默认值
    project_path = 'F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication'#os.getenv('PROJECT_PATH', 'F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication')
    print(f"使用项目路径: {project_path}")
    process_daily_data(Path(project_path))