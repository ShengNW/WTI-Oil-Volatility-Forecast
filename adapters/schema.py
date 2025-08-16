# -*- coding: utf-8 -*-
STD_COLS = ["date", "asset", "RM", "model", "h", "y_true", "y_pred"]

# 常见日期/时间列名（大小写混合也能命中）
COMMON_DATE_COLS = [
    "date","datetime","datadate","time","timestamp","t","dt","date_time",
    "dateTime","Date","DateTime","DATE","Time","Timestamp","DateTimeIndex"
]

# 在 COMMON_RM_VALUE_COLS 里加 'garch_11'
COMMON_RM_VALUE_COLS = [
    "value","RV","TSRV","MedRV","BV","JWTSRV","RK","rk","garch_11"
]

# 在 COMMON_PRED_VALUE_COLS 里也加 'garch_11'
COMMON_PRED_VALUE_COLS = [
    "forecast","yhat","pred","prediction","value","garch_11",
    "RV_ann_1step","TSRV_ann_1step","MedRV_ann_1step","BV_ann_1step","JWTSRV_ann_1step","RK_ann_1step",
    "HAR_forecast","ARFIMA_forecast",
    "cumulative_h1","cumulative_h5","cumulative_h10",
    "*_cumVar_h1","*_cumVar_h5","*_cumVar_h10",
    "*_cumVolRMS_h1","*_cumVolRMS_h5","*_cumVolRMS_h10",
    "*_cumulative_h1","*_cumulative_h5","*_cumulative_h10"
]
