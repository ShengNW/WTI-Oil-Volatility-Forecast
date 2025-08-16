# quick_check_rk_bv.py
import pandas as pd
import numpy as np

from evaluate_forecasts_011_TwoRead import load_forecast

root   = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
asset  = "CL_WTI"
freq   = "5min"

def show(df, tag):
    if df is None:
        print(f"{tag}: None")
        return
    src   = getattr(df, "_source", "n/a")
    ocol  = getattr(df, "_orig_col", "n/a")
    xform = getattr(df, "_xform", "n/a")   # 是否做了 sqrt 等
    cols  = df.columns.tolist()
    head3 = df.iloc[:3, 1].tolist() if df.shape[1] >= 2 else []
    print(f"{tag}: cols={cols}\n  _source={src}\n  _orig_col={ocol}\n  _xform={xform}\n  head3={head3}\n")

for measure in ["RK", "CBV"]:
    for model in ["ANN", "HAR", "ARFIMA"]:
        for h in [1,5,10]:
            df = load_forecast(root, model, asset, measure, freq, h)
            show(df, f"{measure}-{model}-h{h}")
