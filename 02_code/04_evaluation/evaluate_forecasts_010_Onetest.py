from evaluate_forecasts_011_TwoRead import load_forecast
root   = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"
asset  = "CL_WTI"
freq   = "5min"
model  = "ANN"
h      = 5

for measure in ["RK","CBV"]:
    df = load_forecast(root, model, asset, measure, freq, h)
    print(measure, "→", "None" if df is None else (df.columns.tolist(), getattr(df, "_source", "n/a")))
    if df is not None:
        print("head3 =", df.iloc[:3,1].tolist())
