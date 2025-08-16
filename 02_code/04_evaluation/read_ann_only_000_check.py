# ann_label_unit_check.py
import os, pandas as pd, numpy as np

ROOT = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"

ASSET   = "CL_WTI"
MEASURE = "JWTSRV"   # 改：TSRV / RV / ...
FREQ    = "1min"     # 改：1min / 5min
H       = 10         # 改：1 / 5 / 10

# --- 路径（按你调研的保存规则） ---
FP_ANN_H1  = os.path.join(ROOT, "03_results", "final_forecasts", "ANN",
                          f"{ASSET}_{MEASURE}_daily_{FREQ}_ANN_1step_test.parquet")
FP_ANN_H5  = os.path.join(ROOT, "03_results", "intermediate_results", "cumulative_forecasts", "ANN",
                          f"{ASSET}_{MEASURE}_daily_{FREQ}_ANN_cumulative_h5_test.parquet")
FP_ANN_H10 = os.path.join(ROOT, "03_results", "intermediate_results", "cumulative_forecasts", "ANN",
                          f"{ASSET}_{MEASURE}_daily_{FREQ}_ANN_cumulative_h10_test.parquet")

# label（平均方差）的路径（你贴出的命名）
def label_path(h: int):
    if h == 1:
        # 你展示的 h=1 文件没写 _cumulative_h1 后缀，只有列名 JWTSRV
        return os.path.join(ROOT, "03_results", "intermediate_results", "volatility_estimates", "cumulative_forecasts",
                            f"{ASSET}_{MEASURE}_daily_{FREQ}_test_test_cumulative_h1.parquet")
    else:
        return os.path.join(ROOT, "03_results", "intermediate_results", "volatility_estimates", "cumulative_forecasts",
                            f"{ASSET}_{MEASURE}_daily_{FREQ}_test_test_cumulative_h{h}.parquet")

def parse(df):
    # 把 index/列统一成 date,value
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        valcol = [c for c in df.columns if c.lower() not in ("date","datetime","time","timestamp")]
        assert len(valcol)==1, f"ambiguous value columns: {valcol}"
        return df[["date", valcol[0]]].rename(columns={valcol[0]: "v"})
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
        valcol = [c for c in df.columns if c.lower()!="date"]
        assert len(valcol)==1, f"ambiguous value columns: {valcol}"
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", valcol[0]]].rename(columns={valcol[0]: "v"})
    # 兜底：找 float 列
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            out = df.copy()
            if "DateTime" in out.columns:
                out = out.rename(columns={"DateTime":"date"})
            elif "Unnamed: 0" in out.columns:
                out = out.rename(columns={"Unnamed: 0":"date"})
            out["date"] = pd.to_datetime(out.get("date", pd.to_datetime(out.index, errors="coerce")))
            return out[["date", c]].rename(columns={c:"v"})
    raise RuntimeError("cannot parse date/value")

def read_parquet(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_parquet(path)
    return parse(df)

def load_ann(h):
    if h==1: p = FP_ANN_H1
    elif h==5: p = FP_ANN_H5
    elif h==10: p = FP_ANN_H10
    else: raise ValueError(h)
    return read_parquet(p)

def main():
    ann = load_ann(H)
    lab = read_parquet(label_path(H))
    if ann is None:
        print(f"[ERR] ANN file not found for h={H}")
        return
    if lab is None:
        print(f"[ERR] LABEL file not found for h={H}")
        return

    df = pd.merge(ann, lab, on="date", how="inner", suffixes=("_ann","_lab")).dropna()
    if df.empty:
        print("[ERR] no overlap dates")
        return

    ratio = np.median(df["v_ann"].values / df["v_lab"].values)
    print(f"[INFO] rows={len(df)}, median(ANN / LABEL) = {ratio:.6g}")
    # 打印前 5 天对齐检查
    print(df.head().to_string(index=False))

    if H==1:
        # h=1：检查是否 σ^2 ≈ label
        # 读取 1步 σ（ANN_1step）的平方与 label 对比
        one = read_parquet(FP_ANN_H1)
        if one is not None:
            dd = pd.merge(one, lab, on="date", how="inner", suffixes=("_sigma","_lab")).dropna()
            dd["sigma2"] = dd["v_sigma"]**2
            r = np.median(dd["sigma2"]/dd["v_lab"])
            print(f"[H1 check] median( (σ_ann)^2 / LABEL ) = {r:.6g} (≈1 表示一致)")
    else:
        # 对多步：如果你手头有“路径 σ”的文件或能算 RMS-σ，可在此加进一步断言
        pass

if __name__ == "__main__":
    main()
