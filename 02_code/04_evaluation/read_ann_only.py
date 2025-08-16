#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional

import numpy as np
import pandas as pd

# 修改成你的仓库根目录
DEFAULT_ROOT = r"F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication"

# ANN 搜索目录（按需保留其一）
ANN_DIRS = [
    "03_results/final_forecasts/ANN",                              # 1-step 通常在这
    "03_results/intermediate_results/cumulative_forecasts/ANN",    # 累积 h>1 你之前放在这
]

STRICT = True  # 严格模式：文件不匹配就报错

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """把可能的日期列/索引正规化为 'date' 列。"""
    df = df.copy()
    direct = [c for c in df.columns if c.lower() in ("date", "datetime", "time", "timestamp")]
    if direct:
        col = direct[0]
        df[col] = pd.to_datetime(df[col], errors="coerce")
        return df.rename(columns={col: "date"}).sort_values("date").reset_index(drop=True)
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.reset_index().rename(columns={df.index.name or "index": "date"})
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values("date").reset_index(drop=True)
    try:
        idx_parsed = pd.to_datetime(df.index, errors="coerce")
        if getattr(idx_parsed, "notna")().mean() > 0.8:
            out = df.copy()
            out.insert(0, "date", idx_parsed)
            return out.sort_values("date").reset_index(drop=True)
    except Exception:
        pass
    if "Unnamed: 0" in df.columns:
        s = pd.to_datetime(df["Unnamed: 0"], errors="coerce")
        out = df.copy()
        out["date"] = s
        return out.sort_values("date").reset_index(drop=True)
    return df

def try_load_parquet(path: str) -> Optional[pd.DataFrame]:
    """存在就读 parquet；失败返回 None（不抛异常）。"""
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_parquet(path)
        return parse_dates(df)
    except Exception as e:
        print("[READ FAIL]", path, "->", e)
        return None

def pick_forecast_column(df: pd.DataFrame) -> str:
    """挑出预测列名。"""
    candidates = [
        "forecast", "yhat", "pred", "prediction", "value",
        "vol_forecast", "nu_hat", "nu_pred", "vhat", "fcast",
        "cumVar", "cumVolRMS",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # 你的常见命名
    for c in df.columns:
        if re.search(r"_ann_1step$", c, flags=re.I):
            return c
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    for c in float_cols:
        if c.lower() not in ("actual", "target", "truth", "y"):
            return c
    raise ValueError("Could not identify forecast column. cols=" + str(list(df.columns)))

def sqrt_clip(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(x, 0.0, None))

def _exact_ann_name(asset: str, measure: str, freq: str, h: int) -> str:
    if h == 1:
        return "{}_{}_daily_{}_ANN_1step_test.parquet".format(asset, measure, freq)
    else:
        return "{}_{}_daily_{}_ANN_cumulative_h{}_test.parquet".format(asset, measure, freq, h)

def _regex_ann_fallback(asset: str, measure: str, freq: str, h: int) -> str:
    if h == 1:
        return r".*{}.*{}.*{}.*ANN.*1step.*\.parquet$".format(asset, measure, freq)
    else:
        return r".*{}.*{}.*{}.*ANN.*cumulative.*h{}.*\.parquet$".format(asset, measure, freq, h)

def load_ann_forecast(
    root: str,
    asset: str,
    measure: str,
    freq: str,
    h: int,
    strict: bool = STRICT,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    只读取 ANN 的预测，统一返回 σ（标准差）：
      - h=1：文件内容应为 σ（不 √）
      - h>1：若列名含 cumVar/variance => 当方差开方；否则视为已是 σ。
    返回：DataFrame[['date','ANN']]
    """
    # ===== BP1：入口参数 =====
    # 查看 root, asset, measure, freq, h, strict, verbose

    # 搜索目录
    base_dirs = [os.path.join(root, d) for d in ANN_DIRS]
    _seen = set()
    base_dirs = [d for d in base_dirs if not (d in _seen or _seen.add(d))]
    # ===== BP2：确认搜索目录 =====
    # 想只搜 final_forecasts/ANN 就把另一个删掉

    exact_name = _exact_ann_name(asset, measure, freq, h)
    regex_fallback = _regex_ann_fallback(asset, measure, freq, h)

    # 候选路径
    candidates = []
    for d in base_dirs:
        candidates.append(os.path.join(d, exact_name))
    for d in base_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if re.match(regex_fallback, f, flags=re.I):
                candidates.append(os.path.join(d, f))
    _seen = set()
    candidates = [p for p in candidates if not (p in _seen or _seen.add(p))]
    # ===== BP3：看候选列表 =====

    hit_path = None
    hit_df = None
    for path in candidates:
        # ===== BP4：逐个尝试 =====
        df = try_load_parquet(path)
        if df is not None:
            hit_path = path
            hit_df = df
            break

    if hit_df is None:
        msg = "[ANN] Not found for h={}. tried:\n{}".format(h, "\n".join(candidates))
        if strict:
            raise FileNotFoundError(msg)
        else:
            if verbose:
                print(msg)
            return pd.DataFrame(columns=["date", "ANN"])

    # 文件类型一致性检查
    lower_name = os.path.basename(hit_path).lower()
    if h == 1 and "cumulative" in lower_name:
        raise ValueError("[ANN] Expected 1-step but hit cumulative: {}".format(hit_path))
    if h > 1 and "1step" in lower_name:
        raise ValueError("[ANN] Expected cumulative h={} but hit 1-step: {}".format(h, hit_path))

    # 预测列名
    col = pick_forecast_column(hit_df)

    # ===== BP5：单位判断 =====
    name = col.lower()
    if h > 1 and (("cumvar" in name) or re.search(r"(var|variance)", name)):
        treat = "variance"
    else:
        treat = "sigma"

    s = hit_df[col].to_numpy()
    if treat == "variance":
        s = sqrt_clip(s)

    out = parse_dates(hit_df)[["date"]].copy()
    out["ANN"] = s

    if verbose:
        med = float(np.nanmedian(out["ANN"])) if len(out) else float("nan")
        print("[ANN] OK h={} | path={} | col={} | unit={} | N={} | median={:.6g}".format(
            h, hit_path, col, treat, len(out), med
        ))

    # ===== BP6：返回前检查 =====
    return out

# ------------------ 最小示例 ------------------
if __name__ == "__main__":
    root   = DEFAULT_ROOT
    asset  = "CL_WTI"
    measure= "JWTSRV"  # 可改 RV/TSRV/MedRV
    freq   = "1min"    # 或 "5min"
    h      = 5         # 试 1、5、10

    df = load_ann_forecast(root, asset, measure, freq, h, strict=True, verbose=True)
    print(df.head())
