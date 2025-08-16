# adapters/reader_h5.py
from __future__ import annotations
import os, re
from dataclasses import dataclass
from typing import Dict, List, Iterable
import pandas as pd

# ---- 小工具：单位统一 ----
def _to_variance(s: pd.Series, unit: str) -> pd.Series:
    unit = (unit or "").upper()
    if unit == "VOL":   # σ -> σ²
        return s ** 2
    if unit == "VAR":   # σ²
        return s
    raise ValueError(f"Unknown unit='{unit}', expect 'VOL' or 'VAR'.")

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    # 兼容 parquet 中的 DatetimeIndex 或 'DateTime' / 'Date' 列
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df["date"] = df.index
        return df.reset_index(drop=True)
    for c in ("DateTime", "date", "Date"):
        if c in df.columns:
            out = df.copy()
            out["date"] = pd.to_datetime(out[c])
            return out
    # 兜底：尝试把索引转成 datetime
    out = df.copy()
    try:
        out["date"] = pd.to_datetime(out.index)
        out = out.reset_index(drop=True)
        return out
    except Exception:
        raise ValueError("Cannot find/convert datetime index or column in dataframe.")

def _expand_pattern(p: str, freq: str) -> str:
    return p.replace("{FREQ}", freq)

# ---- 读取 Labels（累积标签，h=5）----
def read_labels_h5(cfg: Dict, freqs: Iterable[str]) -> pd.DataFrame:
    base = cfg["labels"]["base_dir"]
    items = cfg["labels"]["items"]
    frames = []
    for freq in freqs:
        for it in items:
            pattern = _expand_pattern(it["pattern"], freq)
            path = os.path.join(base, pattern)
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            df = _ensure_datetime_index(df)
            if it["col"] not in df.columns:
                raise KeyError(f"Label column '{it['col']}' not in {path}.")
            y = _to_variance(df[it["col"]], it["unit"])
            out = pd.DataFrame({
                "date": df["date"].values,
                "freq": freq,
                "rm": it["rm"],
                "y_true_var": y.values
            })
            frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["date","freq","rm","y_true_var"])
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.dropna().sort_values(["rm","freq","date"]).reset_index(drop=True)
    return out

# ---- 读取 Forecasts（各模型累积预测，h=5）----
def read_forecasts_h5(cfg: Dict, freqs: Iterable[str]) -> pd.DataFrame:
    frames = []
    for block in cfg["forecasts"]:
        model = block["model"]
        base_dir = block["base_dir"]
        for ent in block["entries"]:
            rm = ent["rm"]
            for freq in freqs:
                pattern = _expand_pattern(ent["pattern"], freq)
                path = os.path.join(base_dir, pattern)
                if not os.path.exists(path):
                    continue
                df = pd.read_parquet(path)
                df = _ensure_datetime_index(df)
                col = ent["col"]
                if col not in df.columns:
                    raise KeyError(f"Forecast column '{col}' not in {path}.")
                y = _to_variance(df[col], ent["unit"])
                out = pd.DataFrame({
                    "date": df["date"].values,
                    "freq": freq,
                    "rm": rm,
                    "model": model,
                    "y_pred_var": y.values
                })
                frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["date","freq","rm","model","y_pred_var"])
    out = pd.concat(frames, axis=0, ignore_index=True)
    out = out.dropna().sort_values(["rm","model","freq","date"]).reset_index(drop=True)
    return out

# ---- 烟测主入口（可在 main_mz_h5 里直接调用）----
def load_h5_tables(config: Dict) -> Dict[str, pd.DataFrame]:
    freqs = config.get("freqs", ["1min","5min"])
    y_true = read_labels_h5(config, freqs)
    y_pred = read_forecasts_h5(config, freqs)
    return {"y_true": y_true, "y_pred": y_pred}
