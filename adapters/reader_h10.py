# adapters/reader_h10.py
from __future__ import annotations
import os
from typing import Dict, Iterable
import pandas as pd

# ---- 单位统一：VOL(σ) -> VAR(σ²)；VAR(σ²) -> 原样 ----
def _to_variance(s: pd.Series, unit: str) -> pd.Series:
    unit = (unit or "").upper()
    if unit == "VOL":
        return s ** 2
    if unit == "VAR":
        return s
    raise ValueError(f"Unknown unit='{unit}', expect 'VOL' or 'VAR'.")

# ---- 统一时间列为 'date' ----
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy()
        out["date"] = df.index
        return out.reset_index(drop=True)
    for c in ("DateTime", "date", "Date"):
        if c in df.columns:
            out = df.copy()
            out["date"] = pd.to_datetime(out[c])
            return out
    # 兜底：尝试把索引转换为 datetime
    out = df.copy()
    try:
        out["date"] = pd.to_datetime(out.index)
        return out.reset_index(drop=True)
    except Exception:
        raise ValueError("Cannot find/convert datetime index or column in dataframe.")

def _expand_pattern(pattern: str, freq: str) -> str:
    return pattern.replace("{FREQ}", freq)

# ----------------- h=10：读取累积“标签” -----------------
def read_labels_h10(cfg: Dict, freqs: Iterable[str]) -> pd.DataFrame:
    base = cfg["labels"]["base_dir"]
    items = cfg["labels"]["items"]
    frames = []
    for freq in freqs:
        for it in items:
            path = os.path.join(base, _expand_pattern(it["pattern"], freq))
            if not os.path.exists(path):
                continue
            df = pd.read_parquet(path)
            df = _ensure_datetime_index(df)
            col = it["col"]
            if col not in df.columns:
                raise KeyError(f"Label column '{col}' not found in: {path}")
            y = _to_variance(df[col], it["unit"])
            frames.append(pd.DataFrame({
                "date": df["date"].values,
                "freq": freq,
                "rm": it["rm"],
                "y_true_var": y.values
            }))
    if not frames:
        return pd.DataFrame(columns=["date","freq","rm","y_true_var"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna().sort_values(["rm","freq","date"]).reset_index(drop=True)
    return out

# ----------------- h=10：读取各模型累积“预测” -----------------
def read_forecasts_h10(cfg: Dict, freqs: Iterable[str]) -> pd.DataFrame:
    blocks = cfg["forecasts"]
    frames = []
    for block in blocks:
        model = block["model"]
        base_dir = block["base_dir"]
        for ent in block["entries"]:
            rm = ent["rm"]
            for freq in freqs:
                path = os.path.join(base_dir, _expand_pattern(ent["pattern"], freq))
                if not os.path.exists(path):
                    continue
                df = pd.read_parquet(path)
                df = _ensure_datetime_index(df)
                col = ent["col"]
                if col not in df.columns:
                    raise KeyError(f"Forecast column '{col}' not found in: {path}")
                y = _to_variance(df[col], ent["unit"])
                frames.append(pd.DataFrame({
                    "date": df["date"].values,
                    "freq": freq,
                    "rm": rm,
                    "model": model,
                    "y_pred_var": y.values
                }))
    if not frames:
        return pd.DataFrame(columns=["date","freq","rm","model","y_pred_var"])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna().sort_values(["rm","model","freq","date"]).reset_index(drop=True)
    return out

# ----------------- 统一入口 -----------------
def load_h10_tables(config: Dict) -> Dict[str, pd.DataFrame]:
    freqs = config.get("freqs", ["1min","5min"])
    y_true = read_labels_h10(config, freqs)
    y_pred = read_forecasts_h10(config, freqs)
    return {"y_true": y_true, "y_pred": y_pred}
