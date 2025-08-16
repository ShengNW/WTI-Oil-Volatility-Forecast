# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import fnmatch

def to_variance(s: pd.Series, unit: str) -> pd.Series:
    u = (unit or "").upper()
    if u == "VOL":  # σ -> σ^2
        return s.astype("float64") ** 2
    if u == "VAR":
        return s.astype("float64")
    raise ValueError(f"Unknown unit: {unit}")

def _try_parse_datetime_col(df: pd.DataFrame, col: str):
    try:
        out = pd.to_datetime(df[col], errors="raise")
        return out
    except Exception:
        return None

def ensure_datetime(df: pd.DataFrame, date_cols, logger=None, origin_hint: str="") -> pd.DataFrame:
    # 1) 直接命中（忽略大小写）
    cols_lower = {c.lower(): c for c in df.columns}
    for want in date_cols:
        if want.lower() in cols_lower:
            c = cols_lower[want.lower()]
            if logger: logger.info(f"[ensure_datetime] hit '{c}' in {origin_hint}")
            df = df.rename(columns={c: "date"})
            df["date"] = pd.to_datetime(df["date"])
            return df

    # 2) 索引是 DatetimeIndex（不论有没有名字）
    if isinstance(df.index, pd.DatetimeIndex):
        if logger: logger.info(f"[ensure_datetime] using DatetimeIndex as 'date' in {origin_hint}")
        df = df.reset_index()
        # 若 reset 后已有 'date' 列就直接转；否则把第 1 列强制改名为 'date'
        if "date" not in df.columns:
            # 优先用索引原名；否则就用第 1 列（通常就是刚 reset 出来的索引列）
            idx_name = df.columns[0]
            df = df.rename(columns={idx_name: "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df

    # 3) 关键词猜测
    for c in df.columns:
        if any(k in c.lower() for k in ["date","time","timestamp","dt","t"]):
            parsed = _try_parse_datetime_col(df, c)
            if parsed is not None:
                if logger: logger.warning(f"[ensure_datetime] heuristic parsed '{c}' as 'date' in {origin_hint}")
                df = df.rename(columns={c: "date"})
                df["date"] = parsed
                return df

    # 4) 单列兜底
    if df.shape[1] == 1:
        c = df.columns[0]
        parsed = _try_parse_datetime_col(df, c)
        if parsed is not None:
            if logger: logger.warning(f"[ensure_datetime] single-column parsed '{c}' as 'date' in {origin_hint}")
            df = df.rename(columns={c: "date"})
            df["date"] = parsed
            return df

    if logger:
        try_preview = df.head(3).to_dict(orient="list")
        logger.error(f"[ensure_datetime] FAIL in {origin_hint}; columns={list(df.columns)} preview={try_preview}")
    raise KeyError(f"No date column found among {date_cols}")


def pick_first_existing(df: pd.DataFrame, candidates) -> str:
    """
    在 DataFrame 列中按顺序寻找第一个存在的列名。
    支持通配符：如 'BV_*_h1'、'*_ann_1step'
    """
    cols = list(df.columns)
    for pat in candidates:
        # 精确命中
        if pat in df.columns:
            return pat
        # 通配符命中
        if "*" in pat or "?" in pat:
            for c in cols:
                if fnmatch.fnmatch(c, pat):
                    return c
    raise KeyError(f"None of candidate columns present. candidates={candidates}, df_cols={cols}")

def from_cumulative_to_h1(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    若文件是“累积”的，但我们只要 h=1：
      - 若本身有 *_h1 或 cumulative_h1 列，建议上游用 pick_first_existing 选中后传进来；
      - 这里提供兜底：如果 value_col 已经选中某列，则原样抽取 (date, value_col)。
    """
    return df[["date", value_col]].copy()
