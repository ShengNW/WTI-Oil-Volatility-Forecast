# -*- coding: utf-8 -*-
import os, pandas as pd
from .schema import COMMON_DATE_COLS, COMMON_RM_VALUE_COLS
from .unit_ops import ensure_datetime, pick_first_existing

def _rm_value_candidates(rm: str, freq: str):
    """
    根据 RM 名称和频率，返回更精准的候选列。
    目前主要为 RK 做定向：realized_kernel_estimates_test.parquet 有 RK_1min / RK_5min。
    """
    rm_u = (rm or "").upper()
    cands = []
    # RK: 文件里常见 'RK_1min', 'RK_5min'
    if rm_u == "RK":
        # 先给最可能的那个（按 freq）
        if str(freq).lower() in ("1min","1MIN","1Min"):
            cands += ["RK_1min"]
        elif str(freq).lower() in ("5min","5MIN","5Min"):
            cands += ["RK_5min"]
        # 兜底把另一个也加上
        cands += ["RK_1min","RK_5min"]

    # 常规 RM：优先精确名，然后落回通用候选
    cands += [rm_u, rm_u.capitalize(), rm_u.lower()]

    # 末尾追加全局通用候选（value 等）
    # 去重保持顺序
    seen, out = set(), []
    for x in cands + COMMON_RM_VALUE_COLS:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def load_rm_h1(base_dir: str, items: list, asset: str, freq: str, logger=None) -> pd.DataFrame:
    out = []
    for it in items:
        rm = it["rm"]
        path = os.path.join(base_dir, it["pattern"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"RM file missing: {path}")
        df = pd.read_parquet(path)
        if logger: logger.info(f"[RM] reading {path}; cols={list(df.columns)} index={type(df.index)} rows={len(df)}")
        df = ensure_datetime(df, COMMON_DATE_COLS, logger=logger, origin_hint=f"RM:{path}")

        # 根据 rm+freq 生成候选列
        cand_cols = _rm_value_candidates(rm, freq)
        val_col = pick_first_existing(df, cand_cols)
        if logger: logger.info(f"[RM] {path} -> rm={rm} freq={freq} value_col='{val_col}'")

        sub = df[["date", val_col]].copy()
        sub = sub.rename(columns={val_col: "y_true"})
        sub["asset"] = asset
        sub["RM"] = rm
        sub["h"] = 1
        out.append(sub)
    return pd.concat(out, ignore_index=True)
