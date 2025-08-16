# -*- coding: utf-8 -*-
import os, pandas as pd
from .schema import COMMON_DATE_COLS, COMMON_PRED_VALUE_COLS
from .unit_ops import ensure_datetime, pick_first_existing, to_variance, from_cumulative_to_h1

def _value_candidates_for(rm: str, model: str, cumulative: bool):
    """
    根据 rm（如 'BV','RK'）与是否 cumulative，动态拼出更高优先级的候选列名。
    """
    rm = rm.upper()
    cands = []
    # ANN 1-step 常用命名：{RM}_ann_1step
    cands += [f"{rm}_ann_1step", f"{rm}_ANN_1step"]
    # HAR/ARFIMA 1-step 常用：forecast
    cands += ["forecast","value","pred","prediction","yhat"]

    if cumulative:
        # 你给到的多种风格：cumulative_h* / {RM}_cumVar_h* / {RM}_cumVolRMS_h* / {RM}_cumulative_h*
        cands = [
            "cumulative_h1", f"{rm}_cumulative_h1", f"{rm}_cumVar_h1", f"{rm}_cumVolRMS_h1",
            # 也顺带把 h5/h10 放后面，虽然后续图3不用，但留着不伤身
            "cumulative_h5","cumulative_h10",
            f"{rm}_cumulative_h5", f"{rm}_cumulative_h10",
            f"{rm}_cumVar_h5", f"{rm}_cumVar_h10",
            f"{rm}_cumVolRMS_h5", f"{rm}_cumVolRMS_h10",
        ] + cands

    # 最后补上全局通配白名单（带通配符）
    cands += COMMON_PRED_VALUE_COLS
    # 去重但保持顺序
    seen, uniq = set(), []
    for x in cands:
        if x not in seen:
            seen.add(x); uniq.append(x)
    return uniq

def load_preds_h1(forecast_sources: list, asset: str, freq: str, logger=None) -> pd.DataFrame:
    outs = []
    for src in forecast_sources:
        model = src["model"]
        base_dir = src["base_dir"]
        for e in src["entries"]:
            rm = e["rm"]
            path = os.path.join(base_dir, e["pattern"])
            if not os.path.exists(path):
                if logger: logger.warning(f"[PRED] missing file (skip): {path}")
                continue
            df = pd.read_parquet(path)
            if logger: logger.info(f"[PRED] reading {path}; cols={list(df.columns)} index={type(df.index)} rows={len(df)}")
            df = ensure_datetime(df, COMMON_DATE_COLS, logger=logger, origin_hint=f"PRED:{path}")

            cands = _value_candidates_for(rm, model, e.get("cumulative", False))
            val_col = pick_first_existing(df, cands)
            if e.get("cumulative", False):
                df = from_cumulative_to_h1(df, val_col)

            df["y_pred"] = to_variance(df[val_col], e["unit"])
            if logger: logger.info(f"[PRED] {path} model={model} rm={rm} unit={e['unit']} col='{val_col}' -> unified VAR")
            sub = df[["date", "y_pred"]].copy()
            sub["asset"] = asset
            sub["RM"] = rm
            sub["model"] = model
            sub["h"] = 1
            outs.append(sub)
    if not outs:
        raise RuntimeError("No predictions loaded—check your YAML patterns/paths.")
    return pd.concat(outs, ignore_index=True)
