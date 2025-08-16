# -*- coding: utf-8 -*-
import pandas as pd

def build_plain_design(df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：合并后的窄表（含 y_true, y_pred）
    输出：用于回归的 DataFrame：y, x，并保留分组键
    """
    keep = df.dropna(subset=["y_true","y_pred"]).copy()
    keep = keep[["asset","RM","model","h","date","y_true","y_pred"]]
    keep = keep.rename(columns={"y_true":"y", "y_pred":"x"})
    return keep
