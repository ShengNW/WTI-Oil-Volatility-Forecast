# -*- coding: utf-8 -*-
import pandas as pd

def merge_true_pred(rm_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    内连接：date, asset, RM, h
    """
    key = ["date","asset","RM","h"]
    m = pd.merge(rm_df, pred_df, on=key, how="inner", validate="many_to_many")
    # y_true 在 rm_df， y_pred 在 pred_df
    return m
