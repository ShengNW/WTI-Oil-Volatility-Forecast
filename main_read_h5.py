# -*- coding: utf-8 -*-
import os
import sys
import json
import yaml
import pandas as pd
from datetime import datetime

def _mk(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

def _log_write(path, text, mode="a", encoding="utf-8"):
    with open(path, mode, encoding=encoding) as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))

def _read_parquet_strict(path, date_col, value_col, rename_to, log_file=None):
    df = pd.read_parquet(path)
    cols = list(df.columns)

    # 严格检查值列
    if value_col not in cols:
        raise KeyError(f"[STRICT] 列不存在: {value_col} in {path}. 实际列: {cols}")

    # 情况 A：日期就在列里（严格匹配）
    if date_col in cols:
        out = df[[date_col, value_col]].copy()
        out.rename(columns={date_col: "date", value_col: rename_to}, inplace=True)

    else:
        # 情况 B：日期在索引里（可能是无名 DatetimeIndex）
        if isinstance(df.index, pd.DatetimeIndex):
            # 记录一次诊断
            if log_file is not None:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"[INFO] {path} 无列 {date_col}，使用 DatetimeIndex 作为日期。index.name={df.index.name}\n")
            tmp = df.reset_index()           # 索引变成第一列（名为 index 或 0）
            idx_col = tmp.columns[0]
            out = tmp[[idx_col, value_col]].copy()
            out.rename(columns={idx_col: "date", value_col: rename_to}, inplace=True)

        else:
            # 情况 C：既没有目标日期列，也不是 DatetimeIndex —— 严格报错
            raise KeyError(
                f"[STRICT] 日期列不存在: {date_col} in {path}. "
                f"实际列: {cols}, 索引类型: {type(df.index)}, index.name={df.index.name}"
            )

    # 规范日期类型
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"])

    return out.sort_values("date").reset_index(drop=True)


def _append_diag_cols_log(log_file, path, df, expect_col):
    _log_write(log_file, f"文件: {path}")
    _log_write(log_file, f"包含的列名有: {', '.join(df.columns.astype(str))}")
    if expect_col not in df.columns:
        _log_write(log_file, f"[ERROR] 找不到需要的列名: {expect_col}")
    _log_write(log_file, "-"*80)

def load_y_true(cfg, out_dir):
    date_col = cfg["primary_key"]["date_col"]
    value_name = cfg["primary_key"]["value_col_true"]
    freq_value = cfg["primary_key"]["freq_value"]

    logs_cols = os.path.join(out_dir, "logs", "01_columns.txt")
    _log_write(logs_cols, f"=== y_true 列检查 @ {datetime.now()} ===", mode="w")

    frames = []
    for it in cfg["y_true_sources"]:
        rm, path, col = it["rm"], it["path"], it["column"]
        df_raw = pd.read_parquet(path)
        _append_diag_cols_log(logs_cols, path, df_raw, col)
        # load_y_true(...) 中
        df = _read_parquet_strict(path, date_col, col, value_name, log_file=logs_cols)
        df["freq"] = it.get("freq", freq_value)  # ✅ 允许逐条覆盖
        df["rm"] = rm

        frames.append(df)
    y_true = pd.concat(frames, ignore_index=True)
    y_true = y_true[["date", "freq", "rm", value_name]]
    # 诊断
    logs_files = os.path.join(out_dir, "logs", "00_files.txt")
    _log_write(logs_files, f"[y_true] 共加载 {len(cfg['y_true_sources'])} 个文件，合并后 {len(y_true)} 行（去重前）")
    y_true = y_true.drop_duplicates(["date", "freq", "rm"]).sort_values(["rm","date"]).reset_index(drop=True)
    _log_write(logs_files, f"[y_true] 去重后 {len(y_true)} 行；日期范围: {y_true['date'].min()} → {y_true['date'].max()}")
    return y_true

def load_y_pred(cfg, out_dir):
    date_col = cfg["primary_key"]["date_col"]
    value_name = cfg["primary_key"]["value_col_pred"]
    freq_value = cfg["primary_key"]["freq_value"]

    logs_cols = os.path.join(out_dir, "logs", "01_columns.txt")
    _log_write(logs_cols, f"\n=== y_pred 列检查 @ {datetime.now()} ===")

    frames = []
    for it in cfg["y_pred_sources"]:
        model, rm, path, col = it["model"], it["rm"], it["path"], it["column"]
        df_raw = pd.read_parquet(path)
        _append_diag_cols_log(logs_cols, path, df_raw, col)
        # load_y_pred(...) 中
        df = _read_parquet_strict(path, date_col, col, value_name, log_file=logs_cols)
        df["freq"] = it.get("freq", freq_value)  # ✅ 允许逐条覆盖
        df["rm"] = rm
        df["model"] = model

        frames.append(df)
    y_pred = pd.concat(frames, ignore_index=True)
    y_pred = y_pred[["date", "freq", "rm", "model", value_name]]
    # 诊断
    logs_files = os.path.join(out_dir, "logs", "00_files.txt")
    _log_write(logs_files, f"[y_pred] 共加载 {len(cfg['y_pred_sources'])} 个文件，合并后 {len(y_pred)} 行（去重前）")
    y_pred = y_pred.drop_duplicates(["date", "freq", "rm", "model"]).sort_values(["rm","model","date"]).reset_index(drop=True)
    _log_write(logs_files, f"[y_pred] 去重后 {len(y_pred)} 行；模型数: {y_pred['model'].nunique()}；日期范围: {y_pred['date'].min()} → {y_pred['date'].max()}")
    return y_pred

def main():
    # 参数
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "./04_configs/mz_h5_config.yaml"
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "./05_outputs/tables"
    _mk(out_dir)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    y_true = load_y_true(cfg, out_dir)
    y_pred = load_y_pred(cfg, out_dir)

    # 落地 tmp CSV（便于你快速 grep/比对）
    p_true = os.path.join(out_dir, "tmp_h5_y_true.csv")
    p_pred = os.path.join(out_dir, "tmp_h5_y_pred.csv")
    y_true.to_csv(p_true, index=False)
    y_pred.to_csv(p_pred, index=False)

    print(f"[OK] y_true -> {p_true}, shape={y_true.shape}")
    print(f"[OK] y_pred -> {p_pred}, shape={y_pred.shape}")

if __name__ == "__main__":
    main()
