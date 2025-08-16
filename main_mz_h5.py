# -*- coding: utf-8 -*-
import os
import sys
import yaml
import pandas as pd
from datetime import datetime
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import numpy as np
def _mk(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)

def _log(path, text, mode="a"):
    with open(path, mode, encoding="utf-8") as f:
        f.write(text + ("\n" if not text.endswith("\n") else ""))

def _merge_strict(y_true, y_pred, out_dir):
    # A. 基础检查
    need_true = {"date","freq","rm","y_true"}
    need_pred = {"date","freq","rm","model","y_pred"}
    if set(y_true.columns) ^ need_true:
        raise ValueError(f"y_true 列应为 {need_true} ，实际 {set(y_true.columns)}")
    if set(y_pred.columns) ^ need_pred:
        raise ValueError(f"y_pred 列应为 {need_pred} ，实际 {set(y_pred.columns)}")

    # B. 键域交集与反连接（强诊断：02_merge.txt）
    logs_merge = os.path.join(out_dir, "logs", "02_merge.txt")
    _log(logs_merge, f"=== Merge 诊断 @ {datetime.now()} ===", mode="w")

    key3 = ["date","freq","rm"]
    left_keys  = y_true[key3].drop_duplicates()
    right_keys = y_pred[key3].drop_duplicates()

    left_only  = left_keys.merge(right_keys, on=key3, how="left", indicator=True)
    left_only  = left_only[left_only["_merge"]=="left_only"].drop(columns=["_merge"])
    right_only = right_keys.merge(left_keys, on=key3, how="left", indicator=True)
    right_only = right_only[right_only["_merge"]=="left_only"].drop(columns=["_merge"])

    _log(logs_merge, f"[键域] y_true 三键唯一键数: {len(left_keys)}")
    _log(logs_merge, f"[键域] y_pred 三键唯一键数: {len(right_keys)}")
    _log(logs_merge, f"[键域] 仅在 y_true 的键数: {len(left_only)}")
    _log(logs_merge, f"[键域] 仅在 y_pred 的键数: {len(right_only)}")
    if len(left_only):
        _log(logs_merge, "[样例-仅在 y_true 的前10条]"); _log(logs_merge, left_only.head(10).to_string())
    if len(right_only):
        _log(logs_merge, "[样例-仅在 y_pred 的前10条]"); _log(logs_merge, right_only.head(10).to_string())

    # C. 严格内连接（注意：y_true 无 model，merge 后每条 true 会与对应 model 的多条预测对齐）
    merged = y_true.merge(y_pred, on=key3, how="inner")

    # D. 重复键诊断（四键）
    dup4 = merged.duplicated(subset=["date","freq","rm","model"], keep=False)
    logs_files = os.path.join(out_dir, "logs", "00_files.txt")
    _log(logs_files, f"[merge] 合并后 {len(merged)} 行；四键重复行数: {dup4.sum()}")

    return merged

def _mz_gls_r2_panel(df_panel, out_dir):
    """
    对每个 (rm, model) 做一条 MZ 回归： y_true / y_pred = alpha / y_pred + beta
    这里以 OLS 近似（数据量大时与GLS R²差异很小），重点在把空面板/零样本写日志。
    """
    logs_empty = os.path.join(out_dir, "logs", "03_empty_panels.txt")
    _log(logs_empty, f"=== 空面板/异常面板 @ {datetime.now()} ===", mode="w")

    res = []
    for (rm, model), g in df_panel.groupby(["rm","model"], dropna=False):
        g = g.dropna(subset=["y_true","y_pred"])
        if g.empty:
            _log(logs_empty, f"[EMPTY] (rm={rm}, model={model}) 无样本")
            continue
        if (g["y_pred"]==0).all():
            _log(logs_empty, f"[ZERO] (rm={rm}, model={model}) y_pred 全为 0，跳过")
            continue
        eps = 1e-12  # 也可设 1e-10，看你的数值规模

        # 过滤非有限或过小的 y_pred
        g = g[np.isfinite(g["y_pred"]) & np.isfinite(g["y_true"])]
        g = g[g["y_pred"] > eps]
        if g.empty:
            _log(logs_empty, f"[EMPTY-AFTER-FILTER] (rm={rm}, model={model}) 过滤后无样本（y_pred<=eps 或非有限）")
            continue

        # MZ 形式： y_true / y_pred = alpha*(1/y_pred) + beta
        y = g["y_true"] / g["y_pred"]
        X = pd.DataFrame({"inv_pred": 1.0 / g["y_pred"]})
        
        X = add_constant(X)
        try:
            r = OLS(y, X).fit()
            r2 = r.rsquared
            res.append({"rm": rm, "model": model, "R2": r2, "n": len(g)})
        except Exception as e:
            _log(logs_empty, f"[ERROR] (rm={rm}, model={model}) 回归失败: {e}")

    df_res = pd.DataFrame(res).sort_values(["rm","model"]).reset_index(drop=True)
    return df_res

def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "mz_h5_config.yaml"
    tables_dir = sys.argv[2] if len(sys.argv) > 2 else "./05_outputs/tables"
    out_dir    = sys.argv[3] if len(sys.argv) > 3 else "./05_outputs"
    _mk(out_dir)

    # 读取标准化落表（由 main_read_h5.py 生成）
    p_true = os.path.join(tables_dir, "tmp_h5_y_true.csv")
    p_pred = os.path.join(tables_dir, "tmp_h5_y_pred.csv")
    y_true = pd.read_csv(p_true, parse_dates=["date"])
    y_pred = pd.read_csv(p_pred, parse_dates=["date"])

    # 四类日志：00_files、01_columns（在 read 阶段已写）、02_merge、03_empty_panels
    logs_files = os.path.join(out_dir, "logs", "00_files.txt")
    _log(logs_files, f"\n=== main_mz_h5 启动 @ {datetime.now()} ===")
    _log(logs_files, f"[加载] y_true: {p_true}, shape={y_true.shape}")
    _log(logs_files, f"[加载] y_pred: {p_pred}, shape={y_pred.shape}")

    # 严格列名验证（你明确要求：禁止“猜列名”）
    cols_true = ["date","freq","rm","y_true"]
    cols_pred = ["date","freq","rm","model","y_pred"]
    if list(y_true.columns) != cols_true:
        raise ValueError(f"y_true 列不匹配：期望 {cols_true}，实际 {list(y_true.columns)}")
    if list(y_pred.columns) != cols_pred:
        raise ValueError(f"y_pred 列不匹配：期望 {cols_pred}，实际 {list(y_pred.columns)}")

    merged = _merge_strict(y_true, y_pred, out_dir)
    p_merged = os.path.join(tables_dir, "tmp_h5_merged.csv")
    merged.to_csv(p_merged, index=False)
    _log(logs_files, f"[落地] merged -> {p_merged}, shape={merged.shape}")

    # 面板覆盖诊断（每个 RM × model 的样本数）
    cover = (merged
             .groupby(["rm","model"], as_index=False)
             .agg(n=("y_pred","size"),
                  start=("date","min"), end=("date","max")))
    logs_merge = os.path.join(out_dir, "logs", "02_merge.txt")
    _log(logs_merge, "\n[覆盖] 每个 (rm, model) 样本数/日期范围（前20条）")
    _log(logs_merge, cover.head(20).to_string())

    # MZ 回归 & 输出
    df_res = _mz_gls_r2_panel(merged, out_dir)
    p_res = os.path.join(tables_dir, "mz_h5_r2.csv")
    df_res.to_csv(p_res, index=False)
    _log(logs_files, f"[落地] R2 -> {p_res}, shape={df_res.shape}")

    # 对“图里空白”的根因定位：列出样本为 0 的 (rm, model)
    have = set(map(tuple, df_res[["rm","model"]].itertuples(index=False, name=None)))
    all_pairs = set(map(tuple, merged[["rm","model"]].drop_duplicates().itertuples(index=False, name=None)))
    empty_pairs = sorted(list(all_pairs - have))
    logs_empty = os.path.join(out_dir, "logs", "03_empty_panels.txt")
    if empty_pairs:
        _log(logs_empty, f"\n[EMPTY-PAIR] 以下 (rm, model) 无有效样本或被过滤：{empty_pairs}")
    else:
        _log(logs_empty, "[OK] 所有 (rm, model) 均有样本")

    print(f"[OK] merged -> {p_merged}, R2 -> {p_res}")

if __name__ == "__main__":
    main()
