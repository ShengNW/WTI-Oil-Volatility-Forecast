# -*- coding: utf-8 -*-
import os, yaml, logging
from pathlib import Path

# --- import path for local packages ---
import sys
THIS = Path(__file__).resolve()
ROOT = THIS.parent
sys.path.append(str(ROOT / "adapters"))
sys.path.append(str(ROOT / "mz"))

from adapters.schema import STD_COLS
from adapters.rm_adapter import load_rm_h1
from adapters.forecast_adapter import load_preds_h1
from mz.design import build_plain_design
from mz.estimator import run_mz_grouped
from mz.aggregate import merge_true_pred
from mz.plot import plot_fig3_r2

def make_logger():
    logs_dir = ROOT / "05_outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "mz_h1.log"

    logger = logging.getLogger("mz")
    logger.setLevel(logging.INFO)
    # 控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 文件
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    ch.setFormatter(fmt); fh.setFormatter(fmt)
    # 避免重复 Handler
    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    logger.info("==== MZ h=1 run start ====")
    logger.info(f"ROOT={ROOT}")
    logger.info(f"Log file: {log_path}")
    return logger

def main():
    logger = make_logger()

    cfg_path = ROOT / "04_configs" / "mz_h1_config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    project_root = Path(cfg["project_root"])
    asset = cfg["asset"]; freq = cfg["freq"]; h = cfg["h"]

    logger.info(f"asset={asset} freq={freq} h={h}")

    # RM 标签读取
    rm_base = project_root / cfg["rm_sources"]["base_dir"]
    logger.info(f"[PATH] RM base = {rm_base}")
    rm_df = load_rm_h1(str(rm_base), cfg["rm_sources"]["items"], asset, freq, logger=logger)
    logger.info(f"[RM] loaded rows={len(rm_df)} dates=[{rm_df['date'].min()} .. {rm_df['date'].max()}]")

    # 预测读取
    preds_cfg = []
    for src in cfg["forecast_sources"]:
        src = dict(src); src["base_dir"] = str(project_root / src["base_dir"])
        preds_cfg.append(src)
    logger.info(f"[PATH] PRED sources = {len(preds_cfg)} groups")
    pred_df = load_preds_h1(preds_cfg, asset, freq, logger=logger)
    logger.info(f"[PRED] loaded rows={len(pred_df)} dates=[{pred_df['date'].min()} .. {pred_df['date'].max()}]")

    # 合并
    merged = merge_true_pred(rm_df, pred_df)
    logger.info(f"[MERGE] rows={len(merged)} groups={merged.groupby(['RM','model']).size().shape[0]}")

    # 设计矩阵
    design = build_plain_design(merged)
    logger.info(f"[DESIGN] rows={len(design)}")

    # 估计
    res = run_mz_grouped(design)
    logger.info(f"[EST] done; non-NaN R2: {res['R2'].notna().sum()}")

    # 输出
    out_dir = Path(cfg["plot"]["out_dir"])
    out_table = out_dir / cfg["plot"]["table_name"]
    out_fig = out_dir / cfg["plot"]["fig_name"]
    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_fig.parent.mkdir(parents=True, exist_ok=True)

    res.to_csv(out_table, index=False)
    plot_fig3_r2(
        res,
        order_models=cfg["plot"]["order_models"],
        order_rms=cfg["plot"]["order_rms"],
        out_path=str(out_fig),
    )
    logger.info(f"[OK] Saved table: {out_table}")
    logger.info(f"[OK] Saved figure: {out_fig}")
    logger.info("==== MZ h=1 run end ====")

if __name__ == "__main__":
    main()
