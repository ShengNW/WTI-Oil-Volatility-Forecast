# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

TABLES = "./05_outputs/tables"
LOGS   = "./05_outputs/logs"

def main():
    os.makedirs(LOGS, exist_ok=True)
    merged_path = os.path.join(TABLES, "tmp_h5_merged.csv")
    df = pd.read_csv(merged_path, parse_dates=["date"])

    rows = []
    rows_ratio = []
    eps = 1e-12

    for (rm, model), g in df.groupby(["rm","model"]):
        n = len(g)
        pred = g["y_pred"].values
        true = g["y_true"].values

        n_nonfinite = np.sum(~np.isfinite(pred))
        n_le0       = np.sum(np.isfinite(pred) & (pred <= 0))
        q = np.nanquantile(pred[np.isfinite(pred)], [0.0, 0.05, 0.5, 0.95, 1.0]) if np.isfinite(pred).any() else [np.nan]*5

        rows.append({
            "rm": rm, "model": model, "n": n,
            "n_nonfinite": int(n_nonfinite), "pct_nonfinite": n_nonfinite / n if n else np.nan,
            "n_le0": int(n_le0), "pct_le0": n_le0 / n if n else np.nan,
            "min": q[0], "p05": q[1], "p50": q[2], "p95": q[3], "max": q[4],
        })

        # ratio 方差（排除不可用行）
        mask = np.isfinite(pred) & np.isfinite(true) & (pred > eps)
        if mask.any():
            ratio = true[mask] / pred[mask]
            rows_ratio.append({"rm": rm, "model": model, "n_used": int(mask.sum()),
                               "var_ratio": float(np.var(ratio, ddof=1))})
        else:
            rows_ratio.append({"rm": rm, "model": model, "n_used": 0, "var_ratio": np.nan})

    pd.DataFrame(rows).sort_values(["rm","model"]).to_csv(os.path.join(LOGS, "04_pred_sanity.csv"), index=False)
    pd.DataFrame(rows_ratio).sort_values(["rm","model"]).to_csv(os.path.join(LOGS, "04_ratio_const.csv"), index=False)
    print("[OK] wrote 04_pred_sanity.csv & 04_ratio_const.csv")

if __name__ == "__main__":
    main()
