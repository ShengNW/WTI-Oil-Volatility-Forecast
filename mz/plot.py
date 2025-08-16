# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_fig3_r2(df_res: pd.DataFrame, order_models, order_rms, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 透视：行=RM，列=model，值=R2
    pt = df_res.pivot(index="RM", columns="model", values="R2")
    pt = pt.reindex(index=order_rms, columns=order_models, fill_value=float("nan"))

    ax = pt.plot(kind="bar", figsize=(10,5))
    ax.set_ylabel("R² (MZ, h=1)")
    ax.set_xlabel("Realized Measure (RM)")
    ax.set_title("Mincer–Zarnowitz R² (h=1)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
