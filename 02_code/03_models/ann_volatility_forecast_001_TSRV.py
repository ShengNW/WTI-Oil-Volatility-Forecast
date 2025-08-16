#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Rprop

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 20250810
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------------
# Configuration
# -----------------------------
# 只跑 TSRV；频率仍为 1min 与 5min（文件名与目录不变）
MEASURES = ["TSRV"]
FREQS = ["1min", "5min"]

# Default training hyperparameters (unchanged)
DEFAULT_EPOCHS = 400
DEFAULT_PATIENCE = 50
DEFAULT_LR = 0.01
DEFAULT_BATCH = 128
HIDDEN_GRID = [7, 15]
DECAY_GRID = [0.0, 1e-10]


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self._safe_std()

    def decode(self, x: np.ndarray) -> np.ndarray:
        return x * self._safe_std() + self.mean

    def _safe_std(self) -> np.ndarray:
        s = self.std.copy()
        s[s == 0] = 1.0
        return s


class MLP(nn.Module):
    def __init__(self, input_dim: int = 22, hidden: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))
        y = self.fc2(h)
        return y.squeeze(-1)


# -----------------------------
# Data helpers
# -----------------------------

def _measure_col(df: pd.DataFrame, measure: str) -> str:
    if measure in df.columns:
        return measure
    for c in df.columns:
        if c.lower() == measure.lower():
            return c
    float_cols = [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
    if not float_cols:
        raise ValueError(f"No numeric column found for measure={measure}. Columns={list(df.columns)}")
    return float_cols[0]


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception as e:
            alt = path.with_suffix(".csv")
            if alt.exists():
                return pd.read_csv(alt)
            raise RuntimeError(
                f"Failed to read {path} as parquet and no CSV fallback found. Install pyarrow or fastparquet.\n{e}"
            )
    else:
        return pd.read_csv(path)


def _load_series(root: Path, split: str, measure: str, freq: str,
                 override_file: Path | None = None, override_column: str | None = None) -> pd.Series:
    base = root / "03_results" / "intermediate_results" / "volatility_estimates" / f"{split}_set"
    if override_file is not None:
        path = override_file
    else:
        fname = f"CL_WTI_{measure}_daily_{freq}_{split}.parquet"
        path = base / fname
        if not path.exists():
            candidates = list(base.glob(f"CL_WTI_*{measure.lower()}*_daily_{freq}_{split}.*"))
            if candidates:
                path = candidates[0]
            else:
                raise FileNotFoundError(f"Could not find file for measure={measure}, split={split}, freq={freq} under {base}")

    df = _read_parquet_or_csv(path)

    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = df.set_index("DateTime").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    col = override_column if override_column is not None else _measure_col(df, measure)
    s = df[col].astype(float).copy()
    s.name = measure
    return s


def _to_volatility(s: pd.Series) -> pd.Series:
    s2 = s.clip(lower=0.0)
    v = np.sqrt(s2.values)
    return pd.Series(v, index=s.index, name=f"vol_{s.name}")


def make_supervised(vol: pd.Series, lags: int = 22) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
    vals = vol.values.astype(np.float64)
    T = len(vals)
    xs, ys, ts = [], [], []
    for t in range(lags - 1, T - 1):
        x = vals[t - (lags - 1): t + 1][::-1]  # [ν_t, ν_{t-1}, ..., ν_{t-21}]
        y = vals[t + 1]
        xs.append(x)
        ys.append(y)
        ts.append(vol.index[t + 1])
    X = np.asarray(xs)
    y = np.asarray(ys)
    return X, y, ts


# -----------------------------
# Training / Evaluation
# -----------------------------

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def train_one(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    hidden: int,
    l2_lambda: float,
    epochs: int,
    patience: int,
    device: torch.device,
    lr: float = DEFAULT_LR,
) -> Tuple[MLP, float]:
    model = MLP(input_dim=Xtr.shape[1], hidden=hidden).to(device)
    opt = Rprop(model.parameters(), lr=lr)
    best_rmse = math.inf
    best_state = None
    bad = 0

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    Xva_t = torch.from_numpy(Xva).float().to(device)
    yva_t = torch.from_numpy(yva).float().to(device)

    N = Xtr.shape[0]
    batch = min(DEFAULT_BATCH, N)

    for ep in range(1, epochs + 1):
        model.train()
        idx = np.random.permutation(N)
        for start in range(0, N, batch):
            sel = idx[start:start + batch]
            xb = Xtr_t[sel]
            yb = ytr_t[sel]
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            if l2_lambda > 0:
                l2 = 0.0
                for p in model.parameters():
                    l2 = l2 + torch.sum(p.pow(2))
                loss = loss + l2_lambda * l2
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).cpu().numpy()
        cur_rmse = rmse(pv, yva)
        if cur_rmse + 1e-12 < best_rmse:
            best_rmse = cur_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse


@dataclass
class FitResult:
    hidden: int
    l2_lambda: float
    input_norm: Normalizer
    val_rmse: float
    state_dict: Dict[str, np.ndarray]


def fit_ann(X: np.ndarray, y: np.ndarray, ts: List[pd.Timestamp],
            epochs: int, patience: int, device: torch.device) -> FitResult:
    n = X.shape[0]
    n_val = max(60, int(0.15 * n))
    n_tr = n - n_val
    Xtr_raw, ytr = X[:n_tr], y[:n_tr]
    Xva_raw, yva = X[n_tr:], y[n_tr:]

    mu = Xtr_raw.mean(axis=0)
    sd = Xtr_raw.std(axis=0)
    norm = Normalizer(mu, sd)
    Xtr = norm.encode(Xtr_raw)
    Xva = norm.encode(Xva_raw)

    best: FitResult | None = None
    for hidden in HIDDEN_GRID:
        for l2_lambda in DECAY_GRID:
            model, val = train_one(Xtr, ytr, Xva, yva,
                                   hidden=hidden, l2_lambda=l2_lambda,
                                   epochs=epochs, patience=patience,
                                   device=device)
            if (best is None) or (val < best.val_rmse):
                state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
                best = FitResult(hidden=hidden, l2_lambda=l2_lambda,
                                 input_norm=norm, val_rmse=val, state_dict=state)
    assert best is not None
    return best


# -----------------------------
# Target discovery & IO helpers
# -----------------------------
from dataclasses import field
@dataclass
class TargetSpec:
    measure: str
    freq: str
    train_path: Path
    test_path: Path
    column: str | None = None  # kept for RK compatibility


def _match_test_file(train_path: Path, test_dir: Path, rk: bool = False) -> Path:
    if rk and train_path.name.startswith("realized_kernel_estimates"):
        candidate = test_dir / train_path.name.replace("_train", "_test")
        if candidate.exists(): return candidate
        for ext in (".parquet", ".csv"):
            alt = test_dir / f"realized_kernel_estimates_test{ext}"
            if alt.exists(): return alt
    candidate = test_dir / train_path.name.replace("_train", "_test")
    if candidate.exists(): return candidate
    stems = list(test_dir.glob(train_path.stem.replace("_train", "_test") + ".*"))
    if stems: return stems[0]
    raise FileNotFoundError(f"Test file not found for {train_path}")


def discover_targets(root: Path) -> List[TargetSpec]:
    """
    只发现 TSRV 的目标；文件名与路径规则保持不变：
    CL_WTI_TSRV_daily_{freq}_train.parquet / _test.parquet
    """
    train_dir = root / "03_results" / "intermediate_results" / "volatility_estimates" / "train_set"
    test_dir = root / "03_results" / "intermediate_results" / "volatility_estimates" / "test_set"
    specs: List[TargetSpec] = []
    for p in sorted(list(train_dir.glob("*.parquet")) + list(train_dir.glob("*.csv"))):
        name = p.name
        import re
        m = re.match(r"CL_WTI_([A-Za-z]+)_daily_(1min|5min)_train\.(parquet|csv)", name)
        if m:
            measure = m.group(1)
            freq = m.group(2)
            # 仅保留 TSRV
            if measure.upper() != "TSRV":
                continue
            test_p = _match_test_file(p, test_dir, rk=False)
            specs.append(TargetSpec(measure=measure, freq=freq, train_path=p, test_path=test_p))
    return specs


def _to_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path)
    except Exception as e:
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index_label="DateTime")
        print(f"[WARN] Failed to write parquet ({e}); wrote CSV instead: {alt}")


# -----------------------------
# Forecasting utilities
# -----------------------------

def _nn_from_state(state: Dict[str, np.ndarray], input_dim: int, hidden: int, device: torch.device) -> MLP:
    model = MLP(input_dim=input_dim, hidden=hidden).to(device)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    model.eval()
    return model


def predict_1step(model: MLP, norm: Normalizer, last_22: np.ndarray, device: torch.device) -> float:
    x = last_22.astype(np.float64)
    x = norm.encode(x[None, :])
    xt = torch.from_numpy(x).float().to(device)
    with torch.no_grad():
        y = model(xt).cpu().numpy().ravel()[0]
    return float(max(y, 0.0))


def recursive_path(model: MLP, norm: Normalizer, history: np.ndarray, horizon: int, device: torch.device) -> np.ndarray:
    buf = history.copy()
    preds = []
    for _ in range(horizon):
        last_22 = buf[-22:][::-1]
        y1 = predict_1step(model, norm, last_22, device)
        preds.append(y1)
        buf = np.append(buf, y1)
    return np.array(preds, dtype=np.float64)


def cumulative_from_path(path: np.ndarray) -> float:
    h = len(path)
    return float(np.mean(path ** 2))


# -----------------------------
# Orchestration
# -----------------------------

def run_for_measure(root_dir: Path, measure: str, freq: str,
                    epochs: int, patience: int, device: torch.device,
                    out_tag: str = "ANN",
                    override_train: Path | None = None,
                    override_test: Path | None = None,
                    override_column: str | None = None) -> None:
    print(f"[ANN] Measure={measure} Freq={freq}")

    s_tr = _load_series(root_dir, split="train", measure=measure, freq=freq,
                        override_file=override_train, override_column=override_column)
    s_te = _load_series(root_dir, split="test", measure=measure, freq=freq,
                        override_file=override_test, override_column=override_column)
    v_tr = _to_volatility(s_tr)
    v_te = _to_volatility(s_te)

    X, y, ts = make_supervised(v_tr, lags=22)
    fit = fit_ann(X, y, ts, epochs=epochs, patience=patience, device=device)
    print(f"  -> Best hidden={fit.hidden}, decay={fit.l2_lambda:.1e}, val_RMSE={float(fit.val_rmse):.6f}")

    model = _nn_from_state(fit.state_dict, input_dim=22, hidden=fit.hidden, device=device)

    v_all = pd.concat([v_tr, v_te]).sort_index()
    one_step, cum5, cum10, ts_out = [], [], [], []

    for t in range(len(v_tr) - 1, len(v_all) - 1):
        last22_start = t - 21
        if last22_start < 0:
            continue
        history = v_all.values[: t + 1]
        last22 = history[-22:]
        y1 = predict_1step(model, fit.input_norm, last22[::-1], device)
        path5 = recursive_path(model, fit.input_norm, history, horizon=5, device=device)
        path10 = recursive_path(model, fit.input_norm, history, horizon=10, device=device)
        one_step.append(y1)
        cum5.append(cumulative_from_path(path5))
        cum10.append(cumulative_from_path(path10))
        ts_out.append(v_all.index[t + 1])

    df_1 = pd.DataFrame({f"{measure}_ann_1step": one_step}, index=pd.DatetimeIndex(ts_out))
    df_h5 = pd.DataFrame({f"{measure}_cumulative_h5": cum5}, index=pd.DatetimeIndex(ts_out))
    df_h10 = pd.DataFrame({f"{measure}_cumulative_h10": cum10}, index=pd.DatetimeIndex(ts_out))

    params_dir = root_dir / "03_results" / "intermediate_results" / "model_parameters" / out_tag
    params_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir = root_dir / "03_results" / "final_forecasts" / out_tag
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    interm_cumu_dir = root_dir / "03_results" / "intermediate_results" / "cumulative_forecasts" / out_tag
    interm_cumu_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "measure": measure,
        "freq": freq,
        "hidden": fit.hidden,
        "decay": fit.l2_lambda,
        "val_rmse": float(fit.val_rmse),
        "input_norm_mean": fit.input_norm.mean.tolist(),
        "input_norm_std": fit.input_norm.std.tolist(),
        "seed": SEED,
    }
    meta_path = params_dir / f"ANN_{measure}_{freq}_params.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    torch_path = params_dir / f"ANN_{measure}_{freq}_weights.pt"
    torch.save(fit.state_dict, torch_path)

    _to_parquet_safe(df_1,  forecasts_dir   / f"CL_WTI_{measure}_daily_{freq}_ANN_1step_test.parquet")
    _to_parquet_safe(df_h5, interm_cumu_dir / f"CL_WTI_{measure}_daily_{freq}_ANN_cumulative_h5_test.parquet")
    _to_parquet_safe(df_h10,interm_cumu_dir / f"CL_WTI_{measure}_daily_{freq}_ANN_cumulative_h10_test.parquet")

    print(f"  -> Saved params to: {meta_path.name}, {torch_path.name}")
    print(f"  -> Saved 1-step test forecasts: {df_1.shape} -> {forecasts_dir}")
    print(f"  -> Saved cumulative h=5/h=10: {df_h5.shape}, {df_h10.shape} -> {interm_cumu_dir}")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ANN volatility forecaster (WTI project)")
    p.add_argument("--root_dir", type=str, required=False)
    p.add_argument("--measure", type=str, choices=MEASURES, default="TSRV")
    p.add_argument("--freq", type=str, choices=FREQS, default="5min")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    root_dir = Path("F:/CODE/social/JianZhi/future/WTI_volatility_forecast_replication").expanduser().resolve()
    epochs = DEFAULT_EPOCHS
    patience = DEFAULT_PATIENCE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Root: {root_dir}")
    print(f"Device: {device}")

    # 只发现/处理 TSRV 目标
    targets = discover_targets(root_dir)
    if not targets:
        raise RuntimeError("No TSRV training targets discovered in train_set.")

    for t in targets:
        try:
            run_for_measure(root_dir=root_dir,
                            measure=t.measure,   # 'TSRV'
                            freq=t.freq,         # '1min' or '5min'
                            epochs=epochs,
                            patience=patience,
                            device=device,
                            override_train=t.train_path,
                            override_test=t.test_path,
                            override_column=t.column)
        except Exception as e:
            print(f"[ERROR] {t.measure}-{t.freq}: {e}")

if __name__ == "__main__":
    main()
