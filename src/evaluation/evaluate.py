"""Prediction and evaluation helpers shared by every arm."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.evaluation.metrics import all_metrics


@torch.no_grad()
def predict(model, X: np.ndarray, device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def predict_households(model, households, split: str, device, batch_size: int = 4096):
    """Return per-household (y_kwh, yhat_kwh) arrays for a split, after inverse scaling."""
    X = np.concatenate([hd.X[split] for hd in households])
    z = predict(model, X, device, batch_size)
    out, k = [], 0
    for hd in households:
        n = hd.n(split)
        out.append((hd.inverse(hd.y[split][:, 0]), hd.inverse(z[k:k + n, 0])))
        k += n
    return out


def pooled_metrics(pairs, min_actual: float) -> dict:
    """Micro-average over all households' points (each half-hour counts once)."""
    y = np.concatenate([p[0] for p in pairs])
    yhat = np.concatenate([p[1] for p in pairs])
    return all_metrics(y, yhat, min_actual)


def per_household_metrics(households, pairs, min_actual: float) -> pd.DataFrame:
    rows = []
    for hd, (y, yhat) in zip(households, pairs):
        rows.append({"LCLid": hd.hid, **all_metrics(y, yhat, min_actual)})
    return pd.DataFrame(rows)


def evaluate_model(model, households, split, device, cfg):
    pairs = predict_households(model, households, split, device, cfg["train"].get("eval_batch_size", 4096))
    mn = cfg["eval"]["mape_min_actual"]
    return pooled_metrics(pairs, mn), per_household_metrics(households, pairs, mn), pairs


def predictions_frame(households, pairs, split: str = "test") -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"LCLid": hd.hid, "timestamp": hd.t[split],
                                    "y_kwh": y.astype(np.float32), "yhat_kwh": yh.astype(np.float32)})
                      for hd, (y, yh) in zip(households, pairs)], ignore_index=True)


def summarise_household_metrics(df: pd.DataFrame) -> dict:
    """Distributional summary across households (equity view): median and worst decile."""
    return {"hh_mae_median": float(df["mae"].median()),
            "hh_mae_p90": float(df["mae"].quantile(0.9)),
            "hh_wape_median": float(df["wape"].median()),
            "hh_wape_p90": float(df["wape"].quantile(0.9))}
