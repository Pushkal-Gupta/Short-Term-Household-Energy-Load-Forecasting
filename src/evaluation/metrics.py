"""Forecast error metrics. All are computed in kWh per half-hour, after inverse scaling.

MAE   = mean |y - yhat|
RMSE  = sqrt(mean (y - yhat)^2)
MAPE  = 100 * mean |y - yhat| / |y|, over points with y >= mape_min_actual ONLY.
        Household load is often close to zero overnight, and plain MAPE then explodes.
        Points below the threshold are excluded and their share is reported as
        `mape_excluded_frac`. MAPE is therefore not comparable with papers that use another rule.
WAPE  = 100 * sum |y - yhat| / sum |y|. Scale-free and stable near zero; reported alongside.
"""
from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat, min_actual: float = 0.05) -> tuple[float, float]:
    """Return (masked MAPE in %, fraction of points excluded by the mask)."""
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    mask = np.abs(y) >= min_actual
    if not mask.any():
        return float("nan"), 1.0
    return float(100.0 * np.mean(np.abs(y[mask] - yhat[mask]) / np.abs(y[mask]))), float(1.0 - mask.mean())


def wape(y, yhat) -> float:
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    denom = np.abs(y).sum()
    return float(100.0 * np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def all_metrics(y, yhat, min_actual: float = 0.05) -> dict:
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    m, excl = mape(y, yhat, min_actual)
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "mape": m, "wape": wape(y, yhat),
            "mape_excluded_frac": excl, "n_points": int(len(y))}
