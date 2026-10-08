"""Per-household scaling and sliding-window construction.

Forecasting task: the previous `input_window` half-hours predict the load `horizon` steps after
the last input. With the defaults (48, 1) this is "last 24 h -> next 30 min":
    X: (n, 48, channels)    y: (n, 1)

Leakage rules
 - Each household's scaler (mean, std) is fitted on that household's TRAIN period only.
 - A window belongs to the split that contains its TARGET timestamp. Validation and test windows
   may use inputs from the end of the preceding split (those values are already known at
   forecast time). No target is ever drawn from another split.
 - Windows containing any missing value (an unfilled long gap) are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass
class HouseholdData:
    hid: str
    mean: float
    std: float
    X: dict = field(default_factory=dict)      # split -> (n, W, C) float32, scaled
    y: dict = field(default_factory=dict)      # split -> (n, 1) float32, scaled
    t: dict = field(default_factory=dict)      # split -> (n,) datetime64 target timestamps
    train_kwh: np.ndarray | None = None        # raw train-period series (for load-profile features)
    train_index: pd.DatetimeIndex | None = None

    def n(self, split: str = "train") -> int:
        return len(self.y[split])

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


def split_labels(index: pd.DatetimeIndex, train_end: str, val_end: str) -> np.ndarray:
    lab = np.full(len(index), 2, dtype=np.int8)            # test
    lab[index < pd.Timestamp(val_end)] = 1                 # val
    lab[index < pd.Timestamp(train_end)] = 0               # train
    return lab


def calendar_channels(index: pd.DatetimeIndex) -> np.ndarray:
    tod = (index.hour * 60 + index.minute).to_numpy() / 1440.0
    dow = index.dayofweek.to_numpy() / 7.0
    return np.stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                     np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)], axis=1).astype(np.float32)


def make_household(hid: str, x: np.ndarray, index: pd.DatetimeIndex, cfg: dict) -> HouseholdData:
    d, w = cfg["data"], cfg["window"]
    W, h = int(w["input_window"]), int(w["horizon"])
    lab = split_labels(index, d["train_end"], d["val_end"])
    train_vals = x[lab == 0]
    mean = float(np.nanmean(train_vals))
    std = float(np.nanstd(train_vals))
    std = std if std > 1e-6 else 1.0
    z = ((x - mean) / std).astype(np.float32)

    feats = z[:, None]
    if w.get("calendar_features"):
        feats = np.concatenate([feats, calendar_channels(index)], axis=1)

    # target positions t; inputs are [t-h-W+1, t-h]
    t_idx = np.arange(W + h - 1, len(x))
    starts = t_idx - h - W + 1
    win = np.lib.stride_tricks.sliding_window_view(feats, W, axis=0)  # (len-W+1, C, W)
    X_all = win[starts].transpose(0, 2, 1)                           # (n, W, C)
    y_all = z[t_idx]
    valid = np.isfinite(y_all) & np.isfinite(X_all).all(axis=(1, 2))

    hd = HouseholdData(hid, mean, std, train_kwh=train_vals, train_index=index[lab == 0])
    for s, name in enumerate(SPLITS):
        m = valid & (lab[t_idx] == s)
        hd.X[name] = np.ascontiguousarray(X_all[m], dtype=np.float32)
        hd.y[name] = y_all[m].reshape(-1, 1).astype(np.float32)
        hd.t[name] = index[t_idx[m]].to_numpy()
    return hd


def build_households(series: pd.DataFrame, cfg: dict) -> list[HouseholdData]:
    idx = pd.DatetimeIndex(series.index)
    out = [make_household(str(c), series[c].to_numpy(np.float32), idx, cfg) for c in series.columns]
    empty = [hd.hid for hd in out if min(hd.n(s) for s in SPLITS) == 0]
    if empty:
        raise ValueError(f"Households with an empty split (check the split dates): {empty[:5]}")
    return out


def stack(households, split: str):
    """Concatenate windows of several households (used by centralised training and community clients)."""
    X = np.concatenate([hd.X[split] for hd in households])
    y = np.concatenate([hd.y[split] for hd in households])
    return X, y
