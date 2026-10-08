"""Non-learned reference forecasts (E0), so reviewers can see what the LSTM adds.

persistence      yhat(t) = y(t - horizon)    the last observed value
seasonal_naive   yhat(t) = y(t - 48)          the same half-hour yesterday (needs input_window >= 48)

Both read the values straight out of each input window, so they use exactly the same test points.
"""
from __future__ import annotations

import numpy as np


def naive_pairs(households, kind: str, cfg: dict):
    W, h = int(cfg["window"]["input_window"]), int(cfg["window"]["horizon"])
    if kind == "persistence":
        pos = W - 1
    elif kind == "seasonal_naive":
        pos = W + h - 1 - 48
        if not 0 <= pos < W:
            raise ValueError("seasonal_naive needs input_window + horizon - 1 >= 48")
    else:
        raise ValueError(kind)
    return [(hd.inverse(hd.y["test"][:, 0]), hd.inverse(hd.X["test"][:, pos, 0])) for hd in households]
