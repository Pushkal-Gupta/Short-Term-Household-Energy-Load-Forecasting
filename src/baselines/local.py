"""Local-only baseline (E1): every household trains its own LSTM on its own data and nothing else.

There is no collaboration and no communication. It is the reference that federation must
beat to be worth having. The architecture and training loop are the same as in every other arm.
"""
from __future__ import annotations

import numpy as np
import torch

from src.centralized.train import fit_early_stopping
from src.evaluation.evaluate import predict_households
from src.models.lstm import build_model


def run_local(households, cfg, device, seed: int, verbose: bool = True):
    """Return per-household (y, yhat) test pairs and one training history per household."""
    pairs, histories = [], []
    for i, hd in enumerate(households):
        torch.manual_seed(seed + i)
        model = build_model(cfg).to(device)
        rng = np.random.default_rng(seed + i)
        hist = fit_early_stopping(model, (hd.X["train"], hd.y["train"]), [hd], device, cfg, rng)
        histories.append(hist)
        pairs.extend(predict_households(model, [hd], "test", device))
        if verbose:
            print(f"[local] {i + 1:3d}/{len(households)} {hd.hid}  epochs {len(hist)}  "
                  f"best val_mae {min(h['val_mae'] for h in hist):.4f}")
    return pairs, histories
