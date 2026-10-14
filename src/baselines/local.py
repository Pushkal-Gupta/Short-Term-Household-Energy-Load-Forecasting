"""Local-only baseline (E1): every household trains its own LSTM on its own data and nothing else.

There is no collaboration and no communication. It is the reference that federation must
beat to be worth having. The architecture and training loop are the same as in every other arm.
"""
from __future__ import annotations

import numpy as np
import torch

from src.centralized.train import fit_early_stopping, train_epochs
from src.data.window import stack
from src.evaluation.evaluate import predict_households
from src.models.lstm import build_model, get_weights, set_weights


def run_client_local(households, groups, cfg, device, seed: int, verbose: bool = True):
    """Client-only baseline: each client trains one model on its own pooled homes, without federation.

    This is what a community hub or aggregator could do alone with the homes it already pools.
    Same model, optimiser and early stopping (on the client's own validation windows) as local-only.
    """
    pairs = [None] * len(households)
    ft_pairs = [None] * len(households)
    ft_epochs = float(cfg["fed"].get("finetune_epochs", 0))
    histories = []
    for c, g in enumerate(groups):
        members = [households[j] for j in g]
        torch.manual_seed(seed + 1000 + c)
        model = build_model(cfg).to(device)
        rng = np.random.default_rng(seed + 1000 + c)
        hist = fit_early_stopping(model, stack(members, "train"), members, device, cfg, rng)
        histories.append(hist)
        for j in g:
            pairs[j] = predict_households(model, [households[j]], "test", device)
        if ft_epochs > 0:                     # same per-home fine-tuning as FedAvg-FT (common._finetune_eval)
            w = get_weights(model)
            for j in g:
                hd = households[j]
                set_weights(model, w)
                train_epochs(model, hd.X["train"], hd.y["train"], device, cfg, np.random.default_rng(seed + 7 * j), ft_epochs)
                ft_pairs[j] = predict_households(model, [hd], "test", device)
        if verbose:
            print(f"[client-local] client {c + 1}/{len(groups)}  epochs {len(hist)}")
    flat = [p for ps in pairs for p in ps]
    ft = [p for ps in ft_pairs for p in ps] if ft_epochs > 0 else None
    return flat, histories, ft


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
