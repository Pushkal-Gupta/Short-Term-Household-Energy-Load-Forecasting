"""Training loop shared by all arms, and the centralised baseline (E2).

CENTRALISED TRAINING POOLS RAW DATA. In this arm the training windows of every household are
concatenated in one place, as if a utility had collected every home's readings on its own
server. It is the accuracy reference ("upper bound") that federation tries to approach without
moving raw data. The model, preprocessing and per-household scaling are identical to the
federated arms.

    python -m src.centralized.train [--config ...] [--set train.lr=0.0005]
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from src.data.window import stack
from src.evaluation.evaluate import evaluate_model
from src.models.lstm import build_model


def train_epochs(model, X: np.ndarray, y: np.ndarray, device, cfg: dict, rng: np.random.Generator,
                 epochs: float = 1.0, optimizer=None, prox_ref: list | None = None, mu: float = 0.0) -> float:
    """Mini-batch SGD (Adam) over (X, y) for `epochs` passes (fractional = partial work).

    If `prox_ref` is given, the FedProx proximal term (mu/2)*||w - w_ref||^2 is added to the loss.
    Returns the mean training loss (MSE on scaled values, without the proximal term).
    """
    bs = int(cfg["train"]["batch_size"])
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    loss_fn = nn.MSELoss()
    n = len(X)
    n_batches_per_epoch = max(1, int(np.ceil(n / bs)))
    total_batches = max(1, int(round(epochs * n_batches_per_epoch)))
    model.train()
    losses, done = [], 0
    while done < total_batches:
        order = rng.permutation(n)
        for i in range(0, n, bs):
            if done >= total_batches:
                break
            idx = order[i:i + bs]
            xb = torch.from_numpy(X[idx]).to(device)
            yb = torch.from_numpy(y[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            obj = loss
            if prox_ref is not None and mu > 0:
                prox = sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), prox_ref))
                obj = loss + 0.5 * mu * prox
            obj.backward()
            optimizer.step()
            losses.append(loss.item())
            done += 1
    return float(np.mean(losses))


def fit_early_stopping(model, train_xy, val_households, device, cfg, rng, log_prefix=None):
    """Train until validation MAE (kWh) stops improving for `patience` epochs; restore the best weights."""
    X, y = train_xy
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=float(t["lr"]))
    best, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, int(t["max_epochs"]) + 1):
        tr_loss = train_epochs(model, X, y, device, cfg, rng, 1.0, opt)
        val, _, _ = evaluate_model(model, val_households, "val", device, cfg)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_mae": val["mae"],
                        "val_rmse": val["rmse"], "val_mape": val["mape"]})
        if log_prefix:
            print(f"{log_prefix} epoch {epoch:3d}  train_mse {tr_loss:.4f}  val_mae {val['mae']:.4f}")
        if val["mae"] < best - 1e-6:
            best, best_state, bad = val["mae"], copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(t["patience"]):
                break
    model.load_state_dict(best_state)
    return history


def run_centralized(households, cfg, device, seed: int, verbose: bool = True):
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    rng = np.random.default_rng(seed)
    history = fit_early_stopping(model, stack(households, "train"), households, device, cfg, rng,
                                 "[centralised]" if verbose else None)
    return model, history


def main(argv=None):
    from src.experiments.common import run_single
    run_single("centralized", argv)


if __name__ == "__main__":
    main()
