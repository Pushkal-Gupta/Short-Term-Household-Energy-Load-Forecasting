"""Federated Averaging (McMahan et al., 2017): sample-weighted mean of client weights.

    W_global = sum_k n_k * W_k / sum_k n_k        n_k = number of local training windows
"""
from __future__ import annotations

import numpy as np


def weighted_average(results: list[tuple[list, int]]) -> list:
    """results: [(weights_k, n_k), ...] -> aggregated weights (list of float32 arrays)."""
    total = float(sum(n for _, n in results))
    if total <= 0:
        raise ValueError("cannot aggregate: no training samples")
    agg = [np.zeros_like(w, dtype=np.float64) for w in results[0][0]]
    for weights, n in results:
        for a, w in zip(agg, weights):
            a += (n / total) * w
    return [a.astype(np.float32) for a in agg]


def update_dissimilarity(global_w: list, results: list[tuple[list, int]]) -> float:
    """Update-disagreement ratio  rho = sum_k p_k ||d_k - d_bar||^2 / sum_k p_k ||d_k||^2,  in [0, 1].

    d_k = W_k - W_global, p_k = n_k / sum n, d_bar = sum_k p_k d_k (the FedAvg step).
    rho = 0 when every client moves the same way. rho -> 1 when the client updates cancel out,
    which is the symptom of non-IID client drift. It equals 1 - 1/B^2, where B^2 is the
    empirical update-dissimilarity of Li et al. (2020). Unlike B^2 it is bounded, so it does not
    explode when the mean update shrinks near convergence.
    """
    g = np.concatenate([w.ravel() for w in global_w]).astype(np.float64)
    total = float(sum(n for _, n in results))
    deltas = [(np.concatenate([w.ravel() for w in wk]).astype(np.float64) - g, n / total) for wk, n in results]
    d_bar = sum(p * d for d, p in deltas)
    energy = sum(p * float(d @ d) for d, p in deltas)
    if energy <= 1e-30:
        return 0.0
    return float(sum(p * float((d - d_bar) @ (d - d_bar)) for d, p in deltas) / energy)
