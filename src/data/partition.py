"""Assign households to federated clients.

Two modes (`partition.mode`):

household   1 household = 1 client. This is the literal smart-meter setting. Heterogeneity is
            whatever the real homes have; it cannot be dialled.

community   The SAME cohort of households is grouped into clients of
            `households_per_client` homes each (a community energy hub or building gateway).
            Only the grouping changes between heterogeneity levels, so the households and the
            test set are identical in every setting. Any change in error is caused by the
            partition alone.

Strategies for community mode (`partition.strategy`). Every strategy uses real load-profile
clusters (src/data/profiles.py), computed from train-period data:

iid         random grouping. Each client is a mixture of profiles ("IID-ish").
dirichlet   for every client, cluster proportions p ~ Dir(alpha * 1_K). Each of its slots
            draws a cluster from p (renormalised over clusters that still have homes left)
            and takes a random home from that cluster. alpha large -> IID-ish; alpha small ->
            each client dominated by one profile.
cluster     strongest non-IID: homes sorted by (profile cluster, mean load) and chunked, so
            each client holds near-identical homes and clients differ maximally.
"""
from __future__ import annotations

import numpy as np

from src.data.profiles import cluster_profiles


def partition_households(households, cfg: dict, seed: int) -> tuple[list[list[int]], np.ndarray]:
    """Return (list of client -> household indices, profile cluster label per household)."""
    p = cfg["partition"]
    labels = cluster_profiles(households, int(p.get("num_profile_clusters", 5)), seed=0)
    n = len(households)
    if p["mode"] == "household":
        return [[i] for i in range(n)], labels
    m = int(p["households_per_client"])
    if n % m:
        raise ValueError(f"num_households ({n}) must be divisible by households_per_client ({m})")
    rng = np.random.default_rng(seed)
    strategy = p["strategy"]
    alpha = p.get("alpha")
    if strategy == "dirichlet" and (alpha is None or alpha == float("inf") or alpha >= 1e6):
        strategy = "iid"
    if strategy == "dirichlet" and alpha <= 0:
        strategy = "cluster"
    if strategy == "iid":
        order = rng.permutation(n)
        clients = [sorted(order[i:i + m].tolist()) for i in range(0, n, m)]
    elif strategy == "cluster":
        means = np.array([np.nanmean(hd.train_kwh) for hd in households])
        order = np.lexsort((means, labels))
        clients = [sorted(order[i:i + m].tolist()) for i in range(0, n, m)]
    elif strategy == "dirichlet":
        clients = _dirichlet(labels, m, float(alpha), rng)
    else:
        raise ValueError(f"Unknown partition strategy {strategy!r}")
    _check(clients, n)
    return clients, labels


def _dirichlet(labels: np.ndarray, m: int, alpha: float, rng) -> list[list[int]]:
    K = int(labels.max()) + 1
    pools = [list(rng.permutation(np.where(labels == k)[0])) for k in range(K)]
    clients = []
    for _ in range(len(labels) // m):
        p = rng.dirichlet(np.full(K, alpha))
        members = []
        for _ in range(m):
            avail = np.array([len(pl) > 0 for pl in pools], dtype=float)
            q = p * avail
            q = q / q.sum() if q.sum() > 0 else avail / avail.sum()
            k = int(rng.choice(K, p=q))
            members.append(int(pools[k].pop()))
        clients.append(sorted(members))
    return clients


def _check(clients, n):
    flat = sorted(i for c in clients for i in c)
    assert flat == list(range(n)), "partition must use every household exactly once"
