"""FedProx (Li et al., 2020) and HA-FedProx, the heterogeneity-adaptive variant proposed here.

FedProx: each client k minimises
    L_k(w) + (mu / 2) * ||w - w_global||^2
The proximal term keeps the local model close to the global model during local epochs, which
limits client drift on non-IID data. Aggregation is the same sample-weighted average as FedAvg.
FedProx also accepts partial local work from stragglers instead of discarding it.
With mu = 0 and stragglers dropped, it is exactly FedAvg.

HA-FedProx: the server re-tunes mu every round from the update-disagreement ratio
rho_t in [0, 1] of the updates it already receives (fedavg.update_dissimilarity):
    mu_{t+1} = clip(mu_0 * rho_t, mu_min, mu_max)        (round 1 uses mu_0)
High disagreement between client updates (strong non-IID drift) raises mu, and agreeing
updates relax it toward plain FedAvg. Clients share nothing beyond their usual model update.
The FedProx convergence analysis needs a larger mu when the dissimilarity bound B is larger,
and rho_t = 1 - 1/B_t^2 is a bounded, per-round empirical counterpart (see fedavg.py).
mu_0 is therefore the proximal strength used when client updates cancel completely.
The FedProx paper's own adaptive heuristic changes mu from the global training loss.
HA-FedProx instead uses the geometry of the client updates.
"""
from __future__ import annotations

import numpy as np


class MuSchedule:
    def __init__(self, algorithm: str, mu: float, mu_min: float = 1e-4, mu_max: float = 1.0):
        self.algorithm = algorithm
        self.mu0 = float(mu)
        self.mu_min, self.mu_max = float(mu_min), float(mu_max)
        self.current = 0.0 if algorithm == "fedavg" else self.mu0

    def update(self, dissimilarity: float) -> float:
        if self.algorithm == "hafedprox" and np.isfinite(dissimilarity):
            self.current = float(np.clip(self.mu0 * dissimilarity, self.mu_min, self.mu_max))
        return self.current
