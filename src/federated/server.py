"""Deterministic in-process federated simulation (FedAvg / FedProx / HA-FedProx).

One round t:
 1. sample ceil(client_fraction * K) clients (seeded)                 -> partial participation
 2. send them the global weights                                      -> download bytes
 3. each selected client is a straggler with probability straggler_prob and then completes
    only U(straggler_min_work, 1) of its local work. FedAvg drops stragglers; FedProx and
    HA-FedProx keep their partial updates (Li et al., 2020).
 4. surviving clients train locally (FedProx adds (mu/2)||w - w_g||^2) -> upload bytes
 5. sample-weighted aggregation  W = sum n_k W_k / sum n_k
 6. the global model is evaluated on every household's validation windows (pooled, kWh).
    The weights with the lowest validation MAE are kept. Test metrics are logged each round
    for the convergence plots only and are never used for any decision.
 7. HA-FedProx updates mu from the dissimilarity of the round's updates.

The server only ever sees weights, sample counts and scalar losses. It never sees a
household's readings. That is data locality, NOT a formal privacy guarantee: model
updates can still leak information. See the limitations in README.md.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch

from src.evaluation.evaluate import evaluate_model
from src.federated.comm import CommLedger, quantize
from src.federated.fedavg import update_dissimilarity, weighted_average
from src.federated.fedprox import MuSchedule
from src.models.lstm import build_model, count_parameters, get_weights, set_weights


class FederatedServer:
    def __init__(self, clients, households, cfg: dict, device, seed: int, verbose: bool = True):
        self.clients = clients
        self.households = households          # used ONLY for server-side evaluation of the global model
        self.cfg = cfg
        self.device = device
        self.seed = seed
        self.verbose = verbose
        f = cfg["fed"]
        self.algorithm = f["algorithm"]
        if self.algorithm not in ("fedavg", "fedprox", "hafedprox"):
            raise ValueError(f"unknown fed.algorithm {self.algorithm!r}")
        drop = f.get("drop_stragglers")
        self.drop_stragglers = (self.algorithm == "fedavg") if drop is None else bool(drop)
        self.mu = MuSchedule(self.algorithm, f.get("mu", 0.0), f.get("ha_mu_min", 1e-4), f.get("ha_mu_max", 1.0))
        torch.manual_seed(seed)
        self.model = build_model(cfg).to(device)
        self.num_params = count_parameters(self.model)
        self.precision = f.get("update_precision", "float32")
        self.ledger = CommLedger(self.num_params, self.precision)
        self.rng = np.random.default_rng(seed)

    def run(self):
        f = self.cfg["fed"]
        K = len(self.clients)
        n_sel = max(min(int(f.get("min_clients", 1)), K), math.ceil(float(f["client_fraction"]) * K))
        global_w = get_weights(self.model)
        best = {"val_mae": float("inf"), "round": 0, "weights": copy.deepcopy(global_w)}
        history = []
        for rnd in range(1, int(f["rounds"]) + 1):
            selected = sorted(self.rng.choice(K, size=n_sel, replace=False).tolist())
            is_straggler = self.rng.random(n_sel) < float(f.get("straggler_prob", 0.0))
            work = np.where(is_straggler, self.rng.uniform(float(f.get("straggler_min_work", 0.1)), 1.0, n_sel), 1.0)
            sent = quantize(global_w, self.precision)
            mu_t = self.mu.current
            results, losses = [], []
            for j, cid in enumerate(selected):
                if is_straggler[j] and self.drop_stragglers:
                    continue
                cfg_fit = {"local_epochs": f["local_epochs"], "mu": mu_t, "work_fraction": float(work[j]),
                           "seed": self.seed * 100_003 + rnd * 1_009 + cid}
                w_k, n_k, met = self.clients[cid].fit([w.copy() for w in sent], cfg_fit)
                results.append((quantize(w_k, self.precision), n_k))
                losses.append((met["train_loss"], n_k))
            comm = self.ledger.round(n_download=len(selected), n_upload=len(results))
            dissim = float("nan")
            if results:
                dissim = update_dissimilarity(sent, results) if len(results) > 1 else 0.0
                global_w = weighted_average(results)
                self.mu.update(dissim)
            set_weights(self.model, global_w)
            val, _, _ = evaluate_model(self.model, self.households, "val", self.device, self.cfg)
            test, _, _ = evaluate_model(self.model, self.households, "test", self.device, self.cfg)
            train_loss = (sum(l * n for l, n in losses) / sum(n for _, n in losses)) if losses else float("nan")
            row = {"round": rnd, "algorithm": self.algorithm, "mu": mu_t, "clients_selected": len(selected),
                   "clients_returned": len(results), "stragglers": int(is_straggler.sum()),
                   "train_loss": train_loss, "update_disagreement_rho": dissim,
                   "val_loss": val["rmse"] ** 2, "val_mae": val["mae"], "val_rmse": val["rmse"],
                   "val_mape": val["mape"], "val_wape": val["wape"],
                   "test_mae": test["mae"], "test_rmse": test["rmse"], "test_mape": test["mape"],
                   "test_wape": test["wape"], **comm}
            history.append(row)
            if val["mae"] < best["val_mae"]:
                best = {"val_mae": val["mae"], "round": rnd, "weights": copy.deepcopy(global_w),
                        "cumulative_bytes": comm["cumulative_bytes"]}
            if self.verbose:
                print(f"[{self.algorithm}] round {rnd:3d}  clients {len(results)}/{len(selected)}  "
                      f"mu {mu_t:.4g}  rho {dissim:.3f}  val_mae {val['mae']:.4f}  "
                      f"MB {comm['cumulative_bytes'] / 1e6:.2f}")
        set_weights(self.model, best["weights"])
        return self.model, history, best
