"""A simulated smart-meter client.

A client owns the training windows of one household (household mode) or of a few households
(community mode). Those windows never leave the object: the server only calls `fit` and
`evaluate`, with model weights going in and model weights plus scalar statistics coming out.

The method signatures follow Flower's `NumPyClient` (get_parameters / fit / evaluate),
so src/federated/flower_backend.py can hand the same object to Flower unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from src.centralized.train import train_epochs
from src.data.window import stack
from src.evaluation.evaluate import predict_households, pooled_metrics
from src.models.lstm import build_model, get_weights, set_weights


class HouseholdClient:
    def __init__(self, cid: int, households, cfg: dict, device):
        self.cid = cid
        self.households = households
        self.cfg = cfg
        self.device = device
        self._X, self._y = stack(households, "train")   # local, private training data
        self.num_train = len(self._y)
        self.model = build_model(cfg).to(device)

    # --- Flower NumPyClient interface ---------------------------------------------------
    def get_parameters(self, config=None) -> list:
        return get_weights(self.model)

    def fit(self, parameters: list, config: dict):
        """Train locally starting from the global weights.

        config: local_epochs, mu (proximal coefficient, 0 = FedAvg), work_fraction (straggler), seed
        Returns (new weights, number of training windows, metrics dict).
        """
        set_weights(self.model, parameters)
        mu = float(config.get("mu", 0.0))
        prox_ref = [p.detach().clone() for p in self.model.parameters()] if mu > 0 else None
        epochs = float(config.get("local_epochs", 1)) * float(config.get("work_fraction", 1.0))
        rng = np.random.default_rng(int(config.get("seed", 0)))
        torch.manual_seed(int(config.get("seed", 0)))
        loss = train_epochs(self.model, self._X, self._y, self.device, self.cfg, rng, epochs,
                            prox_ref=prox_ref, mu=mu)
        return get_weights(self.model), self.num_train, {"train_loss": loss, "epochs_done": epochs}

    def evaluate(self, parameters: list, config: dict | None = None):
        set_weights(self.model, parameters)
        split = (config or {}).get("split", "val")
        pairs = predict_households(self.model, self.households, split, self.device)
        m = pooled_metrics(pairs, self.cfg["eval"]["mape_min_actual"])
        return float(m["rmse"] ** 2), m["n_points"], {"mae": m["mae"], "rmse": m["rmse"]}
