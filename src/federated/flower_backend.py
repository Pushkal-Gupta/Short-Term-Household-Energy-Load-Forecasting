"""Optional Flower backend: the same clients and the same round logic, driven through Flower's own
`NumPyClient` -> `Client` wrapper, its wire types (FitIns/FitRes/Parameters) and its
`FedAvg` / `FedProx` aggregation strategies.

    python -m src.experiments.run_fedavg --backend flower
    python -m src.experiments.run_fedprox --backend flower

Flower's multi-process simulation engine needs Ray, which is not required here. The backend
therefore calls Flower's client and strategy objects in-process, one client after another, so
no network or Ray is involved. Upload and download bytes are measured from the actual
serialized `Parameters` tensors, which includes Flower's numpy serialization header, and are
logged next to the analytic payload size.
Requires:  pip install -r requirements-flower.txt
"""
from __future__ import annotations

import copy
import math

import numpy as np

from src.evaluation.evaluate import evaluate_model
from src.federated.comm import CommLedger, quantize
from src.federated.fedavg import update_dissimilarity
from src.federated.fedprox import MuSchedule
from src.models.lstm import build_model, count_parameters, get_weights, set_weights


def _flower():
    try:
        import flwr  # noqa: F401
        from flwr.client import NumPyClient
        from flwr.common import Code, FitIns, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
        from flwr.server.strategy import FedAvg, FedProx
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Flower is not installed: pip install -r requirements-flower.txt") from e
    return NumPyClient, Code, FitIns, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays, FedAvg, FedProx


def run_flower(clients, households, cfg, device, seed, verbose=True):
    NumPyClient, Code, FitIns, FitRes, Status, to_params, from_params, FedAvg, FedProx = _flower()

    class FlowerHousehold(NumPyClient):
        """Adapter: Flower calls these methods; they delegate to the HouseholdClient."""
        def __init__(self, inner):
            self.inner = inner

        def get_parameters(self, config):
            return self.inner.get_parameters(config)

        def fit(self, parameters, config):
            cfg_fit = dict(config)
            cfg_fit["mu"] = float(config.get("proximal_mu", 0.0))
            return self.inner.fit(parameters, cfg_fit)

        def evaluate(self, parameters, config):
            return self.inner.evaluate(parameters, config)

    class _Proxy:  # minimal stand-in for flwr's ClientProxy in aggregate_fit's (proxy, FitRes) pairs
        def __init__(self, cid):
            self.cid = str(cid)

    f = cfg["fed"]
    algorithm = f["algorithm"]
    precision = f.get("update_precision", "float32")
    drop = f.get("drop_stragglers")
    drop_stragglers = (algorithm == "fedavg") if drop is None else bool(drop)
    mu = MuSchedule(algorithm, f.get("mu", 0.0), f.get("ha_mu_min", 1e-4), f.get("ha_mu_max", 1.0))
    import torch
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    global_w = get_weights(model)
    strategy = FedAvg() if algorithm == "fedavg" else FedProx(proximal_mu=mu.current)
    fl_clients = [FlowerHousehold(c).to_client() for c in clients]
    ledger = CommLedger(count_parameters(model), precision)
    rng = np.random.default_rng(seed)
    K = len(clients)
    n_sel = max(min(int(f.get("min_clients", 1)), K), math.ceil(float(f["client_fraction"]) * K))
    best = {"val_mae": float("inf"), "round": 0, "weights": copy.deepcopy(global_w), "cumulative_bytes": 0}
    history, wire_cum = [], 0
    for rnd in range(1, int(f["rounds"]) + 1):
        selected = sorted(rng.choice(K, size=n_sel, replace=False).tolist())
        is_straggler = rng.random(n_sel) < float(f.get("straggler_prob", 0.0))
        work = np.where(is_straggler, rng.uniform(float(f.get("straggler_min_work", 0.1)), 1.0, n_sel), 1.0)
        sent = quantize(global_w, precision)
        params = to_params([w.astype(np.float16) if precision == "float16" else w for w in sent])
        down_wire = sum(len(t) for t in params.tensors)
        mu_t = mu.current
        if algorithm != "fedavg":
            strategy.proximal_mu = mu_t
        results, raw, up_wire = [], [], 0
        for j, cid in enumerate(selected):
            if is_straggler[j] and drop_stragglers:
                continue
            config = {"local_epochs": float(f["local_epochs"]), "work_fraction": float(work[j]),
                      "seed": seed * 100_003 + rnd * 1_009 + cid}
            if algorithm != "fedavg":
                config["proximal_mu"] = float(mu_t)
            res = fl_clients[cid].fit(FitIns(params, config))
            w_k = [a.astype(np.float32) for a in from_params(res.parameters)]
            w_k = quantize(w_k, precision)
            up = to_params([w.astype(np.float16) if precision == "float16" else w for w in w_k])
            up_wire += sum(len(t) for t in up.tensors)
            results.append((_Proxy(cid), FitRes(Status(Code.OK, ""), up, res.num_examples, res.metrics)))
            raw.append((w_k, res.num_examples))
        comm = ledger.round(len(selected), len(results))
        wire_cum += down_wire * len(selected) + up_wire
        rho = float("nan")
        if results:
            agg, _ = strategy.aggregate_fit(rnd, results, [])
            global_w = [a.astype(np.float32) for a in from_params(agg)]
            rho = update_dissimilarity(sent, raw) if len(raw) > 1 else 0.0
            mu.update(rho)
        set_weights(model, global_w)
        val, _, _ = evaluate_model(model, households, "val", device, cfg)
        test, _, _ = evaluate_model(model, households, "test", device, cfg)
        history.append({"round": rnd, "algorithm": algorithm, "backend": "flower", "mu": mu_t,
                        "clients_selected": len(selected), "clients_returned": len(results),
                        "stragglers": int(is_straggler.sum()), "update_disagreement_rho": rho,
                        "val_loss": val["rmse"] ** 2, "val_mae": val["mae"], "val_rmse": val["rmse"],
                        "val_mape": val["mape"], "val_wape": val["wape"], "test_mae": test["mae"],
                        "test_rmse": test["rmse"], "test_mape": test["mape"], "test_wape": test["wape"],
                        "flower_wire_cumulative_bytes": wire_cum, **comm})
        if val["mae"] < best["val_mae"]:
            best = {"val_mae": val["mae"], "round": rnd, "weights": copy.deepcopy(global_w),
                    "cumulative_bytes": comm["cumulative_bytes"]}
        if verbose:
            print(f"[flower:{algorithm}] round {rnd:3d}  clients {len(results)}/{len(selected)}  mu {mu_t:.4g}  "
                  f"val_mae {val['mae']:.4f}  payload MB {comm['cumulative_bytes'] / 1e6:.2f}  "
                  f"flower wire MB {wire_cum / 1e6:.2f}")
    set_weights(model, best["weights"])
    return model, history, best
