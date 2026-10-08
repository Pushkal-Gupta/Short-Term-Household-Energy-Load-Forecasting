"""One function that runs any arm and writes its results. Every runner script goes through it.

Output files of one run (name = experiment id, e.g. E4_fedavg_moderate):
  <results>/metrics/runs/<name>__seed<s>.json             summary (pooled test metrics, bytes, H, ...)
  <results>/metrics/runs/<name>__seed<s>_history.csv      per round (federated) or per epoch
  <results>/metrics/runs/<name>__seed<s>_households.csv   per-household test metrics
  <results>/predictions/<name>__seed<s>.parquet           test predictions (y, yhat in kWh)
  <results>/models/<name>__seed<s>.pt                     trained global / centralised weights
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from src.baselines.local import run_local
from src.baselines.naive import naive_pairs
from src.centralized.train import run_centralized, train_epochs
from src.config import base_parser, config_from_args
from src.data.partition import partition_households
from src.data.preprocess import DatasetNotFoundError, load_processed
from src.data.profiles import heterogeneity_index
from src.data.window import build_households
from src.evaluation.evaluate import (evaluate_model, per_household_metrics, pooled_metrics,
                                     predict_households, predictions_frame, summarise_household_metrics)
from src.federated.client import HouseholdClient
from src.federated.comm import raw_data_bytes
from src.federated.server import FederatedServer
from src.models.lstm import build_model, count_parameters, get_weights, set_weights
from src.utils import get_device, results_path, save_json, set_seed, setup_logging

_CACHE: dict = {}


def load_households(cfg: dict):
    d, w = cfg["data"], cfg["window"]
    key = (tuple(sorted((k, str(v)) for k, v in d.items())), tuple(sorted((k, str(v)) for k, v in w.items())))
    if key not in _CACHE:
        series, meta = load_processed(cfg)
        _CACHE.clear()
        _CACHE[key] = (build_households(series, cfg), meta)
    return _CACHE[key]


def run_experiment(cfg: dict, method: str, seed: int, name: str, save_predictions: bool = True,
                   verbose: bool = True) -> dict:
    """method: naive | local | centralized | federated (algorithm taken from cfg['fed'])."""
    set_seed(seed)
    device = get_device(cfg["train"].get("device", "auto"))
    households, meta = load_households(cfg)
    mn = cfg["eval"]["mape_min_actual"]
    tag = f"{name}__seed{seed}"
    t0 = time.perf_counter()
    summary = {"experiment": name, "method": method, "seed": seed, "data_tag": cfg.get("experiment_tag"),
               "num_households": len(households), "input_window": cfg["window"]["input_window"],
               "horizon": cfg["window"]["horizon"], "hidden_size": cfg["model"]["hidden_size"],
               "num_layers": cfg["model"]["num_layers"],
               "num_params": count_parameters(build_model(cfg)),
               "train_windows": int(sum(hd.n("train") for hd in households)),
               "test_windows": int(sum(hd.n("test") for hd in households))}
    history, model, extra_pairs = [], None, {}

    if method == "naive":
        for kind in ("persistence", "seasonal_naive"):
            extra_pairs[kind] = naive_pairs(households, kind, cfg)
        pairs = extra_pairs["seasonal_naive"]
        summary.update({"algorithm": "seasonal_naive", "rounds": 0, "total_bytes": 0})
        for kind, pr in extra_pairs.items():
            summary.update({f"{kind}_{k}": v for k, v in pooled_metrics(pr, mn).items()})
    elif method == "local":
        pairs, hists = run_local(households, cfg, device, seed, verbose)
        summary.update({"algorithm": "local", "rounds": 0, "total_bytes": 0,
                        "mean_epochs": float(np.mean([len(h) for h in hists]))})
        history = [{"LCLid": hd.hid, **h} for hd, hs in zip(households, hists) for h in hs]
    elif method == "centralized":
        model, history = run_centralized(households, cfg, device, seed, verbose)
        _, _, pairs = evaluate_model(model, households, "test", device, cfg)
        summary.update({"algorithm": "centralized", "rounds": 0, "epochs": len(history),
                        "best_val_mae": min(h["val_mae"] for h in history),
                        "total_bytes": raw_data_bytes(households), "bytes_note": "raw training data pooled"})
    elif method == "federated":
        model, history, best, part = _run_federated(households, cfg, device, seed, verbose)
        _, _, pairs = evaluate_model(model, households, "test", device, cfg)
        f = cfg["fed"]
        summary.update({"algorithm": f["algorithm"], "rounds": int(f["rounds"]), "best_round": best["round"],
                        "best_val_mae": best["val_mae"], "local_epochs": f["local_epochs"],
                        "client_fraction": f["client_fraction"], "mu": f.get("mu"),
                        "straggler_prob": f.get("straggler_prob", 0.0),
                        "update_precision": f.get("update_precision", "float32"),
                        "total_bytes": history[-1]["cumulative_bytes"],
                        "bytes_to_best_round": best["cumulative_bytes"],
                        "payload_bytes": history[-1]["payload_bytes"], **part})
        if int(f.get("finetune_epochs", 0)) > 0:
            ft_pairs = _finetune_eval(model, households, cfg, device, seed)
            extra_pairs["finetuned"] = ft_pairs
            summary.update({f"ft_{k}": v for k, v in pooled_metrics(ft_pairs, mn).items()})
    else:
        raise ValueError(method)

    pooled = pooled_metrics(pairs, mn)
    hh = per_household_metrics(households, pairs, mn)
    if "finetuned" in extra_pairs:
        ft = per_household_metrics(households, extra_pairs["finetuned"], mn)
        hh["ft_mae"], hh["ft_wape"] = ft["mae"].values, ft["wape"].values
    hh = hh.merge(meta[["LCLid", "train_mean_kwh", "stratum"]], on="LCLid", how="left")
    if "profile_cluster" in summary:
        hh["profile_cluster"] = summary.pop("profile_cluster")
    if "client_of" in summary:
        hh["client"] = summary.pop("client_of")
    summary.update({f"test_{k}": v for k, v in pooled.items()})
    summary.update(summarise_household_metrics(hh))
    summary["runtime_s"] = round(time.perf_counter() - t0, 1)

    runs = results_path(cfg, "metrics", "runs", f"{tag}.json")
    save_json(summary, runs)
    pd.DataFrame(history).to_csv(runs.with_name(f"{tag}_history.csv"), index=False)
    hh.to_csv(runs.with_name(f"{tag}_households.csv"), index=False)
    if save_predictions:
        predictions_frame(households, pairs).to_parquet(results_path(cfg, "predictions", f"{tag}.parquet"))
    if model is not None:
        torch.save(model.state_dict(), results_path(cfg, "models", f"{tag}.pt"))
    if verbose:
        print(f"[{name} seed {seed}] test MAE {pooled['mae']:.4f}  RMSE {pooled['rmse']:.4f}  "
              f"MAPE {pooled['mape']:.2f}%  ({summary['runtime_s']} s)")
    return summary


def _run_federated(households, cfg, device, seed, verbose):
    if cfg["fed"].get("backend", "sim") == "flower":
        from src.federated.flower_backend import run_flower
        groups, labels = partition_households(households, cfg, seed)
        clients = [HouseholdClient(i, [households[j] for j in g], cfg, device) for i, g in enumerate(groups)]
        model, history, best = run_flower(clients, households, cfg, device, seed, verbose)
    else:
        groups, labels = partition_households(households, cfg, seed)
        clients = [HouseholdClient(i, [households[j] for j in g], cfg, device) for i, g in enumerate(groups)]
        server = FederatedServer(clients, households, cfg, device, seed, verbose)
        model, history, best = server.run()
    p = cfg["partition"]
    hidx = heterogeneity_index([[households[j] for j in g] for g in groups])
    client_of = np.empty(len(households), dtype=int)
    for c, g in enumerate(groups):
        client_of[g] = c
    part = {"partition_mode": p["mode"], "partition_strategy": p["strategy"] if p["mode"] == "community" else "natural",
            "alpha": p.get("alpha") if p["strategy"] == "dirichlet" else None,
            "num_clients": len(groups), "households_per_client": len(groups[0]), **hidx,
            "profile_cluster": labels.tolist(), "client_of": client_of.tolist()}
    return model, history, best, part


def _finetune_eval(global_model, households, cfg, device, seed):
    """FedAvg-FT: each household fine-tunes the final global model on its own train data, at zero extra communication."""
    pairs = []
    epochs = float(cfg["fed"]["finetune_epochs"])
    w = get_weights(global_model)
    model = build_model(cfg).to(device)
    for i, hd in enumerate(households):
        set_weights(model, w)
        train_epochs(model, hd.X["train"], hd.y["train"], device, cfg, np.random.default_rng(seed + 7 * i), epochs)
        pairs.extend(predict_households(model, [hd], "test", device))
    return pairs


# ───────────────────────────── CLI used by the runner scripts ─────────────────────────────

def run_single(method: str, argv=None, algorithm: str | None = None, default_name: str | None = None):
    setup_logging()
    ap = base_parser(f"Run the {algorithm or method} arm")
    ap.add_argument("--name", default=default_name or (algorithm or method))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--backend", choices=["sim", "flower"], default=None)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    if algorithm:
        cfg["fed"]["algorithm"] = algorithm
    if args.backend:
        cfg["fed"]["backend"] = args.backend
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    try:
        s = run_experiment(cfg, method, seed, args.name)
    except DatasetNotFoundError as e:
        raise SystemExit(str(e))
    return s
