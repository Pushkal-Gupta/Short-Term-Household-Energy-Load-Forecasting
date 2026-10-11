"""Run the whole experiment matrix (configs/matrix.yaml) on several seeds, in parallel.

    python -m src.experiments.run_matrix                          # tune phase, then main phase
    python -m src.experiments.run_matrix --phase tune
    python -m src.experiments.run_matrix --phase main --workers 8
    python -m src.experiments.run_matrix --only E4_fedavg_moderate E5_fedprox_moderate
    python -m src.experiments.run_matrix --sample --workers 2     # 10 households, quick check

Finished runs (their summary JSON exists) are skipped, so an interrupted matrix can be resumed
by re-running the same command. After the main phase, run `python -m src.experiments.make_report`.
"""
from __future__ import annotations

import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from src.config import base_parser, config_from_args, resolve, set_by_path
from src.data.preprocess import DatasetNotFoundError, run_preprocessing
from src.utils import results_path, setup_logging


def _apply(cfg: dict, overrides: dict, tuning: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        if isinstance(v, str) and v.startswith("@"):
            if v[1:] not in tuning:
                raise KeyError(f"{v} needs the tune phase first (missing in tuning.json)")
            v = tuning[v[1:]]
        set_by_path(cfg, f"{k}={json.dumps(v)}")
    return cfg


def _job(args):
    cfg, method, seed, name, threads = args
    import torch
    torch.set_num_threads(threads)
    from src.experiments.common import run_experiment
    save_pred = seed == cfg["seeds"][0]
    s = run_experiment(cfg, method, seed, name, save_predictions=save_pred, verbose=False)
    return name, seed, s["test_mae"], s["runtime_s"]


def run_phase(cfg, entries, seeds, tuning, workers, only=None):
    jobs = []
    for e in entries:
        if only and e["name"] not in only:
            continue
        for seed in e.get("seeds", seeds):                 # an entry may carry its own seed list
            out = results_path(cfg, "metrics", "runs", f"{e['name']}__seed{seed}.json")
            if out.exists():
                continue
            jobs.append((_apply(cfg, e.get("set"), tuning), e["method"], seed, e["name"], 1))
    print(f"{len(jobs)} run(s) to do with {workers} worker(s)")
    if not jobs:
        return
    t0 = time.time()
    if workers <= 1:
        for j in jobs:
            print("done", *_job(j))
        return
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                name, seed, mae, rt = f.result()
                print(f"[{k}/{len(jobs)}] {name} seed {seed}: test MAE {mae:.4f} ({rt:.0f} s) "
                      f"| elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
            except Exception as exc:  # keep the rest of the matrix running
                j = futs[f]
                print(f"[{k}/{len(jobs)}] FAILED {j[3]} seed {j[2]}: {exc!r}", flush=True)


def select_tuning(cfg, entries) -> dict:
    """Pick mu (FedProx) and mu_0 (HA-FedProx) with the lowest VALIDATION MAE."""
    seed = cfg["seeds"][0]
    best = {}
    for e in entries:
        p = results_path(cfg, "metrics", "runs", f"{e['name']}__seed{seed}.json")
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        key = f"mu_{s['algorithm']}"
        if key not in best or s["best_val_mae"] < best[key][1]:
            best[key] = (s["mu"], s["best_val_mae"], e["name"])
    tuning = {k: v[0] for k, v in best.items()}
    tuning["_selection"] = {k: {"value": v[0], "val_mae": v[1], "run": v[2]} for k, v in best.items()}
    tuning["_criterion"] = "lowest best-round validation MAE, first seed, strong non-IID (cluster) partition"
    return tuning


def main(argv=None):
    setup_logging()
    ap = base_parser(__doc__)
    ap.add_argument("--matrix", default="configs/matrix.yaml")
    ap.add_argument("--phase", choices=["tune", "main", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    matrix = yaml.safe_load(resolve(args.matrix).read_text())
    try:
        run_preprocessing(cfg)          # once, before the workers start
    except DatasetNotFoundError as e:
        raise SystemExit(str(e))
    tpath = results_path(cfg, "metrics", "tuning.json")
    if args.phase in ("tune", "all"):
        run_phase(cfg, matrix["tune"], cfg["seeds"][:1], {}, args.workers, args.only)
        tuning = select_tuning(cfg, matrix["tune"])
        tpath.write_text(json.dumps(tuning, indent=2))
        print("tuning:", json.dumps(tuning["_selection"], indent=2))
    if args.phase in ("main", "all"):
        tuning = json.loads(tpath.read_text()) if tpath.exists() else {}
        run_phase(cfg, matrix["main"], cfg["seeds"], tuning, args.workers, args.only)


if __name__ == "__main__":
    main()
