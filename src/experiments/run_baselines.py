"""Run the non-federated reference arms: naive forecasts (E0), local-only (E1), centralised (E2).

    python -m src.experiments.run_baselines                 # all three
    python -m src.experiments.run_baselines --only local
    python -m src.experiments.run_baselines --sample        # 10 households, quick check
"""
from src.config import base_parser, config_from_args
from src.data.preprocess import DatasetNotFoundError
from src.experiments.common import run_experiment
from src.utils import setup_logging

ARMS = {"naive": "E0_naive", "local": "E1_local", "centralized": "E2_centralized"}


def main(argv=None):
    setup_logging()
    ap = base_parser(__doc__)
    ap.add_argument("--only", choices=list(ARMS), default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    try:
        for method, name in ARMS.items():
            if args.only in (None, method):
                run_experiment(cfg, method, seed, name)
    except DatasetNotFoundError as e:
        raise SystemExit(str(e))


if __name__ == "__main__":
    main()
