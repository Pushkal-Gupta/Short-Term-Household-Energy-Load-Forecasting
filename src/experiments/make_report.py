"""Build tables, statistics and plots from finished runs.

    python -m src.experiments.make_report                          # real results -> results/ and paper/generated/
    python -m src.experiments.make_report --config configs/demo.yaml
"""
from src.config import base_parser, config_from_args
from src.evaluation.report import build_report
from src.utils import results_path, setup_logging


def main(argv=None):
    setup_logging()
    args = base_parser(__doc__).parse_args(argv)
    cfg = config_from_args(args)
    out = build_report(cfg)
    print(out["comparison"].to_string(index=False))
    print("\nwritten:", results_path(cfg, "metrics", "comparison.md"))


if __name__ == "__main__":
    main()
