"""End-to-end software check on a tiny SYNTHETIC dataset. No download is needed.

    python -m src.experiments.run_demo                # ~2-4 minutes on a laptop CPU
    python -m src.experiments.run_demo --workers 1    # sequential

Steps it exercises: synthetic LCL-format CSV -> cleaning -> household selection -> windows ->
local / centralised / naive baselines -> FedAvg, FedProx and HA-FedProx under IID, Dirichlet and
cluster partitions -> partial participation and stragglers -> float16 updates -> household
clients -> mu tuning on validation -> tables, statistics and plots.

Everything goes to results/demo/ and data/synthetic_demo/, and every plot is stamped
"SYNTHETIC DEMO DATA". These numbers only show that the software runs; they are NOT
project results.
"""
import argparse
import shutil

from src.config import resolve
from src.experiments import make_report, run_matrix


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="do not wipe previous demo output first")
    args = ap.parse_args(argv)
    if not args.keep:
        for p in ("results/demo", "data/synthetic_demo"):
            shutil.rmtree(resolve(p), ignore_errors=True)
    run_matrix.main(["--config", "configs/demo.yaml", "--workers", str(args.workers)])
    make_report.main(["--config", "configs/demo.yaml"])
    print("\nDEMO OK. Synthetic-data output in results/demo/ (NOT project results).")


if __name__ == "__main__":
    main()
