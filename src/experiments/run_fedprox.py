"""Run one FedProx (or HA-FedProx) experiment.

    python -m src.experiments.run_fedprox --set fed.mu=0.01
    python -m src.experiments.run_fedprox --adaptive                        # HA-FedProx (mu re-tuned per round)
    python -m src.experiments.run_fedprox --set partition.strategy=cluster  # strong non-IID
"""
import sys

from src.experiments.common import run_single

if __name__ == "__main__":
    argv = sys.argv[1:]
    adaptive = "--adaptive" in argv
    argv = [a for a in argv if a != "--adaptive"]
    run_single("federated", argv, algorithm="hafedprox" if adaptive else "fedprox")
