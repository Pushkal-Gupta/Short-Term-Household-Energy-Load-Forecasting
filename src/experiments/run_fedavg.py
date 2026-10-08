"""Run one FedAvg experiment.

    python -m src.experiments.run_fedavg                                   # IID-ish community clients
    python -m src.experiments.run_fedavg --set partition.strategy=dirichlet partition.alpha=0.3
    python -m src.experiments.run_fedavg --set partition.mode=household     # 1 household = 1 client
    python -m src.experiments.run_fedavg --set fed.client_fraction=0.5 fed.straggler_prob=0.2
    python -m src.experiments.run_fedavg --backend flower                   # same clients through Flower
"""
from src.experiments.common import run_single

if __name__ == "__main__":
    run_single("federated", algorithm="fedavg")
