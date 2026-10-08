"""Small shared helpers: seeding, device selection, logging, result I/O."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.config import resolve

log = logging.getLogger("fedlstm")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(name: str = "auto") -> torch.device:
    """`auto` picks CUDA, then Apple MPS, then CPU. Bit-exact reproducibility is only guaranteed on CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; using CPU")
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        log.warning("MPS requested but not available; using CPU")
        return torch.device("cpu")
    return torch.device(name)


def results_path(cfg: dict, *parts: str) -> Path:
    p = resolve(cfg.get("results_dir", "results")).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
