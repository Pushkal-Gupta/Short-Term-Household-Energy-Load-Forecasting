"""Configuration loading: YAML file (+ optional `base:` inheritance) and `key.sub=value` overrides."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path = "configs/default.yaml", overrides: list[str] | None = None) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        cfg = _merge(base, cfg)
    for item in overrides or []:
        set_by_path(cfg, item)
    return cfg


def set_by_path(cfg: dict, assignment: str) -> None:
    """Apply one `a.b.c=value` override; the value is parsed as YAML (numbers, bools, null, lists)."""
    if "=" not in assignment:
        raise ValueError(f"Override must look like key.sub=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def with_overrides(cfg: dict, **dotted) -> dict:
    """Return a copy of cfg with python-side overrides, e.g. with_overrides(cfg, **{'fed.mu': 0.1})."""
    out = copy.deepcopy(cfg)
    for k, v in dotted.items():
        node = out
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return out


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def base_parser(description: str, default_config: str = "configs/default.yaml") -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=default_config, help="YAML config file")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. --set fed.rounds=20 fed.mu=0.1")
    ap.add_argument("--sample", action="store_true",
                    help="small-scale test mode: 10 households, few epochs/rounds (real data)")
    return ap


def config_from_args(args) -> dict:
    cfg = load_config(args.config, args.set)
    if getattr(args, "sample", False):
        cfg = with_overrides(cfg, **{
            "data.num_households": 10,
            # separate cache, so a quick check never replaces the full cohort in processed_dir
            "data.processed_dir": str(Path(cfg["data"]["processed_dir"]) / "sample"), "partition.households_per_client": 2,
            "train.max_epochs": 3, "fed.rounds": 3, "seeds": [cfg.get("seed", 42)],
            "experiment_tag": f"{cfg.get('experiment_tag', 'run')}-sample",
            "results_dir": str(Path(cfg.get("results_dir", "results")) / "sample"),
        })
    return cfg
