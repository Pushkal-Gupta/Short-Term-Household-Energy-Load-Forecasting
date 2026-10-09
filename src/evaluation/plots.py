"""Matplotlib figures used in the README, the report and the manuscript.

Every method keeps one fixed colour in every figure, so a colour always means the same arm.
Each chart has a single y-axis, thin lines, and a recessive grid.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

COLORS = {"centralized": "#2a78d6", "fedavg": "#eb6834", "fedprox": "#1baf7a", "local": "#eda100",
          "hafedprox": "#4a3aa7", "naive": "#8a8986", "finetuned": "#e34948"}
LABELS = {"centralized": "Centralised", "fedavg": "FedAvg", "fedprox": "FedProx", "local": "Local-only",
          "hafedprox": "HA-FedProx", "naive": "Seasonal naive", "finetuned": "FedAvg + local FT"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.8, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def _legend_right(ax):
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))


def _rounds_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Communication round")


def _save(fig, path: Path, tag: str | None):
    if tag and "synthetic" in tag.lower():
        fig.text(0.99, 0.01, "SYNTHETIC DEMO DATA — NOT A RESULT", ha="right", va="bottom",
                 color="#e34948", fontsize=8, weight="bold")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def metric_vs_round(curves: dict, metric: str, ylabel: str, path: Path, title: str, tag=None,
                    reference: dict | None = None):
    """curves: label -> (algorithm key, DataFrame with columns round, <metric>_mean, <metric>_std)."""
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for label, (alg, df) in curves.items():
        m, s = df[f"{metric}_mean"].to_numpy(), df[f"{metric}_std"].fillna(0).to_numpy()
        ls = "--" if "strong" in label.lower() else "-"
        ax.plot(df["round"], m, color=COLORS[alg], ls=ls, label=label)
        ax.fill_between(df["round"], m - s, m + s, color=COLORS[alg], alpha=0.12, lw=0)
    for label, (alg, val) in (reference or {}).items():
        ax.axhline(val, color=COLORS[alg], lw=1.0, ls=":", label=label)
    _rounds_axis(ax)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def communication_vs_round(curves: dict, path: Path, tag=None):
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label, (alg, df) in curves.items():
        ax.plot(df["round"], df["cumulative_bytes_mean"] / 1e6, color=COLORS[alg],
                ls="--" if "household" in label.lower() or "partial" in label.lower() else "-", label=label)
    _rounds_axis(ax)
    ax.set_ylabel("Cumulative traffic (MB, up + down)")
    ax.set_title("Communication cost per round", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def error_vs_bytes(curves: dict, path: Path, central_mae: float | None, tag=None):
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for label, (alg, df) in curves.items():
        ax.plot(df["cumulative_bytes_mean"] / 1e6, df["val_mae_mean"], color=COLORS[alg],
                ls="--" if "strong" in label.lower() else "-", label=label)
    if central_mae is not None:
        ax.axhline(central_mae, color=COLORS["centralized"], ls=":", lw=1.0, label="Centralised (val)")
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative traffic (MB, log scale)")
    ax.set_ylabel("Validation MAE (kWh / half-hour)")
    ax.set_title("Accuracy bought per megabyte", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def prediction_example(frames: dict, hid: str, path: Path, days: int = 3, tag=None):
    """frames: label -> (algorithm key, predictions DataFrame with LCLid, timestamp, y_kwh, yhat_kwh)."""
    style()
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    first = None
    for label, (alg, df) in frames.items():
        d = df[df["LCLid"] == hid].sort_values("timestamp")
        d = d[d["timestamp"] < d["timestamp"].iloc[0] + pd.Timedelta(days=days)]
        if first is None:
            first = d
            ax.plot(d["timestamp"], d["y_kwh"], color=INK, lw=1.4, label="Actual")
        ax.plot(d["timestamp"], d["yhat_kwh"], color=COLORS[alg], lw=1.1, label=label)
    ax.set_ylabel("kWh per half-hour")
    ax.set_title(f"Actual vs predicted load — household {hid}, first {days} test days", loc="left")
    _legend_right(ax)
    fig.autofmt_xdate()
    _save(fig, path, tag)


def heterogeneity_response(df: pd.DataFrame, path: Path, central_mae: float, tag=None):
    """df rows: algorithm, H_mean, test_mae_mean, test_mae_std, label."""
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for alg, g in df.groupby("algorithm"):
        g = g.sort_values("H_mean")
        ax.errorbar(g["H_mean"], g["test_mae_mean"], yerr=g["test_mae_std"], color=COLORS[alg],
                    marker="o", ms=5, capsize=2, lw=1.6, label=LABELS[alg])
    for _, r in df[df["algorithm"] == "fedavg"].iterrows():
        ax.annotate(r["label"], (r["H_mean"], r["test_mae_mean"]), textcoords="offset points",
                    xytext=(4, 6), fontsize=7, color=MUTED)
    ax.axhline(central_mae, color=COLORS["centralized"], ls=":", lw=1.0, label="Centralised")
    ax.set_xlabel("Measured heterogeneity index H between clients")
    ax.set_ylabel("Test MAE (kWh / half-hour)")
    ax.set_title("Heterogeneity–error response", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def mu_trajectory(curves: dict, path: Path, tag=None):
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    shades = ["#9a8fe0", "#7565cc", "#4a3aa7", "#30237a", "#1a1150"]
    for (label, df), c in zip(curves.items(), shades):
        ax.plot(df["round"], df["mu_mean"], color=c, label=label)
    _rounds_axis(ax)
    ax.set_ylabel("Proximal coefficient μ_t")
    ax.set_title("HA-FedProx: μ chosen by the server each round", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def equity_by_stratum(df: pd.DataFrame, path: Path, tag=None):
    """df rows: stratum, arm (algorithm key), mae_median. Grouped bars, one group per consumption quintile."""
    style()
    arms = list(dict.fromkeys(df["arm"]))
    strata = sorted(df["stratum"].unique())
    width = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, arm in enumerate(arms):
        g = df[df["arm"] == arm].set_index("stratum").reindex(strata)
        ax.bar(np.arange(len(strata)) + i * width - 0.4 + width / 2, g["mae_median"], width * 0.92,
               color=COLORS[arm], label=LABELS[arm], edgecolor="white", linewidth=0.8)
    ax.set_xticks(np.arange(len(strata)), [f"Q{s + 1}" for s in strata])
    ax.set_xlabel("Household consumption quintile (train-period mean load, Q1 = lowest)")
    ax.set_ylabel("Median household test MAE (kWh)")
    ax.set_title("Who gains from collaboration?", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def benefit_distribution(benefits: dict, path: Path, tag=None):
    """benefits: label -> (algorithm key, array of per-household MAE_local - MAE_fed)."""
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for label, (alg, b) in benefits.items():
        x = np.sort(b)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLORS[alg], label=label)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Federation benefit per household: MAE_local − MAE_fed (kWh)")
    ax.set_ylabel("Share of households (CDF)")
    ax.set_title("Left of zero = household is worse off than training alone", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def rho_vs_round(curves: dict, path: Path, tag=None):
    """curves: label -> DataFrame with round, update_disagreement_rho_mean/_std (FedAvg runs)."""
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    shades = ["#f6b48a", "#eb6834", "#b8461c", "#6e2508", "#2a78d6"]
    for (label, df), c in zip(curves.items(), shades):
        m = df["update_disagreement_rho_mean"].to_numpy()
        s = df["update_disagreement_rho_std"].fillna(0).to_numpy()
        ax.plot(df["round"], m, color=c, label=label)
        ax.fill_between(df["round"], m - s, m + s, color=c, alpha=0.12, lw=0)
    _rounds_axis(ax)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Update disagreement ρ_t")
    ax.set_title("ρ_t follows training progress, not the partition", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)


def fragmentation(df: pd.DataFrame, path: Path, refs: dict, tag=None):
    """df rows: K, test_mae_mean, test_mae_std, ft_mae_mean (FedAvg, random grouping / household clients)."""
    style()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    df = df.sort_values("K")
    ax.errorbar(df["K"], df["test_mae_mean"], yerr=df["test_mae_std"].fillna(0), color=COLORS["fedavg"],
                marker="o", ms=5, capsize=2, label="FedAvg (global model)")
    if "ft_mae_mean" in df and df["ft_mae_mean"].notna().any():
        ax.plot(df["K"], df["ft_mae_mean"], color=COLORS["finetuned"], marker="s", ms=4, ls="--",
                label="FedAvg + 1-epoch local FT")
    for label, (alg, val) in refs.items():
        ax.axhline(val, color=COLORS[alg], lw=1.0, ls=":", label=label)
    ax.set_xscale("log")
    ax.set_xticks(df["K"], [str(int(k)) for k in df["K"]])
    ax.set_xlabel("Number of clients K holding the same 100 homes (log scale)")
    ax.set_ylabel("Test MAE (kWh / half-hour)")
    ax.set_title("Cost of fragmenting the same data", loc="left")
    _legend_right(ax)
    _save(fig, path, tag)
