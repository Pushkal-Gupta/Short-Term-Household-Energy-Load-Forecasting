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
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200) if
             ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
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
    """Two panels against the measured load-shape divergence H_JS between clients:
    (a) final test MAE, (b) megabytes needed to come within eps of centralised validation MAE."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for alg, g in df.groupby("algorithm"):
        g = g.sort_values("H_JS_mean")
        axes[0].errorbar(g["H_JS_mean"], g["test_mae_mean"], yerr=g["test_mae_std"], color=COLORS[alg],
                         marker="o", ms=4.5, capsize=2, lw=1.6, label=LABELS[alg])
        if "B_eps_MB_mean" in g and g["B_eps_MB_mean"].notna().any():
            axes[1].plot(g["H_JS_mean"], g["B_eps_MB_mean"], color=COLORS[alg], marker="o", ms=4.5, lw=1.6,
                         label=LABELS[alg])
    axes[0].axhline(central_mae, color=COLORS["centralized"], ls=":", lw=1.0, label="Centralised")
    fa = df[df["algorithm"] == "fedavg"].sort_values("H_JS_mean")
    for ax in axes:
        ax.set_xlabel("Load-shape divergence between clients, $H_{JS}$")
        top = ax.get_ylim()[1]
        for _, r in fa.iterrows():
            ax.text(r["H_JS_mean"], top, r["label"], ha="center", va="bottom", fontsize=7, color=MUTED)
    axes[0].set_ylabel("Test MAE (kWh / half-hour)")
    axes[0].set_title("(a) Final accuracy", loc="left", pad=14)
    axes[1].set_ylabel("MB to reach centralised + 10 % (B@ε)")
    axes[1].set_title("(b) Communication price of accuracy", loc="left", pad=14)
    axes[1].legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
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
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLORS[alg], label=label,
                ls="--" if "household" in label.lower() else "-")
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
    ax.set_title("Update disagreement ρ_t per round (FedAvg)", loc="left")
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


def graphical_abstract(rows: list, beps: dict, local_gap: float, path: Path):
    """Journal graphical abstract (wide format, >= 1328 x 531 px).
    rows: [(label, gap_to_central_pct, kind)] with kind 'K' (fragmentation) or 'H' (heterogeneity at K=20);
    beps: label -> MB to come within eps of centralised validation MAE."""
    style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.3, 5.6), gridspec_kw={"width_ratios": [1.5, 1.0]})
    fig.subplots_adjust(left=0.2, right=0.98, top=0.8, bottom=0.2, wspace=0.3)
    fig.suptitle("Federated LSTM load forecasting on 100 London homes: same data, same model, one factor changed "
                 "at a time", x=0.02, ha="left", fontsize=13, weight="bold", color=INK)
    rows = rows[::-1]
    shade = {"K": COLORS["fedavg"], "H": "#f6b48a"}
    vals = [g for _, g, _ in rows]
    ax1.barh(range(len(rows)), vals, color=[shade[k] for *_, k in rows], height=0.62, edgecolor="white")
    ax1.axvline(local_gap, color=COLORS["local"], ls=":", lw=1.6)
    ax1.text(local_gap, len(rows) - 1, f"  local-only models ({local_gap:+.1f} %)", color=INK, fontsize=9,
             va="center")
    ax1.set_yticks(range(len(rows)), [lab for lab, *_ in rows], fontsize=10)
    for i, v in enumerate(vals):
        ax1.text(max(v, 0) + 0.15, i, f"{v:+.1f} %", va="center", fontsize=9.5, color=INK)
    ax1.set_xlabel("FedAvg test MAE above centralised training (%)")
    ax1.set_title("Accuracy: the gap grows with the number of clients K;\n"
                  "heterogeneity at fixed K (light bars) barely moves it", loc="left", fontsize=11)
    ax1.set_xlim(0, max(max(vals), local_gap) * 1.22)

    labs = list(beps)
    ax2.bar(range(len(labs)), [beps[k] for k in labs],
            color=[COLORS["centralized"]] + [COLORS["fedavg"]] * (len(labs) - 1), width=0.6, edgecolor="white")
    for i, k in enumerate(labs):
        ax2.text(i, beps[k] + 0.5, f"{beps[k]:.1f} MB", ha="center", fontsize=9.5, color=INK)
    ax2.set_xticks(range(len(labs)), labs, fontsize=9.5)
    ax2.set_ylabel("Megabytes")
    ax2.set_ylim(0, max(beps.values()) * 1.18)
    ax2.set_title("Bandwidth (K = 20): heterogeneity raises the bytes\nneeded to come within 10 % of centralised",
                  loc="left", fontsize=11)
    _save(fig, path, None)
