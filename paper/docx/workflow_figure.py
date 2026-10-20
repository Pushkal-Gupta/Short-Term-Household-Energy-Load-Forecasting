"""Draw the study workflow (Figure 1 of the Word manuscript). A schematic only: it shows no results.

    python paper/docx/workflow_figure.py            -> paper/figures/workflow.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, BLUE, GREY, INK = "#1F3864", "#2F4F7F", "#F2F4F8", "#222222"
CLUSTER = ["#2a78b9", "#d9822b", "#3d9a5b", "#8e5ea2", "#c0504d"]
OUT = Path(__file__).resolve().parents[1] / "figures" / "workflow.png"


def box(ax, x, y, w, h, title, lines, fc=GREY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=NAVY, lw=1.0))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=8.6, weight="bold", color=NAVY)
    ax.text(x + 0.012, y + h - 0.105, "\n".join(lines), ha="left", va="top", fontsize=7.0, color=INK,
            linespacing=1.45)


def arrow(ax, p, q, label=None, color=NAVY, dy=0.025, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, lw=1.1, color=color))
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + dy, label, ha="center", va="bottom", fontsize=6.6,
                color=color)


def main(path: Path = OUT) -> Path:
    fig = plt.figure(figsize=(7.4, 4.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1. data
    box(ax, 0.010, 0.255, 0.190, 0.665, "1. Data (LCL trial)",
        ["3 cohorts × 100 homes", "standard tariff", "half-hourly kWh, 2013", "stratified by load quintile", "",
         "train  Jan–Aug", "valid. Sep–Oct", "test    Nov–Dec", "", "per-home scaling", "window 48 → 1"])

    # 2. partition: 100 homes coloured by load-profile cluster, grouped into clients
    box(ax, 0.230, 0.255, 0.240, 0.665, "2. Partition into K clients", [])
    import numpy as np
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 5, 100)
    for c in range(5):                                   # five example clients, sorted = "cluster" grouping
        cx, cy = 0.258 + c * 0.039, 0.555
        ax.add_patch(FancyBboxPatch((cx, cy), 0.032, 0.235, boxstyle="round,pad=0.003,rounding_size=0.006",
                                    fc="white", ec=BLUE, lw=0.7))
        members = np.sort(clusters[c * 20:(c + 1) * 20])
        for j, m in enumerate(members):
            ax.plot(cx + 0.009 + (j % 2) * 0.014, cy + 0.215 - (j // 2) * 0.021, "o", ms=2.6,
                    color=CLUSTER[m], mec="none")
    ax.text(0.350, 0.535, "homes coloured by load-profile cluster", ha="center", va="top", fontsize=6.3,
            color=INK, style="italic")
    ax.text(0.242, 0.480, "homes: random, Dirichlet($\\alpha$),\ncluster or shape-sorted;\nIID control: windows dealt\nat random\n$K \\in$ {1, 5, 20, 50, 100}\n"
            "measured $H_{JS}$, $H_{CV}$ (train only)", ha="left", va="top", fontsize=7.1, color=INK, linespacing=1.45)

    # 3. federated rounds
    box(ax, 0.500, 0.255, 0.275, 0.665, "3. Federated rounds", [])
    ax.add_patch(FancyBboxPatch((0.530, 0.705), 0.215, 0.110, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc=NAVY, ec=NAVY))
    ax.text(0.6375, 0.775, "Server", ha="center", va="center", fontsize=8, weight="bold", color="white")
    ax.text(0.6375, 0.735, "FedAvg · FedProx · FedAdam", ha="center", va="center", fontsize=6.6,
            color="white")
    for i, x in enumerate([0.515, 0.600, 0.685]):
        ax.add_patch(FancyBboxPatch((x, 0.485), 0.076, 0.090, boxstyle="round,pad=0.003,rounding_size=0.008",
                                    fc="white", ec=BLUE, lw=0.8))
        ax.text(x + 0.038, 0.548, f"client ${'k' if i == 1 else ('1' if i == 0 else 'K')}$", ha="center",
                va="center", fontsize=6.6, color=INK)
        ax.text(x + 0.038, 0.509, "LSTM", ha="center", va="center", fontsize=6.2, color=INK)
        arrow(ax, (x + 0.024, 0.700), (x + 0.024, 0.580), color=BLUE)
        arrow(ax, (x + 0.052, 0.580), (x + 0.052, 0.700), color="#c0504d")
    ax.text(0.620, 0.620, "", ha="center")
    ax.text(0.507, 0.640, "$w_r$", fontsize=7, color=BLUE, ha="left")
    ax.text(0.768, 0.640, "$w_k$", fontsize=7, color="#c0504d", ha="right")
    ax.text(0.510, 0.460, "$R$ = 30 rounds × 1 epoch;\nto a plateau: ≤ 500 rounds\nbytes: $P\\cdot b$ per message,\n"
            "$b$ = 4 (fp32) or 2 (fp16)\nparticipation $C$, stragglers $p_s$\n"
            "client Adam (reset or kept), SGD;\nserver FedAdam", ha="left", va="top",
            fontsize=6.3, color=INK, linespacing=1.3)

    # 4. evaluation
    box(ax, 0.805, 0.255, 0.185, 0.665, "4. Evaluation",
        ["best round on validation", "one fixed test set", "", "MAE, RMSE, WAPE, MASE",
         "R@ε, B@ε, excess", "run-level t-tests", "+ Holm correction", "harm per household", "", "+ local fine-tuning", "  (FT)"])

    for x0, x1 in [(0.200, 0.230), (0.470, 0.500), (0.775, 0.805)]:
        arrow(ax, (x0 + 0.002, 0.61), (x1 - 0.002, 0.61))

    # reference arms
    ax.add_patch(FancyBboxPatch((0.010, 0.025), 0.980, 0.150, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc="white", ec=NAVY, lw=0.8, ls="--"))
    ax.text(0.5, 0.140, "Reference arms on the same homes, split, model and budget", ha="center", va="center",
            fontsize=8, weight="bold", color=NAVY)
    ax.text(0.5, 0.072, "persistence  ·  seasonal naive  ·  local-only (one LSTM per home)  ·  "
            "client-only (each client alone)  ·  centralised (pooled raw data)", ha="center", va="center", fontsize=7.2, color=INK)
    arrow(ax, (0.105, 0.255), (0.105, 0.187), style="-|>")
    arrow(ax, (0.897, 0.187), (0.897, 0.255), style="-|>")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(main())
