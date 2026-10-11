"""Analyses added in response to the internal review (configs/matrix_revision.yaml).

    python -m src.evaluation.revision          (after the revision matrix has finished)

Reads the same run files as report.py and writes
  results/metrics/revision/*.csv            one table per analysis
  results/plots/revision_*.png              figures (also copied to paper/figures/)
  paper/generated/numbers_revision.tex      \\newcommand macros used by the manuscript
  paper/generated/rev_*.tex                 tabular snippets used by the manuscript
Only numbers present in the run files are reported; a missing run shows up as a missing value.
"""
from __future__ import annotations

import io
import json
import math
import shutil
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config, resolve
from src.utils import results_path

FED_K = {1: "E14_fedavg_K1", 5: "E14_fedavg_K5", 20: "E3_fedavg_iid", 50: "E14_fedavg_K50",
         100: "E13_fedavg_household"}
WIN_K = {5: "E15_fedavg_window_K5", 20: "E15_fedavg_window_K20", 50: "E15_fedavg_window_K50",
         100: "E15_fedavg_window_K100"}
LONG_K = {20: "E18_fedavg_K20_long", 50: "E18_fedavg_K50_long", 100: "E18_fedavg_K100_long"}
GROUPINGS = [("window (IID control)", "E15_fedavg_window_K20"), ("random", "E3_fedavg_iid"),
             ("α=1.0", "E10_fedavg_a1.0"), ("α=0.3", "E4_fedavg_moderate"), ("α=0.1", "E10_fedavg_a0.1"),
             ("cluster", "E6_fedavg_strong"), ("shape-sorted", "E16_fedavg_shape")]
COHORT_ARMS = [("Centralised", "centralized"), ("Local-only", "local"), ("FedAvg, K=1", "fedavg_K1"),
               ("FedAvg, K=5", "fedavg_K5"), ("FedAvg, K=20", "fedavg_K20"), ("FedAvg, K=100", "fedavg_K100")]
MAIN_COHORT = {"centralized": "E2_centralized", "local": "E1_local", "fedavg_K1": "E14_fedavg_K1",
               "fedavg_K5": "E14_fedavg_K5", "fedavg_K20": "E3_fedavg_iid", "fedavg_K100": "E13_fedavg_household"}
EPS = [5, 10, 15, 20]
COLORS = ["#2a78b9", "#d9822b", "#3d9a5b", "#8e5ea2", "#c0504d", "#6b6b6b", "#1f3864"]

M: dict[str, str] = {}          # macros


# ───────────────────────────── helpers ─────────────────────────────

def _runs(rundir):
    rows = [json.loads(p.read_text()) for p in sorted(rundir.glob("*.json"))]
    return pd.DataFrame(rows)


def _sel(R, name):
    return R[R["experiment"] == name]


def _ci(x):
    x = np.asarray([v for v in x if pd.notna(v)], float)
    n = len(x)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return x[0], np.nan, 1
    return x.mean(), stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / math.sqrt(n), n


def _hist(rundir, name, seed):
    f = rundir / f"{name}__seed{seed}_history.csv"
    return pd.read_csv(f) if f.exists() else None


def _hh(rundir, name, col="mae"):
    files = sorted(rundir.glob(f"{name}__seed*_households.csv"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    agg = {c: "mean" for c in ["mae", "rmse", "ft_mae"] if c in df}
    agg.update({c: "first" for c in ["train_mean_kwh", "stratum"] if c in df})
    return df.groupby("LCLid").agg(agg)


def _central_val(R):
    c = R[R["experiment"] == "E2_centralized"]
    return dict(zip(c["seed"], c["best_val_mae"])), float(c["best_val_mae"].mean()), float(c["test_mae"].mean())


def _key(s: str) -> str:
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
              "7": "seven", "8": "eight", "9": "nine"}
    return "".join(digits.get(ch, ch) for ch in s if ch.isalnum())


def _m(name, value, fmt="{:.4f}"):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        M["Rv" + _key(name)] = "n/a"
    elif isinstance(value, str):
        M["Rv" + _key(name)] = value
    else:
        M["Rv" + _key(name)] = fmt.format(value)


def _p(p):
    if not np.isfinite(p):
        return "n/a"
    return "$<10^{-4}$" if p < 1e-4 else (f"$={p:.4f}$" if p < 0.01 else f"$={p:.3f}$")


def _signed(x, nd=1):
    return f"{x:+.{nd}f}".replace("-", "−") if np.isfinite(x) else "n/a"


def _tabular(path, cols, rows, align):
    lines = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def _first_hit(h, thr):
    hit = h[h["val_mae"] <= thr]
    return (int(hit["round"].iloc[0]), float(hit["cumulative_bytes"].iloc[0]) / 1e6) if len(hit) else (np.nan, np.nan)


# ───────────────────────────── analyses ─────────────────────────────

def budget(R, rundir, out, gen, figs, central_test):
    rows, tab = [], []
    tw = float(R["train_windows"].dropna().iloc[0])
    for K, name in FED_K.items():
        s = _sel(R, name)
        steps = math.ceil(tw / K / 256)
        L = _sel(R, LONG_K[K]) if K in LONG_K else pd.DataFrame()
        r = {"K": K, "steps_per_client_round": steps, "mae30": s["test_mae"].mean(), "sd30": s["test_mae"].std(),
             "best30": s["best_round"].mean(), "n30": len(s),
             "mae_long": L["test_mae"].mean() if len(L) else np.nan,
             "sd_long": L["test_mae"].std() if len(L) else np.nan,
             "best_long": L["best_round"].mean() if len(L) else np.nan,
             "rounds_run": L["rounds_run"].mean() if len(L) and "rounds_run" in L else np.nan, "n_long": len(L)}
        r["gap30"] = 100 * (r["mae30"] / central_test - 1)
        r["gap_long"] = 100 * (r["mae_long"] / central_test - 1)
        rows.append(r)
        k = {1: "Kone", 5: "Kfive", 20: "Ktwenty", 50: "Kfifty", 100: "Khundred"}[K]
        _m(f"Steps{k}", steps, "{:,}")
        _m(f"BestRound{k}", r["best30"], "{:.1f}")
        if K in LONG_K:
            _m(f"Long{k}MAE", r["mae_long"])
            _m(f"Long{k}Gap", r["gap_long"], "{:.1f}")
            _m(f"Long{k}Best", r["best_long"], "{:.0f}")
            _m(f"Long{k}Run", r["rounds_run"], "{:.0f}")
        tab.append([K, f"{steps:,}", f"{r['best30']:.1f}", f"{r['mae30']:.4f}", _signed(r["gap30"]),
                    "--" if K not in LONG_K else f"{r['best_long']:.0f} / {r['rounds_run']:.0f}",
                    "--" if K not in LONG_K else f"{r['mae_long']:.4f}",
                    "--" if K not in LONG_K else _signed(r["gap_long"])])
    df = pd.DataFrame(rows)
    df.to_csv(out / "budget.csv", index=False)
    _tabular(gen / "rev_budget.tex", ["$K$", "Steps/round", "Best rd. (30)", "MAE (30)", "Gap (\\%)",
                                       "Best / run (100)", "MAE (100)", "Gap (\\%)"], tab, "rrrrrrrr")
    # figure: validation MAE per round, every K (30 rounds) and the long runs
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for c, (K, name) in zip(COLORS, FED_K.items()):
        for nm, ls in ((name, "-"), (LONG_K.get(K), "--")):
            if nm is None:
                continue
            hs = [h for seed in range(42, 47) if (h := _hist(rundir, nm, seed)) is not None]
            if not hs:
                continue
            H = pd.concat(hs).groupby("round")["val_mae"].mean()
            ax.plot(H.index, H.values, ls, color=c, lw=1.4, label=f"K={K}" if ls == "-" else None)
    cv = _central_val(R)[1]
    ax.axhline(cv, color="k", ls=":", lw=1, label="centralised")
    ax.set_xscale("log")
    ax.set_xlabel("Round (log scale); dashed: 100-round runs")
    ax.set_ylabel("Validation MAE (kWh)")
    ax.legend(fontsize=7, ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "revision_budget.png", dpi=300)
    plt.close(fig)
    return df


def window_control(R, out, gen, figs, central_test):
    rows, tab = [], []
    for K in [1, 5, 20, 50, 100]:
        h = _sel(R, FED_K[K])
        w = _sel(R, WIN_K[K]) if K in WIN_K else pd.DataFrame()
        r = {"K": K, "H_JS_home": h["H_JS"].mean(), "mae_home": h["test_mae"].mean(), "sd_home": h["test_mae"].std(),
             "H_JS_win": w["H_JS"].mean() if len(w) else np.nan, "mae_win": w["test_mae"].mean() if len(w) else np.nan,
             "sd_win": w["test_mae"].std() if len(w) else np.nan, "n_win": len(w)}
        r["gap_home"] = 100 * (r["mae_home"] / central_test - 1)
        r["gap_win"] = 100 * (r["mae_win"] / central_test - 1)
        rows.append(r)
        k = {1: "Kone", 5: "Kfive", 20: "Ktwenty", 50: "Kfifty", 100: "Khundred"}[K]
        _m(f"HJSHome{k}", r["H_JS_home"])
        if K in WIN_K:
            _m(f"Win{k}MAE", r["mae_win"])
            _m(f"Win{k}Gap", r["gap_win"], "{:.1f}")
            _m(f"Win{k}HJS", r["H_JS_win"], "{:.4f}")
            _m(f"Win{k}Share", 100 * r["gap_win"] / r["gap_home"], "{:.0f}")
            _m(f"Home{k}Gap", r["gap_home"], "{:.1f}")
        tab.append([K, f"{r['H_JS_home']:.4f}", f"{r['mae_home']:.4f}", _signed(r["gap_home"]),
                    "--" if K not in WIN_K else f"{r['H_JS_win']:.5f}",
                    "--" if K not in WIN_K else f"{r['mae_win']:.4f}",
                    "--" if K not in WIN_K else _signed(r["gap_win"])])
    df = pd.DataFrame(rows)
    big = df[(df["K"] >= 20) & df["gap_win"].notna()]
    if len(big):
        sh = 100 * big["gap_win"] / big["gap_home"]
        _m("WinShareLo", sh.min(), "{:.0f}")
        _m("WinShareHi", sh.max(), "{:.0f}")
    df.to_csv(out / "window_control.csv", index=False)
    _tabular(gen / "rev_window.tex", ["$K$", "$H_{JS}$, homes", "MAE, homes", "Gap (\\%)", "$H_{JS}$, windows",
                                       "MAE, windows", "Gap (\\%)"], tab, "rrrrrrr")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.errorbar(df["K"], df["mae_home"], yerr=df["sd_home"], marker="o", color=COLORS[1], capsize=3,
                label="homes grouped at random")
    d2 = df.dropna(subset=["mae_win"])
    d2 = pd.concat([df[df["K"] == 1].assign(mae_win=df.loc[df["K"] == 1, "mae_home"], sd_win=df.loc[df["K"] == 1, "sd_home"]), d2])
    ax.errorbar(d2["K"], d2["mae_win"], yerr=d2["sd_win"], marker="s", color=COLORS[0], capsize=3, ls="--",
                label="windows dealt at random (IID control)")
    lk = [(K, _sel(R, n)["test_mae"]) for K, n in LONG_K.items()]
    lk = [(K, v) for K, v in lk if len(v)]
    if lk:
        ax.errorbar([K * 1.06 for K, _ in lk], [v.mean() for _, v in lk], yerr=[v.std() for _, v in lk], marker="^",
                    color=COLORS[2], capsize=3, ls=":", label="homes grouped at random, up to 100 rounds")
    ax.axhline(central_test, color="k", ls=":", lw=1, label="centralised")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 20, 50, 100])
    ax.set_xticklabels(["1", "5", "20", "50", "100"])
    ax.set_xlabel("Number of clients K")
    ax.set_ylabel("Test MAE (kWh)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figs / "revision_window.png", dpi=300)
    plt.close(fig)
    return df


def regression(R, out):
    names = list(FED_K.values()) + list(WIN_K.values()) + ["E4_fedavg_moderate", "E6_fedavg_strong",
                                                           "E10_fedavg_a1.0", "E10_fedavg_a0.1", "E16_fedavg_shape"]
    d = R[R["experiment"].isin(names)].dropna(subset=["H_JS", "test_mae", "num_clients"])
    X = np.column_stack([np.ones(len(d)), np.log(d["num_clients"].astype(float)), d["H_JS"].astype(float)])
    y = d["test_mae"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    n, p = X.shape
    s2 = res @ res / (n - p)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tcrit = stats.t.ppf(0.975, n - p)
    pv = 2 * stats.t.sf(np.abs(beta / se), n - p)
    r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    out_df = pd.DataFrame({"term": ["const", "ln_K", "H_JS"], "coef": beta, "se": se,
                           "ci_lo": beta - tcrit * se, "ci_hi": beta + tcrit * se, "p": pv})
    out_df["n"], out_df["r2"] = n, r2
    out_df.to_csv(out / "regression.csv", index=False)
    _m("RegLogK", beta[1] * 1e3, "{:.2f}")
    _m("RegLogKlo", (beta[1] - tcrit * se[1]) * 1e3, "{:.2f}")
    _m("RegLogKhi", (beta[1] + tcrit * se[1]) * 1e3, "{:.2f}")
    _m("RegLogKP", None if not np.isfinite(pv[1]) else pv[1], "{:.2g}")
    M["RvRegLogKP"] = _p(pv[1])
    _m("RegHJS", beta[2] * 1e3, "{:.1f}")
    _m("RegHJSlo", (beta[2] - tcrit * se[2]) * 1e3, "{:.1f}")
    _m("RegHJShi", (beta[2] + tcrit * se[2]) * 1e3, "{:.1f}")
    M["RvRegHJSP"] = _p(pv[2])
    _m("RegN", n, "{:d}")
    # effect sizes over the tested ranges: K from 5 to 100, and H_JS across the K = 20 groupings
    _m("RegEffK", beta[1] * np.log(100 / 5), "{:.4f}")
    g20 = R[R["experiment"].isin([nm for _, nm in GROUPINGS])].groupby("experiment")["H_JS"].mean()
    if len(g20):
        _m("RegEffH", beta[2] * (g20.max() - g20.min()), "{:.4f}")
        _m("RegHJSRangeLo", g20.min(), "{:.4f}")
        _m("RegHJSRangeHi", g20.max(), "{:.4f}")
    _m("RegRtwo", r2, "{:.2f}")
    return out_df


def heterogeneity(R, rundir, out, gen, figs, central_vals):
    cvs, cv_mean, _ = central_vals
    rows, tab = [], []
    for label, name in GROUPINGS:
        s = _sel(R, name)
        if s.empty:
            continue
        mae, ci, n = _ci(s["test_mae"])
        r = {"grouping": label, "experiment": name, "n": n, "H_JS": s["H_JS"].mean(), "H_CV": s["H_CV"].mean(),
             "mae": mae, "mae_ci": ci}
        for e in EPS:
            b, reached = [], 0
            for _, run in s.iterrows():
                h = _hist(rundir, name, run["seed"])
                if h is None:
                    continue
                thr = (1 + e / 100) * cvs.get(run["seed"], cv_mean)
                rr, mb = _first_hit(h, thr)
                if np.isfinite(mb):
                    reached += 1
                    b.append(mb)
            m, c, _ = _ci(b)
            r[f"B{e}"], r[f"B{e}_ci"], r[f"B{e}_n"] = m, c, reached
            r[f"B{e}_vals"] = b
        # excess validation error averaged over the 30 rounds (threshold-free)
        ex = []
        for _, run in s.iterrows():
            h = _hist(rundir, name, run["seed"])
            if h is not None:
                v = cvs.get(run["seed"], cv_mean)
                ex.append(100 * float(np.mean(h["val_mae"].head(30) / v - 1)))
        r["excess"], r["excess_ci"], _ = _ci(ex)
        rows.append(r)
        k = _key(label.split(" ")[0].replace("α=", "a"))
        _m(f"Het{k}HJS", r["H_JS"])
        _m(f"Het{k}MAE", r["mae"])
        _m(f"Het{k}CI", r["mae_ci"])
        _m(f"Het{k}N", n, "{:d}")
        for e in EPS:
            _m(f"Het{k}B{e}", r[f"B{e}"], "{:.1f}")
        _m(f"Het{k}Excess", r["excess"], "{:.1f}")
        tab.append([label, n, f"{r['H_JS']:.4f}", f"{r['H_CV']:.3f}", f"{mae:.4f} $\\pm$ {ci:.4f}"]
                   + [f"{r[f'B{e}']:.1f} ({r[f'B{e}_n']}/{n})" if np.isfinite(r[f"B{e}"]) else f"-- (0/{n})" for e in (5, 10, 20)]
                   + [f"{r['excess']:.1f}"])
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in df if c.endswith("_vals")]).to_csv(out / "heterogeneity_revision.csv", index=False)
    _tabular(gen / "rev_heterogeneity.tex",
             ["Grouping", "Seeds", "$H_{JS}$", "$H_{CV}$", "Test MAE ($\\pm$95\\% CI)", "B@5\\%", "B@10\\%",
              "B@20\\%", "Excess (\\%)"], tab, "lrrrrrrrr")
    # composite-H share and correlation of final error with H_JS at K = 20 (home groupings only)
    k20 = R[R["experiment"].isin([n for l, n in GROUPINGS if not l.startswith("window")])]
    share = (k20["H_CV"] / (1 + k20["H_CV"]) / k20["H"]).mean()
    _m("HcvShare", 100 * share, "{:.0f}")
    rho, p = stats.spearmanr(k20["H_JS"], k20["test_mae"])
    _m("HetSpearman", rho, "{:+.2f}")
    M["RvHetSpearmanP"] = _p(p)
    _m("HetSpearmanN", len(k20), "{:d}")
    lo, hi = df[~df["grouping"].str.startswith("window")]["mae"].agg(["min", "max"])
    _m("HetMaxDiff", hi - lo)
    rnd, clu = df.set_index("grouping").loc["random"], df.set_index("grouping").loc["cluster"]
    _m("HetBepsIncrease", 100 * (clu["B10"] / rnd["B10"] - 1), "{:.0f}")
    # figure: (a) validation curves per grouping, (b) B@eps against eps
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.5))
    top_mb = 1.08 * max([max(v) for v in df[[f"B{e}_vals" for e in EPS]].to_numpy().ravel() if len(v)] + [1])
    for c, (label, name) in zip(COLORS, GROUPINGS):
        s = _sel(R, name)
        hs = [h for sd in s["seed"] if (h := _hist(rundir, name, sd)) is not None]
        if not hs:
            continue
        H = pd.concat(hs).groupby("round")["val_mae"]
        m, sd = H.mean(), H.std()
        axs[0].plot(m.index, m.values, color=c, lw=1.3, label=label)
        axs[0].fill_between(m.index, (m - sd).values, (m + sd).values, color=c, alpha=0.12, lw=0)
        r = df.set_index("grouping").loc[label] if label in set(df["grouping"]) else None
        if r is not None:
            j = (COLORS.index(c) - 3) * 0.35          # small horizontal offset per grouping
            for e in EPS:
                v = r[f"B{e}_vals"]
                axs[1].plot([e + j] * len(v), v, "o", color=c, ms=2.6, alpha=0.55, mec="none")
                miss = int(r["n"]) - len(v)
                if miss:
                    axs[1].plot([e + j], [top_mb], "x", color=c, ms=4)
            means = [r[f"B{e}"] for e in EPS]
            axs[1].plot([e + j for e in EPS], means, color=c, lw=1.2, label=label)
    axs[0].axhline(cv_mean, color="k", ls=":", lw=1)
    axs[0].set_xlabel("Round")
    axs[0].set_ylabel("Validation MAE (kWh)")
    axs[0].set_ylim(cv_mean * 0.98, cv_mean * 1.30)
    axs[0].grid(alpha=0.25)
    axs[0].text(-0.13, 1.04, "(a)", transform=axs[0].transAxes, va="bottom", fontweight="bold")
    axs[1].set_xlabel("ε (% above centralised validation MAE)")
    axs[1].set_ylabel("B@ε (MB; dots: seeds, line: mean)")
    axs[1].set_ylim(0, top_mb * 1.06)
    axs[1].text(0.98, 0.84, "× not reached within 30 rounds", transform=axs[1].transAxes, ha="right", fontsize=7)
    axs[1].set_xticks(EPS)
    axs[1].grid(alpha=0.25)
    axs[1].text(-0.13, 1.04, "(b)", transform=axs[1].transAxes, va="bottom", fontweight="bold")
    axs[1].legend(fontsize=7, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(figs / "revision_heterogeneity.png", dpi=300)
    plt.close(fig)
    return df


def client_only(R, rundir, out, central_test):
    loc = _hh(rundir, "E1_local")
    rows = []
    for K, cl, fed in [(5, "E17_clientlocal_K5", "E14_fedavg_K5"), (20, "E17_clientlocal_K20", "E3_fedavg_iid")]:
        c, f = _sel(R, cl), _sel(R, fed)
        hc, hf = _hh(rundir, cl), _hh(rundir, fed)
        r = {"K": K, "mae_client": c["test_mae"].mean(), "sd_client": c["test_mae"].std(), "n": len(c),
             "mae_fed": f["test_mae"].mean(), "ft_fed": f["ft_mae"].mean()}
        r["gap_client"] = 100 * (r["mae_client"] / central_test - 1)
        if hc is not None and hf is not None and loc is not None:
            j = hc.index.intersection(hf.index)
            r["fed_better_pct"] = 100 * float((hf.loc[j, "mae"] < hc.loc[j, "mae"]).mean())
            r["p_fed_vs_client"] = float(stats.wilcoxon(hf.loc[j, "mae"], hc.loc[j, "mae"]).pvalue)
            j2 = hc.index.intersection(loc.index)
            r["client_better_than_local_pct"] = 100 * float((hc.loc[j2, "mae"] < loc.loc[j2, "mae"]).mean())
            if "ft_mae" in hf:
                r["ft_better_pct"] = 100 * float((hf.loc[j, "ft_mae"] < hc.loc[j, "mae"]).mean())
                r["p_ft_vs_client"] = float(stats.wilcoxon(hf.loc[j, "ft_mae"], hc.loc[j, "mae"]).pvalue)
        rows.append(r)
        k = {5: "Kfive", 20: "Ktwenty"}[K]
        _m(f"Client{k}MAE", r["mae_client"])
        _m(f"Client{k}Gap", r["gap_client"], "{:.1f}")
        _m(f"Client{k}FedBetter", r.get("fed_better_pct", np.nan), "{:.0f}")
        M[f"RvClient{k}FedP"] = _p(r.get("p_fed_vs_client", np.nan))
        _m(f"Client{k}BeatsLocal", r.get("client_better_than_local_pct", np.nan), "{:.0f}")
        _m(f"Client{k}FTBetter", r.get("ft_better_pct", np.nan), "{:.0f}")
        M[f"RvClient{k}FTP"] = _p(r.get("p_ft_vs_client", np.nan))
        _m(f"Client{k}FedMAE", r["mae_fed"])
        _m(f"Client{k}FTMAE", r["ft_fed"])
    df = pd.DataFrame(rows)
    df.to_csv(out / "client_only.csv", index=False)
    return df


def household_scales(cfg_paths):
    """In-sample one-step naive MAE per household (training period), the MASE scale."""
    from src.data.preprocess import load_processed
    out = {}
    for cfg in cfg_paths:
        series, meta = load_processed(cfg)
        tr = series[series.index < pd.Timestamp(cfg["data"]["train_end"])]
        te = series[series.index >= pd.Timestamp(cfg["data"]["val_end"])]
        for c in series.columns:
            out[c] = {"scale": float(np.nanmean(np.abs(np.diff(tr[c].to_numpy())))),
                      "persist_test": float(np.nanmean(np.abs(np.diff(te[c].to_numpy()))))}
    return pd.DataFrame(out).T


def calendar_and_mase(R, rundir, out, gen, scales, persistence_mae):
    pairs = [("Centralised", "E2_centralized", "E19_centralized_cal"), ("Local-only", "E1_local", "E19_local_cal"),
             ("FedAvg, K=20", "E3_fedavg_iid", "E19_fedavg_K20_cal"),
             ("FedAvg, K=100", "E13_fedavg_household", "E19_fedavg_K100_cal")]

    def mase(name):
        h = _hh(rundir, name)
        if h is None:
            return np.nan
        j = h.index.intersection(scales.index)
        return float((h.loc[j, "mae"] / scales.loc[j, "scale"]).mean())

    j = scales.index
    mase_persist = float((scales["persist_test"] / scales["scale"]).mean())
    _m("MASEPersistence", mase_persist, "{:.3f}")
    rows, tab = [], []
    cen_base = _sel(R, "E2_centralized")["test_mae"].mean()
    cen_cal = _sel(R, "E19_centralized_cal")["test_mae"].mean()
    for label, base, cal in pairs:
        b, c = _sel(R, base), _sel(R, cal)
        r = {"arm": label, "mae": b["test_mae"].mean(), "mae_cal": c["test_mae"].mean() if len(c) else np.nan,
             "mase": mase(base), "mase_cal": mase(cal), "n_cal": len(c)}
        r["vs_persist"] = 100 * (1 - r["mae"] / persistence_mae)
        r["vs_persist_cal"] = 100 * (1 - r["mae_cal"] / persistence_mae)
        r["gap"] = 100 * (r["mae"] / cen_base - 1)
        r["gap_cal"] = 100 * (r["mae_cal"] / cen_cal - 1)
        rows.append(r)
        k = _key(label.replace("FedAvg, K=", "Fed").replace("-only", ""))
        for f in ("mae_cal", "mase", "mase_cal"):
            _m(f"Cal{k}{_key(f)}", r[f], "{:.4f}" if f.startswith("mae") else "{:.3f}")
        _m(f"Cal{k}VsPersist", r["vs_persist"], "{:.1f}")
        _m(f"Cal{k}VsPersistCal", r["vs_persist_cal"], "{:.1f}")
        _m(f"Cal{k}GapCal", r["gap_cal"], "{:.1f}")
        tab.append([label, f"{r['mae']:.4f}", f"{r['mase']:.3f}", _signed(r["vs_persist"]),
                    f"{r['mae_cal']:.4f}", f"{r['mase_cal']:.3f}", _signed(r["vs_persist_cal"]),
                    _signed(r["gap_cal"]) if label.startswith("FedAvg") else "--"])
    df = pd.DataFrame(rows)
    df.to_csv(out / "calendar_mase.csv", index=False)
    _tabular(gen / "rev_calendar.tex", ["Arm", "MAE", "MASE", "Gain (\\%)", "MAE, cal.",
                                         "MASE, cal.", "Gain, cal. (\\%)", "Gap, cal. (\\%)"], tab,
             "lrrrrrrr")
    return df


def cohorts(R, out, gen):
    rows, tab = [], []
    for cohort, prefix in [("main (seed 42)", None), ("cohort 43", "C43"), ("cohort 44", "C44")]:
        vals = {}
        for label, key in COHORT_ARMS:
            name = MAIN_COHORT[key] if prefix is None else f"{prefix}_{key}"
            s = _sel(R, name)
            if prefix is None:
                s = s[s["seed"].isin([42, 43])]         # same seeds as the extra cohorts
            vals[key] = s["test_mae"].mean() if len(s) else np.nan
        c = vals["centralized"]
        row = {"cohort": cohort, **{k: v for k, v in vals.items()},
               **{f"gap_{k}": 100 * (v / c - 1) for k, v in vals.items()}}
        rows.append(row)
        tab.append([cohort] + [f"{vals[k]:.4f}" for _, k in COHORT_ARMS]
                   + [", ".join(_signed(row[f"gap_fedavg_K{K}"]) for K in (1, 5, 20, 100))])
    df = pd.DataFrame(rows)
    df.to_csv(out / "cohorts.csv", index=False)
    _tabular(gen / "rev_cohorts.tex", ["Cohort"] + [l for l, _ in COHORT_ARMS] + ["FedAvg gap, K=1/5/20/100 (\\%)"],
             tab, "lrrrrrrl")
    for K in (5, 20, 100):
        g = df[f"gap_fedavg_K{K}"]
        _m(f"CohortGapK{K}Lo", g.min(), "{:.1f}")
        _m(f"CohortGapK{K}Hi", g.max(), "{:.1f}")
    g1 = df["gap_fedavg_K1"]
    _m("CohortGapKoneLo", g1.min(), "{:.1f}")
    _m("CohortGapKoneHi", g1.max(), "{:.1f}")
    cl = 100 * (df["local"] / df["centralized"] - 1)
    _m("CohortLocalGapLo", cl.min(), "{:.1f}")
    _m("CohortLocalGapHi", cl.max(), "{:.1f}")
    return df


def tuning_grid(R, out, gen):
    rows = []
    fedavg_cluster = _sel(R, "E6_fedavg_strong")
    fedavg_cluster = fedavg_cluster[fedavg_cluster["seed"] == 42]
    for algo in ("fedprox", "hafedprox"):
        for prefix in ("T_", "T2_"):
            s = R[R["experiment"].str.startswith(prefix + algo + "_mu")]
            for _, r in s.iterrows():
                rows.append({"algorithm": algo, "mu": float(r["mu"]), "val_mae": r["best_val_mae"], "run": r["experiment"]})
    df = pd.DataFrame(rows).sort_values(["algorithm", "mu"])
    if len(fedavg_cluster):
        df = pd.concat([pd.DataFrame([{"algorithm": "fedavg (mu=0)", "mu": 0.0,
                                       "val_mae": float(fedavg_cluster["best_val_mae"].iloc[0]), "run": "E6_fedavg_strong"}]), df])
    df.to_csv(out / "tuning_grid.csv", index=False)
    tab = []
    for algo, label in (("fedprox", "FedProx $\\mu$"), ("hafedprox", "HA-FedProx $\\mu_0$")):
        d = df[df["algorithm"] == algo]
        tab.append([label] + [f"{m:g}: {v:.5f}" for m, v in zip(d["mu"], d["val_mae"])])
        pos = d[d["mu"] > 0]
        if len(pos):
            bp = pos.loc[pos["val_mae"].idxmin()]
            _m(f"Tune{algo}BestPos", bp["mu"], "{:g}")
            _m(f"Tune{algo}BestPosVal", bp["val_mae"], "{:.5f}")
        best = d.loc[d["val_mae"].idxmin()] if len(d) else None
        if best is not None:
            _m(f"Tune{algo}Best", best["mu"], "{:g}")
            _m(f"Tune{algo}BestVal", best["val_mae"], "{:.5f}")
    if len(fedavg_cluster):
        _m("TuneFedAvgVal", float(fedavg_cluster["best_val_mae"].iloc[0]), "{:.5f}")
    fp = df[df["algorithm"] == "fedprox"]
    if len(fp):
        _m("TuneFedproxSpread", 100 * (fp["val_mae"].max() / fp["val_mae"].min() - 1), "{:.1f}")
    mus = sorted(set(df["mu"]))
    tab = []
    for mu in mus:
        row = [f"{mu:g}"]
        for algo in ("fedprox", "hafedprox"):
            v = df[(df["algorithm"] == algo) & (df["mu"] == mu)]["val_mae"]
            if not len(v) and mu == 0:
                v = df[df["algorithm"].str.startswith("fedavg")]["val_mae"]
            row.append(f"{v.iloc[0]:.5f}" if len(v) else "--")
        tab.append(row)
    _tabular(gen / "rev_tuning.tex", ["$\\mu$ (FedProx) or $\\mu_0$ (adaptive)", "FedProx", "Adaptive (HA-FedProx)"],
             tab, "lrr")
    return df


def equity_extras(R, rundir, out, gen):
    loc = _hh(rundir, "E1_local")
    arms = [("Centralised", "E2_centralized", "mae"), ("FedAvg, random", "E3_fedavg_iid", "mae"),
            ("FedAvg, α=0.3", "E4_fedavg_moderate", "mae"), ("FedAvg, α=0.3 + FT", "E4_fedavg_moderate", "ft_mae"),
            ("FedAvg, K=100", "E13_fedavg_household", "mae"), ("FedAvg, K=100 + FT", "E13_fedavg_household", "ft_mae")]
    rows, tab = [], []
    for label, name, col in arms:
        h = _hh(rundir, name)
        if h is None or loc is None:
            continue
        j = h.index.intersection(loc.index)
        ratio = h.loc[j, col] / loc.loc[j, "mae"]
        r = {"arm": label, "harmed_any": 100 * float((ratio > 1).mean()), "harmed_1pct": 100 * float((ratio > 1.01).mean()),
             "helped_1pct": 100 * float((ratio < 0.99).mean())}
        rows.append(r)
        tab.append([label, f"{r['harmed_any']:.0f}", f"{r['harmed_1pct']:.0f}", f"{r['helped_1pct']:.0f}"])
        k = _key(label.replace("FedAvg, ", "Fed").replace("α=0.3", "Moderate").replace("+ FT", "FT"))
        _m(f"Harm{k}Any", r["harmed_any"], "{:.0f}")
        _m(f"Harm{k}Margin", r["harmed_1pct"], "{:.0f}")
        _m(f"Help{k}Margin", r["helped_1pct"], "{:.0f}")
    pd.DataFrame(rows).to_csv(out / "harm_margin.csv", index=False)
    _tabular(gen / "rev_harm.tex", ["Arm", "Worse than local (\\%)", "Worse by $>$1\\% (\\%)",
                                     "Better by $>$1\\% (\\%)"], tab, "lrrr")
    # median household MAE by consumption quintile
    qarms = [("Local-only", "E1_local", "mae"), ("Centralised", "E2_centralized", "mae"),
             ("FedAvg, α=0.3", "E4_fedavg_moderate", "mae"), ("FedAvg, α=0.3 + FT", "E4_fedavg_moderate", "ft_mae"),
             ("FedAvg, K=100", "E13_fedavg_household", "mae")]
    Q, qtab = {}, []
    for label, name, col in qarms:
        h = _hh(rundir, name)
        if h is None:
            continue
        Q[label] = h.groupby("stratum")[col].median()
    Qdf = pd.DataFrame(Q)
    Qdf.to_csv(out / "quintiles.csv")
    for s, row in Qdf.iterrows():
        qtab.append([f"Q{int(s) + 1}"] + [f"{v:.4f}" for v in row.values])
    _tabular(gen / "rev_quintiles.tex", ["Quintile"] + list(Qdf.columns), qtab, "l" + "r" * len(Qdf.columns))
    if {"Centralised", "Local-only", "FedAvg, α=0.3"} <= set(Qdf.columns):
        top = Qdf.iloc[-1]
        _m("QTopLocal", top["Local-only"])
        _m("QTopCentral", top["Centralised"])
        _m("QTopFed", top["FedAvg, α=0.3"])
        _m("QTopFedFT", top.get("FedAvg, α=0.3 + FT", np.nan))
    return Qdf


def raw_bytes(R, cfg, out):
    s = _sel(R, "E2_centralized")
    raw12 = float(s["total_bytes"].iloc[0])
    readings = raw12 / 12.0
    # bytes per reading of the published CSV release, measured on the raw files of this cohort
    csv_bpr = np.nan
    raw_dir = resolve(cfg["data"]["raw_dir"])
    zips = sorted(raw_dir.glob("*.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as z:
            mem = [m for m in z.namelist() if m.lower().endswith(".csv")][:3]
            nbytes = nlines = 0
            for m in mem:
                with z.open(m) as f:
                    for line in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                        nbytes += len(line.encode())
                        nlines += 1
            csv_bpr = nbytes / max(nlines, 1)
    d = {"readings": readings, "bytes_12": raw12, "bytes_4": 4 * readings, "bytes_2": 2 * readings,
         "csv_bytes_per_row": csv_bpr, "bytes_csv": csv_bpr * readings}
    pd.DataFrame([d]).to_csv(out / "raw_bytes.csv", index=False)
    _m("RawReadings", readings / 1e6, "{:.2f}")
    _m("RawTwelve", raw12 / 1e6, "{:.1f}")
    _m("RawFour", 4 * readings / 1e6, "{:.1f}")
    _m("RawTwo", 2 * readings / 1e6, "{:.1f}")
    _m("CsvBytesPerRow", csv_bpr, "{:.0f}")
    _m("RawCsv", csv_bpr * readings / 1e6, "{:.1f}")
    return d


def precision_and_partial(R, rundir, out):
    base = _hh(rundir, "E4_fedavg_moderate")
    rows = []
    for label, name in (("fp16", "E12_fedavg_fp16_moderate"), ("partial", "E8_fedavg_partial"),
                        ("fp32", "E4_fedavg_moderate")):
        s = _sel(R, name)
        m, ci, n = _ci(s["test_mae"])
        r = {"run": label, "mae": m, "ci": ci, "n": n}
        h = _hh(rundir, name)
        if h is not None and base is not None and name != "E4_fedavg_moderate":
            j = h.index.intersection(base.index)
            r["better_pct"] = 100 * float((h.loc[j, "mae"] < base.loc[j, "mae"]).mean())
            r["median_diff"] = float(np.median(h.loc[j, "mae"] - base.loc[j, "mae"]))
            r["p"] = float(stats.wilcoxon(h.loc[j, "mae"], base.loc[j, "mae"]).pvalue)
        # paired by seed: same partition draw for the same seed
        b = _sel(R, "E4_fedavg_moderate").set_index("seed")["test_mae"]
        ss = s.set_index("seed")["test_mae"]
        jj = ss.index.intersection(b.index)
        r["seed_diffs"] = ";".join(f"{v:+.4f}" for v in (ss.loc[jj] - b.loc[jj]))
        r["seeds_better"] = int(((ss.loc[jj] - b.loc[jj]) < 0).sum())
        r["seeds_n"] = len(jj)
        rows.append(r)
        k = _key(label)
        _m(f"Prec{k}MAE", m)
        _m(f"Prec{k}CI", ci)
        _m(f"Prec{k}N", n, "{:d}")
        _m(f"Prec{k}Better", r.get("better_pct", np.nan), "{:.0f}")
        _m(f"Prec{k}Med", r.get("median_diff", np.nan), "{:+.4f}")
        M[f"RvPrec{k}P"] = _p(r.get("p", np.nan))
        _m(f"Prec{k}SeedsBetter", r["seeds_better"], "{:d}")
        _m(f"Prec{k}SeedsN", r["seeds_n"], "{:d}")
    pd.DataFrame(rows).to_csv(out / "precision_partial.csv", index=False)


# ───────────────────────────── main ─────────────────────────────

def build(cfg_path="configs/default.yaml"):
    cfg = load_config(cfg_path)
    rundir = results_path(cfg, "metrics", "runs", "x").parent
    out = results_path(cfg, "metrics", "revision", "x").parent
    out.mkdir(parents=True, exist_ok=True)
    figs = results_path(cfg, "plots", "x").parent
    gen = resolve("paper/generated")
    R = _runs(rundir)
    cvs = _central_val(R)
    central_test = cvs[2]
    persistence_mae = float(_sel(R, "E0_naive")["persistence_mae"].iloc[0])

    budget(R, rundir, out, gen, figs, central_test)
    window_control(R, out, gen, figs, central_test)
    regression(R, out)
    heterogeneity(R, rundir, out, gen, figs, cvs)
    client_only(R, rundir, out, central_test)
    scales = household_scales([cfg])
    calendar_and_mase(R, rundir, out, gen, scales, persistence_mae)
    cohorts(R, out, gen)
    tuning_grid(R, out, gen)
    equity_extras(R, rundir, out, gen)
    raw_bytes(R, cfg, out)
    precision_and_partial(R, rundir, out)
    _m("PersistenceMAE", persistence_mae)
    nv = _sel(R, "E0_naive").iloc[0]
    for k in ("persistence", "seasonal_naive"):
        for met in ("wape", "rmse"):
            col = f"{k}_{met}"
            if col in nv:
                _m(f"{k.title().replace('_', '')}{met.upper()}", float(nv[col]), "{:.1f}" if met == "wape" else "{:.4f}")
    cen = _sel(R, "E2_centralized")["test_mae"].mean()
    _m("CentralVsPersist", 100 * (1 - cen / persistence_mae), "{:.1f}")

    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())]
    (gen / "numbers_revision.tex").write_text("% generated by src/evaluation/revision.py\n" + "\n".join(lines) + "\n")
    for f in figs.glob("revision_*.png"):
        shutil.copy(f, resolve("paper/figures") / f.name)
    print(f"revision analyses -> {out}; {len(M)} macros -> {gen / 'numbers_revision.tex'}")


if __name__ == "__main__":
    build()
