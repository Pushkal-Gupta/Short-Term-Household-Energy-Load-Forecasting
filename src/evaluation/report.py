"""Aggregate finished runs into tables, statistics, plots and LaTeX snippets.

Reads   <results>/metrics/runs/*.json, *_history.csv, *_households.csv, <results>/predictions/*.parquet
Writes  <results>/metrics/{runs_all,comparison,local_results,centralized_results,fedavg_results,
                            fedprox_results,heterogeneity,stats_tests,equity,convergence}.csv + comparison.md
        <results>/plots/*.png
        paper/generated/*.tex  (only for the real-data results directory)

Only numbers present in the run files are reported. A missing experiment shows up as a missing row.
"""
from __future__ import annotations

import json
import re
import shutil

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import resolve
from src.evaluation import plots
from src.utils import results_path

ALPHA_LABEL = {"iid": "random (IID-ish)", "a1.0": "α=1.0", "moderate": "α=0.3", "a0.3": "α=0.3",
               "a0.1": "α=0.1", "strong": "cluster (strong)"}


def _load_runs(cfg):
    d = results_path(cfg, "metrics", "runs", "x").parent
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
    return pd.DataFrame(rows), d


def _seed_files(rundir, name, suffix):
    return sorted(rundir.glob(f"{name}__seed*{suffix}"))


def _history(rundir, name) -> pd.DataFrame | None:
    files = [f for f in _seed_files(rundir, name, "_history.csv")]
    if not files:
        return None
    df = pd.concat([pd.read_csv(f).assign(seed=i) for i, f in enumerate(files)])
    if "round" not in df:
        return None
    num = df.select_dtypes("number").drop(columns=["seed"])
    g = num.groupby(df["round"])
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    out["round"] = out.index
    return out.reset_index(drop=True)


def _households(rundir, name) -> pd.DataFrame | None:
    files = _seed_files(rundir, name, "_households.csv")
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    agg = {c: "mean" for c in ["mae", "rmse", "mape", "wape", "ft_mae"] if c in df}
    agg.update({c: "first" for c in ["train_mean_kwh", "stratum", "profile_cluster"] if c in df})
    return df.groupby("LCLid").agg(agg)


def _arm_of(name: str, algorithm: str) -> str:
    return {"seasonal_naive": "naive"}.get(algorithm, algorithm)


def _fmt(m, s, nd=4):
    if pd.isna(m):
        return "—"
    return f"{m:.{nd}f} ± {s:.{nd}f}" if pd.notna(s) and s > 0 else f"{m:.{nd}f}"


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = ["test_mae", "test_rmse", "test_mape", "test_wape", "hh_mae_median", "hh_mae_p90",
               "best_round", "total_bytes", "bytes_to_best_round", "H", "H_JS", "H_CV", "ft_mae", "ft_wape",
               "runtime_s", "best_val_mae"]
    metrics = [m for m in metrics if m in runs]
    g = runs.groupby("experiment")
    out = g[metrics].mean().add_suffix("_mean").join(g[metrics].std().add_suffix("_std"))
    meta_cols = [c for c in ["method", "algorithm", "partition_mode", "partition_strategy", "alpha", "mu",
                             "client_fraction", "straggler_prob", "update_precision", "num_clients",
                             "rounds", "num_households", "num_params", "payload_bytes"] if c in runs]
    out = g[meta_cols].first().join(out)
    out["n_seeds"] = g.size()
    return out.reset_index()


def _eps_metrics(rundir, runs, central_val, eps):
    """R@eps / B@eps per run: first round whose val MAE <= (1 + eps/100) * centralised val MAE."""
    rows = []
    for _, r in runs[runs["method"] == "federated"].iterrows():
        f = rundir / f"{r['experiment']}__seed{r['seed']}_history.csv"
        h = pd.read_csv(f)
        cv = central_val.get(r["seed"], np.mean(list(central_val.values())) if central_val else np.nan)
        hit = h[h["val_mae"] <= (1 + eps / 100) * cv]
        rows.append({"experiment": r["experiment"], "seed": r["seed"],
                     "R_eps": int(hit["round"].iloc[0]) if len(hit) else np.nan,
                     "B_eps_MB": float(hit["cumulative_bytes"].iloc[0]) / 1e6 if len(hit) else np.nan})
    return pd.DataFrame(rows)


def _wilcoxon(a: pd.Series, b: pd.Series):
    j = a.index.intersection(b.index)
    d = (a.loc[j] - b.loc[j]).to_numpy()
    if len(j) < 6 or np.allclose(d, 0):
        return np.nan, np.nan, len(j)
    return float(np.median(d)), float(wilcoxon(a.loc[j], b.loc[j]).pvalue), len(j)


def build_report(cfg: dict, write_paper: bool | None = None) -> dict:
    tag = cfg.get("experiment_tag", "")
    runs, rundir = _load_runs(cfg)
    if runs.empty:
        raise SystemExit(f"No finished runs in {rundir}. Run the experiments first.")
    mdir = results_path(cfg, "metrics", "x").parent
    pdir = results_path(cfg, "plots", "x").parent
    runs.to_csv(mdir / "runs_all.csv", index=False)
    main = runs[~runs["experiment"].str.startswith(("T_", "T2_", "t_", "probe", "flower", "sim_", "C43_", "C44_", "E15_", "E16_", "E17_", "E18_", "E19_"))]
    S = summary_table(main)
    eps = float(cfg["eval"].get("epsilon_pct", 10.0))

    central = main[main["experiment"] == "E2_centralized"]
    central_val = dict(zip(central["seed"], central["best_val_mae"])) if len(central) else {}
    if central_val:
        E = _eps_metrics(rundir, main, central_val, eps)
        if len(E):
            ge = E.groupby("experiment")
            S = S.merge(ge[["R_eps", "B_eps_MB"]].mean().add_suffix("_mean")
                        .join(ge[["R_eps", "B_eps_MB"]].count().rename(columns={"R_eps": "R_eps_reached",
                                                                               "B_eps_MB": "_drop"}))
                        .drop(columns="_drop").reset_index(), on="experiment", how="left")
    S.to_csv(mdir / "summary_by_experiment.csv", index=False)

    # ---- named per-method CSVs (what the README promises) ----
    for fname, mask in {"local_results.csv": S["algorithm"] == "local",
                        "centralized_results.csv": S["algorithm"] == "centralized",
                        "naive_results.csv": S["algorithm"] == "seasonal_naive",
                        "fedavg_results.csv": S["algorithm"] == "fedavg",
                        "fedprox_results.csv": S["algorithm"].isin(["fedprox", "hafedprox"])}.items():
        S[mask].to_csv(mdir / fname, index=False)

    # ---- comparison table ----
    comp_rows = []
    nv = main[main["experiment"] == "E0_naive"]
    for kind, lab in [("persistence", "Persistence (naive)"), ("seasonal_naive", "Seasonal naive (t−48)")]:
        if len(nv) and f"{kind}_mae" in nv:
            comp_rows.append({"Method": lab, "Experiment": "E0", "MAE": nv[f"{kind}_mae"].mean(), "MAE_std": 0,
                              "RMSE": nv[f"{kind}_rmse"].mean(), "RMSE_std": 0, "MAPE": nv[f"{kind}_mape"].mean(),
                              "MAPE_std": 0, "WAPE": nv[f"{kind}_wape"].mean(), "Rounds": "n/a",
                              "Communication_MB": 0.0, "Raw data leaves home": "n/a", "Seeds": 1})
    order = ["E1_local", "E2_centralized", "E3_fedavg_iid", "E4_fedavg_moderate", "E5_fedprox_moderate",
             "E10_hafedprox_a0.3", "E6_fedavg_strong", "E7_fedprox_strong", "E10_hafedprox_strong",
             "E8_fedavg_partial", "E9_fedprox_partial", "E11_hafedprox_partial", "E12_fedavg_fp16_moderate",
             "E13_fedavg_household", "E13_fedprox_household", "E13_hafedprox_household",
             "E14_fedavg_K1", "E14_fedavg_K5", "E14_fedavg_K50"]
    for name in order:
        r = S[S["experiment"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        fed = r["method"] == "federated"
        comp_rows.append({
            "Method": describe(name), "Experiment": name.split("_")[0],
            "MAE": r["test_mae_mean"], "MAE_std": r["test_mae_std"], "RMSE": r["test_rmse_mean"],
            "RMSE_std": r["test_rmse_std"], "MAPE": r["test_mape_mean"], "MAPE_std": r["test_mape_std"],
            "WAPE": r["test_wape_mean"],
            "Rounds": f"{r['best_round_mean']:.1f} / {int(r['rounds'])}" if fed else "n/a",
            "Communication_MB": r["total_bytes_mean"] / 1e6,
            "Raw data leaves home": "Yes" if name.startswith("E2") else "No",
            "Seeds": int(r["n_seeds"])})
    C = pd.DataFrame(comp_rows)
    C.to_csv(mdir / "comparison.csv", index=False)
    (mdir / "comparison.md").write_text(_markdown(C, tag))

    # ---- heterogeneity response (community mode sweep) ----
    het_rows = []
    sweep = {"fedavg": ["E3_fedavg_iid", "E10_fedavg_a1.0", "E4_fedavg_moderate", "E10_fedavg_a0.1", "E6_fedavg_strong"],
             "fedprox": ["E10_fedprox_iid", "E10_fedprox_a1.0", "E5_fedprox_moderate", "E10_fedprox_a0.1", "E7_fedprox_strong"],
             "hafedprox": ["E10_hafedprox_iid", "E10_hafedprox_a1.0", "E10_hafedprox_a0.3", "E10_hafedprox_a0.1", "E10_hafedprox_strong"]}
    labels = ["random", "α=1.0", "α=0.3", "α=0.1", "cluster"]
    cmae = S.loc[S["experiment"] == "E2_centralized", "test_mae_mean"]
    cmae = float(cmae.iloc[0]) if len(cmae) else np.nan
    for alg, names in sweep.items():
        for lab, name in zip(labels, names):
            r = S[S["experiment"] == name]
            if r.empty:
                continue
            r = r.iloc[0]
            rho = _mean_rho(rundir, name)
            het_rows.append({"algorithm": alg, "label": lab, "experiment": name, "H_mean": r["H_mean"],
                             "H_JS_mean": r["H_JS_mean"], "H_CV_mean": r["H_CV_mean"],
                             "test_mae_mean": r["test_mae_mean"], "test_mae_std": r["test_mae_std"],
                             "test_rmse_mean": r["test_rmse_mean"], "gap_to_central_pct": 100 * (r["test_mae_mean"] / cmae - 1),
                             "mean_rho": rho, "R_eps_mean": r.get("R_eps_mean", np.nan),
                             "B_eps_MB_mean": r.get("B_eps_MB_mean", np.nan)})
    H = pd.DataFrame(het_rows)
    H.to_csv(mdir / "heterogeneity.csv", index=False)

    # ---- fragmentation: same homes, random grouping, K clients (K=100 is household mode) ----
    frag_names = {1: "E14_fedavg_K1", 5: "E14_fedavg_K5", 20: "E3_fedavg_iid", 50: "E14_fedavg_K50",
                  100: "E13_fedavg_household"}
    fr = []
    for K, name in frag_names.items():
        r = S[S["experiment"] == name]
        if r.empty:
            continue
        r = r.iloc[0]
        fr.append({"K": K, "experiment": name, "test_mae_mean": r["test_mae_mean"], "test_mae_std": r["test_mae_std"],
                   "ft_mae_mean": r.get("ft_mae_mean", np.nan), "H_mean": r["H_mean"], "H_JS_mean": r["H_JS_mean"],
                   "mean_rho": _mean_rho(rundir, name), "best_round_mean": r["best_round_mean"],
                   "gap_to_central_pct": 100 * (r["test_mae_mean"] / cmae - 1)})
    FR = pd.DataFrame(fr)
    FR.to_csv(mdir / "fragmentation.csv", index=False)

    # ---- gap decomposition: walk from centralised to the hardest federated setting, one factor at a time ----
    steps = [("Centralised (pooled raw data)", "E2_centralized"),
             ("+ round protocol, 1 client (K=1)", "E14_fedavg_K1"),
             ("+ split over 20 clients, random grouping", "E3_fedavg_iid"),
             ("+ label-skewed grouping (alpha=0.3)", "E4_fedavg_moderate"),
             ("+ 50% participation, 20% stragglers", "E8_fedavg_partial"),
             ("cluster grouping instead (strong non-IID)", "E6_fedavg_strong"),
             ("1 home = 1 client (K=100)", "E13_fedavg_household")]
    dec, prev = [], None
    for lab, name in steps:
        r = S[S["experiment"] == name]
        if r.empty:
            continue
        m = float(r["test_mae_mean"].iloc[0])
        dec.append({"step": lab, "experiment": name, "test_mae_mean": m,
                    "test_mae_std": float(r["test_mae_std"].iloc[0]) if pd.notna(r["test_mae_std"].iloc[0]) else np.nan,
                    "delta_vs_previous": np.nan if prev is None or "instead" in lab or "1 home" in lab else m - prev,
                    "gap_to_central_pct": 100 * (m / cmae - 1)})
        if "instead" not in lab and "1 home" not in lab:
            prev = m
    DEC = pd.DataFrame(dec)
    DEC.to_csv(mdir / "gap_decomposition.csv", index=False)

    # ---- does rho measure heterogeneity? per-run mean rho (rounds 2..T) against measured H ----
    rr = []
    for _, r in main[(main["method"] == "federated") & (main["algorithm"] == "fedavg")].iterrows():
        f = rundir / f"{r['experiment']}__seed{r['seed']}_history.csv"
        h = pd.read_csv(f)
        late = h[h["round"] > 1]["update_disagreement_rho"]
        rr.append({"experiment": r["experiment"], "seed": r["seed"], "H": r.get("H", np.nan),
                   "H_JS": r.get("H_JS", np.nan), "client_fraction": r.get("client_fraction", np.nan),
                   "num_clients": r.get("num_clients", np.nan), "rho_round1": float(h["update_disagreement_rho"].iloc[0]),
                   "rho_mean": float(late.mean()), "rho_last": float(h["update_disagreement_rho"].iloc[-1])})
    RH = pd.DataFrame(rr)
    RH.to_csv(mdir / "rho_vs_heterogeneity.csv", index=False)

    # ---- fragmentation vs heterogeneity: OLS of per-run FedAvg test MAE on log K and H_JS ----
    REG = {}
    fa = main[(main["method"] == "federated") & (main["algorithm"] == "fedavg") & (main["client_fraction"] == 1.0)
              & (main["update_precision"] == "float32") & (main["num_clients"] > 1)]
    if len(fa) >= 8 and fa["num_clients"].nunique() >= 3:
        from scipy import stats as _st
        X = np.column_stack([np.ones(len(fa)), np.log(fa["num_clients"]), fa["H_JS"]])
        y = fa["test_mae"].to_numpy()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ b
        dof = len(y) - X.shape[1]
        se = np.sqrt(np.diag(resid @ resid / dof * np.linalg.inv(X.T @ X)))
        pv = 2 * _st.t.sf(np.abs(b / se), dof)
        k20 = fa[fa["num_clients"] == 20]
        rs20 = _st.spearmanr(k20["H_JS"], k20["test_mae"])
        REG = {"b_logK": b[1], "p_logK": pv[1], "b_HJS": b[2], "p_HJS": pv[2],
               "r2": 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())), "n": len(y),
               "rho_k20": rs20.statistic, "p_k20": rs20.pvalue, "n_k20": len(k20)}
        pd.DataFrame([REG]).to_csv(mdir / "fragmentation_regression.csv", index=False)

    # ---- paired statistics over households ----
    hh = {n: _households(rundir, n) for n in S["experiment"]}
    pairs = [("E4_fedavg_moderate", "E1_local"), ("E4_fedavg_moderate", "E2_centralized"),
             ("E3_fedavg_iid", "E2_centralized"), ("E5_fedprox_moderate", "E4_fedavg_moderate"),
             ("E7_fedprox_strong", "E6_fedavg_strong"), ("E9_fedprox_partial", "E8_fedavg_partial"),
             ("E10_hafedprox_a0.3", "E4_fedavg_moderate"), ("E10_hafedprox_a0.3", "E5_fedprox_moderate"),
             ("E10_hafedprox_strong", "E6_fedavg_strong"), ("E10_hafedprox_strong", "E7_fedprox_strong"),
             ("E11_hafedprox_partial", "E8_fedavg_partial"), ("E11_hafedprox_partial", "E9_fedprox_partial"),
             ("E6_fedavg_strong", "E3_fedavg_iid"), ("E13_fedavg_household", "E1_local"),
             ("E13_fedavg_household", "E2_centralized"), ("E13_hafedprox_household", "E13_fedavg_household"),
             ("E12_fedavg_fp16_moderate", "E4_fedavg_moderate"), ("E2_centralized", "E1_local"),
             ("E14_fedavg_K1", "E2_centralized"), ("E3_fedavg_iid", "E14_fedavg_K1"),
             ("E4_fedavg_moderate:ft", "E1_local"), ("E4_fedavg_moderate:ft", "E2_centralized"),
             ("E13_fedavg_household:ft", "E1_local")]

    def _col(key):
        name, _, ft = key.partition(":")
        h = hh.get(name)
        col = "ft_mae" if ft else "mae"
        return None if h is None or col not in h else h[col]

    st = []
    for a, b in pairs:
        A, B = _col(a), _col(b)
        if A is None or B is None:
            continue
        med, p, n = _wilcoxon(A, B)
        st.append({"A": a, "B": b, "median_diff_mae_A_minus_B": med, "p_value": p, "n_households": n,
                   "share_households_A_better": float((A < B.reindex(A.index)).mean())})
    ST = pd.DataFrame(st)
    if len(ST):
        ST["p_holm"] = _holm(ST["p_value"].to_numpy())
    ST.to_csv(mdir / "stats_tests.csv", index=False)

    # ---- equity: who benefits from federation ----
    eq_rows, strata_rows, benefits = [], [], {}
    loc = hh.get("E1_local")
    for name in ["E2_centralized", "E3_fedavg_iid", "E4_fedavg_moderate", "E6_fedavg_strong",
                 "E10_hafedprox_a0.3", "E13_fedavg_household", "E13_hafedprox_household"]:
        h = hh.get(name)
        if h is None or loc is None:
            continue
        ben = loc["mae"] - h["mae"].reindex(loc.index)
        row = {"experiment": name, "share_harmed": float((ben < 0).mean()), "median_benefit_kwh": float(ben.median()),
               "worst_decile_mae": float(h["mae"].quantile(0.9)), "local_worst_decile_mae": float(loc["mae"].quantile(0.9))}
        if "ft_mae" in h:
            ben_ft = loc["mae"] - h["ft_mae"].reindex(loc.index)
            row.update({"ft_share_harmed": float((ben_ft < 0).mean()), "ft_median_benefit_kwh": float(ben_ft.median()),
                        "ft_test_mae_hh_mean": float(h["ft_mae"].mean())})
        eq_rows.append(row)
    EQ = pd.DataFrame(eq_rows)
    EQ.to_csv(mdir / "equity.csv", index=False)
    for name, arm in [("E1_local", "local"), ("E2_centralized", "centralized"), ("E4_fedavg_moderate", "fedavg"),
                      ("E10_hafedprox_a0.3", "hafedprox")]:
        h = hh.get(name)
        if h is None or "stratum" not in h:
            continue
        for s, g in h.groupby("stratum"):
            strata_rows.append({"stratum": int(s), "arm": arm, "experiment": name, "mae_median": g["mae"].median(),
                                "mae_mean": g["mae"].mean(), "wape_median": g["wape"].median()})
    STR = pd.DataFrame(strata_rows)
    STR.to_csv(mdir / "equity_by_stratum.csv", index=False)

    _make_plots(cfg, S, H, STR, hh, rundir, pdir, tag, central_val)
    hist_rho = {lab: _history(rundir, n) for n, lab in [("E3_fedavg_iid", "random"), ("E4_fedavg_moderate", "α=0.3"),
                                                        ("E6_fedavg_strong", "cluster"),
                                                        ("E13_fedavg_household", "1 home = 1 client")]}
    hist_rho = {k: v for k, v in hist_rho.items() if v is not None}
    if hist_rho:
        plots.rho_vs_round(hist_rho, pdir / "rho_vs_round.png", tag)
    if len(FR) >= 2:
        refs = {}
        if len(cmae_s := S.loc[S["experiment"] == "E2_centralized", "test_mae_mean"]):
            refs["Centralised"] = ("centralized", float(cmae_s.iloc[0]))
        if len(lm := S.loc[S["experiment"] == "E1_local", "test_mae_mean"]):
            refs["Local-only"] = ("local", float(lm.iloc[0]))
        plots.fragmentation(FR, pdir / "fragmentation.png", refs, tag)
    if len(DEC) >= 3 and "synthetic" not in tag.lower():
        def _b(n):
            r = S.loc[S["experiment"] == n, "B_eps_MB_mean"] if "B_eps_MB_mean" in S else []
            return float(r.iloc[0]) if len(r) and pd.notna(r.iloc[0]) else None
        beps = {"Raw data upload\n(centralised)": float(S.loc[S["experiment"] == "E2_centralized",
                                                             "total_bytes_mean"].iloc[0]) / 1e6}
        for n, lab in [("E3_fedavg_iid", "FedAvg\nrandom"), ("E4_fedavg_moderate", "FedAvg\nα=0.3"),
                       ("E6_fedavg_strong", "FedAvg\nclustered")]:
            if (b := _b(n)) is not None:
                beps[lab] = b
        lg = S.loc[S["experiment"] == "E1_local", "test_mae_mean"]
        gap = {n: float(100 * (S.loc[S["experiment"] == n, "test_mae_mean"].iloc[0] / cmae - 1))
               for n in S["experiment"]}
        ga_rows = [(lab, gap[n], kind) for n, lab, kind in [
            ("E14_fedavg_K1", "K = 1 (all homes in one client)", "K"), ("E14_fedavg_K5", "K = 5, random", "K"),
            ("E3_fedavg_iid", "K = 20, random", "K"), ("E4_fedavg_moderate", "K = 20, Dirichlet α = 0.3", "H"),
            ("E6_fedavg_strong", "K = 20, clustered by profile", "H"), ("E14_fedavg_K50", "K = 50, random", "K"),
            ("E13_fedavg_household", "K = 100, one home per client", "K")] if n in gap]
        plots.graphical_abstract(ga_rows, beps, float(100 * (lg.iloc[0] / cmae - 1)) if len(lg) else np.nan,
                                 pdir / "graphical_abstract.png")
    if write_paper is None:
        write_paper = "synthetic" not in tag.lower() and "sample" not in tag.lower()
    if write_paper:
        _latex(S, C, H, ST, EQ, STR, cfg, central_val, FR, DEC, RH, REG)
    return {"comparison": C, "summary": S}


def _mean_rho(rundir, name):
    files = _seed_files(rundir, name, "_history.csv")
    vals = [pd.read_csv(f)["update_disagreement_rho"].mean() for f in files]
    return float(np.nanmean(vals)) if vals else np.nan


def _holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out


def describe(name: str) -> str:
    fixed = {"E1_local": "Local-only LSTM", "E2_centralized": "Centralised LSTM (pooled raw data)",
             "E12_fedavg_fp16_moderate": "FedAvg, α=0.3, float16 updates"}
    if name in fixed:
        return fixed[name]
    base, _, ft = name.partition(":")
    if ft:
        return describe(base) + " + local FT"
    if name.startswith("E14_"):
        return f"FedAvg, K={name.rsplit('K', 1)[1]} client(s), random"
    alg = "HA-FedProx" if "hafedprox" in name else "FedProx" if "fedprox" in name else "FedAvg"
    if "household" in name:
        return f"{alg}, 1 household = 1 client"
    if "partial" in name:
        return f"{alg}, α=0.3, 50% participation + 20% stragglers"
    key = name.split("_")[-1]
    return f"{alg}, {ALPHA_LABEL.get(key, key)}"


def _markdown(C: pd.DataFrame, tag: str) -> str:
    head = ("| Method | MAE (kWh) | RMSE (kWh) | MAPE* (%) | WAPE (%) | Best round / rounds | Communication (MB) | Raw data leaves home | Seeds |\n"
            "|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r.Method} | {_fmt(r.MAE, r.MAE_std)} | {_fmt(r.RMSE, r.RMSE_std)} | {_fmt(r.MAPE, r.MAPE_std, 2)} | "
        f"{r.WAPE:.2f} | {r.Rounds} | {r.Communication_MB:.1f} | {r._12} | {r.Seeds} |\n"
        for r in C.itertuples())
    note = ("\n*MAPE over half-hours with actual load ≥ the configured threshold (0.05 kWh by default). "
            "Mean ± std over seeds. Centralised 'communication' is the raw training data volume (value + timestamp).\n")
    banner = "> **SYNTHETIC DEMO DATA — these numbers are not project results.**\n\n" if "synthetic" in tag.lower() else ""
    return f"# Comparison ({tag})\n\n{banner}{head}{body}{note}"


def _make_plots(cfg, S, H, STR, hh, rundir, pdir, tag, central_val):
    hist = {n: _history(rundir, n) for n in S["experiment"] if (S.loc[S["experiment"] == n, "method"] == "federated").any()}
    curve_sets = [("E4_fedavg_moderate", "fedavg", "FedAvg, α=0.3"), ("E5_fedprox_moderate", "fedprox", "FedProx, α=0.3"),
                  ("E10_hafedprox_a0.3", "hafedprox", "HA-FedProx, α=0.3"), ("E6_fedavg_strong", "fedavg", "FedAvg, strong"),
                  ("E7_fedprox_strong", "fedprox", "FedProx, strong"), ("E10_hafedprox_strong", "hafedprox", "HA-FedProx, strong")]
    curves = {lab: (alg, hist[n]) for n, alg, lab in curve_sets if hist.get(n) is not None}
    if not curves:
        curves = {n: (S.loc[S["experiment"] == n, "algorithm"].iloc[0], h) for n, h in hist.items() if h is not None}
    ref = {}
    c = S[S["experiment"] == "E2_centralized"]
    loc = S[S["experiment"] == "E1_local"]
    if curves:
        for metric, ylab, fname, refcol in [("val_loss", "Validation loss (MSE, kWh²)", "loss_vs_round.png", None),
                                            ("test_mae", "Test MAE (kWh / half-hour)", "mae_vs_round.png", "test_mae_mean"),
                                            ("test_rmse", "Test RMSE (kWh / half-hour)", "rmse_vs_round.png", "test_rmse_mean"),
                                            ("test_mape", "Test MAPE (%)", "mape_vs_round.png", "test_mape_mean")]:
            ref = {}
            if refcol and len(c):
                ref["Centralised"] = ("centralized", float(c[refcol].iloc[0]))
            if refcol and len(loc):
                ref["Local-only"] = ("local", float(loc[refcol].iloc[0]))
            plots.metric_vs_round(curves, metric, ylab, pdir / fname,
                                  f"{ylab.split(' (')[0]} per round (mean ± std over seeds)", tag, ref)
        comm = {lab: v for lab, v in curves.items() if "strong" not in lab}
        for n, alg, lab in [("E8_fedavg_partial", "fedavg", "FedAvg, 50% participation"),
                            ("E13_fedavg_household", "fedavg", "FedAvg, household clients"),
                            ("E12_fedavg_fp16_moderate", "fedavg", "FedAvg, float16")]:
            if hist.get(n) is not None:
                comm[lab] = (alg, hist[n])
        plots.communication_vs_round(comm, pdir / "communication_vs_round.png", tag)
        cv = float(np.mean(list(central_val.values()))) if central_val else None
        plots.error_vs_bytes(curves, pdir / "mae_vs_cumulative_bytes.png", cv, tag)
        mus = {lab: hist[n] for n, lab in [("E10_hafedprox_iid", "random"), ("E10_hafedprox_a1.0", "α=1.0"),
                                         ("E10_hafedprox_a0.3", "α=0.3"), ("E10_hafedprox_a0.1", "α=0.1"),
                                         ("E10_hafedprox_strong", "cluster")] if hist.get(n) is not None}
        if mus:
            plots.mu_trajectory(mus, pdir / "hafedprox_mu_vs_round.png", tag)
    if len(H) and len(c):
        plots.heterogeneity_response(H, pdir / "heterogeneity_response.png", float(c["test_mae_mean"].iloc[0]), tag)
    if len(STR):
        plots.equity_by_stratum(STR, pdir / "equity_by_stratum.png", tag)
    loc_hh = hh.get("E1_local")
    if loc_hh is not None:
        ben = {}
        for n, alg, lab, col in [("E2_centralized", "centralized", "Centralised", "mae"),
                                 ("E4_fedavg_moderate", "fedavg", "FedAvg, α=0.3", "mae"),
                                 ("E4_fedavg_moderate", "finetuned", "FedAvg, α=0.3 + local FT", "ft_mae"),
                                 ("E13_fedavg_household", "fedavg", "FedAvg, household clients", "mae")]:
            if hh.get(n) is not None and col in hh[n]:
                ben[lab] = (alg, (loc_hh["mae"] - hh[n][col].reindex(loc_hh.index)).to_numpy())
        if ben:
            plots.benefit_distribution(ben, pdir / "federation_benefit_cdf.png", tag)
    _prediction_plot(cfg, S, pdir, tag)


def _prediction_plot(cfg, S, pdir, tag):
    pred_dir = results_path(cfg, "predictions", "x").parent
    frames = {}
    for n, alg, lab in [("E1_local", "local", "Local-only"), ("E2_centralized", "centralized", "Centralised"),
                        ("E4_fedavg_moderate", "fedavg", "FedAvg"), ("E10_hafedprox_a0.3", "hafedprox", "HA-FedProx")]:
        files = sorted(pred_dir.glob(f"{n}__seed*.parquet"))
        if files:
            frames[lab] = (alg, pd.read_parquet(files[0]))
    if not frames:
        files = sorted(pred_dir.glob("*.parquet"))
        if not files:
            return
        frames = {files[0].stem: ("fedavg", pd.read_parquet(files[0]))}
    first = next(iter(frames.values()))[1]
    ids = sorted(first["LCLid"].unique())
    # the household closest to the cohort's median load, chosen without looking at any error
    means = first.groupby("LCLid")["y_kwh"].mean()
    hid = (means - means.median()).abs().idxmin() if len(means) else ids[0]
    plots.prediction_example(frames, hid, pdir / "prediction_example.png", 3, tag)


# ───────────────────────────── LaTeX for the manuscript ─────────────────────────────

def _latex(S, C, H, ST, EQ, STR, cfg, central_val, FR=None, DEC=None, RH=None, REG=None):
    out = resolve("paper/generated")
    out.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & MAE & RMSE & MAPE$^\ast$ & WAPE & Best rd. & Traffic (MB) \\", r"\midrule"]
    for r in C.itertuples():
        lines.append(f"{_tex(_short(r.Method))} & {_pm(r.MAE, r.MAE_std)} & {_pm(r.RMSE, r.RMSE_std)} & "
                     f"{_pm(r.MAPE, r.MAPE_std, 1)} & {r.WAPE:.1f} & {_tex(str(r.Rounds))} & {r.Communication_MB:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table_main.tex").write_text("\n".join(lines) + "\n")

    if len(H):
        L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Partition & Method & $H$ & $H_{JS}$ & $\bar\rho$ & Test MAE & Gap to central (\%) \\", r"\midrule"]
        for r in H.sort_values(["H_mean", "algorithm"]).itertuples():
            L.append(f"{_tex(r.label)} & {plots.LABELS[r.algorithm]} & {r.H_mean:.3f} & {r.H_JS_mean:.4f} & "
                     f"{r.mean_rho:.3f} & {_pm(r.test_mae_mean, r.test_mae_std)} & {r.gap_to_central_pct:+.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "table_heterogeneity.tex").write_text("\n".join(L) + "\n")

    if len(ST):
        L = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"A & B & median $\Delta$MAE (A$-$B) & A better (\%) & $p$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
        for r in ST.itertuples():
            L.append(f"{_tex(_short(describe(r.A)))} & {_tex(_short(describe(r.B)))} & {r.median_diff_mae_A_minus_B:+.4f} & "
                     f"{100 * r.share_households_A_better:.0f} & {_p(r.p_value)} & {_p(r.p_holm)} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "table_stats.tex").write_text("\n".join(L) + "\n")

    if len(EQ):
        L = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Arm & Harmed vs local (\%) & Median benefit (kWh) & P90 MAE & Harmed after FT (\%) \\", r"\midrule"]
        for r in EQ.itertuples():
            ft = f"{100 * r.ft_share_harmed:.0f}" if hasattr(r, "ft_share_harmed") and pd.notna(r.ft_share_harmed) else "--"
            L.append(f"{_tex(_short(describe(r.experiment)))} & {100 * r.share_harmed:.0f} & {r.median_benefit_kwh:+.4f} & "
                     f"{r.worst_decile_mae:.4f} & {ft} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "table_equity.tex").write_text("\n".join(L) + "\n")

    # \newcommand macros so the manuscript text cites numbers straight from the results
    mac = {}
    for r in S.itertuples():
        key = _mkey(r.experiment)
        mac[f"{key}MAE"] = f"{r.test_mae_mean:.4f}"
        mac[f"{key}RMSE"] = f"{r.test_rmse_mean:.4f}"
        mac[f"{key}MAPE"] = f"{r.test_mape_mean:.1f}"
        mac[f"{key}WAPE"] = f"{r.test_wape_mean:.1f}"
        mac[f"{key}MB"] = f"{r.total_bytes_mean / 1e6:.1f}"
        if pd.notna(getattr(r, "H_mean", np.nan)):
            mac[f"{key}H"] = f"{r.H_mean:.3f}"
        if pd.notna(getattr(r, "best_round_mean", np.nan)):
            mac[f"{key}Round"] = f"{r.best_round_mean:.1f}"
        if pd.notna(getattr(r, "ft_mae_mean", np.nan)):
            mac[f"{key}FT"] = f"{r.ft_mae_mean:.4f}"
        if pd.notna(getattr(r, "test_mae_std", np.nan)):
            mac[f"{key}MAEsd"] = f"{r.test_mae_std:.4f}"
        if pd.notna(getattr(r, "R_eps_mean", np.nan)):
            mac[f"{key}Reps"] = f"{r.R_eps_mean:.1f}"
            mac[f"{key}Beps"] = f"{r.B_eps_MB_mean:.1f}"
        if pd.notna(getattr(r, "hh_mae_p90_mean", np.nan)):
            mac[f"{key}PninetyMAE"] = f"{r.hh_mae_p90_mean:.4f}"
        cm = S.loc[S["experiment"] == "E2_centralized", "test_mae_mean"]
        if len(cm):
            mac[f"{key}Gap"] = f"{100 * (r.test_mae_mean / float(cm.iloc[0]) - 1):+.1f}"
    if len(C):
        for lab, key in [("Persistence (naive)", "Persistence"), ("Seasonal naive (t−48)", "SeasonalNaive")]:
            row = C[C["Method"] == lab]
            if len(row):
                mac[f"{key}MAE"] = f"{row['MAE'].iloc[0]:.4f}"
                mac[f"{key}RMSE"] = f"{row['RMSE'].iloc[0]:.4f}"
    any_run = S.iloc[0]
    mac.update({"NumHouseholds": str(int(any_run["num_households"])), "NumParams": f"{int(any_run['num_params']):,}".replace(",", "{,}"),
                "PayloadKB": f"{4 * int(any_run['num_params']) / 1e3:.1f}"})
    rep = json.loads(resolve(cfg["data"]["processed_dir"]).joinpath("preprocess_report.json").read_text())
    mac.update({"HouseholdsSeen": f"{rep['households_seen']:,}".replace(",", "{,}"),
                "HouseholdsEligible": f"{rep['households_eligible']:,}".replace(",", "{,}"),
                "RowsInPeriod": f"{rep['rows_in_period'] / 1e6:.1f}"})
    tpath = results_path(cfg, "metrics", "tuning.json")
    if tpath.exists():
        t = json.loads(tpath.read_text())
        mac.update({"TunedMuFedProx": f"{t.get('mu_fedprox', float('nan')):g}",
                    "TunedMuHA": f"{t.get('mu_hafedprox', float('nan')):g}"})
    if FR is not None and len(FR):
        L = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$K$ & homes/client & $H_{JS}$ & Test MAE & FT MAE & $\bar\rho$ & Gap to central (\%) \\", r"\midrule"]
        for r in FR.itertuples():
            ft = f"{r.ft_mae_mean:.4f}" if pd.notna(r.ft_mae_mean) else "--"
            L.append(f"{r.K} & {100 // r.K} & {r.H_JS_mean:.4f} & {_pm(r.test_mae_mean, r.test_mae_std)} & {ft} & "
                     f"{r.mean_rho:.3f} & {r.gap_to_central_pct:+.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "table_fragmentation.tex").write_text("\n".join(L) + "\n")
    if DEC is not None and len(DEC):
        L = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Setting & Test MAE & $\Delta$ vs previous & Gap to central (\%) \\", r"\midrule"]
        for r in DEC.itertuples():
            d = f"{r.delta_vs_previous:+.4f}" if pd.notna(r.delta_vs_previous) else "--"
            L.append(f"{_tex(r.step).replace('alpha', '$\\alpha$')} & {_pm(r.test_mae_mean, r.test_mae_std)} & {d} & "
                     f"{r.gap_to_central_pct:+.1f} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        (out / "table_decomposition.tex").write_text("\n".join(L) + "\n")
    if RH is not None and len(RH) >= 4:
        from scipy.stats import spearmanr
        # same K and full participation, so only the grouping differs between runs
        sub = RH[(RH["num_clients"] == 20) & (RH["client_fraction"] == 1.0)].dropna(subset=["H_JS"])
        if len(sub) >= 4:
            rs, rp = spearmanr(sub["H_JS"], sub["rho_mean"])
            r1s, r1p = spearmanr(sub["H_JS"], sub["rho_round1"])
            mac.update({"RhoHSpearman": f"{rs:+.2f}", "RhoHSpearmanP": _p(rp), "RhoHRuns": str(len(sub)),
                        "RhoOneHSpearman": f"{r1s:+.2f}", "RhoOneHSpearmanP": _p(r1p)})
            per_part = sub.groupby("experiment")["rho_mean"].mean()
            between = float(per_part.max() - per_part.min())
            within = float((sub["rho_last"] - sub["rho_round1"]).mean())
            mac.update({"RhoBetween": f"{between:.3f}", "RhoWithin": f"{within:.2f}",
                        "RhoWithinOverBetween": f"{within / between:.0f}" if between > 0 else "--"})
        comp = RH[(RH["num_clients"] == 20) & (RH["client_fraction"] == 1.0)]
        if len(comp):
            mac.update({"RhoRoundOne": f"{comp['rho_round1'].mean():.2f}", "RhoLast": f"{comp['rho_last'].mean():.2f}",
                        "RhoMin": f"{comp['rho_mean'].min():.2f}", "RhoMax": f"{comp['rho_mean'].max():.2f}"})
    if REG:
        mac.update({"RegLogK": f"{1e3 * REG['b_logK']:.2f}", "RegLogKP": _p(REG["p_logK"]),
                    "RegHJS": f"{REG['b_HJS']:.3f}", "RegHJSP": _p(REG["p_HJS"]), "RegRtwo": f"{REG['r2']:.2f}",
                    "RegN": str(REG["n"]), "KtwentySpearman": f"{REG['rho_k20']:+.2f}",
                    "KtwentySpearmanP": _p(REG["p_k20"]), "KtwentyN": str(REG["n_k20"])})
    if len(ST):
        for r in ST.itertuples():
            k = f"{_mkey(r.A)}vs{_mkey(r.B)}"
            mac[f"St{k}Better"] = f"{100 * r.share_households_A_better:.0f}"
            mac[f"St{k}P"] = _p(r.p_holm)
            mac[f"St{k}Med"] = f"{r.median_diff_mae_A_minus_B:+.4f}"
    if len(EQ):
        for r in EQ.itertuples():
            k = _mkey(r.experiment)
            mac[f"Eq{k}Harmed"] = f"{100 * r.share_harmed:.0f}"
            if hasattr(r, "ft_share_harmed") and pd.notna(r.ft_share_harmed):
                mac[f"Eq{k}FTHarmed"] = f"{100 * r.ft_share_harmed:.0f}"
    (out / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    figdir = resolve("paper/figures")
    figdir.mkdir(parents=True, exist_ok=True)
    for png in results_path(cfg, "plots", "x").parent.glob("*.png"):
        shutil.copy(png, figdir / png.name)


def _short(label: str) -> str:
    """Compact arm names for wide LaTeX tables."""
    for long, short in [("Centralised LSTM (pooled raw data)", "Centralised"), ("Local-only LSTM", "Local-only"),
                        (", 50% participation + 20% stragglers", ", partial"), (", 1 household = 1 client", ", K=100"),
                        (" (IID-ish)", ""), (" (strong)", ""), (" client(s), random", ", random"),
                        (", float16 updates", ", fp16")]:
        label = label.replace(long, short)
    return label


_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _mkey(name: str) -> str:
    """Experiment name -> LaTeX-safe macro stem, e.g. E10_fedavg_a0.1 -> Eonezerofedavgazeropone;
    a ':ft' suffix (fine-tuned variant) becomes 'FT'."""
    name, _, ft = name.partition(":")
    name = name.replace(".", "p")
    out = "".join(_DIGITS[int(c)] if c.isdigit() else c for c in name)
    return re.sub(r"[^A-Za-z]", "", out) + ("FT" if ft else "")


def _tex(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("α", r"$\alpha$")
            .replace("−", "--").replace("±", r"$\pm$"))


def _pm(m, s, nd=4):
    if pd.isna(m):
        return "--"
    return f"{m:.{nd}f}" + (f" $\\pm$ {s:.{nd}f}" if pd.notna(s) and s > 0 else "")


def _p(p):
    if pd.isna(p):
        return "--"
    return "$<10^{-4}$" if p < 1e-4 else f"{p:.4f}"
