"""Load, clean and select Low Carbon London households.

    python -m src.data.preprocess                       # real data, configs/default.yaml
    python -m src.data.preprocess --config configs/demo.yaml

Steps (all choices are config values, see `data:` in configs/default.yaml):
 1. Find raw CSVs under `data.raw_dir`, either loose or still inside the downloaded zip(s).
    The London Datastore format (LCLid, stdorToU, DateTime, KWH/hh) and the Kaggle derivative
    (LCLid, tstp, energy(kWh/hh)) are both recognised.
 2. Keep readings inside the study period. Parse timestamps and floor them to the 30-min grid.
 3. Readings that are `Null`, negative, or above `max_kwh_per_hh` become missing.
 4. Duplicate (household, timestamp) rows are averaged.
 5. Each household is re-indexed to the complete half-hourly grid, and coverage is measured.
 6. Gaps of at most `max_gap_steps` are linearly interpolated. Longer gaps stay missing, and
    any window that touches them is dropped later (window.py).
 7. Households are selected by tariff (`Std`) and coverage. A seeded sample of
    `num_households` is drawn, stratified by quintile of TRAIN-period mean load (no
    test-period information is used to choose the cohort).

Outputs go to `data.processed_dir`:
 all_households_summary.csv   every household seen: tariff, coverage, train-period mean
 eligible_series.parquet      cleaned series of every eligible household (cache)
 series.parquet               cleaned series of the selected households (wide: timestamp x LCLid)
 households.csv               the selected cohort and its stratum
 preprocess_report.json       row counts for every cleaning rule
"""
from __future__ import annotations

import io
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import base_parser, config_from_args, resolve
from src.utils import log, setup_logging

ID_COLS = ("LCLid",)
TIME_COLS = ("DateTime", "tstp")
TARIFF_COLS = ("stdorToU",)


class DatasetNotFoundError(FileNotFoundError):
    pass


# ───────────────────────────── raw file discovery ─────────────────────────────

def find_members(raw_dir: Path) -> list[tuple[str, str | None]]:
    """Return (path, zip_member) pairs for every CSV under raw_dir (zip_member=None for loose files)."""
    if not raw_dir.exists():
        raise DatasetNotFoundError(_missing_msg(raw_dir))
    members: list[tuple[str, str | None]] = []
    for p in sorted(raw_dir.rglob("*")):
        if p.suffix.lower() == ".csv" and "informations_households" not in p.name:
            members.append((str(p), None))
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for name in sorted(z.namelist()):
                    if name.lower().endswith(".csv") and "__MACOSX" not in name \
                            and "informations_households" not in name:
                        members.append((str(p), name))
    if not members:
        raise DatasetNotFoundError(_missing_msg(raw_dir))
    # A full single-file export and the partitioned files may both be present: prefer partitioned.
    partitioned = [m for m in members if "LCL-June2015v2_" in (m[1] or m[0])]
    # The Kaggle derivative also ships daily/weather CSVs; keep only its half-hourly blocks.
    kaggle = [m for m in members if "halfhourly" in (m[1] or m[0]).lower()]
    return partitioned or kaggle or members


def _missing_msg(raw_dir: Path) -> str:
    return (f"No Low Carbon London CSV/zip found under {raw_dir}.\n"
            f"  Download it with:  python -m src.data.download\n"
            f"  or place 'Partitioned LCL Data.zip' (or its CSVs) in {raw_dir}/ — see data/README.md.\n"
            f"  For a software-only check without the dataset run:  python -m src.experiments.run_demo")


def _open(path: str, member: str | None):
    if member is None:
        return open(path, "rb")
    with zipfile.ZipFile(path) as z:
        return io.BytesIO(z.read(member))


def _pick(cols, options):
    for c in cols:
        if c.strip() in options:
            return c
    return None


# ───────────────────────────── per-file cleaning ─────────────────────────────

def _clean_member(args) -> dict:
    path, member, start, end, max_kwh, interval = args
    with _open(path, member) as fh:
        df = pd.read_csv(fh, dtype=str, keep_default_na=False)
    id_c = _pick(df.columns, ID_COLS)
    t_c = _pick(df.columns, TIME_COLS)
    tar_c = _pick(df.columns, TARIFF_COLS)
    kwh_c = next((c for c in df.columns if "kwh" in c.lower() or "energy" in c.lower()), None)
    if not (id_c and t_c and kwh_c):
        raise ValueError(f"Unrecognised columns in {member or path}: {list(df.columns)}")
    t = df[t_c].str.strip()
    in_period = (t >= start) & (t < end)
    df, t = df[in_period], t[in_period]
    stats = {"rows_in_period": int(len(df))}
    ts = pd.to_datetime(t.str[:19], format="%Y-%m-%d %H:%M:%S").dt.floor(interval)
    raw = df[kwh_c].str.strip()
    kwh = pd.to_numeric(raw, errors="coerce")
    stats["null_or_unparseable"] = int(kwh.isna().sum())
    stats["negative"] = int((kwh < 0).sum())
    stats["above_max"] = int((kwh > max_kwh).sum())
    kwh = kwh.where((kwh >= 0) & (kwh <= max_kwh))
    out = pd.DataFrame({"id": df[id_c].str.strip().values, "ts": ts.values, "kwh": kwh.values})
    g = out.groupby(["id", "ts"], sort=False)["kwh"].mean()
    stats["duplicates_averaged"] = int(len(out) - len(g))
    tariffs = {}
    if tar_c:
        tariffs = df.groupby(df[id_c].str.strip())[tar_c].agg(lambda s: s.str.strip().mode().iat[0]).to_dict()
    g = g.reset_index()
    return {"id": g["id"].to_numpy(), "ts": g["ts"].to_numpy("datetime64[ns]"),
            "kwh": g["kwh"].to_numpy(np.float32), "tariffs": tariffs, "stats": stats}


def fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate runs of NaN no longer than max_gap that have data on both sides."""
    x = x.astype(np.float32).copy()
    isnan = np.isnan(x)
    if not isnan.any():
        return x
    n = len(x)
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = np.interp(np.arange(i, j), [i - 1, j], [x[i - 1], x[j]])
            i = j
        else:
            i += 1
    return x


# ───────────────────────────── main pipeline ─────────────────────────────

def build_eligible(cfg: dict, workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = cfg["data"]
    raw_dir = resolve(d["raw_dir"])
    if d["source"] == "synthetic":
        from src.data.synthetic import make_synthetic_lcl
        make_synthetic_lcl(raw_dir, d["num_households"], d["start"], d["end"], seed=d.get("selection_seed", 42))
    members = find_members(raw_dir)
    log.info("Reading %d raw CSV file(s) from %s", len(members), raw_dir)
    jobs = [(p, m, d["start"], d["end"], d["max_kwh_per_hh"], d["interval"]) for p, m in members]
    pieces: dict[str, list[pd.Series]] = {}
    tariffs: dict[str, str] = {}
    report = {"files": len(members)}
    ex = ProcessPoolExecutor(max_workers=min(workers, len(jobs))) if min(workers, len(jobs)) > 1 else None
    try:
        for k, res in enumerate(ex.map(_clean_member, jobs) if ex else map(_clean_member, jobs)):
            for key, v in res["stats"].items():
                report[key] = report.get(key, 0) + v
            tariffs.update(res["tariffs"])
            s = pd.Series(res["kwh"], index=pd.MultiIndex.from_arrays([res["id"], res["ts"]]))
            for hid, part in s.groupby(level=0, sort=False):
                pieces.setdefault(hid, []).append(part.droplevel(0))
            if (k + 1) % 20 == 0:
                log.info("  %d/%d files", k + 1, len(jobs))
    finally:
        if ex:
            ex.shutdown()

    grid = pd.date_range(d["start"], d["end"], freq=d["interval"], inclusive="left")
    train_mask = grid < pd.Timestamp(d["train_end"])
    info = _load_household_info(d.get("household_info"))
    rows, eligible = [], {}
    for hid, parts in pieces.items():
        s = pd.concat(parts)
        if s.index.duplicated().any():  # the same household split across two files
            s = s.groupby(level=0).mean()
        x = s.reindex(grid).to_numpy(np.float32)
        coverage = float(np.isfinite(x).mean())
        tariff = tariffs.get(hid) or info.get(hid, {}).get("stdorToU")
        train_mean = float(np.nanmean(x[train_mask])) if np.isfinite(x[train_mask]).any() else np.nan
        ok_tariff = d.get("tariff") in (None, tariff) or tariff is None
        ok = ok_tariff and coverage >= d["min_coverage"] and np.isfinite(train_mean)
        rows.append({"LCLid": hid, "tariff": tariff, "coverage": coverage,
                     "train_mean_kwh": train_mean, "eligible": ok, **_acorn(info, hid)})
        if ok:
            eligible[hid] = fill_short_gaps(x, d["max_gap_steps"])
    summary = pd.DataFrame(rows).sort_values("LCLid").reset_index(drop=True)
    series = pd.DataFrame(eligible, index=grid).sort_index(axis=1)
    report.update({"households_seen": int(len(summary)),
                   "households_eligible": int(summary["eligible"].sum()),
                   "tariff_filter": d.get("tariff"), "min_coverage": d["min_coverage"]})
    if d.get("tariff") and not tariffs and not info:
        log.warning("No tariff column or household-info file: tariff filter could not be applied")
        report["tariff_filter_applied"] = False
    return series, summary, report


def _load_household_info(path) -> dict:
    if not path:
        return {}
    p = resolve(path)
    if not p.exists():
        log.warning("household_info file %s not found; continuing without ACORN labels", p)
        return {}
    df = pd.read_csv(p)
    return df.set_index("LCLid").to_dict("index")


def _acorn(info: dict, hid: str) -> dict:
    rec = info.get(hid, {})
    return {"acorn": rec.get("Acorn"), "acorn_grouped": rec.get("Acorn_grouped")}


def select_households(summary: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Seeded sample of n eligible homes, stratified by quintile of train-period mean load."""
    el = summary[summary["eligible"]].copy()
    if len(el) < n:
        raise ValueError(f"Only {len(el)} eligible households, {n} requested. "
                         f"Lower data.num_households or data.min_coverage.")
    q = min(5, n)
    el["stratum"] = pd.qcut(el["train_mean_kwh"].rank(method="first"), q, labels=False)
    rng = np.random.default_rng(seed)
    sizes = el["stratum"].value_counts().sort_index().to_numpy()
    quota = n * sizes / sizes.sum()                       # proportional allocation, largest remainder
    per = np.floor(quota).astype(int)
    per[np.argsort(-(quota - per), kind="stable")[: n - per.sum()]] += 1
    chosen = []
    for s, k in zip(range(q), per):
        ids = el.loc[el["stratum"] == s, "LCLid"].to_numpy()
        chosen.extend(rng.choice(ids, size=k, replace=False))
    return el[el["LCLid"].isin(chosen)].sort_values("LCLid").reset_index(drop=True)


def _signature(cfg: dict) -> dict:
    d = cfg["data"]
    keys = ["source", "raw_dir", "tariff", "start", "end", "train_end", "min_coverage",
            "max_gap_steps", "max_kwh_per_hh", "interval", "household_info"]
    return {k: d.get(k) for k in keys}


def run_preprocessing(cfg: dict, force: bool = False, workers: int = 8) -> dict:
    d = cfg["data"]
    out = resolve(d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    sig = _signature(cfg)
    rep_path = out / "preprocess_report.json"
    cached = rep_path.exists() and (out / "eligible_series.parquet").exists() and \
        json.loads(rep_path.read_text()).get("signature") == sig
    if cached and not force:
        report = json.loads(rep_path.read_text())
        series = pd.read_parquet(out / "eligible_series.parquet")
        summary = pd.read_csv(out / "all_households_summary.csv")
    else:
        series, summary, report = build_eligible(cfg, workers)
        series.to_parquet(out / "eligible_series.parquet")
        summary.to_csv(out / "all_households_summary.csv", index=False)
        report["signature"] = sig
    chosen = select_households(summary, d["num_households"], d.get("selection_seed", 42))
    chosen.to_csv(out / "households.csv", index=False)
    series[chosen["LCLid"].tolist()].to_parquet(out / "series.parquet")
    report.update({"households_selected": int(len(chosen)), "selection_seed": d.get("selection_seed", 42),
                   "selected_num_households": d["num_households"]})
    rep_path.write_text(json.dumps(report, indent=2))
    log.info("Preprocessing: %d seen, %d eligible, %d selected -> %s",
             report["households_seen"], report["households_eligible"], len(chosen), out)
    return report


def load_processed(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (series, households) for the configured cohort, running preprocessing if needed."""
    d = cfg["data"]
    out = resolve(d["processed_dir"])
    rep_path = out / "preprocess_report.json"
    ok = rep_path.exists() and (out / "series.parquet").exists()
    if ok:
        rep = json.loads(rep_path.read_text())
        ok = rep.get("signature") == _signature(cfg) and \
            rep.get("selected_num_households") == d["num_households"] and \
            rep.get("selection_seed") == d.get("selection_seed", 42)
    if not ok:
        run_preprocessing(cfg)
    return pd.read_parquet(out / "series.parquet"), pd.read_csv(out / "households.csv")


def main(argv=None):
    setup_logging()
    ap = base_parser("Preprocess Low Carbon London smart-meter data")
    ap.add_argument("--force", action="store_true", help="re-scan the raw files even if cached")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    try:
        rep = run_preprocessing(cfg, force=args.force, workers=args.workers)
    except DatasetNotFoundError as e:
        raise SystemExit(str(e))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
