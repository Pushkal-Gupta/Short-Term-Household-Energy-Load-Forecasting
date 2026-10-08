# Data

## Primary dataset: Low Carbon London (LCL)

| | |
|---|---|
| Publisher | UK Power Networks, via the London Datastore |
| Page | <https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d> |
| File used | `Partitioned LCL Data.zip`: 795,722,689 bytes, 168 CSVs, 8.5 GB uncompressed |
| Content | 5,567 London households, half-hourly kWh, Nov 2011 – Feb 2014 |
| Columns | `LCLid, stdorToU, DateTime, KWH/hh (per half hour)` |
| Licence | See the dataset page. The data is **not** redistributed in this repository or its ZIP. |

### Getting it

```bash
python -m src.data.download          # resumable download into data/raw/ (uses curl)
```

Or download `Partitioned LCL Data.zip` by hand from the page above and put it in `data/raw/`.
**Do not unzip it.** `src/data/preprocess.py` streams the CSVs directly from the zip.

Also accepted in `data/raw/`:
- loose `LCL-June2015v2_*.csv` files (the same data, extracted);
- the single-file `CC_LCL-FullData.csv` from `LCL-FullData.zip`;
- the Kaggle derivative "Smart meters in London" (`halfhourly_dataset/block_*.csv`, with columns
  `LCLid, tstp, energy(kWh/hh)`). Point `data.household_info` at its `informations_households.csv`
  to attach ACORN socio-demographic groups and tariff labels.

If nothing is found, every command stops with a message that explains where to put the data.

## What preprocessing does (`python -m src.data.preprocess`)

| Step | Rule (config key) | Count on the real data (2013) |
|---|---|---|
| Study period | 2013-01-01 to 2013-12-31 (`data.start`, `data.end`) | 93.1 M readings in period |
| Unparseable / `Null` readings | set to missing | 12 |
| Negative readings | set to missing | 0 |
| Readings > 10 kWh per half-hour | set to missing (`data.max_kwh_per_hh`) | 13 |
| Duplicate (household, half-hour) rows | averaged | 63,613 |
| Grid | timestamps floored to 30 min, full grid re-indexed (`data.interval`) | |
| Short gaps ≤ 2 h | linear interpolation (`data.max_gap_steps`) | |
| Longer gaps | left missing; windows touching them are dropped | |
| Tariff | standard-tariff homes only (`data.tariff: Std`); ToU homes had dynamic prices in 2013 | 4,411 Std, 1,117 ToU |
| Coverage | ≥ 95 % non-missing half-hours in 2013 (`data.min_coverage`) | 4,034 eligible |
| Cohort | 100 homes (`data.num_households`), seeded and stratified by quintile of **train-period** mean load (`data.selection_seed`) | 20 per quintile |

The counts come from `data/processed/preprocess_report.json`, which is regenerated on every run.

**Splits (chronological, per household):** train Jan–Aug 2013, validation Sep–Oct 2013,
test Nov–Dec 2013 (`data.train_end`, `data.val_end`).

**Scaling:** each household gets a standard scaler fitted on its own train split only. The same
per-household scaling is used by every arm, the centralised one included. Metrics are reported in
kWh after inverse scaling.

## Synthetic demo data

`python -m src.experiments.run_demo` generates `data/synthetic_demo/raw/synthetic_LCL_format_NOT_REAL.csv`.
It contains 12 fake households with ids `SYNTH_000…`, written in the LCL column format and seeded with
deliberate defects (Null, negative, spike, duplicates, gaps). It exists only to test the software. It is
never mixed with `data/raw/`, and its results go to `results/demo/` with every figure stamped
"SYNTHETIC DEMO DATA".

## Folders

| Folder | Contents | In git? |
|---|---|---|
| `raw/` | downloaded LCL zip | no |
| `processed/` | cleaned series (parquet), cohort, summaries, cleaning report | no (regenerable) |
| `clients/` | kept empty on purpose: clients are built in memory from `processed/` every run | README only |
| `synthetic_demo/` | demo data | no |
