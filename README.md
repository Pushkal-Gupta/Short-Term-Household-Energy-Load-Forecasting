# Training budget and fragmentation, more than heterogeneity, drive the accuracy cost of federated household load forecasting

Code, experiments and manuscript for a controlled, communication-priced study of federated
short-term load forecasting on the Low Carbon London (LCL) smart-meter data.

**Pushkal Gupta** · **Vanka Kaushik** · **Kalaavathi B**\*  
School of Computer Science and Engineering, Vellore Institute of Technology, Vellore 632014, Tamil Nadu, India  
\*Corresponding author: Kalaavathi B, kalaavathi.b@vit.ac.in  
ORCID: 0009-0004-7032-6847 (P. Gupta), 0009-0003-0735-264X (V. Kaushik), 0000-0003-4027-3137 (Kalaavathi B)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23493458.svg)](https://doi.org/10.5281/zenodo.23493458)
Archived release: v1.0.0 on Zenodo, https://doi.org/10.5281/zenodo.23493458 (all versions).

| | |
|---|---|
| **Data** | Low Carbon London (UK Power Networks), half-hourly kWh, 2013; main cohort of 100 standard-tariff homes and two further cohorts |
| **Task** | one home's last 24 h (48 half-hours) → its next half-hour |
| **Model** | one LSTM (64 hidden units, 17,217 parameters) shared by every arm |
| **Arms** | persistence · seasonal naive · local-only · client-only · centralised · FedAvg (homes or IID window control) · FedProx · HA-FedProx · FedAdam · FedAvg + local fine-tuning |
| **Stack** | PyTorch; deterministic in-process FL simulator; optional Flower backend that reproduces it exactly |
| **Manuscript** | [`submission/main.pdf`](submission/main.pdf) / [`submission/main.docx`](submission/main.docx) (built from [`paper/docx/content.tex`](paper/docx/content.tex)), supplement [`submission/supplementary_material.pdf`](submission/supplementary_material.pdf); `paper/manuscript.tex` (LaTeX) and `manuscript_long.docx` predate the revisions |

---

## 1. The question

Federated learning (FL) lets homes train a shared forecaster without sending their readings
anywhere. Two claims are usually made for it: that household data are so heterogeneous that plain
FedAvg needs a fix, and that FL saves bandwidth. This repository tests both claims under one
controlled protocol. The households, the split, the test set and the model stay fixed, and
one factor changes at a time:

| Factor | Settings |
|---|---|
| Pooling | local-only · client-only · centralised · federated |
| Fragmentation | the same 100 homes split into K = 1, 5, 20, 50 or 100 clients |
| Heterogeneity | IID control (training windows dealt at random) → random grouping of homes → Dirichlet α ∈ {1, 0.3, 0.1} over load-profile clusters → cluster or shape-sorted grouping |
| Training budget | 30 rounds · up to 100 rounds · to a plateau (up to 500 rounds, stopped after 50 without improvement) |
| Optimiser | client Adam reset each round (default) or kept · client SGD · server FedAdam |
| Participation | all clients, or 50 % per round with 20 % stragglers |
| Aggregation | FedAvg · FedProx · HA-FedProx |
| Update precision | float32 or float16 |

Because every arm is scored on the same test half-hours, any difference in error comes from the
factor that changed.

## 2. Contributions

1. **Gap decomposition with an IID control.** Splitting the same homes over more clients is
   compared with dealing their training windows at random to the same number of clients, so that
   clients do not differ. This separates fragmentation and training budget from heterogeneity.
2. **Byte ledger.** Exact upload and download traffic per round, B@ε and R@ε (bytes and rounds until
   validation MAE stays within ε of centralised), compared with four encodings of the raw readings.
3. **Who loses from federation.** Per-household harm with a 1 % margin, a client-only baseline and
   local fine-tuning.
4. **Negative results.** Calendar inputs, proximal regularisation (validation-tuned μ = 0) and client
   optimiser changes do not help; a server-side adaptive optimiser helps partly.

Statistics use the training run as the unit (paired t-tests by seed, Welch otherwise) with one Holm
correction over all 31 comparisons. Positioning against prior work is in
[`paper/LITERATURE_NOTES.md`](paper/LITERATURE_NOTES.md).

## 3. Results

<!-- RESULTS -->
All numbers are real-data results on the main cohort of 100 LCL households, means over seeds (five
for the main arms, three for the plateau and optimiser runs), on the same 292,052 test half-hours.
They are produced by `make report` and `make docx` (`results/metrics/`, `paper/generated/`); the
manuscript gives confidence intervals and corrected p-values for each.

| Arm | K | Test MAE (kWh) | Gap to centralised | Traffic, 30 rounds (MB) |
|---|---|---|---|---|
| Persistence | – | 0.0890 | – | 0 |
| Local-only | – | 0.0839 | +3.3 % | 0 |
| Centralised | – | 0.0812 | 0.0 % | 14.0 (raw data, 12 B/reading) |
| FedAvg, one client | 1 | 0.0809 | −0.4 % | 4.1 |
| FedAvg, homes at random | 5 | 0.0826 | +1.8 % | 20.7 |
| FedAvg, homes at random | 20 | 0.0854 | +5.2 % | 82.6 |
| FedAvg, homes at random | 50 | 0.0870 | +7.2 % | 206.6 |
| FedAvg, one home per client | 100 | 0.0893 | +10.0 % | 413.2 |

### Key findings

1. **Fragmentation costs accuracy even without heterogeneity.** With training windows dealt at
   random (identical clients), the 30-round gap was 1.1, 4.4, 5.4 and 7.6 % at K = 5, 20, 50 and
   100: 76–85 % of the gap with homes at K ≥ 20, and 71–96 % across three cohorts.
2. **Much of the gap is training budget.** Trained to a plateau, the gap fell to 1.8, 4.4 and 5.4 %
   with homes and 1.3, 1.9 and 3.5 % with windows at K = 20, 50 and 100. A K = 100 run with homes
   moved about 4,711 MB on average (6,887 MB for the run stopped at the 500-round cap).
3. **Optimisers.** Keeping client Adam state changed nothing; client SGD was worse (significantly at
   K = 20). Server FedAdam narrowed the one-home-per-client gap from 7.3 to 5.3 % over 100 rounds,
   in 3 of 3 seeds but not significantly.
4. **Heterogeneity mainly slows early training.** At K = 20, groupings of homes from random to
   shape-sorted left final error within 0.0851–0.0860 kWh; heterogeneity showed up as a larger rise
   of validation error in the first rounds and more disagreement between client updates.
5. **FL is not the bandwidth-light option at this scale.** The training readings occupy 2.3–58.1 MB
   depending on encoding; 30 rounds at K ≥ 20 moved more than any of them.
6. **Equity.** FedAvg at K = 20 harmed (by more than 1 %) 51–56 % of households and 73 % at K = 100;
   one epoch of local fine-tuning reduced this to 40 and 46 %. With one home per client, fine-tuned
   FedAvg (0.0849 kWh) still lost to local models (0.0839 kWh).

Figures: `paper/figures/` and `results/plots/`.
<!-- /RESULTS -->

## 4. Pipeline

```
LCL zip ─► preprocess ─► per-home scaler (train only) ─► 48→1 windows ─┬─► naive / local / centralised
                                                                         └─► partition ─► K clients
                                                    weights only  ▲            │ (data stays here)
                                                                  │            ▼
                                       server: sample → broadcast → local fit (+prox) → n_k-weighted mean (or FedAdam step)
                                               → best round on validation → HA-FedProx μ update
                                               → byte ledger → metrics, statistics, plots, LaTeX tables
```

Implementation details, leakage controls and every formula are in [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## 5. Installation

Tested with Python 3.14.7 and torch 2.14.1 on macOS (CPU and Apple MPS). It should also work on any
Python ≥ 3.10 with these wheels, on CPU-only Linux/Windows or CUDA.

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-flower.txt                # optional Flower backend
python -m pytest                                      # unit tests
```

The device is picked automatically (CUDA, then MPS, then CPU). Force one with `--set train.device=cpu`.

## 6. Data

The LCL data are public but **not redistributed here** (≈ 800 MB zipped, 8.5 GB of CSV).

```bash
python -m src.data.download      # "Partitioned LCL Data.zip" (795,722,689 bytes) → data/raw/
python -m src.data.preprocess    # ≈ 40 s; cleaning report in data/processed/preprocess_report.json
```

You can also download `Partitioned LCL Data.zip` from the
[London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)
and put it, unzipped, in `data/raw/`. Cleaning rules and their counts are in [`data/README.md`](data/README.md).

## 7. Reproducing the paper

```bash
python -m src.experiments.run_demo                # synthetic-data smoke test, ~2 min → results/demo/
python -m src.experiments.run_matrix --phase tune --workers 8   # μ / μ0 selection on validation
python -m src.experiments.run_matrix --phase main --workers 8   # E0–E14 × 3 seeds (resumable)
python -m src.experiments.make_report             # CSVs, statistics, plots, paper/generated/*.tex
make paper                                        # paper/manuscript.pdf (tectonic or pdflatex)
make docx                                         # revision analyses, Word manuscript, supplement, long version, review response
```

The revision runs (IID control, budget and plateau runs, client-only, optimisers, further cohorts)
use their own matrices and are analysed by `src/evaluation/revision.py` (run by `make docx`):

```bash
python -m src.experiments.run_matrix --matrix configs/matrix_revision.yaml --workers 8
python -m src.experiments.run_matrix --matrix configs/matrix_revision2.yaml --workers 8
python -m src.experiments.run_matrix --matrix configs/matrix_revision2_opt.yaml --workers 8
```

The full matrix takes about 8 h on one Apple M5 Max with 8 parallel runs. Finished runs are skipped
on restart. Individual arms can be run on their own:

```bash
python -m src.experiments.run_baselines                       # naive + local + centralised
python -m src.experiments.run_fedavg  --set partition.strategy=dirichlet partition.alpha=0.3
python -m src.experiments.run_fedprox --set fed.mu=0.001
python -m src.experiments.run_fedprox --adaptive --set fed.mu=0.01      # HA-FedProx
python -m src.experiments.run_fedavg  --set partition.mode=household    # 1 home = 1 client
python -m src.experiments.run_fedavg  --set fed.client_fraction=0.5 fed.straggler_prob=0.2
python -m src.experiments.run_fedavg  --backend flower                  # same run through Flower
```

Every command accepts `--config`, `--set key=value …`, `--seed` and `--sample`. `--sample` runs on 10
real households with 3 rounds as a quick check.

### Experiment matrix (`configs/matrix.yaml`)

| ID | Arm | Factor |
|---|---|---|
| E0 | persistence, seasonal naive | reference |
| E1 | local-only | no collaboration |
| E2 | centralised | pooled raw data |
| E3 | FedAvg | random grouping, K = 20 |
| E4 / E5 | FedAvg / FedProx | Dirichlet α = 0.3 |
| E6 / E7 | FedAvg / FedProx | cluster grouping (strong non-IID) |
| E8 / E9 / E11 | FedAvg / FedProx / HA-FedProx | α = 0.3, 50 % participation, 20 % stragglers |
| E10 | FedAvg, FedProx, HA-FedProx | α ∈ {∞, 1, 0.3, 0.1, 0} sweep |
| E12 | FedAvg | α = 0.3, float16 updates |
| E13 | FedAvg, FedProx, HA-FedProx | one household per client (K = 100) |
| E14 | FedAvg | random grouping, K ∈ {1, 5, 50} |

Every row uses seeds 42, 43 and 44; the revision matrices add seeds 45 and 46 for the main arms. FedProx μ and HA-FedProx μ₀ are tuned on validation data only
(`results/metrics/tuning.json`).

### Outputs

| Path | Content |
|---|---|
| `results/metrics/comparison.{csv,md}` | main table: MAE · RMSE · MAPE · WAPE · best round · traffic |
| `results/metrics/summary_by_experiment.csv` | every experiment, mean/std over seeds, R@ε, B@ε |
| `results/metrics/gap_decomposition.csv`, `fragmentation.csv` | where the federated gap comes from |
| `results/metrics/heterogeneity.csv`, `rho_vs_heterogeneity.csv` | H, ρ and error per partition |
| `results/metrics/stats_tests.csv` | per-household Wilcoxon tests (descriptive) |
| `results/metrics/revision/` | run-level tests with one Holm correction, plateau, optimiser, cohort and budget tables |
| `results/metrics/equity.csv`, `equity_by_stratum.csv` | who gains or loses from federation |
| `results/metrics/runs/` | every run: summary JSON, per-round history, per-household errors |
| `results/plots/*.png` | all figures (copied to `paper/figures/`) |
| `paper/generated/*.tex` | tables and `\newcommand` numbers used by the manuscript |

## 8. Metrics

| Metric | Definition | Note |
|---|---|---|
| MAE | mean \|y − ŷ\| (kWh per half-hour) | primary metric |
| RMSE | √mean(y − ŷ)² | weights large misses (peaks) |
| WAPE | 100 · Σ\|y − ŷ\| / Σy | scale-free, stable near zero |
| MAPE | 100 · mean \|y − ŷ\|/y over half-hours with y ≥ 0.05 kWh | plain MAPE is unbounded at night-time loads; the excluded share is reported |

Metrics are pooled over every test half-hour of every household (micro-average), in kWh after
inverse scaling.

## 9. Repository layout

```
configs/      default.yaml (real data) · demo.yaml (synthetic) · matrix.yaml · matrix_revision*.yaml (experiments)
data/         raw/ (LCL zip, not in git) · processed/ · clients/ · README.md
src/
  config.py utils.py
  data/         download · preprocess · window · partition · profiles · synthetic
  models/       lstm.py (the one model)
  baselines/    local · naive
  centralized/  train.py (shared training loop + centralised arm)
  federated/    client · server · fedavg · fedprox (HA-FedProx) · comm · flower_backend
  evaluation/   metrics · evaluate · plots · report
  experiments/  run_baselines · run_fedavg · run_fedprox · run_matrix · run_demo · make_report · common
  evaluation/revision.py  run-level tests, IID control, plateau, optimiser, cohort and harm analyses
results/      metrics/ · plots/ · logs/ · demo/ (synthetic only)
paper/        manuscript.tex · sections/ · refs.bib · generated/ · figures/ · LITERATURE_NOTES.md
  docx/       template.docx · content.tex (15 pages) · content_supp.tex · content_long.tex · build_docx.py · response.py (first review, frozen) · response2.py · workflow_figure.py
docs/         TECHNICAL.md
tests/        test_core.py
submission/   main.docx/.pdf (manuscript) · supplementary_material · response_to_review_2 · cover_letter · highlights
CITATION.cff, .zenodo.json   citation metadata for the repository
```

## 10. Limitations

- **Data locality is not formal privacy.** Weight updates can leak information. Neither differential
  privacy nor secure aggregation is implemented.
- **The network is simulated.** Clients run in one process, and stragglers and dropouts are sampled.
  Traffic counts tensor payload only; Flower measures its framing overhead at about 1 %.
- **One dataset, one city, one year, one horizon.** The cohort is 100 of 4,034 eligible
  standard-tariff London homes, tested on Nov–Dec 2013, at a 30-minute horizon.
- **Evaluation is server-side.** For speed, the simulator scores the global model on every
  household's validation and test windows. In deployment this would be client-side `evaluate` calls.
- **Determinism.** Runs are bit-exact on CPU. On MPS and CUDA they may differ in the last digits.
- **Fixed hyper-parameters.** Learning rate, hidden size and window are fixed a priori for every arm.
  Only μ, μ₀ and the SGD and FedAdam step sizes are tuned, on validation data.

## 11. Data and licence

The LCL data are published by UK Power Networks on the London Datastore under their own licence
and are not included here. The code is released for research use. Please cite the manuscript if
you use it; citation metadata for this repository are in [`CITATION.cff`](CITATION.cff). To cite the code:

> Gupta, P., Kaushik, V., Kalaavathi B (2026). Training budget and fragmentation, more than heterogeneity,
> drive the accuracy cost of federated household load forecasting: code and results (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.23493458
