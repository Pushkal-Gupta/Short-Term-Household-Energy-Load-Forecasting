# Technical description

This document describes what the code in `src/` actually does. Every statement points to the
module that implements it.

## 1. System architecture

```
 data/raw/ (LCL zip)                           configs/*.yaml (+ --set overrides)
        │                                                │
        ▼                                                ▼
 src/data/preprocess.py ── clean, grid, gap-fill, select ─► data/processed/series.parquet
        │
        ▼
 src/data/window.py  ── per-household scaler (train only), 48→1 windows, chronological split
        │
        ├─► src/baselines/naive.py        E0  persistence, seasonal naive
        ├─► src/baselines/local.py        E1  one LSTM per household
        ├─► src/centralized/train.py      E2  one LSTM on pooled windows
        └─► src/data/partition.py ─► src/federated/client.py (K clients, data stays inside)
                                           ▲          │ weights, n_k, loss
                                 weights   │          ▼
                                     src/federated/server.py   (FedAvg / FedProx / HA-FedProx)
                                     src/federated/comm.py     (byte ledger)
                                     src/federated/flower_backend.py (same loop through Flower)
        │
        ▼
 src/evaluation/{metrics,evaluate}.py ─► results/metrics/runs/*.json|csv
 src/evaluation/report.py, plots.py   ─► results/metrics/*.csv, results/plots/*.png, paper/generated/*.tex
```

Every arm is launched through `src/experiments/common.py::run_experiment`, so all of them share
data loading, evaluation and result writing.

## 2. Data pipeline

The cleaning rules and their counts on the real data are listed in `data/README.md`.
In short: study year 2013, standard-tariff homes with ≥ 95 % coverage, gaps of up to 2 h
interpolated, longer gaps dropped, and a cohort of 100 households (seeded, stratified by
train-period load quintile).

Leakage controls (`src/data/window.py`):
- Each household's scaler is fitted on its own train-period readings.
- A window belongs to the split of its **target** timestamp. Validation and test inputs may
  reach back into the previous split, since those readings are known at forecast time.
- Cohort selection, profile clustering and the heterogeneity index use train-period data only.
- Model selection uses validation MAE: early stopping for local and centralised, best round for
  federated. Test metrics are logged per round for the convergence plots but never drive a decision.

## 3. Forecasting task and model

The input is the last 48 half-hours (24 h) of one household's scaled load, `X ∈ R^{48×1}`.
The target is the next half-hour, `y ∈ R^1`. Both lengths are configurable (`window.input_window`,
`window.horizon`). Calendar channels are optional and off by default.

`src/models/lstm.py::LoadLSTM`: LSTM(input 1, hidden 64, 1 layer) → last hidden state →
Linear(64→1). It has 17,217 parameters, about 68.9 kB in float32. The architecture comes from one
`model:` config block, which is identical for all arms. Training uses Adam (lr 1e-3), batch 256 and MSE.

## 4. Clients and server

- **Client** (`src/federated/client.py::HouseholdClient`) owns the training windows of its
  household(s) and exposes Flower's `NumPyClient` methods: `get_parameters`, `fit` and `evaluate`.
  `fit` loads the global weights and trains for `local_epochs × work_fraction` epochs, adding the
  proximal term when `mu > 0`. It returns `(weights, n_k, {train_loss})`.
- **Server** (`src/federated/server.py::FederatedServer`) runs the rounds and never touches any
  client's training arrays. For model selection and reporting it evaluates the global model on
  every household's validation and test windows. In a deployment, this evaluation would be the
  clients' `evaluate` calls; in the simulation it is done centrally for speed. The evaluation
  produces numbers only and never changes the weights.
- **Flower backend** (`--backend flower`, `src/federated/flower_backend.py`) wraps the same clients
  with `NumPyClient.to_client()` and aggregates with Flower's own `FedAvg` and `FedProx` strategies
  through its `FitIns`/`FitRes`/`Parameters` types, in-process, without Ray. On the demo config,
  FedAvg through Flower reproduces the simulator's per-round validation MAE and final test MAE
  exactly. The backend also logs the measured Flower wire size, about 1 % above the analytic payload
  because of numpy serialization headers.

**Privacy scope.** Raw readings stay inside the client object. The server receives weights,
sample counts and a scalar loss. That is data locality, not a formal privacy guarantee: weight
updates can leak information through gradient inversion and membership inference. Differential
privacy and secure aggregation are not implemented.

## 5. FedAvg

Each round `t` (`server.py`):
1. Select `max(min_clients, ⌈C·K⌉)` clients uniformly without replacement (seeded).
2. Send them `w_t` (download).
3. Stragglers are drawn with probability `straggler_prob` and do `U(straggler_min_work, 1)` of
   their local work. Under FedAvg they are **dropped**: they download but never upload.
4. Clients train for `E` local epochs and return `w_k` and `n_k` (upload).
5. Aggregate `w_{t+1} = Σ_k n_k w_k / Σ_k n_k` (`fedavg.py::weighted_average`).
6. Evaluate. Keep the weights with the best validation MAE.

## 6. FedProx and HA-FedProx

FedProx (`client.py` + `centralized/train.py::train_epochs`): the local objective is
`L_k(w) + (μ/2)‖w − w_t‖²`, with `w_t` frozen at the received global weights. Stragglers'
partial updates are **kept**. With μ = 0 and stragglers dropped, the code path is bit-identical
to FedAvg, which a unit test checks.

**HA-FedProx (contribution of this project, `fedprox.py::MuSchedule`).** After aggregating round t,
the server computes the update-disagreement ratio

ρ_t = Σ_k p_k‖Δ_k − Δ̄‖² / Σ_k p_k‖Δ_k‖²,  Δ_k = w_k − w_t,  Δ̄ = Σ p_k Δ_k,  p_k = n_k/Σn,

which lies in [0, 1] (`fedavg.py::update_dissimilarity`) and equals 1 − 1/B_t², where B_t² is
the empirical local-dissimilarity of Li et al. (2020). The next round uses
μ_{t+1} = clip(μ_0 · ρ_t, μ_min, μ_max). The server computes ρ_t from updates it already holds,
so the clients send nothing extra. μ_0 is tuned on validation exactly like FedProx's fixed μ.

Hyper-parameter tuning (`configs/matrix.yaml`, `tune:`): μ ∈ {0.001, 0.01, 0.1} for FedProx and
μ_0 ∈ {0.01, 0.1, 1.0} for HA-FedProx. Tuning runs under the strongest non-IID partition, with
the first seed, selecting on best-round validation MAE. The winners are stored in
`results/metrics/tuning.json` and used for every FedProx and HA-FedProx run.

## 7. Non-IID partitioning (`src/data/partition.py`, `src/data/profiles.py`)

- **Load-profile features** (train period): a normalised 24-bin mean daily shape plus log mean
  load, standardised, then k-means with k = 5. Clusters therefore separate, for example,
  evening-peaking homes from flat or daytime ones and high users from low users. These are real
  consumption characteristics, not invented labels.
- **Household mode:** one household per client (K = 100). This is natural heterogeneity, which
  cannot be dialled.
- **Community mode:** the same 100 households grouped into K = 20 clients of 5 homes each.
  - `iid`: random grouping.
  - `dirichlet(α)`: each client draws proportions p ~ Dir(α·1_K) over clusters. Each of its 5
    slots draws a cluster from p, renormalised to clusters with homes left, and takes a random
    home from it.
  - `cluster`: homes sorted by (cluster, mean load) and chunked, which is the maximal
    separation.

  Because the households, and so the test windows, are identical across strategies, differences
  in error are caused by the partition alone.
- **Heterogeneity index H:** H_JS is the mean pairwise Jensen–Shannon divergence (base 2) between
  clients' hour-of-week consumption distributions. H_CV is the coefficient of variation of client
  mean load. H = H_JS + H_CV/(1 + H_CV). It is computed from train data and reported for every run,
  so results can be plotted against measured rather than nominal heterogeneity.

## 8. Metrics (`src/evaluation/metrics.py`)

All metrics are in kWh per half-hour, after inverse scaling, and are pooled over every test
half-hour of every household (micro-average).
- MAE = mean|y − ŷ|
- RMSE = √mean(y − ŷ)²
- MAPE = 100·mean(|y − ŷ|/|y|) **only over points with y ≥ 0.05 kWh** (`eval.mape_min_actual`).
  The excluded share is stored as `mape_excluded_frac`. Without the mask, near-zero overnight
  load makes MAPE unbounded. The numbers are not comparable with papers that use a different rule.
- WAPE = 100·Σ|y − ŷ|/Σ|y|, which stays scale-free and stable near zero.
- Per household: the same metrics, plus the median and 90th-percentile household MAE (equity view).

## 9. Communication cost (`src/federated/comm.py`)

payload = P × b bytes (P = 17,217 parameters; b = 4 for float32, 2 for float16).
Per round: download = payload × |selected|, upload = payload × |returned|, round = down + up;
the cumulative total is logged per round. Protocol framing is not counted (Flower's measured
overhead is about 1 %). Local-only training costs 0. For the centralised arm, the reported figure
is the raw training-data volume: (4 B value + 8 B timestamp) × readings.
Derived metrics (`report.py`): **R@ε** and **B@ε** are the round and the cumulative bytes at
which validation MAE first comes within ε = 10 % of the centralised validation MAE.

## 10. Experimental design (`configs/matrix.yaml`)

| ID | Arm | Partition | Notes |
|---|---|---|---|
| E0 | persistence, seasonal naive | – | no learning |
| E1 | local-only | – | 100 separate LSTMs, early stopping |
| E2 | centralised | – | pooled raw windows, early stopping |
| E3 | FedAvg | random | IID-ish |
| E4 / E5 | FedAvg / FedProx | Dirichlet α=0.3 | moderate non-IID |
| E6 / E7 | FedAvg / FedProx | cluster | strong non-IID |
| E8 / E9 | FedAvg / FedProx | α=0.3, C=0.5, 20 % stragglers | partial participation |
| E10 | FedAvg, FedProx, HA-FedProx | random, α ∈ {1, 0.3, 0.1}, cluster | heterogeneity sweep |
| E11 | HA-FedProx | as E8 | partial participation |
| E12 | FedAvg | α=0.3, float16 updates | half the bytes |
| E13 | FedAvg, FedProx, HA-FedProx | 1 household = 1 client | natural heterogeneity |

Budget: federated runs use 30 rounds × 1 local epoch, the same number of passes over the data
as the centralised arm's 30-epoch cap. Each runs on 3 seeds (42, 43, 44); the seed changes model
initialisation, client sampling, stragglers and random/Dirichlet groupings. The household cohort
stays fixed. FedAvg-FT, where each household fine-tunes the final global model for one local
epoch, is evaluated for every federated run and adds no communication.

Statistics (`report.py`): paired Wilcoxon signed-rank tests on per-household test MAE (averaged
over seeds) for the comparisons listed in `stats_tests.csv`, with Holm correction across the family.

## 11. Reproducibility

- `src/utils.py::set_seed` seeds Python, NumPy and PyTorch. Each client's local RNG is derived
  from (seed, round, client id).
- Runs are bit-reproducible on CPU (unit test `test_seeded_runs_are_reproducible`). On MPS and
  CUDA, kernels may differ in the last digits between runs.
- Every run writes its full config-derived metadata into its summary JSON. The processed-data
  signature is checked before reuse.
- Real results were produced on macOS 26 (Apple M5 Max, MPS), Python 3.14.7, torch 2.14.1.

## 12. Run status and observations so far

The real-data matrix runs as `python -m src.experiments.run_matrix --workers 8`, and its log is
`results/logs/matrix_real.log`. Each federated community run takes about 45–50 min with 8 runs in
parallel on the M5 Max (MPS). The matrix can be resumed: re-running the command skips any run whose
summary JSON already exists.

Observations from completed runs only. The README results section holds the live table.
- **Tuning** (`tuning.json`) picked the smallest grid value for both proximal methods (μ = 0.001; μ₀ = 0.01).
  Larger μ consistently hurt validation MAE.
- **The update-disagreement ratio ρ_t was high and nearly flat (≈ 0.80–0.87)** in every tuning run. So
  μ_t = μ₀·ρ_t is close to a constant, and HA-FedProx behaves like fixed-μ FedProx. The probable cause: with
  1 local epoch on 5 noisy homes, client updates are dominated by stochastic-gradient noise, and ρ cannot
  separate that noise from distribution shift. Whether mean ρ rises with H across the α sweep is reported in
  `heterogeneity.csv` (column `mean_rho`) when E10 completes. That column is the direct test of the
  rule's premise.
- **Communication is not a saving at this scale.** The federated arms upload and download 82.6 MB over
  30 rounds, about 6× the 14.0 MB of raw training readings the centralised arm would ship. The bytes-to-ε
  metrics (`R_eps`, `B_eps_MB` in `summary_by_experiment.csv`) quantify how much of that budget is needed. So far: about 12.9 MB with random
  grouping vs about 21 MB at α = 0.3.
