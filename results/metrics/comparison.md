# Comparison (real)

| Method | MAE (kWh) | RMSE (kWh) | MAPE* (%) | WAPE (%) | Best round / rounds | Communication (MB) | Raw data leaves home | Seeds |
|---|---|---|---|---|---|---|---|---|
| Persistence (naive) | 0.0890 | 0.1987 | 41.06 | 37.64 | n/a | 0.0 | n/a | 1 |
| Seasonal naive (t−48) | 0.1332 | 0.2773 | 63.62 | 56.33 | n/a | 0.0 | n/a | 1 |
| Local-only LSTM | 0.0840 ± 0.0003 | 0.1779 ± 0.0006 | 37.27 ± 0.40 | 35.52 | n/a | 0.0 | No | 3 |
| Centralised LSTM (pooled raw data) | 0.0811 ± 0.0003 | 0.1718 ± 0.0015 | 36.66 ± 0.81 | 34.30 | n/a | 14.0 | Yes | 3 |
| FedAvg, random (IID-ish) | 0.0853 ± 0.0009 | 0.1758 ± 0.0007 | 38.97 ± 0.61 | 36.07 | 23.0 / 30 | 82.6 | No | 3 |
| FedAvg, α=0.3 | 0.0861 ± 0.0004 | 0.1758 ± 0.0007 | 39.72 ± 0.15 | 36.42 | 24.0 / 30 | 82.6 | No | 3 |
| FedProx, α=0.3 | 0.0858 ± 0.0001 | 0.1755 ± 0.0006 | 39.50 ± 0.11 | 36.31 | 26.0 / 30 | 82.6 | No | 3 |
| FedAvg, cluster (strong) | 0.0857 ± 0.0003 | 0.1758 ± 0.0005 | 39.19 ± 0.17 | 36.24 | 22.0 / 30 | 82.6 | No | 3 |
| FedProx, cluster (strong) | 0.0858 ± 0.0002 | 0.1759 ± 0.0006 | 39.33 ± 0.16 | 36.29 | 23.3 / 30 | 82.6 | No | 3 |
| FedAvg, α=0.3, 50% participation + 20% stragglers | 0.0849 ± 0.0004 | 0.1757 ± 0.0010 | 38.74 ± 0.50 | 35.92 | 24.0 / 30 | 37.4 | No | 3 |
| FedProx, α=0.3, 50% participation + 20% stragglers | 0.0857 ± 0.0003 | 0.1764 ± 0.0012 | 39.10 ± 0.25 | 36.24 | 20.7 / 30 | 41.3 | No | 3 |

*MAPE over half-hours with actual load ≥ the configured threshold (0.05 kWh by default). Mean ± std over seeds. Centralised 'communication' is the raw training data volume (value + timestamp).
