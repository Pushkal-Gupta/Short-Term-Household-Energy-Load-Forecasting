# Comparison (real)

| Method | MAE (kWh) | RMSE (kWh) | MAPE* (%) | WAPE (%) | Best round / rounds | Communication (MB) | Raw data leaves home | Seeds |
|---|---|---|---|---|---|---|---|---|
| Persistence (naive) | 0.0890 | 0.1987 | 41.06 | 37.64 | n/a | 0.0 | n/a | 1 |
| Seasonal naive (t−48) | 0.1332 | 0.2773 | 63.62 | 56.33 | n/a | 0.0 | n/a | 1 |
| Local-only LSTM | 0.0840 ± 0.0003 | 0.1779 ± 0.0006 | 37.27 ± 0.40 | 35.52 | n/a | 0.0 | No | 3 |
| Centralised LSTM (pooled raw data) | 0.0811 ± 0.0003 | 0.1718 ± 0.0015 | 36.66 ± 0.81 | 34.30 | n/a | 14.0 | Yes | 3 |
| FedAvg, random (IID-ish) | 0.0853 ± 0.0009 | 0.1758 ± 0.0007 | 38.97 ± 0.61 | 36.07 | 23.0 / 30 | 82.6 | No | 3 |
| FedAvg, α=0.3 | 0.0861 ± 0.0004 | 0.1758 ± 0.0007 | 39.72 ± 0.15 | 36.42 | 24.0 / 30 | 82.6 | No | 3 |
| FedProx, α=0.3 | 0.0859 ± 0.0002 | 0.1758 ± 0.0007 | 39.50 ± 0.16 | 36.32 | 24.0 / 30 | 82.6 | No | 2 |

*MAPE over half-hours with actual load ≥ the configured threshold (0.05 kWh by default). Mean ± std over seeds. Centralised 'communication' is the raw training data volume (value + timestamp).
