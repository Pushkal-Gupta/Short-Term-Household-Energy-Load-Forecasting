# Comparison (real)

| Method | MAE (kWh) | RMSE (kWh) | MAPE* (%) | WAPE (%) | Best round / rounds | Communication (MB) | Raw data leaves home | Seeds |
|---|---|---|---|---|---|---|---|---|
| Persistence (naive) | 0.0890 | 0.1987 | 41.06 | 37.64 | n/a | 0.0 | n/a | 1 |
| Seasonal naive (t−48) | 0.1332 | 0.2773 | 63.62 | 56.33 | n/a | 0.0 | n/a | 1 |
| Local-only LSTM | 0.0839 ± 0.0003 | 0.1777 ± 0.0006 | 37.25 ± 0.31 | 35.49 | n/a | 0.0 | No | 5 |
| Centralised LSTM (pooled raw data) | 0.0812 ± 0.0004 | 0.1721 ± 0.0013 | 36.54 ± 0.79 | 34.35 | n/a | 14.0 | Yes | 5 |
| FedAvg, random (IID-ish) | 0.0854 ± 0.0008 | 0.1758 ± 0.0006 | 39.04 ± 0.57 | 36.12 | 22.0 / 30 | 82.6 | No | 5 |
| FedAvg, α=0.3 | 0.0860 ± 0.0003 | 0.1757 ± 0.0005 | 39.68 ± 0.16 | 36.37 | 24.0 / 30 | 82.6 | No | 5 |
| FedProx, α=0.3 | 0.0857 ± 0.0002 | 0.1759 ± 0.0009 | 39.30 ± 0.36 | 36.27 | 22.8 / 30 | 82.6 | No | 5 |
| HA-FedProx, α=0.3 | 0.0863 ± 0.0003 | 0.1765 ± 0.0005 | 39.62 ± 0.11 | 36.52 | 20.7 / 30 | 82.6 | No | 3 |
| FedAvg, cluster (strong) | 0.0857 ± 0.0003 | 0.1759 ± 0.0004 | 39.20 ± 0.18 | 36.24 | 21.4 / 30 | 82.6 | No | 5 |
| FedProx, cluster (strong) | 0.0858 ± 0.0002 | 0.1759 ± 0.0006 | 39.33 ± 0.16 | 36.29 | 23.3 / 30 | 82.6 | No | 3 |
| HA-FedProx, cluster (strong) | 0.0862 ± 0.0004 | 0.1768 ± 0.0008 | 39.44 ± 0.35 | 36.48 | 18.0 / 30 | 82.6 | No | 5 |
| FedAvg, α=0.3, 50% participation + 20% stragglers | 0.0850 ± 0.0005 | 0.1759 ± 0.0008 | 38.70 ± 0.50 | 35.97 | 22.4 / 30 | 37.2 | No | 5 |
| FedProx, α=0.3, 50% participation + 20% stragglers | 0.0857 ± 0.0003 | 0.1764 ± 0.0012 | 39.10 ± 0.25 | 36.24 | 20.7 / 30 | 41.3 | No | 3 |
| HA-FedProx, α=0.3, 50% participation + 20% stragglers | 0.0857 ± 0.0003 | 0.1769 ± 0.0011 | 39.14 ± 0.18 | 36.25 | 23.0 / 30 | 41.3 | No | 3 |
| FedAvg, α=0.3, float16 updates | 0.0857 ± 0.0004 | 0.1755 ± 0.0007 | 39.51 ± 0.20 | 36.27 | 25.2 / 30 | 41.3 | No | 5 |
| FedAvg, 1 household = 1 client | 0.0893 ± 0.0009 | 0.1810 ± 0.0009 | 40.94 ± 1.08 | 37.77 | 29.2 / 30 | 413.2 | No | 5 |
| FedProx, 1 household = 1 client | 0.0893 ± 0.0013 | 0.1817 ± 0.0011 | 40.68 ± 1.58 | 37.77 | 30.0 / 30 | 413.2 | No | 3 |
| HA-FedProx, 1 household = 1 client | 0.0901 ± 0.0005 | 0.1823 ± 0.0009 | 41.23 ± 0.80 | 38.11 | 29.3 / 30 | 413.2 | No | 3 |
| FedAvg, K=1 client(s), random | 0.0809 ± 0.0004 | 0.1709 ± 0.0005 | 36.76 ± 0.26 | 34.23 | 18.6 / 30 | 4.1 | No | 5 |
| FedAvg, K=5 client(s), random | 0.0826 ± 0.0005 | 0.1721 ± 0.0005 | 38.05 ± 0.30 | 34.96 | 21.8 / 30 | 20.7 | No | 5 |
| FedAvg, K=50 client(s), random | 0.0870 ± 0.0005 | 0.1780 ± 0.0005 | 39.76 ± 0.48 | 36.81 | 26.6 / 30 | 206.6 | No | 5 |

*MAPE over half-hours with actual load ≥ the configured threshold (0.05 kWh by default). Mean ± std over seeds. Centralised 'communication' is the raw training data volume (value + timestamp).
