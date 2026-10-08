"""The single forecasting model used by EVERY arm (local, centralised, FedAvg, FedProx, HA-FedProx).

    input (batch, W, C) -> LSTM(hidden_size, num_layers) -> last hidden state -> Linear -> (batch, horizon_out)

Only the learning strategy differs between arms. The architecture is built from the same
`model:` config block every time, through `build_model`.
"""
from __future__ import annotations

import torch
from torch import nn


class LoadLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1,
                 dropout: float = 0.0, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_model(cfg: dict) -> LoadLSTM:
    m = cfg["model"]
    input_size = 1 + (4 if cfg["window"].get("calendar_features") else 0)
    return LoadLSTM(input_size, int(m["hidden_size"]), int(m["num_layers"]), float(m.get("dropout", 0.0)), 1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_weights(model: nn.Module) -> list:
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def set_weights(model: nn.Module, weights: list) -> None:
    state = {k: torch.as_tensor(w) for k, w in zip(model.state_dict().keys(), weights)}
    model.load_state_dict(state, strict=True)
