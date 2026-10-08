"""Explicit communication accounting.

The payload of a model transfer is the raw tensor data:
    payload_bytes = num_parameters * bytes_per_value      (float32: 4, float16: 2)
Per round:
    download_bytes = payload_bytes * (clients the global model was sent to)
    upload_bytes   = payload_bytes * (clients that returned an update)
A dropped straggler has downloaded the model but uploads nothing.
Serialization framing, TLS and gRPC headers are NOT counted. They add a roughly constant
per-message overhead, which is small next to a ~69 kB payload. Local-only training transfers
0 bytes. Centralised training uploads raw data instead of models, and `raw_data_bytes`
measures that volume for comparison.
"""
from __future__ import annotations

import numpy as np

BYTES = {"float32": 4, "float16": 2}


def payload_bytes(num_params: int, precision: str = "float32") -> int:
    return int(num_params) * BYTES[precision]


def quantize(weights: list, precision: str) -> list:
    """Simulate the wire format: values are rounded to `precision` and restored to float32 on receipt."""
    if precision == "float32":
        return [w.astype(np.float32) for w in weights]
    return [w.astype(np.float16).astype(np.float32) for w in weights]


def raw_data_bytes(households, bytes_per_reading: int = 4, timestamp_bytes: int = 8) -> int:
    """Bytes needed to ship every training-period reading (value + timestamp) to a central server."""
    return int(sum(len(hd.train_kwh) for hd in households) * (bytes_per_reading + timestamp_bytes))


class CommLedger:
    def __init__(self, num_params: int, precision: str = "float32"):
        self.payload = payload_bytes(num_params, precision)
        self.cumulative = 0

    def round(self, n_download: int, n_upload: int) -> dict:
        down, up = self.payload * n_download, self.payload * n_upload
        self.cumulative += down + up
        return {"payload_bytes": self.payload, "download_bytes": down, "upload_bytes": up,
                "round_bytes": down + up, "cumulative_bytes": self.cumulative}
