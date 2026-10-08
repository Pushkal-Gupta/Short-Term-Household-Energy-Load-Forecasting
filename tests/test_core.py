"""Unit tests for the parts where a silent bug would invalidate the comparison.

    python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import load_config, with_overrides
from src.data.partition import partition_households
from src.data.preprocess import fill_short_gaps, select_households
from src.data.window import make_household
from src.evaluation.metrics import all_metrics, mae, mape, rmse, wape
from src.federated.client import HouseholdClient
from src.federated.comm import CommLedger, payload_bytes, quantize
from src.federated.fedavg import update_dissimilarity, weighted_average
from src.federated.server import FederatedServer
from src.models.lstm import build_model, count_parameters, get_weights


@pytest.fixture(scope="module")
def cfg():
    c = load_config("configs/demo.yaml")
    return with_overrides(c, **{"train.device": "cpu", "data.start": "2013-01-01", "data.end": "2013-01-11",
                                "data.train_end": "2013-01-07", "data.val_end": "2013-01-09"})


def _series(days=10, seed=0, mean=0.3):
    idx = pd.date_range("2013-01-01", periods=48 * days, freq="30min")
    rng = np.random.default_rng(seed)
    x = mean + 0.2 * np.sin(np.arange(len(idx)) * 2 * np.pi / 48) + rng.normal(0, 0.05, len(idx))
    return np.clip(x, 0.01, None).astype(np.float32), idx


@pytest.fixture(scope="module")
def households(cfg):
    out = []
    for i in range(6):
        x, idx = _series(seed=i, mean=0.2 + 0.1 * (i % 3))
        out.append(make_household(f"H{i}", x, idx, cfg))
    return out


# ---------------- metrics ----------------
def test_metrics_hand_computed():
    y, yh = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 2.0])
    assert mae(y, yh) == pytest.approx(1.0)
    assert rmse(y, yh) == pytest.approx(np.sqrt(5 / 3))
    assert mape(y, yh, 0.05)[0] == pytest.approx(100 * (1 + 0 + 0.5) / 3)
    assert wape(y, yh) == pytest.approx(100 * 3 / 7)


def test_mape_masks_near_zero():
    y, yh = np.array([0.0, 0.01, 1.0]), np.array([0.5, 0.5, 1.1])
    m, excluded = mape(y, yh, min_actual=0.05)
    assert m == pytest.approx(10.0)
    assert excluded == pytest.approx(2 / 3)
    assert np.isnan(mape(np.zeros(3), np.ones(3))[0])
    assert np.isfinite(all_metrics(y, yh)["mape"])


# ---------------- preprocessing / windows ----------------
def test_fill_short_gaps_only():
    x = np.array([1, np.nan, 3, np.nan, np.nan, np.nan, 7], dtype=np.float32)
    out = fill_short_gaps(x, max_gap=2)
    assert out[1] == pytest.approx(2.0)
    assert np.isnan(out[3:6]).all()


def test_stratified_selection_sizes():
    s = pd.DataFrame({"LCLid": [f"h{i}" for i in range(23)], "eligible": True,
                      "train_mean_kwh": np.linspace(0.1, 1, 23)})
    sel = select_households(s, 12, seed=1)
    assert len(sel) == 12 and sel["LCLid"].is_unique


def test_windows_targets_stay_in_split_and_scaler_uses_train_only(cfg):
    x, idx = _series()
    x_mod = x.copy()
    x_mod[idx >= pd.Timestamp(cfg["data"]["train_end"])] += 100.0  # change val/test only
    a, b = make_household("a", x, idx, cfg), make_household("b", x_mod, idx, cfg)
    assert a.mean == pytest.approx(b.mean) and a.std == pytest.approx(b.std)
    tr_end, va_end = pd.Timestamp(cfg["data"]["train_end"]), pd.Timestamp(cfg["data"]["val_end"])
    assert (a.t["train"] < tr_end).all()
    assert ((a.t["val"] >= tr_end) & (a.t["val"] < va_end)).all()
    assert (a.t["test"] >= va_end).all()
    W = cfg["window"]["input_window"]
    assert a.X["train"].shape[1:] == (W, 1) and a.y["train"].shape[1] == 1
    # first train target sits exactly W + h - 1 steps into the series and its input is the preceding W values
    z = (x - a.mean) / a.std
    np.testing.assert_allclose(a.X["train"][0, :, 0], z[:W], rtol=1e-5)
    np.testing.assert_allclose(a.y["train"][0, 0], z[W], rtol=1e-5)


def test_windows_with_missing_values_are_dropped(cfg):
    x, idx = _series()
    full = make_household("f", x, idx, cfg).n("train")
    x[100] = np.nan
    assert make_household("g", x, idx, cfg).n("train") == full - 49  # 48 windows use it as input, 1 as target


# ---------------- partitioning ----------------
@pytest.mark.parametrize("strategy,alpha", [("iid", None), ("dirichlet", 0.3), ("dirichlet", 0.0), ("cluster", None)])
def test_partition_uses_every_household_once(cfg, households, strategy, alpha):
    c = with_overrides(cfg, **{"partition.strategy": strategy, "partition.alpha": alpha,
                               "partition.households_per_client": 2, "partition.num_profile_clusters": 3})
    clients, labels = partition_households(households, c, seed=3)
    assert sorted(i for g in clients for i in g) == list(range(len(households)))
    assert all(len(g) == 2 for g in clients) and len(labels) == len(households)


def test_household_mode_is_one_client_per_home(cfg, households):
    c = with_overrides(cfg, **{"partition.mode": "household"})
    clients, _ = partition_households(households, c, seed=0)
    assert clients == [[i] for i in range(len(households))]


# ---------------- aggregation / communication ----------------
def test_fedavg_is_sample_weighted_mean():
    a, b = [np.ones((2, 2)), np.zeros(3)], [np.full((2, 2), 4.0), np.ones(3)]
    out = weighted_average([(a, 1), (b, 3)])
    np.testing.assert_allclose(out[0], np.full((2, 2), 3.25))
    np.testing.assert_allclose(out[1], np.full(3, 0.75))
    eq = weighted_average([(a, 5), (b, 5)])
    np.testing.assert_allclose(eq[0], np.full((2, 2), 2.5))


def test_update_disagreement_bounds():
    g = [np.zeros(4)]
    same = [([np.ones(4)], 1), ([np.ones(4)], 1)]
    opposite = [([np.ones(4)], 1), ([-np.ones(4)], 1)]
    assert update_dissimilarity(g, same) == pytest.approx(0.0)
    assert update_dissimilarity(g, opposite) == pytest.approx(1.0)


def test_byte_accounting(cfg):
    P = count_parameters(build_model(cfg))
    assert payload_bytes(P) == 4 * P and payload_bytes(P, "float16") == 2 * P
    led = CommLedger(P)
    r = led.round(n_download=5, n_upload=3)
    assert r["round_bytes"] == 8 * 4 * P and led.cumulative == 8 * 4 * P
    w = [np.array([0.1234567], dtype=np.float32)]
    assert quantize(w, "float16")[0].dtype == np.float32 and quantize(w, "float16")[0][0] != w[0][0]


# ---------------- federated equivalences / reproducibility ----------------
def _fed_run(cfg, households, algorithm, mu, seed=0):
    c = with_overrides(cfg, **{"fed.algorithm": algorithm, "fed.mu": mu, "fed.rounds": 2,
                               "fed.drop_stragglers": True, "partition.households_per_client": 2})
    clients, _ = partition_households(households, c, seed)
    cl = [HouseholdClient(i, [households[j] for j in g], c, torch.device("cpu")) for i, g in enumerate(clients)]
    model, hist, _ = FederatedServer(cl, households, c, torch.device("cpu"), seed, verbose=False).run()
    return get_weights(model), hist


def test_fedprox_mu0_equals_fedavg(cfg, households):
    wa, ha = _fed_run(cfg, households, "fedavg", 0.0)
    wp, hp = _fed_run(cfg, households, "fedprox", 0.0)
    for x, y in zip(wa, wp):
        np.testing.assert_array_equal(x, y)
    assert [r["val_mae"] for r in ha] == [r["val_mae"] for r in hp]


def test_seeded_runs_are_reproducible(cfg, households):
    w1, h1 = _fed_run(cfg, households, "hafedprox", 0.1, seed=5)
    w2, h2 = _fed_run(cfg, households, "hafedprox", 0.1, seed=5)
    for x, y in zip(w1, w2):
        np.testing.assert_array_equal(x, y)
    assert h1[-1]["cumulative_bytes"] == h2[-1]["cumulative_bytes"] > 0
