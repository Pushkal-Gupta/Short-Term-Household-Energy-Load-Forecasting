"""Load-profile features, profile clustering and the heterogeneity index H.

Everything here is computed from TRAIN-period consumption only.

Profile features (one vector per household)
 - normalised mean daily shape: 24 hourly bins of the average day, divided by their sum
 - log of mean half-hourly consumption (the "size" of the household)
These describe *how* and *how much* a home consumes (morning-heavy vs evening-heavy,
low vs high users). They are real household characteristics, not synthetic labels.

Heterogeneity index (between federated clients)
 - H_JS  = mean pairwise Jensen-Shannon divergence (base 2, in [0, 1]) between clients'
           hour-of-week consumption distributions (168 bins, each normalised to sum to 1).
           It measures differences in WHEN clients consume.
 - H_CV  = coefficient of variation of client mean load. It measures differences in HOW MUCH.
 - H     = H_JS + H_CV / (1 + H_CV). This is the headline scalar: both parts lie in [0, 1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans


def _binned(hd, key: str, nbins: int) -> np.ndarray:
    idx = hd.train_index
    x = hd.train_kwh
    if key == "hour":
        b = idx.hour.to_numpy()
    else:
        b = idx.dayofweek.to_numpy() * 24 + idx.hour.to_numpy()
    ok = np.isfinite(x)
    sums = np.bincount(b[ok], weights=x[ok], minlength=nbins)
    counts = np.bincount(b[ok], minlength=nbins)
    return sums / np.maximum(counts, 1)


def profile_features(households) -> np.ndarray:
    feats = []
    for hd in households:
        day = _binned(hd, "hour", 24)
        shape = day / max(day.sum(), 1e-9)
        feats.append(np.concatenate([shape * 24.0, [np.log(np.nanmean(hd.train_kwh) + 1e-3)]]))
    F = np.asarray(feats)
    return (F - F.mean(0)) / (F.std(0) + 1e-9)


def cluster_profiles(households, k: int, seed: int) -> np.ndarray:
    k = min(k, len(households))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    return km.fit_predict(profile_features(households))


def week_shape_pc1(households) -> np.ndarray:
    """Score of each home on the first principal component of its normalised hour-of-week shape."""
    S = np.array([_binned(hd, "week", 168) for hd in households])
    S = S / np.maximum(S.sum(1, keepdims=True), 1e-12)
    S = S - S.mean(0)
    _, _, vt = np.linalg.svd(S, full_matrices=False)
    return S @ vt[0]


def window_shape_mass(kwh: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Mean consumption per hour-of-week bin of a set of target readings (window-level clients)."""
    t = pd.DatetimeIndex(times)
    b = t.dayofweek.to_numpy() * 24 + t.hour.to_numpy()
    sums = np.bincount(b, weights=kwh, minlength=168)
    return sums / np.maximum(np.bincount(b, minlength=168), 1)


def heterogeneity_from_masses(dists: list[np.ndarray], means: list[float]) -> dict:
    dists = [d / max(d.sum(), 1e-12) for d in dists]
    K = len(dists)
    js = [jensenshannon(dists[i], dists[j], base=2) ** 2 for i in range(K) for j in range(i + 1, K)]
    h_js = float(np.mean(js)) if js else 0.0
    means = np.asarray(means)
    h_cv = float(means.std() / means.mean()) if means.mean() > 0 else 0.0
    return {"H_JS": h_js, "H_CV": h_cv, "H": h_js + h_cv / (1.0 + h_cv)}


def hour_of_week_mass(households) -> np.ndarray:
    """Total consumption per hour-of-week bin, summed over the given households (not normalised)."""
    return np.sum([_binned(hd, "week", 168) for hd in households], axis=0)


def heterogeneity_index(client_households: list[list]) -> dict:
    dists, means = [], []
    for members in client_households:
        m = hour_of_week_mass(members)
        dists.append(m / max(m.sum(), 1e-12))
        means.append(np.mean([np.nanmean(hd.train_kwh) for hd in members]))
    K = len(dists)
    js = [jensenshannon(dists[i], dists[j], base=2) ** 2 for i in range(K) for j in range(i + 1, K)]
    h_js = float(np.mean(js)) if js else 0.0
    means = np.asarray(means)
    h_cv = float(means.std() / means.mean()) if means.mean() > 0 else 0.0
    return {"H_JS": h_js, "H_CV": h_cv, "H": h_js + h_cv / (1.0 + h_cv)}
