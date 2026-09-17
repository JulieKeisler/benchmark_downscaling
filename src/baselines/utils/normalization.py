"""Normalization for DL downscaling baselines.

iDownscale / ConsistencyModel: log-transform + min-max to [-1, 1].
"""
from __future__ import annotations

import numpy as np

LOG_VARS: frozenset[str] = frozenset(["pr"])
LOG_EPS:  float = 1e-6


def _to_log(x: np.ndarray) -> np.ndarray:
    return np.log(x + LOG_EPS) - np.log(LOG_EPS)


def _from_log(x: np.ndarray) -> np.ndarray:
    return np.exp(x + np.log(LOG_EPS)) - LOG_EPS


# ── Min-max (iDownscale, ConsistencyModel) ────────────────────────────────────

def encode(x: np.ndarray, stats: dict, var: str) -> np.ndarray:
    """Physical → normalised [-1, 1]."""
    if var in LOG_VARS:
        x = _to_log(np.clip(x, 0.0, None))
    xmin, xmax = stats[var]["xmin"], stats[var]["xmax"]
    return (2.0 * (x - xmin) / (xmax - xmin) - 1.0).astype(np.float32)


def decode(x: np.ndarray, stats: dict, var: str) -> np.ndarray:
    """Normalised → physical."""
    xmin, xmax = stats[var]["xmin"], stats[var]["xmax"]
    x = (x + 1.0) / 2.0 * (xmax - xmin) + xmin
    if var in LOG_VARS:
        x = _from_log(x)
    return x.astype(np.float64)


def compute_stats(raw: np.ndarray, variables: list[str]) -> dict:
    """Compute per-variable xmin/xmax stats from raw [T, N_VARS, H, W]."""
    stats = {}
    for i, var in enumerate(variables):
        arr = raw[:, i]
        if var in LOG_VARS:
            arr = _to_log(np.clip(arr, 0.0, None))
        stats[var] = {
            "xmin": float(arr.min()),
            "xmax": float(arr.max()),
            "log_transform": var in LOG_VARS,
        }
    return stats

