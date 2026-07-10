"""Stateless downscaling baselines: bicubic, BCSD.

Each function signature: (gcm, obs, variable) -> xr.DataArray.
Import target for METHOD_REGISTRY in __init__.py and for evaluate.py.

CDF-t       lives in inference_cdft.py       (variable-specific hyperparameters, separate calib).
iDownscale  lives in inference_idownscale.py  (CERFACS U-Net + coarse-scale CDF-t bias correction).
SerpentFlow lives in inference_serpentflow.py (flow-matching U-Net with attention).
CM          lives in train_consistency.py + inference_cm.py (consistency distillation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from data_processing import bicubic_regrid

# ── Bicubic ───────────────────────────────────────────────────────────────────

def bicubic_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                      variable: str) -> xr.DataArray:
    """Cubic spline interpolation to the ERA5 grid. Stateless baseline."""
    pred = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
    pred = pred.assign_coords(time=obs.time)
    if variable == "pr":
        pred.values[:] = np.clip(pred.values, 0.0, None)
    return pred


# ── BC (QM) ───────────────────────────────────────────────────────────────────

def _qmap_1d(src: np.ndarray, ref: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """Empirical quantile mapping: fit CDF from src→ref, apply to apply.

    Returns apply unchanged if src or ref is empty (e.g. a month with no data).
    """
    if len(src) == 0 or len(ref) == 0:
        return apply.copy()
    q = np.linspace(0, 1, min(len(src), 200))
    return np.interp(apply, np.quantile(src, q), np.quantile(ref, q))
