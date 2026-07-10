import numpy as np
import xarray as xr

# ------------------------------------------------------------------
# Inter-variable correlation
# ------------------------------------------------------------------

def _pointwise_corr(a: xr.DataArray, b: xr.DataArray) -> np.ndarray:
    """Pearson r along time at each grid cell, returned as a plain [lat, lon] array."""
    a_np = a.values.astype(float)
    b_np = b.values.astype(float)
    a_c  = a_np - a_np.mean(axis=0, keepdims=True)
    b_c  = b_np - b_np.mean(axis=0, keepdims=True)
    num   = (a_c * b_c).mean(axis=0)
    denom = a_np.std(axis=0) * b_np.std(axis=0)
    return np.where(denom > 1e-10, num / denom, np.nan)


def intervar_corr_mae(obs1: xr.DataArray, obs2: xr.DataArray,
                      pred1: xr.DataArray, pred2: xr.DataArray) -> float:
    """Mean absolute difference in pointwise temporal correlation between two variable pairs."""
    obs_corr  = _pointwise_corr(obs1, obs2)
    pred_corr = _pointwise_corr(pred1, pred2)
    return float(np.nanmean(np.abs(obs_corr - pred_corr)))
