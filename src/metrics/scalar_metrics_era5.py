from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import ks_2samp

# ------------------------------------------------------------------
# Scalar metrics: obs vs pred
# ------------------------------------------------------------------

def bias(obs: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Temporal-mean bias at each grid point (pred − obs)."""
    return pred.mean("time") - obs.mean("time")


def clim_rmse(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """RMSE of the time-mean spatial field."""
    return float(np.sqrt(((pred.mean("time") - obs.mean("time")) ** 2).mean().values))


def spatial_mean_abs_bias(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """Mean absolute temporal-mean bias across all grid points."""
    return float(np.abs(pred.mean("time") - obs.mean("time")).mean().values)


def std_error(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """Mean absolute error of temporal std across all grid points."""
    return float(np.abs(pred.std("time") - obs.std("time")).mean().values)


def ks_stat(obs: xr.DataArray, pred: xr.DataArray, quantile: float | None = None) -> float:
    """KS distance between flattened distributions (or tail above/below quantile)."""
    o = obs.values.ravel()
    p = pred.values.ravel()
    o = o[np.isfinite(o)]
    p = p[np.isfinite(p)]
    if quantile is not None:
        if quantile >= 0.5:
            thr = np.quantile(o, quantile)
            o = o[o > thr]
            p = p[p > thr]
        else:
            thr = np.quantile(o, quantile)
            # Guard against negative thresholds from CDS rounding artifacts
            # (e.g. ERA5 pr has tiny negative values; 5th-pct can be < 0,
            # making o[o<=thr] empty and the tail test meaningless).
            if thr < 0:
                return float("nan")
            o = o[o <= thr]
            p = p[p <= thr]
    if len(o) < 2 or len(p) < 2:
        return float("nan")
    return float(ks_2samp(o, p).statistic)


def seasonal_mae(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """MAE of the monthly climatology (12 months) averaged over space.

    Despite the name, this computes monthly—not seasonal—climatology error.
    The name is retained for backward compatibility with saved metrics.json files.
    """
    obs_clim = obs.groupby("time.month").mean("time")
    pred_clim = pred.groupby("time.month").mean("time")
    return float(np.abs(pred_clim - obs_clim).mean().values)


def extreme_freq_bias(obs: xr.DataArray, pred: xr.DataArray, q: float = 0.95) -> float:
    """Spatial mean of bias in fraction of days above obs q-quantile.

    The threshold is a single global quantile computed over all space × time
    (not per grid cell). This is a deliberate choice: it tests how well the
    method captures domain-wide extreme events rather than local tail shape.
    """
    thr = float(obs.quantile(q))
    obs_freq = (obs > thr).mean("time")
    pred_freq = (pred > thr).mean("time")
    return float((pred_freq - obs_freq).mean().values)


def extreme_intensity_bias(obs: xr.DataArray, pred: xr.DataArray, q: float = 0.95) -> float:
    """Spatial mean of bias in mean intensity above obs q-quantile.

    Uses the same global threshold as extreme_freq_bias (see that docstring).
    """
    thr = float(obs.quantile(q))
    obs_int = obs.where(obs > thr).mean("time")
    pred_int = pred.where(pred > thr).mean("time")
    return float((pred_int - obs_int).mean().values)
