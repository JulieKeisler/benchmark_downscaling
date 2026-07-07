from __future__ import annotations

import numpy as np
import xarray as xr
# ------------------------------------------------------------------
# Scalar metrics: GCM vs pred (input consistency)
# ------------------------------------------------------------------

def temporal_corr_map(gcm: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Pearson correlation over time at each grid point (daily)."""
    return xr.corr(gcm, pred, dim="time")


def mean_temporal_corr(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Spatial mean of daily temporal Pearson correlation."""
    return float(temporal_corr_map(gcm, pred).mean().values)


def temporal_corr_annual_map(gcm: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Pearson correlation of annual-mean anomalies at each grid point.

    More informative than daily correlation for free-running GCMs: daily
    temporal_corr is trivially ~1 for any deterministic function of the GCM input.
    Annual correlation tests whether the method preserves interannual variability.
    """
    gcm_ann  = gcm.groupby("time.year").mean("time")
    pred_ann = pred.groupby("time.year").mean("time")
    return xr.corr(gcm_ann, pred_ann, dim="year")


def mean_temporal_corr_annual(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Spatial mean of annual temporal Pearson correlation."""
    return float(temporal_corr_annual_map(gcm, pred).mean().values)


def lag1_autocorr(da: xr.DataArray) -> xr.DataArray:
    """Lag-1 temporal autocorrelation at each grid point."""
    da0 = da.isel(time=slice(None, -1))
    da1 = da.isel(time=slice(1, None))
    da0 = da0.assign_coords(time=da1.time)
    return xr.corr(da0, da1, dim="time")


def lag1_autocorr_mae(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Mean absolute error of lag-1 autocorrelation maps."""
    return float(np.abs(lag1_autocorr(pred) - lag1_autocorr(gcm)).mean().values)


def annual_anomaly_bias(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """MAE of annual-mean anomalies (climate trend preservation)."""
    gcm_anom  = gcm.groupby("time.year").mean("time")  - gcm.mean("time")
    pred_anom = pred.groupby("time.year").mean("time") - pred.mean("time")
    return float(np.abs(pred_anom - gcm_anom).mean().values)

