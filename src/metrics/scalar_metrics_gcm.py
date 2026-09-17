from __future__ import annotations

import numpy as np
import xarray as xr

# ------------------------------------------------------------------
# Scalar metrics: GCM vs pred (input consistency)
#
# PRECONDITION: all functions assume gcm and pred are on the same grid
# (same lat/lon coordinates). The caller must regrid gcm to the ERA5 grid
# (e.g. via bicubic_regrid) before passing it here.
# ------------------------------------------------------------------

def temporal_corr_map(gcm: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Pearson correlation over time at each grid point (daily)."""
    return xr.corr(gcm, pred, dim="time")


def mean_temporal_corr(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Spatial mean of daily temporal Pearson correlation.

    Returns 0.0 when all grid points have zero temporal variance in pred
    (e.g. a stochastic model that predicts monthly climatology only).
    """
    v = float(temporal_corr_map(gcm, pred).mean().values)
    return 0.0 if np.isnan(v) else v


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


def delta_change_score(
    pred_hist: xr.DataArray,
    pred_fut:  xr.DataArray,
    gcm_hist:  xr.DataArray,
    gcm_fut:   xr.DataArray,
) -> float:
    """Climate change signal preservation metric (lower = better).

    Computes the mean absolute error between the method's relative change
    and the GCM's relative change over a spatial grid:

        score = mean_grid | Δ_method - Δ_gcm |

    where Δ = (future_mean - hist_mean) / |hist_mean| × 100 (% change).
    """
    hist_pred = pred_hist.mean("time").values
    fut_pred  = pred_fut.mean("time").values
    hist_gcm  = gcm_hist.mean("time").values
    fut_gcm   = gcm_fut.mean("time").values

    with np.errstate(invalid="ignore", divide="ignore"):
        delta_pred = np.where(
            np.abs(hist_pred) > 1e-8,
            (fut_pred - hist_pred) / np.abs(hist_pred) * 100.0,
            np.nan,
        )
        delta_gcm = np.where(
            np.abs(hist_gcm) > 1e-8,
            (fut_gcm - hist_gcm) / np.abs(hist_gcm) * 100.0,
            np.nan,
        )

    return float(np.nanmean(np.abs(delta_pred - delta_gcm)))

