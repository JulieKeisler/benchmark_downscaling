from __future__ import annotations

import numpy as np
import xarray as xr

from .scalar_metrics_era5 import daily_anomalies

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


def block_daily_anomalies(da: xr.DataArray, block_years: int | None = 20,
                          window: int = 31) -> xr.DataArray:
    """Daily anomalies from a climatology refitted in each block of ``block_years`` years.

    Over 2000-2099 one climatology would leave the forced trend in the anomalies
    (about 4 K for tas under ssp370), and the correlation would reward keeping
    the trend, which the change metrics already score. The blocks start at the
    first year; a last block shorter than half a block joins the previous one.
    ``block_years=None`` uses one climatology for the whole series.
    """
    if block_years is None:
        return daily_anomalies(da, window)
    years = da.time.dt.year.values
    starts = list(range(int(years.min()), int(years.max()) + 1, block_years))
    if len(starts) > 1 and int(years.max()) + 1 - starts[-1] < block_years / 2:
        starts.pop()
    edges = starts + [int(years.max()) + 1]
    parts = [daily_anomalies(da.isel(time=(years >= a) & (years < b)), window)
             for a, b in zip(edges[:-1], edges[1:])]
    return xr.concat(parts, dim="time")


def acc_daily_map(gcm: xr.DataArray, pred: xr.DataArray, block_years: int | None = 20,
                  window: int = 31) -> xr.DataArray:
    """Anomaly correlation coefficient (ACC) of daily values with the GCM at each grid cell.

    Each series' own daily climatology (``block_daily_anomalies``) is removed
    before the Pearson correlation over time, so the seasonal cycle, which
    makes the raw daily correlation close to 1 for any monotone correction,
    does not count. It tells how much of the GCM's day-to-day weather the
    method keeps.
    """
    return xr.corr(block_daily_anomalies(gcm, block_years, window),
                   block_daily_anomalies(pred, block_years, window), dim="time")


def mean_acc_daily(gcm: xr.DataArray, pred: xr.DataArray, block_years: int | None = 20,
                   window: int = 31) -> float:
    """Spatial mean of acc_daily_map (higher = better, bicubic = 1 by construction)."""
    return float(acc_daily_map(gcm, pred, block_years, window).mean().values)


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

