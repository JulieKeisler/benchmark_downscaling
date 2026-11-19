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


_DAYS_BEFORE_MONTH = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])


def _day_of_year(da: xr.DataArray) -> np.ndarray:
    """Day 1..365 of each time step, Feb 29 mapped onto Feb 28 (day 59)."""
    month, day = da.time.dt.month.values, da.time.dt.day.values
    return np.where((month == 2) & (day == 29), 59, _DAYS_BEFORE_MONTH[month - 1] + day)


def daily_climatology(da: xr.DataArray, window: int = 31) -> xr.DataArray:
    """Mean of each day of a 365-day year, smoothed by a circular running mean.

    Feb 29 is left out, so every calendar maps to days 1..365. The running mean
    (``window`` days, wrapping Dec 31 to Jan 1; 1 = no smoothing) removes most
    of the sampling noise of one-day means over ~20 years.
    """
    month, day = da.time.dt.month.values, da.time.dt.day.values
    keep = ~((month == 2) & (day == 29))
    da = da.isel(time=keep)
    clim = da.assign_coords(doy=("time", _day_of_year(da))).groupby("doy").mean("time")
    if clim.sizes["doy"] != 365:
        raise ValueError(f"daily_climatology: {clim.sizes['doy']} days of year, need 365")
    half = window // 2
    padded = clim.pad(doy=half, mode="wrap")
    return padded.rolling(doy=window, center=True).mean().isel(doy=slice(half, half + 365))


def harmonic_climatology(da: xr.DataArray, n_harmonics: int = 2) -> xr.DataArray:
    """Least-squares fit of a mean and the first ``n_harmonics`` annual harmonics to the 365-day climatology.

    Two harmonics (annual and semi-annual) keep the shape of the seasonal cycle
    and drop the day-to-day noise that a running mean leaves in precipitation
    and pressure: over 21 years, two periods of ERA5 differ less with the fit.
    """
    raw = daily_climatology(da, window=1)
    d = np.arange(1, 366)
    basis = [np.ones(365)]
    for k in range(1, n_harmonics + 1):
        basis += [np.cos(2 * np.pi * k * d / 365), np.sin(2 * np.pi * k * d / 365)]
    x = np.stack(basis, axis=1)                                   # (365, 1 + 2n)
    proj = x @ np.linalg.pinv(x)                                  # projection on the harmonics
    fitted = np.tensordot(proj, raw.transpose("doy", ...).values, axes=1)
    return raw.transpose("doy", ...).copy(data=fitted)


def seasonal_corr_map(obs: xr.DataArray, pred: xr.DataArray, n_harmonics: int = 2) -> xr.DataArray:
    """Pearson correlation of the harmonic climatologies at each grid point.

    Scores the shape and timing of the seasonal cycle, whatever its offset and
    amplitude, which seasonal_mae scores.
    """
    return xr.corr(harmonic_climatology(obs, n_harmonics), harmonic_climatology(pred, n_harmonics), dim="doy")


def seasonal_corr(obs: xr.DataArray, pred: xr.DataArray, n_harmonics: int = 2) -> float:
    """Spatial mean of seasonal_corr_map (higher = better, 1 = same shape)."""
    return float(seasonal_corr_map(obs, pred, n_harmonics).mean().values)


def daily_anomalies(da: xr.DataArray, window: int = 31) -> xr.DataArray:
    """Daily values minus the series' own smoothed 365-day climatology (Feb 29 takes Feb 28's)."""
    clim = daily_climatology(da, window)
    on_days = clim.sel(doy=xr.DataArray(_day_of_year(da), dims="time")).drop_vars("doy")
    return da - on_days.assign_coords(time=da.time)


def lag1_autocorr_anom_mae(obs: xr.DataArray, pred: xr.DataArray, window: int = 31) -> float:
    """MAE of the lag-1 autocorrelation of daily anomalies (persistence), averaged over space.

    The seasonal cycle is removed from each series first: left in, it dominates
    the autocorrelation of temperature and radiation (ERA5 radiation over France:
    0.84 with the cycle, 0.33 without), and the score would measure the size of
    the cycle against the daily variance instead of persistence.
    """
    def ac1(a: xr.DataArray) -> xr.DataArray:
        a0 = a.isel(time=slice(None, -1))
        a1 = a.isel(time=slice(1, None)).assign_coords(time=a0.time)
        return xr.corr(a0, a1, dim="time")
    return float(np.abs(ac1(daily_anomalies(pred, window)) - ac1(daily_anomalies(obs, window))).mean().values)


def extreme_freq_bias(obs: xr.DataArray, pred: xr.DataArray, q: float = 0.95) -> float:
    """Spatial mean of |bias| in fraction of days above obs q-quantile.

    The threshold is a single global quantile computed over all space × time
    (not per grid cell). This is a deliberate choice: it tests how well the
    method captures domain-wide extreme events rather than local tail shape.
    Signed bias is folded into an absolute value: direction (over/under
    prediction) is lost, magnitude is what matters.
    """
    thr = float(obs.quantile(q))
    obs_freq = (obs > thr).mean("time")
    pred_freq = (pred > thr).mean("time")
    return float(np.abs(pred_freq - obs_freq).mean().values)


def extreme_maps(obs: xr.DataArray, pred: xr.DataArray, q_high: float = 0.99,
                 q_low: float | None = 0.01, q_freq: float = 0.95) -> dict[str, np.ndarray]:
    """Per-cell maps of where the extremes are, for ERA5 and the prediction.

    - ``obs_q99`` / ``pred_q99``: the ``q_high`` quantile of daily values at each cell;
    - ``obs_q01`` / ``pred_q01``: the ``q_low`` quantile (left out when ``q_low`` is None,
      e.g. precipitation, whose low quantiles are 0 mm/day everywhere);
    - ``obs_freq95`` / ``pred_freq95``: fraction of days above the ERA5 ``q_freq``
      quantile of the whole region, the map that extreme_freq_bias averages.
    """
    thr = float(obs.quantile(q_freq))
    out = {
        "obs_q99": obs.quantile(q_high, dim="time").values,
        "pred_q99": pred.quantile(q_high, dim="time").values,
        "obs_freq95": (obs > thr).mean("time").values,
        "pred_freq95": (pred > thr).mean("time").values,
        "freq_threshold": np.float64(thr),
    }
    if q_low is not None:
        out["obs_q01"] = obs.quantile(q_low, dim="time").values
        out["pred_q01"] = pred.quantile(q_low, dim="time").values
    return out


def extreme_intensity_bias(obs: xr.DataArray, pred: xr.DataArray, q: float = 0.95) -> float:
    """Spatial mean of |bias| in mean intensity above obs q-quantile.

    Uses the same global threshold as extreme_freq_bias (see that docstring).
    Signed bias is folded into an absolute value.
    """
    thr = float(obs.quantile(q))
    obs_int = obs.where(obs > thr).mean("time")
    pred_int = pred.where(pred > thr).mean("time")
    return float(np.abs(pred_int - obs_int).mean().values)
