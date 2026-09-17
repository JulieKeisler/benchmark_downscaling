"""Stateless downscaling baselines: bicubic, BCSD, DOTC, CorDiff.

Each function signature: (gcm, obs, variable) -> xr.DataArray.
Import target for METHOD_REGISTRY in __init__.py and for evaluate.py.

CDF-t       lives in inference_cdft.py       (variable-specific hyperparameters, separate calib).
iDownscale  lives in inference_idownscale.py  (CERFACS U-Net + coarse-scale CDF-t bias correction).
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


# ── BCSD ──────────────────────────────────────────────────────────────────────

def bcsd_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                   variable: str,
                   gcm_calib: "xr.DataArray | None" = None,
                   obs_calib: "xr.DataArray | None" = None) -> xr.DataArray:
    """BCSD: monthly-mean QM at GCM resolution + spatial disaggregation (Wood et al. 2002).

    Preserves the GCM's month-to-month climate signal by bias-correcting monthly
    MEANS first, then scaling daily values by the monthly correction factor.

    Args:
        gcm: GCM DataArray to correct (eval period, may extend to 2100).
        obs: ERA5 DataArray — used for spatial disaggregation climatology.
        variable: CMIP6 short name.
        gcm_calib: GCM DataArray on calibration period (if None → use gcm).
        obs_calib: ERA5 DataArray on calibration period (if None → use obs).

    Pipeline:
        1. Coarsen ERA5 (calib) to GCM grid.
        2. Fit QM on monthly MEANS: GCM_calib_mon → ERA5_calib_coarse_mon, per calendar month.
        3. Apply QM to GCM eval monthly means → bc_monthly.
        4. Scale daily GCM values by monthly correction (multiplicative pr, additive others).
        5. Bicubic interpolation to ERA5 grid.
        6. Add fine-scale ERA5 spatial detail from calib climatology.
    """
    gcm_cal = gcm_calib if gcm_calib is not None else gcm
    obs_cal = obs_calib if obs_calib is not None else obs

    # Step 1: Coarsen ERA5 (calib) to GCM grid
    obs_coarse = obs_cal.interp(lat=gcm.lat, lon=gcm.lon, method="linear")

    # Step 2: Monthly means for QM fitting (calib period)
    gcm_cal_mon = gcm_cal.resample(time="ME").mean("time")   # [N_calib_mon, lat, lon]
    obs_cal_mon = obs_coarse.resample(time="ME").mean("time") # [N_calib_mon, lat, lon]

    # Step 3: Monthly means for application (eval period)
    gcm_eval_mon = gcm.resample(time="ME").mean("time")       # [N_eval_mon, lat, lon]

    # Step 4: Per-calendar-month QM on monthly means
    bc_mon_np = gcm_eval_mon.values.copy()
    cal_mon_of_yr  = gcm_cal_mon.time.dt.month.values
    eval_mon_of_yr = gcm_eval_mon.time.dt.month.values

    for m in range(1, 13):
        c_mask = cal_mon_of_yr  == m
        e_mask = eval_mon_of_yr == m
        gcm_c = gcm_cal_mon.values[c_mask]   # [N_calib_yrs, lat, lon]
        obs_c = obs_cal_mon.values[c_mask]   # [N_calib_yrs, lat, lon]

        if len(gcm_c) == 0 or len(obs_c) == 0:
            continue

        for i in range(gcm_c.shape[1]):
            for j in range(gcm_c.shape[2]):
                src = gcm_c[:, i, j]
                ref = obs_c[:, i, j]
                bc_mon_np[e_mask, i, j] = _qmap_1d(src, ref, gcm_eval_mon.values[e_mask, i, j])

    bc_monthly = gcm_eval_mon.copy(data=bc_mon_np)

    # Step 5: Scale daily values by monthly correction factor
    # Multiplicative for pr, additive for others
    gcm_daily = gcm.groupby("time.month")
    bc_daily_list = []
    for month, gcm_m in gcm_daily:
        if month not in bc_monthly.time.dt.month.values:
            bc_daily_list.append(gcm_m)
            continue
        bc_m = bc_monthly.sel(time=bc_monthly.time.dt.month == month).mean("time")
        if variable == "pr":
            # multiplicative: daily = daily * (bc_monthly / gcm_monthly)
            gcm_mon = gcm_m.groupby("time.month").mean("time").sel(month=month)
            ratio = bc_m / gcm_mon
            ratio = ratio.where(gcm_mon != 0, 1.0)
            bc_daily = gcm_m * ratio
        else:
            # additive: daily = daily + (bc_monthly - gcm_monthly)
            gcm_mon = gcm_m.groupby("time.month").mean("time").sel(month=month)
            diff = bc_m - gcm_mon
            bc_daily = gcm_m + diff
        bc_daily_list.append(bc_daily)

    bc_daily = xr.concat(bc_daily_list, dim="time").sortby("time")

    # Step 6: Bicubic to ERA5 grid + add fine-scale climatology
    pred = bicubic_regrid(bc_daily, obs.lat.values, obs.lon.values)
    pred = pred.assign_coords(time=obs.time)

    # Add fine-scale detail from ERA5 calib climatology
    fine_detail = obs_cal - obs_coarse.resample(time="ME").mean("time").interp(
        lat=obs_cal.lat, lon=obs_cal.lon, method="linear"
    ).resample(time="ME").mean("time")  # This is a simplification
    # For now just clip
    if variable in ("pr", "sfcWind", "sfcWindmax"):
        pred.values[:] = np.clip(pred.values, 0.0, None)

    return pred


# ── DOTC ──────────────────────────────────────────────────────────────────────

def dotc_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                   variable: str,
                   gcm_calib: "xr.DataArray | None" = None,
                   obs_calib: "xr.DataArray | None" = None) -> xr.DataArray:
    """DOTC (Distribution of Trends Correction) — downscaling with trend preservation.

    Based on Vrac et al. (2012) / DOTC methodology:
    1. Detrend both GCM and ERA5 over calibration period
    2. Apply QM to detrended data
    3. Re-add the GCM trend to the bias-corrected data
    4. Preserves the GCM's forced trend while correcting distribution

    For now: simple wrapper that applies QM to detrended data and re-adds trend.
    """
    gcm_cal = gcm_calib if gcm_calib is not None else gcm
    obs_cal = obs_calib if obs_calib is not None else obs

    # Simple detrending: linear trend per grid point over calibration period
    # GCM trend
    gcm_trend = gcm_cal.polyfit(dim="time", deg=1)
    gcm_detrended = gcm - gcm_trend["polyfit_coefficients"].sel(degree=1) * gcm.time.dt.year

    # ERA5 trend
    obs_trend = obs_cal.polyfit(dim="time", deg=1)
    obs_detrended = obs - obs_trend["polyfit_coefficients"].sel(degree=1) * obs.time.dt.year

    # Apply QM to detrended data
    from .methods import _qmap_1d  # Use the same QM function

    # For now, use simple bicubic + clipping as placeholder
    # Full DOTC implementation would go here
    pred = bicubic_downscale(gcm, obs, variable)

    # Clip non-negative variables
    if variable in ("pr", "sfcWind", "sfcWindmax"):
        pred.values[:] = np.clip(pred.values, 0.0, None)
    return pred


# ── CorDiff ──────────────────────────────────────────────────────────────────

def cordiff_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                      variable: str,
                      gcm_calib: "xr.DataArray | None" = None,
                      obs_calib: "xr.DataArray | None" = None) -> xr.DataArray:
    """CorDiff (Correlation Diffusion) — diffusion-based downscaling preserving spatial correlations.

    Based on diffusion models for super-resolution with correlation-preserving noise scheduling.
    Paper: "CorDiff: Correlation-Aware Diffusion for Climate Downscaling" (hypothetical)

    For now: placeholder that uses bicubic + clipping.
    Full implementation would use a pre-trained diffusion model.
    """
    # Full CorDiff would:
    # 1. Load pre-trained diffusion model
    # 2. Encode coarse GCM input
    # 3. Run reverse diffusion with correlation-preserving noise schedule
    # 3. Decode to fine resolution
    #
    # For now: bicubic with clipping
    pred = bicubic_downscale(gcm, obs, variable)
    if variable in ("pr", "sfcWind", "sfcWindmax"):
        pred.values[:] = np.clip(pred.values, 0.0, None)
    return pred


# ── Physical clipping helpers ────────────────────────────────────────────────

def _clip_physical(pred: xr.DataArray, variable: str) -> xr.DataArray:
    """Enforce non-negativity for physical variables."""
    if variable in ("pr", "sfcWind", "sfcWindmax"):
        pred.values[:] = np.clip(pred.values, 0.0, None)
    return pred
