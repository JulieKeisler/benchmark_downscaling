#!/usr/bin/env python3
"""DOTC (Distribution of Trends Correction) baseline inference using SBCK.

DOTC (Vrac et al. 2012) preserves the GCM's forced trend while correcting the distribution:
    1. Fit linear trend over calibration period for both GCM and ERA5
    2. Detrend: x' = x - trend * (year - ref_year)
    3. Apply CDF-t (SBCK) to detrended values
    4. Re-add the original GCM trend: x_bc = x'_bc + trend_GCM * (year - ref_year)

Applied month-by-month per grid cell with sliding window over evaluation period.

Requires: pip install sbck

Usage:
    python inference_dotc.py \
        --era5-dir /path/to/era5 \
        --gcm-dir  /path/to/cmip6/ipsl_cm6a_lr \
        --out-dir  results/dotc \
        --calib-start 1980 --calib-end 1999 \
        --eval-start  2000 --eval-end  2100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

try:
    import SBCK
except ImportError:
    SBCK = None  # guarded inside _cdft_apply; fails there, not at import time

# Physical lower bounds applied after correction
CLIP_MIN: dict[str, float] = {
    "pr":         0.0,
    "sfcWind":    0.0,
    "sfcWindmax": 0.0,
    "rsds":       0.0,
}


# ── Core CDF-t (SBCK) ─────────────────────────────────────────────────────────

def _cdft_apply(gp: np.ndarray, obs: np.ndarray, gf: np.ndarray,
                clip_min: float | None = None) -> np.ndarray:
    """CDF-t for one grid cell, one calendar month — delegates to SBCK.CDFt.

    gp:  GCM calibration values  [n_calib]
    obs: ERA5 calibration values  [n_calib]
    gf:  GCM evaluation values   [n_eval]

    Uses version=2 with normalize_cdf=True (SBCK defaults).
    """
    if SBCK is None:
        raise ImportError("SBCK is required: pip install sbck")
    bc = SBCK.CDFt(version=2, normalize_cdf=True)
    bc.fit(obs.reshape(-1, 1), gp.reshape(-1, 1), gf.reshape(-1, 1))
    out = np.asarray(bc.predict(gf.reshape(-1, 1)), dtype=np.float32).ravel()
    if clip_min is not None:
        out = np.maximum(out, clip_min)
    return out


def _detrend_linear(da: xr.DataArray, ref_year: float) -> xr.DataArray:
    """Remove linear trend per grid point: x' = x - slope * (year - ref_year)."""
    coeffs = da.polyfit(dim="time", deg=1)
    slope = coeffs["polyfit_coefficients"].sel(degree=1)
    years = da.time.dt.year.astype(float)
    detrended = da - slope * (years - ref_year)
    return detrended


def _add_trend(da: xr.DataArray, trend: xr.DataArray, ref_year: float) -> xr.DataArray:
    """Re-add linear trend: x = x' + trend * (year - ref_year)."""
    years = da.time.dt.year.astype(float)
    return da + trend * (years - ref_year)


def _dotc_field(gcm_calib: xr.DataArray, obs_calib: xr.DataArray,
                gcm_eval: xr.DataArray, variable: str) -> xr.DataArray:
    """Apply DOTC month-by-month to a single-variable field on the ERA5 grid."""
    clip_min = CLIP_MIN.get(variable)
    ref_year = float(gcm_calib.time.dt.year.values.mean())  # center of calibration period

    def _apply(gp, obs, gf):
        # gp: GCM calib [n_calib], obs: ERA5 calib [n_calib], gf: GCM eval [n_eval]
        # Step 1: detrend calibration data
        gp_da = xr.DataArray(gp, dims=["time"])
        obs_da = xr.DataArray(obs, dims=["time"])
        gf_da = xr.DataArray(gf, dims=["time"])

        # Simple linear detrending on the 1D arrays
        gp_t = np.arange(len(gp))
        obs_t = np.arange(len(obs))
        gf_t = np.arange(len(gf)) + len(gp)

        # Fit trends
        gp_slope = np.polyfit(gp_t, gp, 1)[0]
        obs_slope = np.polyfit(obs_t, obs, 1)[0]

        gp_detrended = gp - gp_slope * (gp_t - gp_t.mean())
        obs_detrended = obs - obs_slope * (obs_t - obs_t.mean())
        gf_detrended = gf - gp_slope * (gf_t - gf_t.mean())

        # Step 2: Apply CDF-t to detrended data
        bc = _cdft_apply(gp_detrended, obs_detrended, gf_detrended, clip_min)

        # Step 3: Re-add GCM trend
        bc = bc + gp_slope * (gf_t - gp_t.mean())
        return bc

    results = []
    for m in range(1, 13):
        gp  = gcm_calib.isel(time=gcm_calib.time.dt.month == m).drop_vars("time")
        obs = obs_calib.isel(time=obs_calib.time.dt.month == m).drop_vars("time")
        gf  = gcm_eval.isel(time=gcm_eval.time.dt.month == m)
        if gp.sizes["time"] == 0 or obs.sizes["time"] == 0 or gf.sizes["time"] == 0:
            continue
        corrected = xr.apply_ufunc(
            _apply,
            gp, obs, gf.drop_vars("time"),
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[np.float32],
            exclude_dims={"time"},
        )
        corrected = corrected.assign_coords(time=gf.time)
        results.append(corrected)

    return xr.concat(results, dim="time").sortby("time").transpose("time", "lat", "lon")


# ── Sliding-window wrapper ────────────────────────────────────────────────────

def _sliding_windows(eval_start: int, eval_end: int,
                     win_size: int = 20, step: int = 10, trim: int = 5,
                     ) -> list[tuple[int, int, int, int]]:
    """Return (win_start, win_end, keep_start, keep_end) tuples."""
    windows: list[tuple[int, int, int, int]] = []
    win_start = eval_start - trim
    while True:
        win_end    = win_start + win_size - 1
        keep_start = max(win_start + trim, eval_start)
        keep_end   = min(win_end   - trim, eval_end)
        if keep_start > eval_end:
            break
        windows.append((win_start, win_end, keep_start, keep_end))
        if keep_end >= eval_end:
            break
        win_start += step
    return windows


def _dotc_field_sliding(gcm_calib: xr.DataArray, obs_calib: xr.DataArray,
                         gcm_eval: xr.DataArray, variable: str,
                         win_size: int = 20, step: int = 10, trim: int = 5,
                         ) -> xr.DataArray:
    """DOTC with sliding windows over the evaluation period."""
    eval_start = int(gcm_eval.time.dt.year.values.min())
    eval_end   = int(gcm_eval.time.dt.year.values.max())

    windows = _sliding_windows(eval_start, eval_end, win_size, step, trim)
    if len(windows) == 1:
        return _dotc_field(gcm_calib, obs_calib, gcm_eval, variable)

    pieces = []
    for win_start, win_end, keep_start, keep_end in windows:
        yr = gcm_eval.time.dt.year.values
        gcm_win = gcm_eval.isel(time=(yr >= win_start) & (yr <= win_end))
        if gcm_win.sizes["time"] == 0:
            continue
        corrected_win  = _dotc_field(gcm_calib, obs_calib, gcm_win, variable)
        yr_c = corrected_win.time.dt.year.values
        corrected_keep = corrected_win.isel(
            time=(yr_c >= keep_start) & (yr_c <= keep_end)
        )
        if corrected_keep.sizes["time"] > 0:
            pieces.append(corrected_keep)

    if not pieces:
        return _dotc_field(gcm_calib, obs_calib, gcm_eval, variable)
    return xr.concat(pieces, dim="time").sortby("time")


# ── Convenience wrapper (for inline METHOD_REGISTRY use) ─────────────────────

def dotc_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                   variable: str) -> xr.DataArray:
    """DOTC on the ERA5 grid (degenerate: calib = eval period)."""
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent / "evaluation"))
    from data_processing import bicubic_regrid  # noqa: PLC0415

    gcm_fine = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
    gcm_fine = gcm_fine.assign_coords(time=obs.time)
    return _dotc_field(gcm_fine, obs, gcm_fine, variable)


# ── Standalone inference ────────────────────────────────────────────────────

def run(era5_dir: Path, gcm_dir: Path, out_dir: Path,
        variables: list[str],
        calib_start: int, calib_end: int,
        eval_start: int, eval_end: int,
        scenario: str = "ssp370",
        win_size: int = 20, step: int = 10, trim: int = 5) -> None:

    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent / "evaluation"))
    from data_processing import (  # noqa: PLC0415
        DEFAULT_VARIABLES, load_era5, load_gcm, align_time,
        save_predictions, bicubic_regrid,
    )
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calibration: {calib_start}–{calib_end}  |  Evaluation: {eval_start}–{eval_end}"
          f"  |  Window: {win_size}yr step {step}yr trim {trim}yr")

    era5_calib, gcm_calib, gcm_eval = {}, {}, {}
    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        era5_calib[var] = load_era5(era5_dir, var, calib_start, calib_end)
        gcm_calib[var]  = load_gcm(gcm_dir, var, calib_start, calib_end, "historical")
        gcm_eval[var]   = load_gcm(gcm_dir, var, eval_start,  eval_end,   scenario)

    all_calib = {f"era5_{v}": era5_calib[v] for v in variables}
    all_calib.update({f"gcm_{v}": gcm_calib[v] for v in variables})
    aligned_calib = align_time(all_calib)
    era5_calib = {v: aligned_calib[f"era5_{v}"] for v in variables}
    gcm_calib  = {v: aligned_calib[f"gcm_{v}"]  for v in variables}

    ref_era5 = next(iter(era5_calib.values()))
    target_lat, target_lon = ref_era5.lat.values, ref_era5.lon.values

    arrays = []
    ref_da = None
    for var in variables:
        print(f"  DOTC {var} ...", flush=True)
        gc_c = bicubic_regrid(gcm_calib[var], target_lat, target_lon)
        gc_c = gc_c.assign_coords(time=era5_calib[var].time)
        gc_e = bicubic_regrid(gcm_eval[var], target_lat, target_lon)
        gc_e = gc_e.assign_coords(time=gcm_eval[var].time)

        corrected = _dotc_field_sliding(gc_c, era5_calib[var], gc_e, var,
                                         win_size=win_size, step=step, trim=trim)
        if ref_da is None:
            ref_da = corrected
        arrays.append(torch.tensor(corrected.values, dtype=torch.float32))

    tensor = torch.stack(arrays, dim=1)
    save_predictions(tensor, variables, out_dir, ref_da)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",    required=True)
    p.add_argument("--gcm-dir",     required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--variables",   nargs="+", default=None)
    p.add_argument("--calib-start", type=int, default=None)
    p.add_argument("--calib-end",   type=int, default=None)
    p.add_argument("--eval-start",  type=int, default=2000)
    p.add_argument("--eval-end",    type=int, default=2100)
    p.add_argument("--scenario",    default="ssp370")
    p.add_argument("--win-size",    type=int, default=20)
    p.add_argument("--step",        type=int, default=10)
    p.add_argument("--trim",        type=int, default=5)
    args = p.parse_args()

    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent / "evaluation"))
    from data_processing import DEFAULT_VARIABLES  # noqa: PLC0415

    variables   = args.variables or DEFAULT_VARIABLES
    calib_start = args.calib_start if args.calib_start is not None else args.eval_start
    calib_end   = args.calib_end   if args.calib_end   is not None else args.eval_end

    run(Path(args.era5_dir), Path(args.gcm_dir), Path(args.out_dir),
        variables, calib_start, calib_end,
        args.eval_start, args.eval_end, args.scenario,
        args.win_size, args.step, args.trim)


if __name__ == "__main__":
    main()