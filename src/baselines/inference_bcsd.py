#!/usr/bin/env python3
"""BCSD baseline inference — Bias-Corrected Spatial Disaggregation (Wood et al. 2002, 2004).

Monthly-mean QM at GCM resolution, then daily temporal disaggregation and spatial
interpolation to the ERA5 grid. A separate calibration period is used to fit QM.

As in Wood et al. (2004), the GCM's tas trend is removed before the quantile map
and added back after it: within each 20-year window the monthly-mean change
relative to the calibration period is subtracted, mapped, and re-added, so
warmer future months are not clamped to the top of the 1980-1999 range.
`--no-detrend` reproduces the earlier variant, kept in the benchmark as
"EQM (no detrending)" (`--method eqm`), which loses part of the warming.

Usage:
    python baselines/inference_bcsd.py \
        --era5-dir /scratch/jkeisler/benchmark_downscaling/era5 \
        --gcm-dir  /scratch/jkeisler/benchmark_downscaling/cmip6/ipsl_cm6a_lr \
        --out-dir  /scratch/jkeisler/downscaling_benchmark/results/bcsd \
        --calib-start 1980 --calib-end 1999 \
        --eval-start  2000 --eval-end  2100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from data_processing import (
    DEFAULT_VARIABLES, load_era5, load_gcm, align_time, save_predictions,
)
from methods import _qmap_1d


def _box_average(obs: xr.DataArray, lat_c: np.ndarray, lon_c: np.ndarray) -> xr.DataArray:
    """Average ERA5 onto GCM cells: each ERA5 point goes to its nearest GCM centre.

    Wood et al. (2002) aggregate the observations to the GCM grid; point
    interpolation at cell centres would keep sub-grid ERA5 variance and
    orography in the quantile-map target.
    """
    da = np.abs(obs.lat.values[:, None] - lat_c[None, :])
    db = np.abs(obs.lon.values[:, None] - lon_c[None, :])
    ia, ib = da.argmin(1), db.argmin(1)
    # Points beyond half a cell from every centre belong to no GCM cell. Edge
    # cells only partly inside the ERA5 box average their inner part.
    in_a = da.min(1) <= 0.5 * np.abs(np.diff(lat_c)).mean() + 1e-6
    in_b = db.min(1) <= 0.5 * np.abs(np.diff(lon_c)).mean() + 1e-6
    keep = (in_a[:, None] & in_b[None, :]).ravel()
    cell = (ia[:, None] * len(lon_c) + ib[None, :]).ravel()
    n_cells = len(lat_c) * len(lon_c)
    counts = np.bincount(cell[keep], minlength=n_cells).astype(np.float64)
    vals = obs.values.reshape(obs.shape[0], -1)
    sums = np.zeros((obs.shape[0], n_cells))
    for c in np.unique(cell[keep]):
        sums[:, c] = vals[:, keep & (cell == c)].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        coarse = (sums / counts).reshape(obs.shape[0], len(lat_c), len(lon_c))
    out = xr.DataArray(coarse, dims=("time", "lat", "lon"),
                       coords={"time": obs.time, "lat": lat_c, "lon": lon_c})
    if np.isnan(coarse).any():  # GCM cells centred outside the ERA5 box: nearest ERA5 value
        near = obs.interp(lat=out.lat, lon=out.lon, method="nearest",
                          kwargs={"fill_value": None})
        out = out.fillna(near)
    if np.isnan(out.values).any():
        raise ValueError("_box_average: NaN left in the QM target")
    return out
from inference_cdft import _sliding_windows

def bcsd_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                   variable: str,
                   gcm_calib: "xr.DataArray | None" = None,
                   obs_calib: "xr.DataArray | None" = None,
                   detrend: bool = True) -> xr.DataArray:
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

    # Longitude convention: GCM crops may use 0–360° while ERA5 uses −180..180°.
    # Interpolating ERA5 at out-of-range targets silently yields all-NaN, so map
    # the GCM grid to the ERA5 convention first (keep output on the ERA5 grid).
    if float(gcm.lon.max()) > 180.0 and float(obs.lon.min()) < 0.0:
        gcm = gcm.assign_coords(lon=gcm.lon.values - 360.0).sortby("lon")
        if gcm_cal is not None and gcm_cal is not gcm:
            gcm_cal = gcm_cal.assign_coords(lon=gcm_cal.lon.values - 360.0).sortby("lon")

    # Step 1: Coarsen ERA5 (calib) to GCM grid — area average (Wood 2002) for
    # BCSD; the EQM variant keeps the original point interpolation.
    if detrend:
        obs_coarse = _box_average(obs_cal, gcm.lat.values, gcm.lon.values)
    else:
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
        gcm_e = gcm_eval_mon.values[e_mask]  # [N_eval_yrs, lat, lon]
        bc_e  = gcm_e.copy()
        # Wood et al. (2004): remove the GCM tas change before mapping, so the map
        # sees calibration-like values, then re-add it (shift per coarse cell).
        if detrend and variable == "tas":
            shift = gcm_e.mean(axis=0) - gcm_c.mean(axis=0)
        else:
            shift = np.zeros(gcm_e.shape[1:], dtype=gcm_e.dtype)
        nlat, nlon = gcm_c.shape[1], gcm_c.shape[2]
        for i in range(nlat):
            for j in range(nlon):
                bc_e[:, i, j] = _qmap_1d(gcm_c[:, i, j], obs_c[:, i, j],
                                         gcm_e[:, i, j] - shift[i, j]) + shift[i, j]
        bc_mon_np[e_mask] = bc_e

    # Step 5: Daily disaggregation — vectorized index lookup
    ym_daily = gcm.time.dt.year.values * 100 + gcm.time.dt.month.values
    ym_mon   = gcm_eval_mon.time.dt.year.values * 100 + gcm_eval_mon.time.dt.month.values
    ym_to_i  = {ym: i for i, ym in enumerate(ym_mon)}
    day_i    = np.array([ym_to_i.get(ym, -1) for ym in ym_daily])

    gcm_np     = gcm.values                        # [T, lat_gcm, lon_gcm]
    bc_daily   = gcm_np.copy()
    valid      = day_i >= 0
    gcm_mon_d  = gcm_eval_mon.values[day_i[valid]] # [T_valid, lat, lon]
    bc_mon_d   = bc_mon_np[day_i[valid]]            # [T_valid, lat, lon]

    if variable == "pr":
        denom = np.where(np.abs(gcm_mon_d) > 1e-8, gcm_mon_d, 1e-8)
        scale = np.clip(bc_mon_d / denom, 0.0, 5.0)
        bc_daily[valid] = gcm_np[valid] * scale
        bc_daily = np.clip(bc_daily, 0.0, None)
    else:
        bc_daily[valid] = gcm_np[valid] + (bc_mon_d - gcm_mon_d)

    bc = gcm.copy(data=bc_daily)

    # Step 6: Bicubic interpolation to ERA5 grid
    bc_fine = bc.interp(lat=obs.lat, method="cubic",
                        kwargs={"fill_value": "extrapolate"})
    bc_fine = bc_fine.interp(lon=obs.lon, method="cubic",
                              kwargs={"fill_value": "extrapolate"})
    # Step 7: Fine-scale ERA5 spatial detail from ERA5 calib climatology
    obs_clim = obs_cal.groupby("time.month").mean("time")           # [12, lat_fine, lon_fine]
    obs_clim_coarse = obs_coarse.groupby("time.month").mean("time") # [12, lat_gcm, lon_gcm]
    obs_clim_coarse_fine = obs_clim_coarse.interp(
        lat=obs.lat, method="cubic", kwargs={"fill_value": "extrapolate"}
    ).interp(lon=obs.lon, method="cubic", kwargs={"fill_value": "extrapolate"})

    month_idx = bc_fine.time.dt.month.values - 1
    pred_np = bc_fine.values.copy()

    if variable == "pr":
        denom = np.where(obs_clim_coarse_fine.values > 1e-6,
                         obs_clim_coarse_fine.values, 1e-6)
        ratio = np.clip(obs_clim.values / denom, 0.0, 5.0)
        pred_np *= ratio[month_idx]
        pred_np = np.clip(pred_np, 0.0, None)
    else:
        detail = obs_clim.values - obs_clim_coarse_fine.values
        pred_np += detail[month_idx]

    if variable == "rsds":
        pred_np = np.clip(pred_np, 0.0, None)

    return xr.DataArray(pred_np, coords=bc_fine.coords, dims=bc_fine.dims)


def run(
    era5_dir: Path,
    gcm_dir:  Path,
    out_dir:  Path,
    variables: list[str],
    calib_start: int,
    calib_end:   int,
    eval_start:  int,
    eval_end:    int,
    scenario: str = "ssp370",
    win_size: int = 20, step: int = 10, trim: int = 5,
    detrend: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calib: {calib_start}–{calib_end}  |  Eval: {eval_start}–{eval_end}"
          f"  |  Window: {win_size}yr step {step}yr trim {trim}yr", flush=True)

    era5_cal, gcm_cal, gcm_eval = {}, {}, {}
    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        era5_cal[var] = load_era5(era5_dir, var, calib_start, calib_end)
        gcm_cal[var]  = load_gcm(gcm_dir, var, calib_start, calib_end, "historical")
        # From eval_start - trim, so the first window (e.g. 1995-2014) is full.
        gcm_eval[var] = load_gcm(gcm_dir, var, eval_start - trim, eval_end, scenario)

    all_cal = {}
    for var in variables:
        all_cal[f"era5_{var}"] = era5_cal[var]
        all_cal[f"gcm_{var}"]  = gcm_cal[var]
    aligned_cal = align_time(all_cal)
    era5_cal = {v: aligned_cal[f"era5_{v}"] for v in variables}
    gcm_cal  = {v: aligned_cal[f"gcm_{v}"]  for v in variables}

    windows = _sliding_windows(eval_start, eval_end, win_size, step, trim)
    ref_da  = None
    arrays  = []
    for var in variables:
        print(f"  BCSD {var} ({len(windows)} windows, detrend={detrend}) ...", flush=True)
        gcm_c = gcm_cal[var]
        obs_c = era5_cal[var]

        pieces = []
        for win_start, win_end, keep_start, keep_end in windows:
            yr = gcm_eval[var].time.dt.year.values
            gcm_win  = gcm_eval[var].isel(time=(yr >= win_start) & (yr <= win_end))
            if gcm_win.sizes["time"] == 0:
                continue
            pred_win = bcsd_downscale(gcm_win, obs_c, var, gcm_calib=gcm_c, obs_calib=obs_c,
                                      detrend=detrend)
            yr_p     = pred_win.time.dt.year.values
            pieces.append(pred_win.isel(time=(yr_p >= keep_start) & (yr_p <= keep_end)))

        pred = xr.concat(pieces, dim="time").sortby("time")
        if ref_da is None:
            ref_da = pred
        arrays.append(torch.tensor(pred.values, dtype=torch.float32))

    tensor = torch.stack(arrays, dim=1)
    save_predictions(tensor, variables, out_dir, ref_da)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",    required=True)
    p.add_argument("--gcm-dir",     required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--variables",   nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--calib-start", type=int, default=1980)
    p.add_argument("--calib-end",   type=int, default=1999)
    p.add_argument("--eval-start",  type=int, default=2000)
    p.add_argument("--eval-end",    type=int, default=2100)
    p.add_argument("--scenario",    default="ssp370")
    p.add_argument("--win-size",    type=int, default=20)
    p.add_argument("--step",        type=int, default=10)
    p.add_argument("--trim",        type=int, default=5)
    p.add_argument("--no-detrend",  action="store_true",
                   help="EQM variant: quantile-map tas without removing the GCM trend")
    args = p.parse_args()
    run(
        Path(args.era5_dir), Path(args.gcm_dir), Path(args.out_dir),
        args.variables,
        args.calib_start, args.calib_end,
        args.eval_start,  args.eval_end,
        args.scenario,
        args.win_size, args.step, args.trim,
        not args.no_detrend,
    )


if __name__ == "__main__":
    main()
