#!/usr/bin/env python3
"""BCSD baseline inference — Bias-Corrected Spatial Disaggregation (Wood et al. 2002).

Monthly-mean QM at GCM resolution, then daily temporal disaggregation and spatial
interpolation to the ERA5 grid. A separate calibration period is used to fit QM.

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
        gcm_e = gcm_eval_mon.values[e_mask]  # [N_eval_yrs, lat, lon]
        bc_e  = gcm_e.copy()
        nlat, nlon = gcm_c.shape[1], gcm_c.shape[2]
        for i in range(nlat):
            for j in range(nlon):
                bc_e[:, i, j] = _qmap_1d(gcm_c[:, i, j], obs_c[:, i, j], gcm_e[:, i, j])
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
        scale = np.clip(bc_mon_d / denom, 0.0, 10.0)
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
    if bc_fine.sizes["time"] == obs.sizes["time"]:
        bc_fine = bc_fine.assign_coords(time=obs.time)

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
        ratio = np.clip(obs_clim.values / denom, 0.1, 10.0)
        pred_np *= ratio[month_idx]
        pred_np = np.clip(pred_np, 0.0, None)
    else:
        detail = obs_clim.values - obs_clim_coarse_fine.values
        pred_np += detail[month_idx]

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
    scenario: str = "ssp585",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    separate = (calib_start, calib_end) != (eval_start, eval_end)
    print(f"Calib: {calib_start}–{calib_end}  |  Eval: {eval_start}–{eval_end}", flush=True)

    era5_cal, gcm_cal, gcm_eval = {}, {}, {}
    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        if separate:
            era5_cal[var] = load_era5(era5_dir, var, calib_start, calib_end)
            gcm_cal[var]  = load_gcm(gcm_dir, var, calib_start, calib_end, "historical")
        gcm_eval[var] = load_gcm(gcm_dir, var, eval_start, eval_end, scenario)

    if separate:
        all_cal = {}
        for var in variables:
            all_cal[f"era5_{var}"] = era5_cal[var]
            all_cal[f"gcm_{var}"]  = gcm_cal[var]
        aligned_cal = align_time(all_cal)
        era5_cal = {v: aligned_cal[f"era5_{v}"] for v in variables}
        gcm_cal  = {v: aligned_cal[f"gcm_{v}"]  for v in variables}

    ref_da = None
    arrays = []
    for var in variables:
        print(f"  BCSD {var} ...", flush=True)
        gcm_c = gcm_cal.get(var)
        obs_c = era5_cal.get(var)
        # obs (for spatial disaggregation) = era5 calib; gcm eval extends to eval_end
        obs   = obs_c if obs_c is not None else load_era5(era5_dir, var, eval_start, eval_end)
        pred = bcsd_downscale(gcm_eval[var], obs, var, gcm_calib=gcm_c, obs_calib=obs_c)
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
    p.add_argument("--scenario",    default="ssp585")
    args = p.parse_args()
    run(
        Path(args.era5_dir), Path(args.gcm_dir), Path(args.out_dir),
        args.variables,
        args.calib_start, args.calib_end,
        args.eval_start,  args.eval_end,
        args.scenario,
    )


if __name__ == "__main__":
    main()
