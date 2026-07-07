#!/usr/bin/env python3
"""Prepare ERA5 (training) and GCM (inference) tensors for baseline and evaluation.
Should take some arguments into account.

Outputs (all saved to --out-dir):
  ERA5_all5_train.pt   [T_train, 5, H, W]   float32, physical units
  GCM_all5_eval.pt     [T_eval,  5, H, W]   float32, physical units (GCM on ERA5 grid)
  sf_norm_stats.json   ERA5 train channel-wise mean/std (for denormalisation) # only if serpentflow
  sf_eval_dates.npy    ISO date strings for the eval time axis

Variable order: tas, pr, sfcWind, sfcWindmax, psl   (same for all baselines)

Run on Cleps (CPU node):
  python scripts/data_processing.py \
      --era5-dir /scratch/jkeisler/era5 \
      --ipsl-dir /scratch/jkeisler/cmip6/ipsl_cm6a_lr \
      --out-dir  /scratch/jkeisler/serpenflow/data \
      --train-start 1979 --train-end 2003 \
      --eval-start  2004 --eval-end   2014
"""

### TODO: add spectral interpolation option (for serpentflow method, keep bicubic for consistency model, quantile matching etc.)
### TODO simplify + reorganize data to have a single data processing pipeline for all methods (currently some duplication, mismatch between methods)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]

ERA5_UNITS: dict = {
    "pr": None,   # handled inline — detect m vs m/day
}
CMIP6_UNITS: dict = {
    "pr": lambda da: da * 86400,   # kg m⁻² s⁻¹ → mm/day
}


def _era5_pr_convert(da: xr.DataArray) -> xr.DataArray:
    ref = float(np.nanmean(da.isel(time=0).values))
    if ref > 0.05:
        return da * 1000 / 365.25   # annual total in m → mm/day
    else:
        return da * 1000            # daily value in m → mm/day


def load_era5(era5_dir: Path, variable: str, start: int, end: int) -> xr.DataArray:
    files = sorted(era5_dir.glob(f"{variable}_*.nc"))
    files = [f for f in files if start <= int(f.stem.split("_")[-1]) <= end]
    if not files:
        raise FileNotFoundError(f"No ERA5 files for {variable} {start}-{end} in {era5_dir}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        if "latitude" in ds.dims:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        parts.append(ds[variable])
    da = xr.concat(parts, dim="time")
    if variable == "pr":
        da = _era5_pr_convert(da)
    return da.load()


def load_ipsl(ipsl_dir: Path, variable: str, start: int, end: int) -> xr.DataArray:
    hist = ipsl_dir / f"{variable}_historical_1960-2014.nc"
    ssp  = ipsl_dir / f"{variable}_ssp585_2015-2100.nc"
    parts = []
    if hist.exists() and start <= 2014:
        parts.append(str(hist))
    if ssp.exists() and end >= 2015:
        parts.append(str(ssp))
    if not parts:
        raise FileNotFoundError(f"No IPSL files for {variable} in {ipsl_dir}")
    das = []
    for f in parts:
        ds = xr.open_dataset(f, use_cftime=True)
        if "latitude" in ds.dims:
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
        das.append(ds[variable])
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    da = da.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    if variable in CMIP6_UNITS:
        da = CMIP6_UNITS[variable](da)
    return da.load()


def bicubic_regrid(gcm: xr.DataArray, target_lat, target_lon) -> xr.DataArray:
    out = gcm.interp(lat=target_lat, method="cubic", kwargs={"fill_value": "extrapolate"})
    return out.interp(lon=target_lon, method="cubic", kwargs={"fill_value": "extrapolate"})


def align_time(das: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Return subset of each DataArray to the common date intersection."""
    date_sets = [set(da.time.dt.strftime("%Y-%m-%d").values) for da in das.values()]
    common = sorted(set.intersection(*date_sets))
    print(f"  Common timesteps: {len(common)}")
    result = {}
    for var, da in das.items():
        dates = da.time.dt.strftime("%Y-%m-%d").values
        idx = [i for i, d in enumerate(dates) if d in set(common)]
        result[var] = da.isel(time=idx)
    return result


def stack_to_tensor(das: dict[str, xr.DataArray], variables: list[str]) -> torch.Tensor:
    """Stack dict of [T,H,W] DataArrays → [T, C, H, W] float32 tensor."""
    arrays = [torch.tensor(das[v].values, dtype=torch.float32) for v in variables]
    return torch.stack(arrays, dim=1)   # [T, C, H, W]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--ipsl-dir", required=True)
    parser.add_argument("--out-dir",  required=True)
    parser.add_argument("--train-start", type=int, default=1979)
    parser.add_argument("--train-end",   type=int, default=2003)
    parser.add_argument("--eval-start",  type=int, default=2004)
    parser.add_argument("--eval-end",    type=int, default=2014)
    args = parser.parse_args()

    era5_dir = Path(args.era5_dir)
    ipsl_dir = Path(args.ipsl_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── ERA5 training tensor ───────────────────────────────────────────────
    print(f"\n=== ERA5 training {args.train_start}–{args.train_end} ===")
    era5_train = {}
    for var in VARIABLES:
        print(f"  Loading {var} ...", flush=True)
        era5_train[var] = load_era5(era5_dir, var, args.train_start, args.train_end)
        print(f"    shape={era5_train[var].shape}", flush=True)

    era5_train = align_time(era5_train)
    era5_tensor = stack_to_tensor(era5_train, VARIABLES)   # [T, 5, H, W]
    print(f"ERA5 training tensor: {era5_tensor.shape}")

    # Compute & save normalization stats (per channel, over all T*H*W)
    norm_stats = {}
    for c, var in enumerate(VARIABLES):
        ch = era5_tensor[:, c]
        mask = torch.isfinite(ch)
        m = ch[mask].mean().item()
        s = ch[mask].std().item()
        norm_stats[var] = {"mean": m, "std": s}
        print(f"  {var}: mean={m:.4g}, std={s:.4g}")

    train_path = out_dir / "ERA5_all5_train.pt"
    torch.save(era5_tensor, train_path)
    print(f"Saved {train_path}")

    stats_path = out_dir / "sf_norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"Saved {stats_path}")

    # ── IPSL evaluation tensor ─────────────────────────────────────────────
    print(f"\n=== IPSL eval {args.eval_start}–{args.eval_end} ===")
    era5_eval = {}
    ipsl_eval = {}
    for var in VARIABLES:
        print(f"  Loading {var} ...", flush=True)
        era5_eval[var] = load_era5(era5_dir, var, args.eval_start, args.eval_end)
        ipsl_eval[var] = load_ipsl(ipsl_dir, var, args.eval_start, args.eval_end)

    # Align ERA5 and IPSL to common dates
    all_das = {f"era5_{v}": era5_eval[v] for v in VARIABLES}
    all_das.update({f"ipsl_{v}": ipsl_eval[v] for v in VARIABLES})
    aligned = align_time(all_das)
    era5_eval = {v: aligned[f"era5_{v}"] for v in VARIABLES}
    ipsl_eval = {v: aligned[f"ipsl_{v}"] for v in VARIABLES}

    # Reference ERA5 lat/lon (from first variable)
    ref_era5 = next(iter(era5_eval.values()))
    target_lat = ref_era5.lat.values
    target_lon = ref_era5.lon.values

    # Bicubic interpolation of IPSL to ERA5 grid
    #TODO use version in args.interpolation == "spectral" from nc_to_tensors.py from serpentflow
    ipsl_on_era5 = {}
    for var in VARIABLES:
        print(f"  Regridding IPSL {var} ...", flush=True)
        da = bicubic_regrid(ipsl_eval[var], target_lat, target_lon)
        da = da.assign_coords(time=ref_era5.time)
        if var == "pr":
            da.values[:] = np.clip(da.values, 0.0, None)
        ipsl_on_era5[var] = da

    ipsl_tensor = stack_to_tensor(ipsl_on_era5, VARIABLES)   # [T, 5, H, W]
    print(f"IPSL eval tensor: {ipsl_tensor.shape}")

    eval_path = out_dir / "IPSL_all5_eval.pt"
    torch.save(ipsl_tensor, eval_path)
    print(f"Saved {eval_path}")

    # Save eval date strings (ISO, from ERA5 side)
    dates = ref_era5.time.dt.strftime("%Y-%m-%d").values
    dates_path = out_dir / "sf_eval_dates.npy"
    np.save(dates_path, dates)
    print(f"Saved {dates_path}  ({len(dates)} dates)")

    # Also save lat/lon for xarray reconstruction
    np.save(out_dir / "sf_lat.npy", target_lat)
    np.save(out_dir / "sf_lon.npy", target_lon)
    print("Done.")


if __name__ == "__main__":
    main()
