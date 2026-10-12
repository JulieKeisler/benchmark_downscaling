#!/usr/bin/env python3
"""Crop the JZ ERA5 WB2 archive to any regional domain and write daily NetCDF.

The 0.25° WeatherBench2 ERA5 archive stored on Jean-Zay lustre is read
directly by this script (no GCS). Only the requested domain is decompressed
(contiguous h5py slices), so per-year CPU/IO cost is small.

Source layout (0.25°, WB2, 6-hourly):
    /lustre/fsstor/projects/rech/idl/uzm26qx/era5/1440x721/weatherbench2/4xyearly/
        {year}_{00|06|12|18}h.h5    # one file per year per UTC hour
        Each file: 365 daily snapshots at that fixed UTC hour, global 1440x721

Strategy:
    - Open the 4 hourly files per year, concatenate along time
    - Crop to the region before computing (only the region is read)
    - Daily mean:  tas, psl, ps
    - Daily sum:   pr  (total_precipitation_6hr summed over 4 snapshots/day)
    - Daily mean:  sfcWind    = mean of sqrt(u10^2 + v10^2) over 4 snapshots
    - Daily max:   sfcWindmax = max   of sqrt(u10^2 + v10^2) over 4 snapshots
    rsds: NOT available in this dataset (no radiation variable).

NOTE on sfcWindmax: with 6-hourly input the daily max uses 4 samples/day and
slightly underestimates true hourly maxima. This matches the existing France
pipeline. Use the hourly WB2 dataset (download_wb2.py --hourly-wind) only if
24-sample extremes are essential.

Usage on Jean-Zay:
    module load python/3.10.4  # or pytorch-gpu which ships h5py+hdf5plugin
    python src/data_download/wb2_archive/crop_jz_archive.py \
        --region australia --output-dir $WORK/era5_australia
    python src/data_download/wb2_archive/crop_jz_archive.py \
        --area -24 136 -38 152 --output-dir $WORK/era5_australia \
        --years 1979 2022 --variables tas psl pr sfcWind sfcWindmax

Regions are defined in src/configs/data/regions.yaml.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import hdf5plugin  # must be imported before any HDF5 file is opened  # noqa: F401

from data_download_regions import REGIONS  # noqa: F401  (regions.yaml loader)

SOURCE_DIR = Path(
    "/lustre/fsstor/projects/rech/idl/uzm26qx/era5/1440x721/weatherbench2/4xyearly"
)
HOURS = ["00", "06", "12", "18"]

YEARS_AVAILABLE = list(range(1979, 2023))  # 1979–2022 on disk

SURFACE_VARS = [
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "2m_temperature", "mean_sea_level_pressure", "surface_pressure",
    "total_precipitation_6hr",
]


def region_slices(area: list[float], path: Path) -> tuple:
    """Return contiguous h5py slices for the [N, W, S, E] crop.

    Uses slice() objects (not fancy arrays) so h5py decompresses only the
    requested Blosc2 chunks. Handles both meridian-crossing and fully east /
    fully west domains. Returns (lat_sl, lon_sl_list, lat_vals, lon_vals)
    where lon_sl_list is a list of 1 or 2 slices concatenated in order.
    """
    import h5py
    north, west, south, east = area
    with h5py.File(path, "r") as f:
        lat = f["latitude"][:]   # decreasing 90→-90, step -0.25
        lon = f["longitude"][:]  # increasing 0→359.75, step 0.25

    i0 = int(np.searchsorted(-lat, -north, side="left"))
    i1 = int(np.searchsorted(-lat, -south, side="right"))
    lat_sl = slice(i0, i1)
    lat_vals = lat[lat_sl]

    # Longitude in 0..360 representation of the requested box
    lo = west if west >= 0 else 360.0 + west
    hi = east if east >= 0 else 360.0 + east
    if lo < hi:
        j0 = int(np.searchsorted(lon, lo))
        j1 = int(np.searchsorted(lon, hi + 0.001))
        lon_sl_list = [slice(j0, j1)]
        lon_chunks = [lon[lon_sl_list[0]]]
    else:
        # Box crosses the 0°/360° meridian: west piece [lo, 360) + east [0, hi)
        jw0 = int(np.searchsorted(lon, lo))
        jw1 = int(np.searchsorted(lon, 359.751))
        je0 = 0
        je1 = int(np.searchsorted(lon, hi + 0.001))
        lon_sl_list = [slice(jw0, jw1), slice(je0, je1)]
        lon_chunks = [lon[s] for s in lon_sl_list]

    lon_vals = np.concatenate([(c + 180) % 360 - 180 for c in lon_chunks])
    return lat_sl, lon_sl_list, lat_vals, lon_vals


def h5_to_dataset(path: Path, slices: tuple) -> xr.Dataset:
    """Read one HDF5 file into xarray, loading only the region."""
    import h5py
    import pandas as pd

    lat_sl, lon_sl_list, lat_vals, lon_vals = slices
    with h5py.File(path, "r") as f:
        raw_time = f["time"][:]
        units_attr = f["time"].attrs.get("units", b"hours since 1959-01-01")
        if isinstance(units_attr, bytes):
            units_attr = units_attr.decode()
        m = re.match(r"(\w+) since (\S+)", units_attr)
        if m:
            _unit_map = {"hours": "h", "days": "D", "seconds": "s",
                         "minutes": "min"}
            unit_short = _unit_map.get(m.group(1), "h")
            origin = pd.Timestamp(m.group(2))
            times = pd.to_datetime(raw_time, unit=unit_short, origin=origin)
        else:
            times = pd.to_datetime(raw_time, unit="h",
                                   origin=pd.Timestamp("1959-01-01"))

        data_vars = {}
        for name in SURFACE_VARS:
            if name not in f:
                continue
            arr = np.concatenate(
                [f[name][:, lat_sl, s] for s in lon_sl_list], axis=2)
            data_vars[name] = xr.DataArray(
                arr,
                dims=["time", "latitude", "longitude"],
                coords={"time": times, "latitude": lat_vals,
                        "longitude": lon_vals},
            )
    return xr.Dataset(data_vars)


def open_year(year: int, slices: tuple) -> xr.Dataset:
    """Open all 4 hourly files for one year, cropping to the region."""
    paths = [SOURCE_DIR / f"{year}_{h}h.h5" for h in HOURS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {[str(p) for p in missing]}")
    datasets = [h5_to_dataset(p, slices) for p in paths]
    return xr.concat(datasets, dim="time").sortby("time")


def compute_daily(ds: xr.Dataset) -> xr.Dataset:
    """Aggregate 4 snapshots/day → daily variables with CMIP6 short names."""
    out = xr.Dataset()
    if "2m_temperature" in ds:
        out["tas"] = ds["2m_temperature"].resample(time="1D").mean()
    if "mean_sea_level_pressure" in ds:
        out["psl"] = ds["mean_sea_level_pressure"].resample(time="1D").mean()
    if "surface_pressure" in ds:
        out["ps"] = ds["surface_pressure"].resample(time="1D").mean()
    if "total_precipitation_6hr" in ds:
        out["pr"] = ds["total_precipitation_6hr"].resample(time="1D").sum()
    u = ds.get("10m_u_component_of_wind")
    v = ds.get("10m_v_component_of_wind")
    if u is not None and v is not None:
        ws = np.sqrt(u ** 2 + v ** 2)
        out["sfcWind"]    = ws.resample(time="1D").mean()
        out["sfcWindmax"] = ws.resample(time="1D").max()
    return out


def process_year(year: int, slices: tuple, output_dir: Path,
                 variables: list[str]) -> str | None:
    print(f"[{year}] opening and cropping...")
    try:
        ds_reg = open_year(year, slices)
        print(f"  grid: {dict(ds_reg.sizes)}")

        daily = compute_daily(ds_reg)
        for short_name, da in daily.data_vars.items():
            if short_name not in variables:
                continue
            path = output_dir / f"{short_name}_{year}.nc"
            da.to_netcdf(path)
            print(f"  -> {path}")
        ds_reg.close()
        del ds_reg
    except Exception as exc:
        return f"{year}: {exc}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir", required=True,
        help="Where to write per-variable per-year NetCDF")
    parser.add_argument(
        "--region", default=None,
        help="Region name from src/configs/data/regions.yaml "
             "(e.g. australia, us_central, france)")
    parser.add_argument(
        "--area", nargs=4, type=float, default=None,
        metavar=("N", "W", "S", "E"),
        help="Crop area [North, West, South, East]; overrides --region")
    parser.add_argument(
        "--years", nargs=2, type=int, default=None,
        help=f"Year range (default: {YEARS_AVAILABLE[0]}–{YEARS_AVAILABLE[-1]})")
    parser.add_argument(
        "--variables", nargs="+", default=None,
        help="Variables to write (default: all). "
             "Available: tas psl ps pr sfcWind sfcWindmax")
    args = parser.parse_args(argv)

    if args.area is None and args.region is None:
        sys.exit("Provide --region NAME or --area N W S E")
    if args.area is not None:
        area = list(args.area)
        label = f"{area}"
    else:
        if args.region not in REGIONS:
            sys.exit(f"Unknown region '{args.region}'; known: "
                     f"{sorted(REGIONS)}")
        area = list(REGIONS[args.region]["area"])
        label = f"{args.region} ({REGIONS[args.region]['label']})"

    all_vars = ["tas", "psl", "ps", "pr", "sfcWind", "sfcWindmax"]
    variables = args.variables or all_vars
    years = (list(range(args.years[0], args.years[1] + 1))
             if args.years else YEARS_AVAILABLE)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("ERA5 JZ crop  (0.25°, WB2 4xyearly)")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Area   : {label}  (N W S E)")
    print(f"  Years  : {years[0]}–{years[-1]}  ({len(years)} years)")
    print(f"  Vars   : {variables}")
    print(f"  Output : {output_dir}")
    print(f"  NOTE   : rsds not available in this dataset")
    print()

    # Precompute slices once (uses the first year's file for grid metadata)
    probe = SOURCE_DIR / f"{years[0]}_00h.h5"
    if not probe.exists():
        sys.exit(f"Missing source archive: {probe} (run this on Jean-Zay)")
    slices = region_slices(area, probe)
    print(f"  Grid : lat {len(slices[2])}  lon {len(slices[3])}\n")

    todo, skipped = [], 0
    for year in years:
        missing = [v for v in variables
                   if not (output_dir / f"{v}_{year}.nc").exists()]
        if missing:
            todo.append((year, missing))
        else:
            skipped += 1

    if skipped:
        print(f"{skipped} year(s) already complete, skipping")
    if not todo:
        print("Nothing to do.")
        return 0
    print(f"{len(todo)} year(s) to process\n")

    failures = []
    for year, missing_vars in todo:
        err = process_year(year, slices, output_dir, missing_vars)
        if err:
            failures.append(err)
            print(f"  FAILED: {err}")

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())