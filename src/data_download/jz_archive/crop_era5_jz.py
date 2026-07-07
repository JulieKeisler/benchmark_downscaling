#!/usr/bin/env python3
"""Crop the JZ ERA5 WB2 archive to the France domain and write daily NetCDF.

Source layout (0.25°, WB2):
    /lustre/fsstor/projects/rech/idl/uzm26qx/era5/1440x721/weatherbench2/4xyearly/
        {year}_{00|06|12|18}h.h5    # one file per year per UTC hour
        Each file: 365 daily snapshots at that fixed UTC hour, global 1440x721

Strategy:
    - Open the 4 hourly files per year, concatenate along time
    - Crop to France [55,-6,41,10] before computing (no data loaded yet)
    - Daily mean:  tas, psl
    - Daily sum:   pr  (total_precipitation_6hr summed over 4 snapshots/day)
    - Daily mean:  sfcWind   = mean of sqrt(u10^2 + v10^2) over 4 snapshots
    - Daily max:   sfcWindmax = max  of sqrt(u10^2 + v10^2) over 4 snapshots
    rsds: NOT available in this dataset (no radiation variable).

Usage on Jean-Zay:
    module load python/3.10.4
    python scripts/crop_era5_jz.py --output-dir $ALL_CCFRWORK/era5_france
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import re

import hdf5plugin  # must be imported before any HDF5 file is opened  # noqa: F401
import numpy as np
import xarray as xr

SOURCE_DIR = Path(
    "/lustre/fsstor/projects/rech/idl/uzm26qx/era5/1440x721/weatherbench2/4xyearly"
)
HOURS = ["00", "06", "12", "18"]

# France bounding box [N, W, S, E]
AREA = [55.0, -6.0, 41.0, 10.0]

YEARS_AVAILABLE = list(range(1979, 2023))  # 1979–2022 on disk


def _france_slices(path: Path) -> tuple:
    """Return contiguous h5py slices for the France crop.

    Uses slice() objects (not fancy arrays) so h5py decompresses only the
    requested Blosc2 chunks — avoids loading the full 721×1440 grid.
    Returns (lat_sl, lon_sl_west, lon_sl_east, lat_vals, lon_vals).
    """
    import h5py
    north, west, south, east = AREA
    with h5py.File(path, "r") as f:
        lat = f["latitude"][:]   # decreasing 90→-90, step -0.25
        lon = f["longitude"][:]  # increasing 0→359.75, step 0.25

    # Latitude: north=55 → index=(90-55)/0.25=140, south=41 → 196
    i0 = int(np.searchsorted(-lat, -north, side="left"))   # first index ≤ north
    i1 = int(np.searchsorted(-lat, -south, side="right"))  # include south boundary
    lat_sl = slice(i0, i1)
    lat_vals = lat[lat_sl]

    # Longitude west piece: 354.0 to 359.75
    lon_w360 = 360.0 + west                   # 354.0
    j0w = int(np.searchsorted(lon, lon_w360))
    j1w = int(np.searchsorted(lon, 359.751))  # include 359.75
    lon_sl_west = slice(j0w, j1w)

    # Longitude east piece: 0.0 to 10.0
    j0e = 0
    j1e = int(np.searchsorted(lon, east + 0.001))  # include 10.0
    lon_sl_east = slice(j0e, j1e)

    lon_vals = np.concatenate([
        (lon[lon_sl_west] + 180) % 360 - 180,
        (lon[lon_sl_east] + 180) % 360 - 180,
    ])
    return lat_sl, lon_sl_west, lon_sl_east, lat_vals, lon_vals


def _h5_to_dataset(path: Path, slices: tuple) -> xr.Dataset:
    """Read one HDF5 file into xarray, loading only the France region.

    netCDF4 engine cannot decompress Blosc2; h5py + hdf5plugin can.
    Contiguous slices let h5py skip unneeded compressed chunks.
    """
    import h5py
    import pandas as pd

    SURFACE_VARS = [
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "mean_sea_level_pressure", "surface_pressure",
        "total_precipitation_6hr",
    ]
    lat_sl, lon_sl_w, lon_sl_e, lat_vals, lon_vals = slices

    with h5py.File(path, "r") as f:
        raw_time = f["time"][:]
        # Decode time using the units attribute (e.g. "hours since 1959-01-01").
        # Without this, pd.to_datetime treats integers as nanoseconds, collapsing
        # all 365 daily timestamps to a single date near 1970-01-01.
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
            arr = np.concatenate([
                f[name][:, lat_sl, lon_sl_w],
                f[name][:, lat_sl, lon_sl_e],
            ], axis=2)
            data_vars[name] = xr.DataArray(
                arr,
                dims=["time", "latitude", "longitude"],
                coords={"time": times, "latitude": lat_vals,
                        "longitude": lon_vals},
            )
    return xr.Dataset(data_vars)


def open_year(year: int) -> xr.Dataset:
    """Open all 4 hourly files for one year, cropping to France while reading."""
    paths = [SOURCE_DIR / f"{year}_{h}h.h5" for h in HOURS]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {[str(p) for p in missing]}")
    slices = _france_slices(paths[0])
    datasets = [_h5_to_dataset(p, slices) for p in paths]
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


def process_year(year: int, output_dir: Path, variables: list[str]) -> str | None:
    """Process one year. Returns an error string or None on success."""
    print(f"[{year}] opening and cropping...")
    try:
        ds_fr = open_year(year)
        print(f"  grid: {dict(ds_fr.sizes)}")

        print(f"[{year}] computing daily statistics...")
        daily = compute_daily(ds_fr)

        for short_name, da in daily.data_vars.items():
            if short_name not in variables:
                continue
            path = output_dir / f"{short_name}_{year}.nc"
            da.to_netcdf(path)
            print(f"  -> {path}")

        ds_fr.close()
        del ds_fr
    except Exception as exc:
        return f"{year}: {exc}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--output-dir", required=True,
        help="Where to write per-variable per-year NetCDF "
             "(e.g. $ALL_CCFRWORK/era5_france)"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help=f"Years to process (default: all {YEARS_AVAILABLE[0]}–{YEARS_AVAILABLE[-1]})"
    )
    parser.add_argument(
        "--variables", nargs="+", default=None,
        help="Variables to write (default: all). "
             "Available: tas psl ps pr sfcWind sfcWindmax"
    )
    args = parser.parse_args(argv)

    all_vars = ["tas", "psl", "ps", "pr", "sfcWind", "sfcWindmax"]
    variables = args.variables or all_vars
    years = args.years or YEARS_AVAILABLE

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ERA5 JZ crop  (0.25°, WB2 4xyearly)")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Area   : {AREA}  (N W S E — France)")
    print(f"  Years  : {years[0]}–{years[-1]}  ({len(years)} years)")
    print(f"  Vars   : {variables}")
    print(f"  Output : {output_dir}")
    print(f"  NOTE   : rsds not available in this dataset")
    print()

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
        err = process_year(year, output_dir, missing_vars)
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
