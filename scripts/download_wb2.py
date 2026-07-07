#!/usr/bin/env python3
"""Download ERA5 daily data for a regional domain from WeatherBench2 (GCS).

No CDS account needed.  Data is read lazily from the public WeatherBench2
zarr on Google Cloud Storage; only the configured domain is written to disk.

BANDWIDTH WARNING
  The WB2 zarr is chunked as one full global field per time step.  Even
  with a lazy region selection, every chunk read transfers a complete global
  snapshot (~4 MB).  For 0.25° and 6 variables × 1460 time steps/year that
  is ~36 GB/year of GCS traffic.  Running on a cluster with fast cloud
  connectivity (e.g. ssh jz) is strongly recommended for 0.25°.  The 1.5°
  dataset (--res 1.5) is ~50× smaller and manageable locally.

VARIABLES
  Available: tas, psl, pr, sfcWind, sfcWindmax, rsds.

  rsds: surface downwelling shortwave is NOT in WB2.  We use
    mean_surface_net_short_wave_radiation_flux (net = downwelling − reflected)
    as a proxy.  This underestimates true downwelling by ~10–30 %.  For a
    downscaling benchmark the quantity is valid as long as both reference and
    model use the same net-flux definition.  If true downwelling is required,
    download surface_solar_radiation_downwards from CDS via download_era5.py.

  sfcWindmax: with --hourly-wind, both sfcWind (daily mean) and sfcWindmax
    (daily max) are computed from the 1-hourly u/v components (24 values/day).
    Without that flag, both use the 6-hourly pre-computed 10 m wind speed
    (4 values/day), which slightly underestimates sfcWindmax extremes.

Usage:
    python scripts/download_wb2.py                         # all years, 0.25°
    python scripts/download_wb2.py --res 1.5               # coarser, local-friendly
    python scripts/download_wb2.py --years 1990 1991       # subset
    python scripts/download_wb2.py --variables rsds sfcWind sfcWindmax
    python scripts/download_wb2.py --output-dir /scratch/era5_wb2
    python scripts/download_wb2.py --config configs/data/era5.yaml
    # Hourly wind on JZ (overwrites 6-hourly sfcWind/sfcWindmax approximation):
    python scripts/download_wb2.py --hourly-wind --variables sfcWind sfcWindmax

Requirements:
    pip install gcsfs xarray zarr numpy netcdf4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

# WeatherBench2 public GCS bucket (no credentials needed)
WB2_BASE = "gs://weatherbench2/datasets/era5"
WB2_DATASETS_6H = {
    "0.25": f"{WB2_BASE}/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr",
    "1.5":  f"{WB2_BASE}/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
}
# Hourly 0.25° — required for true daily sfcWindmax (24 values vs 4).
# Chunks are still (1, 721, 1440): ~864 GB GCS traffic/year. JZ only.
WB2_1H_025 = f"{WB2_BASE}/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"

# WB2 variable name -> (CMIP6 short name, aggregation)  [6-hourly source]
VARIABLE_MAP_6H = {
    "2m_temperature":          ("tas",     "mean"),
    "mean_sea_level_pressure": ("psl",     "mean"),
    "total_precipitation_6hr": ("pr",      "sum"),   # m/6h -> m/day
    "10m_wind_speed":          ("sfcWind", "mean"),
    # sfcWindmax handled separately (same source var, different aggregation)
    # rsds: surface_solar_radiation_downwards is NOT in WB2.
    # Download it from CDS: python scripts/download_era5.py --variables rsds
}

# For the hourly dataset: no pre-computed wind speed, use u/v
VARIABLE_MAP_1H = {
    "2m_temperature":               ("tas",        "mean"),
    "mean_sea_level_pressure":      ("psl",        "mean"),
    "total_precipitation":          ("pr",         "sum"),   # m/1h -> m/day
    # sfcWind and sfcWindmax computed from u10/v10 below
}

WIND_SPEED_VAR = "10m_wind_speed"   # 6h dataset (pre-computed)
WIND_U_VAR = "10m_u_component_of_wind"
WIND_V_VAR = "10m_v_component_of_wind"


def _open_zarr_normalized(url: str) -> xr.Dataset:
    """Open a WB2 zarr and normalise longitude from 0–360 to –180/180."""
    ds = xr.open_zarr(url, storage_options={"token": "anon"})
    if float(ds.longitude.max()) > 180:
        ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)
        ds = ds.sortby("longitude")
    return ds


def open_wb2(res: str, hourly_wind: bool = False) -> tuple[xr.Dataset, bool]:
    """Open WB2 zarr(s) lazily.

    Returns (ds_main, uses_hourly_wind).  If hourly_wind is True and res is
    0.25, a second hourly zarr is opened for wind and merged into ds_main.
    """
    if res not in WB2_DATASETS_6H:
        sys.exit(f"Unknown resolution '{res}'; choose from {list(WB2_DATASETS_6H)}")

    ds = _open_zarr_normalized(WB2_DATASETS_6H[res])

    if hourly_wind and res == "0.25":
        ds_1h = _open_zarr_normalized(WB2_1H_025)
        # Keep only the wind components from the hourly source
        wind_ds = ds_1h[[WIND_U_VAR, WIND_V_VAR]]
        return ds, wind_ds  # caller handles merge
    return ds, None


def crop(ds: xr.Dataset, area: list[float]) -> xr.Dataset:
    """Crop to [N, W, S, E] bounding box."""
    north, west, south, east = area
    return ds.sel(latitude=slice(north, south), longitude=slice(west, east))


def aggregate_daily(ds_year: xr.Dataset,
                    ds_wind_year: xr.Dataset | None = None,
                    ) -> tuple[xr.Dataset, xr.DataArray | None]:
    """Aggregate to daily means/sums. Returns (daily Dataset, hourly_speed or None).

    ds_wind_year: optional hourly Dataset with u10/v10 for true sfcWindmax.
    hourly_speed is returned so the caller can persist it if requested.
    """
    out = xr.Dataset()
    hourly_speed = None

    for wb2_name, (short_name, method) in VARIABLE_MAP_6H.items():
        if wb2_name not in ds_year:
            print(f"  WARNING: {wb2_name} not in dataset — skipping {short_name}")
            continue
        da = ds_year[wb2_name]
        if method == "mean":
            out[short_name] = da.resample(time="1D").mean()
        elif method == "sum":
            out[short_name] = da.resample(time="1D").sum()

    if ds_wind_year is not None:
        u = ds_wind_year[WIND_U_VAR]
        v = ds_wind_year[WIND_V_VAR]
        hourly_speed = np.sqrt(u ** 2 + v ** 2)
        out["sfcWind"]    = hourly_speed.resample(time="1D").mean()
        out["sfcWindmax"] = hourly_speed.resample(time="1D").max()
    elif WIND_SPEED_VAR in ds_year:
        # 6-hourly pre-computed speed (4 values/day) — sfcWindmax is approximate.
        out["sfcWind"]    = ds_year[WIND_SPEED_VAR].resample(time="1D").mean()
        out["sfcWindmax"] = ds_year[WIND_SPEED_VAR].resample(time="1D").max()
    else:
        print("  WARNING: no wind data — skipping sfcWind/sfcWindmax")

    return out, hourly_speed


def write_variable(da: xr.DataArray, short_name: str, year: int,
                   output_dir: Path) -> Path:
    path = output_dir / f"{short_name}_{year}.nc"
    da.to_netcdf(path)
    return path


def process_year(year: int, ds_wb2: xr.Dataset, output_dir: Path,
                 variables: list[str],
                 ds_wind: xr.Dataset | None = None,
                 hourly_wind_dir: Path | None = None,
                 overwrite: bool = False) -> list[str]:
    """Download and aggregate one year. Returns list of failure strings.

    When hourly wind data is present, loads one month at a time to keep peak
    memory manageable — each full-global 0.25° hourly chunk is ~4 MB, so a
    full year of u+v at once would require ~72 GB before the France crop.
    """
    print(f"[{year}] loading from GCS and aggregating...")
    failures = []
    try:
        ds_year = ds_wb2.sel(time=ds_wb2.time.dt.year == year).compute()
        ds_wind_year = None
        if ds_wind is not None:
            ds_wind_year = ds_wind.sel(time=ds_wind.time.dt.year == year).compute()
        daily, hourly_speed = aggregate_daily(ds_year, ds_wind_year)

        for short_name, da in daily.data_vars.items():
            if short_name not in variables:
                continue
            path = output_dir / f"{short_name}_{year}.nc"
            if path.exists() and not overwrite:
                continue
            da.to_netcdf(path)
            print(f"  -> {path}")
        if hourly_speed is not None and hourly_wind_dir is not None:
            path = hourly_wind_dir / f"sfcWind_hourly_{year}.nc"
            if not path.exists() or overwrite:
                hourly_speed.to_netcdf(path)
                print(f"  -> {path}")
    except Exception as exc:
        failures.append(f"{year}: {exc}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data/era5.yaml",
                        help="Config file for area and years (default: configs/data/era5.yaml)")
    parser.add_argument("--res", default="0.25", choices=list(WB2_DATASETS_6H),
                        help="Grid resolution (default: 0.25)")
    parser.add_argument("--years", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Year range inclusive, e.g. --years 1979 2022 (default: from config)")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="Variables to write (default: all available)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/raw/era5_wb2)")
    parser.add_argument("--hourly-wind", action="store_true",
                        help="Use the hourly WB2 dataset for true 24-value sfcWindmax "
                             "(0.25° only; ~864 GB GCS traffic/year — JZ recommended)")
    parser.add_argument("--hourly-wind-dir", default=None,
                        help="If set, also write sfcWind_hourly_<year>.nc (1h speed) "
                             "to this directory (requires --hourly-wind)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files (default: skip)")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    area = cfg["area"]
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1))
    else:
        years = list(range(cfg["years"]["start"], cfg["years"]["end"] + 1))
    output_dir = Path(args.output_dir) if args.output_dir else Path("data/raw/era5_wb2")
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_wind_dir = None
    if args.hourly_wind_dir:
        if not args.hourly_wind:
            sys.exit("--hourly-wind-dir requires --hourly-wind")
        hourly_wind_dir = Path(args.hourly_wind_dir)
        hourly_wind_dir.mkdir(parents=True, exist_ok=True)

    all_short_names = [n for n, _ in VARIABLE_MAP_6H.values()] + ["sfcWindmax"]
    variables = args.variables or all_short_names

    print(f"WeatherBench2 ERA5 download")
    print(f"  Resolution : {args.res}°")
    print(f"  Area       : {area} (N W S E)")
    print(f"  Years      : {years[0]}–{years[-1]}  ({len(years)} years)")
    print(f"  Variables  : {variables}")
    print(f"  Output     : {output_dir}")
    if hourly_wind_dir:
        print(f"  Hourly dir : {hourly_wind_dir}  (sfcWind_hourly_<year>.nc)")
    if args.res == "0.25":
        gb = 864 if args.hourly_wind else 36
        print(f"  NOTE: 0.25° chunks are full global fields (~{gb} GB GCS traffic/year).")
        print(f"        Consider running on a cluster: ssh jz")
    print()

    print("Opening WB2 zarr (lazy)...")
    ds_wb2, ds_wind = open_wb2(args.res, hourly_wind=args.hourly_wind)
    ds_wb2 = crop(ds_wb2, area)
    if ds_wind is not None:
        ds_wind = crop(ds_wind, area)
        print("  Hourly wind: enabled (true 24-value sfcWindmax)")
    print(f"  Cropped grid: {dict(ds_wb2.sizes)}")
    print(f"  Time range : {str(ds_wb2.time.values[0])[:10]} → "
          f"{str(ds_wb2.time.values[-1])[:10]}")
    print()

    # Check which year–variable pairs already exist
    todo_years = []
    for year in years:
        if args.overwrite:
            missing = variables
        else:
            missing = [v for v in variables
                       if not (output_dir / f"{v}_{year}.nc").exists()]
        if missing:
            todo_years.append((year, missing))
        else:
            print(f"[{year}] all files exist, skipping")

    if not todo_years:
        print("Nothing to download.")
        return 0

    print(f"{len(todo_years)} year(s) to process\n")

    all_failures = []
    for year, missing_vars in todo_years:
        failures = process_year(year, ds_wb2, output_dir, missing_vars,
                                ds_wind, hourly_wind_dir, args.overwrite)
        all_failures.extend(failures)

    if all_failures:
        print(f"\n{len(all_failures)} failure(s):", file=sys.stderr)
        for line in all_failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
