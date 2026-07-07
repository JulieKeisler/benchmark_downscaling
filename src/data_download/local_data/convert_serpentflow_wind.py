#!/usr/bin/env python3
"""Convert serpentflow hourly 10m wind speed (ff10) to daily ERA5 format files.

Reads  serpentflow/data/RAW/Reanalysis/ERA5/ERA5_ff10fc_FR.nc (hourly, 1979–2023)
and writes one annual NetCDF per year:
  sfcWind_<year>.nc    — daily mean 10m wind speed  (m/s)
  sfcWindmax_<year>.nc — daily maximum 10m wind speed (m/s)

Output is compatible with the benchmark's load_era5() function (lat/lon dims,
one variable per file named 'sfcWind' or 'sfcWindmax').

NOTE: The source file covers lat 41–52°N; the benchmark ERA5 data from JZ covers
41–55°N.  The three northernmost degree bands are absent here.  For a full-domain
evaluation use the Cleps ERA5 files; use this for local quick runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input",
        default="/home/jkeisler/Documents/Code/serpentflow/data/RAW/Reanalysis/ERA5/ERA5_ff10fc_FR.nc",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/jkeisler/Documents/Code/benchmark_downscaling/data/raw/era5",
    )
    parser.add_argument("--start", type=int, default=1979)
    parser.add_argument("--end",   type=int, default=2022)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.input, chunks={"time": 8760})
    da = ds["ff10"].rename({"latitude": "lat", "longitude": "lon"})

    for year in range(args.start, args.end + 1):
        mean_out = out_dir / f"sfcWind_{year}.nc"
        max_out  = out_dir / f"sfcWindmax_{year}.nc"
        if mean_out.exists() and max_out.exists():
            print(f"  {year} already done, skipping")
            continue

        print(f"  Processing {year} ...", flush=True)
        da_year = da.sel(time=str(year))
        if da_year.sizes["time"] == 0:
            print(f"    no data for {year}, skipping")
            continue

        daily_mean = da_year.resample(time="1D").mean()
        daily_max  = da_year.resample(time="1D").max()

        daily_mean = daily_mean.rename("sfcWind")
        daily_max  = daily_max.rename("sfcWindmax")

        enc = {"zlib": True, "complevel": 4}
        daily_mean.to_dataset(name="sfcWind").to_netcdf(
            mean_out, encoding={"sfcWind": enc}
        )
        daily_max.to_dataset(name="sfcWindmax").to_netcdf(
            max_out, encoding={"sfcWindmax": enc}
        )
        print(f"    saved {mean_out.name}  {max_out.name}")

    # skip ds.close() — netCDF4/dask segfaults on close of large chunked files
    print("Done.")


if __name__ == "__main__":
    main()
