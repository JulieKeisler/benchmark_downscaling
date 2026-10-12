#!/usr/bin/env python3
"""Daily ERA5 tas/pr/psl/sfcWind/sfcWindmax from the public WeatherBench2 zarr on GCS.

Same product and the same daily aggregation as `crop_jz_archive.py` (the 6-hourly
0.25° WB2 archive mirrored on Jean-Zay, which only covers 1979-2022), so years
fetched here are bit-identical to the Jean-Zay-derived files where they overlap
(verified on 1979). Use it for years outside the JZ mirror — notably 1960-1978,
needed by SerpentFlow-CLIM's 20-year trailing climatology.

The zarr is chunked one global field per 6-hour step: each chunk is read once and
cropped for every requested region (same streaming design as
`arco/download_era5_arco.py`).

Daily aggregation (4 snapshots/day, UTC), copied from `crop_jz_archive.compute_daily`:
    tas, psl   = mean of the 00/06/12/18 snapshots
    pr         = sum of total_precipitation_6hr at 00/06/12/18 (m/day)
    sfcWind    = mean of sqrt(u10^2 + v10^2);  sfcWindmax = max of the same 4 values

Usage:
    python src/data_download/wb2_archive/download_wb2_gcs.py --years 1960 1978
    python src/data_download/wb2_archive/download_wb2_gcs.py --regions france --years 1960 1960

Requirements:
    pip install xarray zarr gcsfs netcdf4 pyyaml
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

WB2_STORE = ("gs://weatherbench2/datasets/era5/"
             "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
REPO_ROOT = Path(__file__).resolve().parents[3]
REGIONS_YAML = REPO_ROOT / "src" / "configs" / "data" / "regions.yaml"

# CMIP6 short name -> WB2 source variables it is built from
SOURCES = {
    "tas": ["2m_temperature"],
    "psl": ["mean_sea_level_pressure"],
    "pr": ["total_precipitation_6hr"],
    "sfcWind": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
    "sfcWindmax": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
}


def region_index(area: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer indices into the WB2 0.25° grid (lat 90→-90, lon 0→359.75) for [N, W, S, E]."""
    north, west, south, east = area
    lat = np.arange(north, south - 0.125, -0.25)
    lon = np.arange(west, east + 0.125, 0.25)
    ilat = np.rint((90.0 - lat) / 0.25).astype(int)
    ilon = np.rint((lon % 360.0) / 0.25).astype(int)
    return ilat, ilon, lat.astype(np.float32), lon.astype(np.float32)


def compute_daily(ds: xr.Dataset) -> xr.Dataset:
    """Verbatim copy of crop_jz_archive.compute_daily (importing it needs hdf5plugin)."""
    out = xr.Dataset()
    if "2m_temperature" in ds:
        out["tas"] = ds["2m_temperature"].resample(time="1D").mean()
    if "mean_sea_level_pressure" in ds:
        out["psl"] = ds["mean_sea_level_pressure"].resample(time="1D").mean()
    if "total_precipitation_6hr" in ds:
        out["pr"] = ds["total_precipitation_6hr"].resample(time="1D").sum()
    u = ds.get("10m_u_component_of_wind")
    v = ds.get("10m_v_component_of_wind")
    if u is not None and v is not None:
        ws = np.sqrt(u ** 2 + v ** 2)
        out["sfcWind"] = ws.resample(time="1D").mean()
        out["sfcWindmax"] = ws.resample(time="1D").max()
    return out


def process_year(ds: xr.Dataset, year: int, targets: dict, variables: list[str],
                 threads: int, overwrite: bool) -> None:
    todo = {name: t for name, t in targets.items()
            if overwrite or any(not (t["out_dir"] / f"{v}_{year}.nc").exists() for v in variables)}
    if not todo:
        print(f"  {year}: all files exist, skipping")
        return
    sources = sorted({s for v in variables for s in SOURCES[v]})
    times = ds.time.sel(time=slice(f"{year}-01-01", f"{year}-12-31T18:00")).values
    if len(times) != 4 * (366 if year % 4 == 0 else 365):
        raise RuntimeError(f"{year}: expected 4 steps/day, got {len(times)}")
    i0 = int(np.searchsorted(ds.time.values, times[0]))

    def read(args):
        src, k = args
        field = ds[src].isel(time=i0 + k).values  # (721, 1440) global
        return src, k, {n: field[np.ix_(t["index"][0], t["index"][1])] for n, t in todo.items()}

    buf = {n: {s: np.empty((len(times), len(t["index"][0]), len(t["index"][1])), np.float32)
               for s in sources} for n, t in todo.items()}
    t0 = time.time()
    jobs = [(s, k) for s in sources for k in range(len(times))]
    with ThreadPoolExecutor(threads) as pool:
        for src, k, crops in pool.map(read, jobs):
            for n, c in crops.items():
                buf[n][src][k] = c

    for name, t in todo.items():
        _, _, lat, lon = t["index"]
        coords = {"time": times, "latitude": lat, "longitude": lon}
        raw = xr.Dataset({s: (("time", "latitude", "longitude"), a) for s, a in buf[name].items()},
                         coords=coords)
        daily = compute_daily(raw)
        t["out_dir"].mkdir(parents=True, exist_ok=True)
        for v in variables:
            da = daily[v].astype(np.float32)
            if bool(da.isnull().any()):
                raise RuntimeError(f"{name} {v} {year}: NaNs in source")
            da.attrs = {"source": f"ERA5 via WeatherBench2 ({WB2_STORE})",
                        "history": "daily aggregation by src/data_download/wb2_archive/download_wb2_gcs.py "
                                   "(= crop_jz_archive.compute_daily)"}
            path = t["out_dir"] / f"{v}_{year}.nc"
            tmp = path.with_suffix(".nc.part")
            da.to_netcdf(tmp)
            tmp.rename(path)
    print(f"  {year}: {', '.join(todo)} written ({time.time() - t0:.0f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--regions", nargs="+", default=None,
                        help="region names from regions.yaml (default: all)")
    parser.add_argument("--years", nargs=2, type=int, default=[1960, 1978], metavar=("START", "END"))
    parser.add_argument("--variables", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--threads", type=int, default=32, help="concurrent chunk reads")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    regions = yaml.safe_load(REGIONS_YAML.read_text())["regions"]
    names = args.regions or list(regions)
    unknown = set(names) - set(regions)
    if unknown:
        parser.error(f"unknown regions {sorted(unknown)}; known: {sorted(regions)}")
    targets = {n: {"index": region_index(regions[n]["area"]),
                   "out_dir": REPO_ROOT / regions[n]["era5_dir"]} for n in names}

    ds = xr.open_zarr(WB2_STORE, storage_options={"token": "anon"}, chunks=None)
    print(f"{args.variables} <- WB2 6-hourly  regions={names}  years={args.years}")
    for year in range(args.years[0], args.years[1] + 1):
        process_year(ds, year, targets, args.variables, args.threads, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
