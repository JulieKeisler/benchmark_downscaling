#!/usr/bin/env python3
"""Download daily ERA5 fields from Google's ARCO-ERA5 public Zarr (no CDS).

ARCO-ERA5 (Carver & Merose 2023, https://github.com/google-research/arco-era5)
is the full hourly 0.25° ERA5 single-level archive on a public GCS bucket:
anonymous access, no account, no queue.  It is the preferred source for
variables the WB2 archive on Jean-Zay does not carry — notably `rsds`
(surface solar radiation downwards) — and for any years outside WB2's
1979–2022 coverage (e.g. the 1960–1978 climatology history SF-CLIM needs).

The store is chunked one global field per hour, so every chunk read is
shared across all requested regions (a single pass over the archive serves
france + australia + us_central at the same cost as one region).

Output matches the WB2/jz crop layout exactly:
    <era5_dir>/<var>_<year>.nc   (time, latitude, longitude)
    latitude north→south, longitude in [-180, 180), daily datetime64 at 00:00 UTC

Daily aggregation (UTC days), consistent with crop_jz_archive.py:
    rsds  = mean over the 24 hourly accumulations ending in (D 00:00, D+1 00:00],
            J m-2 / 3600 → W m-2 (same definition as CMIP6 day-table `rsds`).
    pr    = sum of the 24 hourly `total_precipitation` accumulations ending in
            (D 00:00, D+1 00:00] → daily total in m (loaders convert to mm/day)
    tas   = mean of hourly 2m temperature over day D
    psl   = mean of hourly mean-sea-level pressure over day D
    sfcWind    = mean over day D of sqrt(u10^2 + v10^2)      [m/s]
    sfcWindmax = max  over day D of sqrt(u10^2 + v10^2)      [m/s]
Instantaneous fields use hour stamps (D 00:00, D 23:00]; accumulated fields
use (D 00:00, D+1 00:00] (ERA5 accumulations are stamped at their END hour).

Usage:
    python src/data_download/arco/download_era5_arco.py                # all regions, 1979-2022, rsds
    python src/data_download/arco/download_era5_arco.py --regions france australia us_central \\
        --years 1960 1978 --variables tas pr sfcWind sfcWindmax psl rsds   # SF-CLIM backfill
    python src/data_download/arco/download_era5_arco.py \\
        --area 55 -6 41 10 --output-dir data/era5 --years 1979 2022

Requirements:
    pip install xarray zarr gcsfs netcdf4 pyyaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
REPO_ROOT = Path(__file__).resolve().parents[3]
REGIONS_YAML = REPO_ROOT / "src" / "configs" / "data" / "regions.yaml"

# CMIP6 short name -> spec dict.
#   sources : ARCO variable(s) to read; 2 names => combined into a single field
#             (the only combination used today is sqrt(u10^2 + v10^2)).
#   agg     : accum_mean / accum_sum  (hourly accumulations, window (D,D+1])
#             instant_mean / instant_max (instantaneous fields, window (D,D+1) 00:00..23:00)
#   scale   : multiplier applied to the aggregated value
#   attrs   : netCDF attrs on the output variable
VARIABLES = {
    "rsds": dict(
        sources=("surface_solar_radiation_downwards",), agg="accum_mean", scale=1.0 / 3600.0,
        attrs={"units": "W m-2", "long_name": "Surface downwelling shortwave radiation",
               "standard_name": "surface_downwelling_shortwave_flux_in_air"},
    ),
    "pr": dict(
        sources=("total_precipitation",), agg="accum_sum", scale=1.0,
        attrs={"units": "m", "long_name": "Daily total precipitation",
               "standard_name": "precipitation_amount"},
    ),
    "tas": dict(
        sources=("2m_temperature",), agg="instant_mean", scale=1.0,
        attrs={"units": "K", "long_name": "2 metre temperature",
               "standard_name": "air_temperature"},
    ),
    "psl": dict(
        sources=("mean_sea_level_pressure",), agg="instant_mean", scale=1.0,
        attrs={"units": "Pa", "long_name": "Mean sea level pressure",
               "standard_name": "air_pressure_at_mean_sea_level"},
    ),
    "sfcWind": dict(
        sources=("10m_u_component_of_wind", "10m_v_component_of_wind"), agg="instant_mean", scale=1.0,
        attrs={"units": "m s-1", "long_name": "10 metre wind speed",
               "standard_name": "wind_speed"},
    ),
    "sfcWindmax": dict(
        sources=("10m_u_component_of_wind", "10m_v_component_of_wind"), agg="instant_max", scale=1.0,
        attrs={"units": "m s-1", "long_name": "Daily maximum 10 metre wind speed",
               "standard_name": "wind_speed_of_gust"},
    ),
}


def region_index(area: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integer indices into the ARCO 0.25° grid (lat 90→-90, lon 0→359.75) for [N, W, S, E]."""
    north, west, south, east = area
    lat = np.arange(north, south - 0.125, -0.25)
    lon = np.arange(west, east + 0.125, 0.25)
    ilat = np.rint((90.0 - lat) / 0.25).astype(int)
    ilon = np.rint((lon % 360.0) / 0.25).astype(int)
    return ilat, ilon, lat.astype(np.float32), lon.astype(np.float32)


def write_year(daily: xr.DataArray, var: str, attrs: dict, year: int, out_dir: Path) -> Path:
    ds = daily.astype(np.float32).rename(var).to_dataset()
    ds[var].attrs = {**attrs, "source": f"ERA5 via ARCO-ERA5 ({ARCO_STORE})"}
    ds.attrs["history"] = "daily aggregation by src/data_download/arco/download_era5_arco.py"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{var}_{year}.nc"
    tmp = path.with_suffix(".nc.part")
    ds.to_netcdf(tmp, encoding={"time": {"units": f"days since {year}-01-01", "dtype": "int64"}})
    tmp.rename(path)
    return path


def process_year(src_vars: dict, times: np.ndarray, year: int, targets: dict,
                 var: str, threads: int, overwrite: bool) -> None:
    """Stream one year of hourly global fields once, accumulating every region's daily value.

    Each hourly chunk is a full global field; it is read once, cropped for all
    regions in numpy and discarded, so memory stays at ~2 x threads x 4 MB.
    """
    from concurrent.futures import ThreadPoolExecutor

    spec = VARIABLES[var]
    how, scale = spec["agg"], spec["scale"]
    todo = {name: t for name, t in targets.items()
            if overwrite or not (t["out_dir"] / f"{var}_{year}.nc").exists()}
    if not todo:
        print(f"  {var} {year}: all regions exist, skipping")
        return

    days = np.arange(f"{year}-01-01", f"{year + 1}-01-01", dtype="datetime64[D]")
    n_hours = 24 * len(days)

    # accumulated fields: hours (D 00:00, D+1 00:00] are stamped 01:00..24:00 of
    # the following day boundary; instantaneous fields: D 00:00..23:00.
    if how in ("accum_mean", "accum_sum"):
        first = np.datetime64(f"{year}-01-01T01:00")
    else:
        first = np.datetime64(f"{year}-01-01T00:00")
    i0 = int(np.searchsorted(times, first))
    if times[i0] != first or i0 + n_hours > len(times):
        raise RuntimeError(f"{year}: hourly axis does not cover the year")

    # one accumulator per region per day
    acc = {name: {"sum": np.zeros((len(days), len(t["index"][0]), len(t["index"][1])), np.float64),
                  "max": np.full((len(days), len(t["index"][0]), len(t["index"][1])), -np.inf, np.float64),
                  "cnt": np.zeros(len(days), np.int64)} for name, t in todo.items()}

    def combine(spec, crops):
        if len(spec["sources"]) == 1:
            return crops[0]
        u, v = crops
        return np.sqrt(u ** 2 + v ** 2)

    def read(h: int):
        fields = [src.isel(time=i0 + h).values for src in src_vars]  # (721,1440) or two
        crops = [[f[np.ix_(t["index"][0], t["index"][1])] for f in fields] for t in todo.values()]
        return h, [combine(spec, c) for c in crops]

    t0 = time.time()
    with ThreadPoolExecutor(threads) as pool:
        for k, (h, combined) in enumerate(pool.map(read, range(n_hours)), 1):
            d, i_h = divmod(h, 24)
            for name, c in zip(todo, combined):
                a = acc[name]
                if how in ("accum_mean", "accum_sum", "instant_mean"):
                    a["sum"][d] += c
                    a["cnt"][d] += 1
                else:  # instant_max
                    a["max"][d] = np.maximum(a["max"][d], c)
                    a["cnt"][d] += 1
            if k % (24 * 30) == 0:
                print(f"    {year}: {k}/{n_hours} h ({time.time() - t0:.0f}s)", flush=True)

    for name, t in todo.items():
        _, _, lat, lon = t["index"]
        a = acc[name]
        if how in ("accum_mean", "instant_mean"):
            vals = a["sum"] / np.maximum(a["cnt"][:, None, None], 1)
        else:  # instant_max
            vals = a["max"]
        if how == "accum_sum":
            vals = a["sum"]
        vals = vals * scale
        vals = np.clip(vals, 0.0, None) if how.startswith("accum") else vals  # tiny negatives from packing
        if np.isnan(vals).any():
            raise RuntimeError(f"{name} {var} {year}: NaNs in source")
        daily = xr.DataArray(vals, dims=("time", "latitude", "longitude"),
                             coords={"time": days.astype("datetime64[ns]"),
                                     "latitude": lat, "longitude": lon})
        path = write_year(daily, var, spec["attrs"], year, t["out_dir"])
        print(f"  {var} {year} {name}: {path} (mean {vals.mean():.1f} {spec['attrs']['units']})")
    print(f"  {var} {year}: {time.time() - t0:.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--regions", nargs="+", default=None,
                        help="region names from regions.yaml (default: all)")
    parser.add_argument("--area", nargs=4, type=float, metavar=("N", "W", "S", "E"),
                        help="explicit domain instead of --regions (requires --output-dir)")
    parser.add_argument("--output-dir", type=Path, help="output dir for --area")
    parser.add_argument("--years", nargs=2, type=int, default=[1979, 2022], metavar=("START", "END"))
    parser.add_argument("--variables", nargs="+", default=["rsds"], choices=sorted(VARIABLES))
    parser.add_argument("--threads", type=int, default=32, help="concurrent chunk reads")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.area:
        if not args.output_dir:
            parser.error("--area requires --output-dir")
        targets = {"custom": {"area": args.area, "out_dir": args.output_dir}}
    else:
        regions = yaml.safe_load(REGIONS_YAML.read_text())["regions"]
        names = args.regions or list(regions)
        unknown = set(names) - set(regions)
        if unknown:
            parser.error(f"unknown regions {sorted(unknown)}; known: {sorted(regions)}")
        targets = {n: {"area": regions[n]["area"], "out_dir": REPO_ROOT / regions[n]["era5_dir"]}
                   for n in names}
    for t in targets.values():
        t["index"] = region_index(t["area"])

    # trust_env=True makes aiohttp honor HTTP(S)_PROXY (Jean-Zay requires the
    # IDRIS proxy for external egress); without it gcsfs hangs on direct
    # connections. The ClientTimeout turns a hung connection into an error
    # instead of blocking a worker thread forever (jobs with many long-lived
    # GCS connections would otherwise crawl to ~1 h/s and stall).
    import aiohttp

    ds = xr.open_zarr(
        ARCO_STORE,
        storage_options={
            "token": "anon",
            "session_kwargs": {
                "trust_env": True,
                "timeout": aiohttp.ClientTimeout(total=120),
            },
        },
        chunks=None,
    )
    times = ds.time.values
    for var in args.variables:
        src_vars = [ds[name].reset_coords(drop=True) for name in VARIABLES[var]["sources"]]
        print(f"{var} <- {VARIABLES[var]['sources']}  regions={list(targets)}  years={args.years}")
        for year in range(args.years[0], args.years[1] + 1):
            process_year(src_vars, times, year, targets, var, args.threads, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())