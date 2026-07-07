#!/usr/bin/env python3
"""Download ERA5 daily data for a regional domain from the Copernicus CDS.

Two pipelines driven by configs/data/era5.yaml:

  1. Daily statistics (derived-era5-single-levels-daily-statistics):
     One request per variable per year (cost 365 variable-days, limit 400).
     Writes <short_name>_<year>.nc.

  2. Wind (reanalysis-era5-single-levels, hourly):
     Downloads 10m u+v for one year per request (cost ~105 k, limit 121 k),
     computes sqrt(u²+v²) at each hour, aggregates to daily mean (sfcWind)
     and daily max (sfcWindmax), deletes the raw hourly file unless
     keep_hourly: true.

Pre-flight: each pipeline sends one representative request to the /costing
endpoint (no download) and aborts if the cost would exceed the limit.

Requirements:
    pip install cdsapi pyyaml xarray netcdf4 numpy
    ~/.cdsapirc with your Copernicus API key

Usage:
    python scripts/download_era5.py
    python scripts/download_era5.py --variables rsds
    python scripts/download_era5.py --variables rsds --years 1979 2022
    python scripts/download_era5.py --config configs/data/era5.yaml --jobs 4
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

# These variables have a known ECMWF data issue (June 2026, +1 h window shift
# in the _since_previous_post_processing accumulation window).  Remove entries
# once ECMWF announces the fix.
KNOWN_ISSUE_VARIABLES = {
    "maximum_2m_temperature_since_previous_post_processing",
    "minimum_2m_temperature_since_previous_post_processing",
    "10m_wind_gust_since_previous_post_processing",
    "maximum_total_precipitation_rate_since_previous_post_processing",
    "minimum_total_precipitation_rate_since_previous_post_processing",
}

DAILY_STATS_DATASET = "derived-era5-single-levels-daily-statistics"
HOURLY_DATASET = "reanalysis-era5-single-levels"

# Cost limit for daily-stats requests (per /costing API)
DAILY_STATS_COST_LIMIT = 400
# Cost limit for hourly requests (per /costing API)
HOURLY_COST_LIMIT = 121_000

_thread_local = threading.local()


def _client():
    """Return a thread-local cdsapi.Client (lazy init, one per worker)."""
    if not hasattr(_thread_local, "client"):
        import cdsapi
        _thread_local.client = cdsapi.Client()
    return _thread_local.client


# ---------------------------------------------------------------------------
# Cost pre-flight
# ---------------------------------------------------------------------------

def _check_cost(request: dict, dataset: str, limit: int) -> None:
    """Abort if one representative request exceeds the CDS cost limit."""
    import urllib.request, json, urllib.parse

    base = "https://cds.climate.copernicus.eu/api/retrieve/v1/processes"
    url = f"{base}/{dataset}/costing"
    body = json.dumps({"inputs": request}).encode()
    req = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        cost = data.get("cost", 0)
        actual_limit = data.get("limit", limit)
        if cost > actual_limit:
            sys.exit(
                f"Cost check FAILED for {dataset}:\n"
                f"  Request cost : {cost}\n"
                f"  Server limit : {actual_limit}\n"
                "Reduce --variables, or split into smaller year ranges."
            )
        print(f"  Cost check OK: {cost} / {actual_limit}")
    except Exception as exc:
        print(f"  WARNING: cost check skipped ({exc})")


# ---------------------------------------------------------------------------
# Daily statistics pipeline
# ---------------------------------------------------------------------------

def _daily_stats_request(variable: str, year: int, area: list) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": variable,
        "statistic": "daily_mean",
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time_zone": "UTC+00:00",
        "frequency": "1_hourly",
        "area": area,
        "data_format": "netcdf",
    }


def download_daily_stat(short_name: str, era5_name: str, year: int,
                         area: list, output_dir: Path) -> str | None:
    """Download one variable × one year. Returns error string or None.

    Retries with 5-minute waits when the CDS queue limit is hit (old requests
    from previous failed runs can block new ones for a while).
    """
    import time
    path = output_dir / f"{short_name}_{year}.nc"
    if path.exists():
        return None
    tmp = path.with_suffix(".tmp.nc")
    req = _daily_stats_request(era5_name, year, area)
    for attempt in range(1, 13):  # up to 1 hour of retries
        try:
            _client().retrieve(DAILY_STATS_DATASET, req, str(tmp))
            tmp.rename(path)
            print(f"  -> {path}")
            return None
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if "temporarily limited" in str(exc) or "queued" in str(exc).lower():
                wait = 5 * 60
                print(f"  [{short_name} {year}] queue limit hit, waiting {wait//60} min "
                      f"(attempt {attempt}/12)...")
                time.sleep(wait)
            else:
                return f"{short_name} {year}: {exc}"
    return f"{short_name} {year}: queue limit exceeded after 12 retries"


# ---------------------------------------------------------------------------
# Wind pipeline (hourly → daily)
# ---------------------------------------------------------------------------

def _hourly_wind_request(year: int, area: list) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area,
        "data_format": "netcdf",
    }


def download_wind_year(year: int, area: list, output_dir: Path,
                        keep_hourly: bool = False) -> str | None:
    """Download hourly u/v → sfcWind + sfcWindmax. Returns error or None."""
    wind_path = output_dir / f"sfcWind_{year}.nc"
    wmax_path = output_dir / f"sfcWindmax_{year}.nc"
    if wind_path.exists() and wmax_path.exists():
        return None
    hourly_path = output_dir / f"_hourly_uv_{year}.nc"
    try:
        if not hourly_path.exists():
            req = _hourly_wind_request(year, area)
            _client().retrieve(HOURLY_DATASET, req, str(hourly_path))
        ds = xr.open_dataset(hourly_path)
        # time dimension may be named 'valid_time' on the new CDS
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        if time_dim == "valid_time":
            ds = ds.rename({"valid_time": "time"})
        u = ds["u10"] if "u10" in ds else ds["10m_u_component_of_wind"]
        v = ds["v10"] if "v10" in ds else ds["10m_v_component_of_wind"]
        spd = np.sqrt(u ** 2 + v ** 2)
        if not wind_path.exists():
            spd.resample(time="1D").mean().to_netcdf(wind_path)
            print(f"  -> {wind_path}")
        if not wmax_path.exists():
            spd.resample(time="1D").max().to_netcdf(wmax_path)
            print(f"  -> {wmax_path}")
        ds.close()
        if not keep_hourly:
            hourly_path.unlink(missing_ok=True)
    except Exception as exc:
        return f"wind {year}: {exc}"
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data/era5.yaml")
    parser.add_argument("--variables", nargs="+", default=None,
                        help="CMIP6 short names to download (default: all in config). "
                             "Wind (sfcWind/sfcWindmax) is driven by hourly_wind config.")
    parser.add_argument("--years", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Year range, e.g. --years 1979 2022 (default: from config)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: from config)")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Parallel CDS requests (default: 4)")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    area = cfg["area"]
    y_start, y_end = args.years or (cfg["years"]["start"], cfg["years"]["end"])
    years = list(range(y_start, y_end + 1))
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the daily-stats variable table from config
    daily_vars: dict[str, str] = {}  # short_name -> era5_name
    for short_name, era5_name in cfg.get("daily_statistics", {}).get("daily_mean", {}).items():
        if era5_name in KNOWN_ISSUE_VARIABLES:
            print(f"SKIP {short_name}: {era5_name} has a known ECMWF data issue "
                  f"(June 2026 +1h window shift) — remove from KNOWN_ISSUE_VARIABLES once fixed")
            continue
        daily_vars[short_name] = era5_name

    do_wind = cfg.get("hourly_wind", {}).get("enabled", False)
    keep_hourly = cfg.get("hourly_wind", {}).get("keep_hourly", False)

    # Filter to requested variables
    requested = set(args.variables) if args.variables else None
    if requested:
        daily_vars = {k: v for k, v in daily_vars.items() if k in requested}
        if "sfcWind" not in requested and "sfcWindmax" not in requested:
            do_wind = False

    print(f"ERA5 CDS download")
    print(f"  Config     : {args.config}")
    print(f"  Area       : {area} (N W S E)")
    print(f"  Years      : {y_start}–{y_end}  ({len(years)} years)")
    print(f"  Daily vars : {list(daily_vars)}")
    print(f"  Wind       : {'yes (hourly → daily)' if do_wind else 'no'}")
    print(f"  Jobs       : {args.jobs}")
    print(f"  Output     : {output_dir}")
    print()

    # Cost pre-flight for daily stats
    if daily_vars:
        rep_var = next(iter(daily_vars.values()))
        rep_req = _daily_stats_request(rep_var, years[0], area)
        print(f"Pre-flight cost check ({DAILY_STATS_DATASET})...")
        _check_cost(rep_req, DAILY_STATS_DATASET, DAILY_STATS_COST_LIMIT)

    # Cost pre-flight for hourly wind
    if do_wind:
        rep_wind_req = _hourly_wind_request(years[0], area)
        print(f"Pre-flight cost check ({HOURLY_DATASET})...")
        _check_cost(rep_wind_req, HOURLY_DATASET, HOURLY_COST_LIMIT)

    # Build task list: (short_name, era5_name, year) or ("wind", year)
    tasks: list[tuple] = []
    skipped = 0
    for short_name, era5_name in daily_vars.items():
        for year in years:
            if (output_dir / f"{short_name}_{year}.nc").exists():
                skipped += 1
            else:
                tasks.append(("stat", short_name, era5_name, year))
    if do_wind:
        for year in years:
            wind_ok = (output_dir / f"sfcWind_{year}.nc").exists()
            wmax_ok = (output_dir / f"sfcWindmax_{year}.nc").exists()
            if wind_ok and wmax_ok:
                skipped += 1
            else:
                tasks.append(("wind", year))

    if skipped:
        print(f"{skipped} file(s) already exist, skipping")
    if not tasks:
        print("Nothing to download.")
        return 0
    print(f"{len(tasks)} task(s) to submit\n")

    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for task in tasks:
            if task[0] == "stat":
                _, short_name, era5_name, year = task
                f = pool.submit(download_daily_stat,
                                short_name, era5_name, year, area, output_dir)
                futures[f] = task
            else:
                _, year = task
                f = pool.submit(download_wind_year,
                                year, area, output_dir, keep_hourly)
                futures[f] = task
        for f in as_completed(futures):
            err = f.result()
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
