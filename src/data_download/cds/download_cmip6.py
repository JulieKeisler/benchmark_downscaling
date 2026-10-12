#!/usr/bin/env python3
"""Download daily CMIP6 GCM data from the Copernicus Climate Data Store.

Requires a CDS account and API key in ~/.cdsapirc (see README.md), and:

    pip install cdsapi pyyaml

Usage:

    python scripts/download_cmip6.py                            # everything
    python scripts/download_cmip6.py --models cesm2             # one model
    python scripts/download_cmip6.py --variables tas sfcWindmax # a subset
    python scripts/download_cmip6.py --jobs 8                   # more parallelism

The requested period is split into the `historical` experiment (up to 2014)
and the configured scenario (2015 onwards). The CDS delivers a zip per
request; its NetCDF content is extracted to
data/raw/cmip6/<model>/<variable>_<experiment>_<start>-<end>.nc.
Existing files are skipped, so an interrupted download can be re-run.

Requests are submitted in parallel (--jobs, default 4): the CDS queues
requests server-side and processes several per user concurrently.

Not every model publishes every variable at daily resolution on the CDS;
failed requests are reported at the end and do not stop the others.
To add a variable, extend CDS_VARIABLES below and list its short name in
the config.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

DATASET = "projections-cmip6"

# CMIP6 short name -> CDS variable name
CDS_VARIABLES = {
    "tas": "near_surface_air_temperature",
    "tasmax": "daily_maximum_near_surface_air_temperature",
    "tasmin": "daily_minimum_near_surface_air_temperature",
    "sfcWind": "near_surface_wind_speed",
    "sfcWindmax": "daily_maximum_near_surface_wind_speed",
    "pr": "precipitation",
    "rsds": "surface_downwelling_shortwave_radiation",
    "psl": "sea_level_pressure",
}

HISTORICAL_END = 2014

_thread_local = threading.local()


def get_client():
    """One cdsapi client per worker thread."""
    if not hasattr(_thread_local, "client"):
        import cdsapi

        _thread_local.client = cdsapi.Client()
    return _thread_local.client


def experiment_spans(start: int, end: int, scenario: str) -> list[tuple[str, int, int]]:
    """Split [start, end] into (experiment, start, end) chunks."""
    spans = []
    if start <= HISTORICAL_END:
        spans.append(("historical", start, min(end, HISTORICAL_END)))
    if end > HISTORICAL_END:
        spans.append((scenario, max(start, HISTORICAL_END + 1), end))
    return spans


def download_one(model: str, variable: str, experiment: str,
                 start: int, end: int, area: list[float], target: Path) -> None:
    # CDS API v2 (2025): "date" replaced by "year"/"month" arrays; "format" removed.
    request = {
        "temporal_resolution": "daily",
        "experiment": experiment,
        "variable": CDS_VARIABLES[variable],
        "model": model,
        "year": [str(y) for y in range(start, end + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "area": area,
    }
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "download.zip"
        get_client().retrieve(DATASET, request, str(zip_path))
        with zipfile.ZipFile(zip_path) as zf:
            nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not nc_members:
                raise RuntimeError("no NetCDF file in CDS zip")
            if len(nc_members) > 1:
                raise RuntimeError(f"expected one NetCDF file, got {nc_members}")
            zf.extract(nc_members[0], tmp)
            shutil.move(str(Path(tmp) / nc_members[0]), str(target))


def fetch(task: tuple[str, str, str, int, int, list[float], Path]) -> str | None:
    """Download one model-variable-experiment. Returns an error string, or None."""
    import time
    model, variable, experiment, start, end, area, target = task
    print(f"[get ] {model} {variable} {experiment} {start}-{end}")
    for attempt in range(1, 13):  # up to 1 hour of retries
        try:
            download_one(model, variable, experiment, start, end, area, target)
            print(f"[done] {target}")
            return None
        except Exception as exc:
            msg = str(exc)
            if "temporarily limited" in msg or "queued" in msg.lower():
                wait = 5 * 60
                print(f"  [{model} {variable} {experiment}] queue limit, "
                      f"waiting {wait//60} min (attempt {attempt}/12)...")
                time.sleep(wait)
            else:
                return f"{model} {variable} {experiment}: {exc}"
    return f"{model} {variable} {experiment}: queue limit exceeded after 12 retries"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="src/configs/data/cmip6.yaml")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models from the config (default: all)")
    parser.add_argument("--variables", nargs="*", default=None,
                        help="Subset of variables from the config (default: all)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory from config")
    parser.add_argument("--jobs", type=int, default=4,
                        help="Number of CDS requests submitted in parallel (default: 4)")
    args = parser.parse_args(argv)

    import cdsapi  # noqa: F401  imported here so --help works without it installed

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = args.models or cfg["models"]
    variables = args.variables or cfg["variables"]
    unknown = set(variables) - set(CDS_VARIABLES)
    if unknown:
        sys.exit(f"No CDS name known for {sorted(unknown)}; extend CDS_VARIABLES "
                 "in scripts/download_cmip6.py")

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    spans = experiment_spans(cfg["years"]["start"], cfg["years"]["end"],
                             cfg["scenario"])

    tasks = []
    skipped = 0
    for model in models:
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for variable in variables:
            for experiment, start, end in spans:
                target = model_dir / f"{variable}_{experiment}_{start}-{end}.nc"
                if target.exists():
                    skipped += 1
                    continue
                tasks.append((model, variable, experiment, start, end,
                              cfg["area"], target))

    if skipped:
        print(f"{skipped} file(s) already present, skipped")
    if not tasks:
        print("Nothing to download.")
        return 0
    print(f"{len(tasks)} request(s) to submit with {args.jobs} parallel job(s)\n")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        failures = [err for err in executor.map(fetch, tasks) if err]

    if failures:
        print(f"\n{len(failures)} request(s) failed:", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
