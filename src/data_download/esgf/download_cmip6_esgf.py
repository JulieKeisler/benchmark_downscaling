#!/usr/bin/env python3
"""Download daily CMIP6 GCM data from ESGF (no CDS queue).

Uses the DKRZ ESGF node for search and downloads files directly via HTTP.
Files are global at native model resolution; they are cropped to the
configured domain with xarray after download.  Temporary global files are
deleted unless --keep-global is set.

Usage:

    python scripts/download_cmip6_esgf.py
    python scripts/download_cmip6_esgf.py --models cesm2
    python scripts/download_cmip6_esgf.py --variables tas pr
    python scripts/download_cmip6_esgf.py --jobs 3

Requirements:
    pip install requests xarray netcdf4 pyyaml
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import xarray as xr
import yaml

ESGF_SEARCH = "https://esgf-data.dkrz.de/esg-search/search"

# CDS model id -> ESGF source_id
SOURCE_IDS = {
    "ipsl_cm6a_lr": "IPSL-CM6A-LR",
    "cnrm_cm6_1":   "CNRM-CM6-1",
    "cesm2":        "CESM2",
}

# Default ensemble member per model (overridden by config `members` key).
# If a member has no files for a specific experiment, auto-detect is used.
DEFAULT_MEMBERS = {
    "ipsl_cm6a_lr": "r1i1p1f1",
    "cnrm_cm6_1":   "r1i1p1f2",
    "cesm2":        "r1i1p1f1",
}

HISTORICAL_END = 2014
# How many mirrors to try per file before giving up
MAX_MIRROR_TRIES = 3

_session_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_session_local, "s"):
        _session_local.s = requests.Session()
        _session_local.s.headers["User-Agent"] = "download_cmip6_esgf/1.0"
    return _session_local.s


def experiment_spans(start: int, end: int, scenario: str) -> list[tuple[str, int, int]]:
    spans = []
    if start <= HISTORICAL_END:
        spans.append(("historical", start, min(end, HISTORICAL_END)))
    if end > HISTORICAL_END:
        spans.append((scenario, max(start, HISTORICAL_END + 1), end))
    return spans


def _file_years(title: str) -> tuple[int, int] | None:
    """Parse start/end year from CMIP6 filename like ..._18500101-20141231.nc"""
    m = re.search(r"_(\d{4})\d{4}-(\d{4})\d{4}\.nc$", title)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def search_files(source_id: str, variable: str, experiment: str,
                 member: str, start_year: int, end_year: int
                 ) -> list[tuple[str, str]]:
    """Return list of (url, title) for files overlapping [start_year, end_year].

    If member yields no results, tries without member filter (auto-detect).
    Returns deduplicated URLs, preferring vesg.ipsl.upmc.fr then esgf3.dkrz.de.
    """
    def _query(member_filter: str | None) -> list[dict]:
        params = {
            "project": "CMIP6",
            "source_id": source_id,
            "variable_id": variable,
            "table_id": "day",
            "experiment_id": experiment,
            "type": "File",
            "format": "application/solr+json",
            "limit": 200,
            "fields": "title,url,size,member_id",
        }
        if member_filter:
            params["member_id"] = member_filter
        r = _session().get(ESGF_SEARCH, params=params, timeout=30)
        r.raise_for_status()
        return r.json()["response"]["docs"]

    docs = _query(member)
    if not docs:
        # Auto-detect: take first available member
        docs = _query(None)
        if docs:
            detected = docs[0].get("member_id", ["?"])[0]
            print(f"  NOTE: {source_id} {experiment}: member {member} not found, "
                  f"using {detected}")

    # Filter to files overlapping our period; collect all mirrors per title
    relevant: dict[str, list[str]] = {}  # title -> ordered list of mirror URLs
    for doc in docs:
        title = doc["title"]
        years = _file_years(title)
        if years is None:
            continue
        file_start, file_end = years
        if file_end < start_year or file_start > end_year:
            continue
        urls = [u.split("|")[0] for u in doc.get("url", []) if "HTTPServer" in u]
        if not urls:
            continue
        if title not in relevant:
            # Order mirrors: IPSL vesg first, then DKRZ, then rest
            ordered = (
                [u for u in urls if "vesg.ipsl" in u]
                + [u for u in urls if "dkrz.de" in u]
                + [u for u in urls if "vesg.ipsl" not in u and "dkrz.de" not in u]
            )
            relevant[title] = ordered
        else:
            # Merge additional mirrors from duplicate docs
            existing = set(relevant[title])
            relevant[title] += [u for u in urls if u not in existing]

    return list(relevant.items())  # (title, [url, ...])


def _download_file(url: str, dest: Path) -> None:
    """Stream-download url to dest."""
    r = _session().get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    written = 0
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
            f.write(chunk)
            written += len(chunk)
    if total and written < total * 0.99:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"incomplete download: {written}/{total} bytes")


def fetch(task: tuple) -> str | None:
    """Download, crop and save one model-variable-experiment. Returns error or None."""
    (model, source_id, variable, experiment, start_year, end_year,
     member, area, target, keep_global) = task

    print(f"[get ] {model} {variable} {experiment} {start_year}-{end_year}")

    files = search_files(source_id, variable, experiment, member,
                         start_year, end_year)
    if not files:
        return (f"{model} {variable} {experiment}: no files found on ESGF "
                f"(source_id={source_id}, member={member})")

    north, west, south, east = area

    tmp_base = target.parent.parent / "_tmp"
    tmp_base.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmp:
        tmp = Path(tmp)
        local_files = []
        for title, urls in sorted(files):  # sorted by filename = chronological
            dest = tmp / title
            print(f"  downloading {title} ...")
            last_exc = None
            for url in urls:
                try:
                    _download_file(url, dest)
                    local_files.append(dest)
                    last_exc = None
                    break
                except Exception as exc:
                    print(f"  mirror failed ({url.split('/')[2]}): {exc}")
                    dest.unlink(missing_ok=True)
                    last_exc = exc
            if last_exc is not None:
                return f"{model} {variable} {experiment} {title}: all mirrors failed"

        if not local_files:
            return f"{model} {variable} {experiment}: no files downloaded"

        ds = xr.open_mfdataset(local_files, combine="by_coords", use_cftime=True)

        # Crop to domain — handle both -180..180 and 0..360 lon conventions
        lat_dim = "lat" if "lat" in ds.dims else "latitude"
        lon_dim = "lon" if "lon" in ds.dims else "longitude"
        lon_vals = ds[lon_dim].values
        if lon_vals.max() > 180:
            # 0-360 convention: convert domain bounds and roll so -180..180 is contiguous
            ds = ds.assign_coords({lon_dim: ((ds[lon_dim] + 180) % 360) - 180})
            ds = ds.sortby(lon_dim)
        ds = ds.sel(
            {lat_dim: slice(south, north), lon_dim: slice(west, east)}
        )

        # Subset to our period
        ds = ds.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))

        # Keep only the variable of interest
        if variable in ds.data_vars:
            ds = ds[[variable]]

        tmp_out = target.with_suffix(".tmp.nc")
        ds.to_netcdf(tmp_out)
        ds.close()
        tmp_out.rename(target)
        print(f"[done] {target}")

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/data/cmip6.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--jobs", type=int, default=3,
                        help="Parallel downloads (default: 3)")
    parser.add_argument("--keep-global", action="store_true",
                        help="Keep raw global files in --output-dir/global/")
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = args.models or cfg["models"]
    variables = args.variables or cfg["variables"]
    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["output_dir"])
    spans = experiment_spans(cfg["years"]["start"], cfg["years"]["end"],
                             cfg["scenario"])

    # Member IDs: config > DEFAULT_MEMBERS
    cfg_members = cfg.get("members", {})

    tasks = []
    skipped = 0
    for model in models:
        if model not in SOURCE_IDS:
            print(f"WARNING: unknown model {model}, skipping")
            continue
        source_id = SOURCE_IDS[model]
        member = cfg_members.get(model, DEFAULT_MEMBERS.get(model, "r1i1p1f1"))
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for variable in variables:
            for experiment, start, end in spans:
                target = model_dir / f"{variable}_{experiment}_{start}-{end}.nc"
                if target.exists():
                    skipped += 1
                    continue
                tasks.append((model, source_id, variable, experiment,
                               start, end, member, cfg["area"],
                               target, args.keep_global))

    if skipped:
        print(f"{skipped} file(s) already present, skipped")
    if not tasks:
        print("Nothing to download.")
        return 0
    print(f"{len(tasks)} task(s) with {args.jobs} parallel job(s)\n")

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        failures = [err for err in executor.map(fetch, tasks) if err]

    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
