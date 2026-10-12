#!/usr/bin/env python3
"""Download daily CMIP6 GCM data from ESGF (no CDS queue).

Uses the DKRZ ESGF node for search and downloads files directly via HTTP.
Files are global at native model resolution; they are cropped to the
configured domain with xarray after download.  Temporary global files are
deleted unless --keep-global is set.

Usage:

    python src/data_download/esgf/download_cmip6_esgf.py
    python src/data_download/esgf/download_cmip6_esgf.py --config src/configs/data/cmip6.yaml
    python src/data_download/esgf/download_cmip6_esgf.py --models cesm2
    python src/data_download/esgf/download_cmip6_esgf.py --variables tas pr
    python src/data_download/esgf/download_cmip6_esgf.py --area -24 136 -38 152
    python src/data_download/esgf/download_cmip6_esgf.py --jobs 3

Config keys (beyond the original single `scenario`):
  scenarios: [ssp370, ssp245]   # list of future experiments (historical is always added)
  ignore_missing: [sfcWind:ssp370, sfcWind:ssp245]   # known GCM gaps, downgraded to warnings
  areas: {model: [N, W, S, E]}  # optional per-model override of the global area
  stars/end are taken from the `years` key; scenario spans cover 2015+.
  ESGF_MIRROR_ORDER env var can reorder mirror preference (comma-separated
  host fragments, e.g. "dkrz.de,vesg.ipsl").

Requirements:
    pip install requests xarray netcdf4 pyyaml
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import xarray as xr
import yaml

ESGF_SEARCH = "https://esgf-data.dkrz.de/esg-search/search"
ESGF_SEARCHES = [
    ESGF_SEARCH,
    "https://esgf-index1.ceda.ac.uk/esg-search/search",
    "https://esgf-node.llnl.gov/esg-search/search",
]

# CDS model id -> ESGF source_id
SOURCE_IDS = {
    "ipsl_cm6a_lr":  "IPSL-CM6A-LR",
    "cnrm_cm6_1":    "CNRM-CM6-1",
    "cesm2":         "CESM2",
    "access_esm1_5": "ACCESS-ESM1-5",
}

# Default ensemble member per model (overridden by config `members` key).
# If a member has no files for a specific experiment, auto-detect is used.
DEFAULT_MEMBERS = {
    "ipsl_cm6a_lr":  "r1i1p1f1",
    "cnrm_cm6_1":    "r1i1p1f2",
    "cesm2":         "r1i1p1f1",
    "access_esm1_5": "r1i1p1f1",
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


def experiment_spans(start: int, end: int,
                     scenarios: str | list[str]) -> list[tuple[str, int, int]]:
    """Split [start, end] into (experiment, start, end) chunks.

    The historical experiment covers <= 2014; each entry in `scenarios`
    (accepts a single string for backward compatibility) covers 2015+.
    """
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    spans = []
    if start <= HISTORICAL_END:
        spans.append(("historical", start, min(end, HISTORICAL_END)))
    if end > HISTORICAL_END:
        for scenario in scenarios:
            spans.append((scenario, max(start, HISTORICAL_END + 1), end))
    return spans


def _mirror_sort_key(urls: list[str]) -> list[str]:
    """Order mirrors by the ESGF_MIRROR_ORDER env var, then host fragment."""
    import os
    env = os.environ.get("ESGF_MIRROR_ORDER", "")
    preferred = [f for f in env.split(",") if f]
    if preferred:
        return sorted(urls, key=lambda u: next(
            (i for i, frag in enumerate(preferred) if frag in u),
            len(preferred)))
    # Default order tuned for European sites (~22 MB/s to dkrz measured 2026-09):
    # dkrz first, then IPSL vesg, then everything else.
    ordered = (
        [u for u in urls if "dkrz.de" in u]
        + [u for u in urls if "vesg.ipsl" in u]
        + [u for u in urls if "dkrz.de" not in u and "vesg.ipsl" not in u]
    )
    return ordered


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
    Returns deduplicated URLs (one version, one grid per title), preferring
    the dkrz fast mirror then other replicas.
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
            "fields": "title,url,size,member_id,institution_id,version",
        }
        if member_filter:
            params["member_id"] = member_filter
        last_exc: Exception | None = None
        for endpoint in ESGF_SEARCHES:
            try:
                r = _session().get(endpoint, params=params, timeout=30)
                r.raise_for_status()
                docs = r.json()["response"]["docs"]
                if docs:
                    return docs
            except Exception as exc:
                last_exc = exc
        if last_exc:
            raise last_exc
        return []

    # The dkrz search index intermittently returns empty/failing result sets;
    # retry a few times before treating it as a genuinely missing dataset.
    docs: list[dict] = []
    for attempt in range(4):
        try:
            docs = _query(member)
        except Exception:
            docs = []
        if docs:
            break
        time.sleep(5 * (attempt + 1))
    if not docs:
        # Auto-detect: take first available member. If every mirror fails
        # (e.g. the dataset genuinely does not exist and mirrors return HTTP
        # errors rather than empty results), treat it as missing so callers
        # can classify it via ignore_missing.
        try:
            docs = _query(None)
        except Exception as exc:
            print(f"  WARNING: {source_id} {experiment} {variable}: "
                  f"no files found and all mirrors failed ({exc})")
            docs = []
        if docs:
            # Pick the lowest-numbered member deterministically (search order
            # is arbitrary) and filter to it below: leaving `member` at the
            # missing value made the member filter a no-op, so every member
            # got concatenated (CESM2 ssp245 pr came out as r4+r10+r11, 3x time).
            def _rank(m: str) -> tuple:
                return tuple(int(x) for x in re.findall(r"\d+", m))
            detected = min((d.get("member_id", ["?"])[0] for d in docs), key=_rank)
            print(f"  NOTE: {source_id} {experiment}: member {member} not found, "
                  f"using {detected}")
            member = detected

    # Collapse duplicate file docs to ONE version per title. Docs sharing the
    # same title can come from distinct published versions (e.g., CESM2 psl
    # exists at both v20240821 and v20241119); concatenating more than one
    # version of the same period doubles the time axis. Keep the newest.
    _recent = {}
    for doc in docs:
        title = doc["title"]
        v = doc.get("version")
        if isinstance(v, list):
            v = v[0] if v else ""
        v = str(v or "")
        vnum = int(re.sub(r"\D", "", v)[:8] or 0)
        if title not in _recent or vnum > _recent[title][1]:
            _recent[title] = (doc, vnum)
    docs = [d for d, _ in _recent.values()]

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
        dkrz = _dkrz_candidate(urls[0], title, experiment)
        if dkrz and not any("dkrz.de" in u for u in urls):
            print(f"  dkrz fast mirror: {Path(dkrz).name}")
            urls.insert(0, dkrz)
        if title not in relevant:
            relevant[title] = _mirror_sort_key(urls)
        else:
            # Merge additional mirrors from duplicate docs
            existing = set(relevant[title])
            relevant[title] += [u for u in urls if u not in existing]

    # Member filter (defence in depth): search may return several members
    # (esp. after auto-detect fallback); concatenating different members of the
    # same period doubles the time axis with incompatible values. When a
    # specific member was requested and present, keep only that member's titles.
    if member:
        keep = []
        for t in list(relevant):
            m = re.search(r"_(r\d+i\d+p\d+f\d+)_", t)
            if m and m.group(1) == member:
                keep.append(t)
        if keep:
            relevant = {t: relevant[t] for t in keep}

    # Deduplicate by grid family: keep only one grid (prefer 'gn') to avoid
    # contaminating the concatenated result with incompatible coordinate systems
    # (e.g., CESM2 publishes psl_day_* in both gn and gr grids, leading to
    # doubled time when both are concatenated).
    grids: dict[str, list[str]] = {}
    for t in list(relevant):
        m = re.search(r"_([a-z]{2})_", t)
        g = m.group(1) if m else "?"
        grids.setdefault(g, []).append(t)
    if len(grids) > 1:
        preferred = "gn" if "gn" in grids else next(iter(grids))
        relevant = {t: relevant[t] for t in grids[preferred]}

    return list(relevant.items())  # (title, [url, ...])


# Canonical dkrz THREDDS layout (verified 200 for CNRM-CM6-1/CESM2); the
# ESGF search index often omits dkrz replicas even though dkrz hosts them
# at ~22 MB/s (measured 2026-09), far faster than CEDA (~1.3 MB/s).
_DKRZ_URL = ("http://esgf3.dkrz.de/thredds/fileServer/cmip6/"
             "{activity}/{institute}/{source}/{experiment}/{member}/{table}/"
             "{variable}/{grid}/{version}/{title}")

# experiment_id -> CMIP6 DRS activity directory name on the server
_ACTIVITY_DIR = {"historical": "CMIP", "ssp370": "ScenarioMIP",
                 "ssp245": "ScenarioMIP", "ssp585": "ScenarioMIP"}


def _dkrz_candidate(url: str, title: str, experiment: str) -> str | None:
    """Build the canonical dkrz THREDDS URL for a file from an existing mirror URL.

    The ESGF search index often omits dkrz replicas even though dkrz hosts them
    at ~22 MB/s (measured vs CEDA ~1.3 MB/s). All path parts are derived from an
    already-returned mirror URL (member / table / version are in its THREDDS
    path), so it does not depend on the flaky `version`/`institution_id` doc fields.
    Returns None when the layout is unrecognised (the mirror fallback then applies).
    """
    m = re.search(r"/(r\d+i\d+p\d+f\d+)/([A-Za-z0-9]+)/[^/]+/[^/]+/(v\d+)/[^/]+/?" + r"[^/]*$", url)
    if not m:
        return None
    parts = title.split("_")
    if len(parts) < 3:
        return None
    source = parts[2]
    i = url.find(source)
    if i <= 0:
        return None
    institute = url[:i].rstrip("/").rsplit("/", 1)[-1]
    activity = _ACTIVITY_DIR.get(experiment)
    if not activity:
        return None
    g = re.search(r"_([a-z]{2})_", title)
    return _DKRZ_URL.format(
        activity=activity, institute=institute, source=source,
        experiment=experiment, member=m.group(1), table=m.group(2),
        variable=parts[0], grid=g.group(1) if g else "gn",
        version=m.group(3), title=title)


_NETCDF_MAGIC = (b"\x89HDF", b"CDF\x01", b"CDF\x02", b"CDF\x05")


def _download_file(url: str, dest: Path) -> None:
    """Stream-download url to dest (connect timeout 10 s: dead mirrors should
    fall through quickly)."""
    r = _session().get(url, stream=True, timeout=(10, 300))
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
            f.write(chunk)
    with open(dest, "rb") as f:
        magic = f.read(4)
    if not any(magic.startswith(m) for m in _NETCDF_MAGIC):
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"not a NetCDF file (server returned HTML — "
                           f"got {magic!r} from {url.split('/')[2]})")


def _download_stage(task: tuple) -> tuple | str:
    """Threaded stage: resolve file list and download all globals.

    netCDF4/HDF5 is not safe to run in multiple threads of one process
    (concurrent crops of large fields hard-kill the process), so downloads
    run in parallel but cropping is done serially on the main thread.
    Returns (task, tmp_dir) on success, or an error string.
    """
    (model, source_id, variable, experiment, start_year, end_year,
     member, area, target, keep_global) = task

    print(f"[get ] {model} {variable} {experiment} {start_year}-{end_year}")
    files = search_files(source_id, variable, experiment, member,
                         start_year, end_year)
    if not files:
        return (f"{model} {variable} {experiment}: no files found on ESGF "
                f"(source_id={source_id}, member={member})")

    tmp_base = target.parent.parent / "_tmp"
    tmp_base.mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=tmp_base))
    for title, urls in sorted(files):  # sorted by filename = chronological
        dest = tmp / title
        print(f"  downloading {title} ...")
        last_exc = None
        for url in urls:
            try:
                _download_file(url, dest)
                last_exc = None
                break
            except Exception as exc:
                print(f"  mirror failed ({url.split('/')[2]}): {exc}")
                dest.unlink(missing_ok=True)
                last_exc = exc
        if last_exc is not None:
            shutil.rmtree(tmp, ignore_errors=True)
            return f"{model} {variable} {experiment} {title}: all mirrors failed"
    if not any(tmp.glob("*.nc")):
        shutil.rmtree(tmp, ignore_errors=True)
        return f"{model} {variable} {experiment}: no files downloaded"
    return task, tmp


def _crop_stage(task: tuple, tmp: Path) -> str | None:
    """Serial stage (main thread): crop globals to the domain and save.

    Crops per-file BEFORE concatenating — a full-global per-decade concat
    materialises ~100 GB for variables like pr (65 yr × 1M cells) and
    OOM-kills the process; per-file cropping keeps memory ≈ crop size.
    """
    (model, source_id, variable, experiment, start_year, end_year,
     member, area, target, keep_global) = task
    north, west, south, east = area

    local_files = sorted(tmp.glob("*.nc"))
    # Defensive: multiple published versions of the same dataset can carry
    # different period suffixes in the title (1850-2014 vs 1960-2014), so fully
    # overlapping time axes would double on concat. Keep the widest file per
    # coverage and drop any file whose period lies entirely inside a kept one.
    def _period(f: Path) -> tuple[int, int] | None:
        m = re.search(r"_(\d{8})-(\d{8})\.nc$", f.name)
        if not m:
            return None
        return (int(m.group(1)), int(m.group(2)))

    covered: list[tuple[int, int]] = []
    uniq_files = []
    for f in sorted(local_files,
                     key=lambda p: -((_period(p) or (0, 0))[1] -
                                     (_period(p) or (0, 0))[0])):
        period = _period(f)
        if period and any(s <= period[0] and e >= period[1]
                          for s, e in covered):
            continue
        if period:
            covered.append(period)
        uniq_files.append(f)
    local_files = sorted(uniq_files)
    try:
        _coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        _open_kw: dict = dict(decode_times=_coder)
    except AttributeError:
        _open_kw = dict(use_cftime=True)

    lat_dim, lon_dim = "lat", "lon"
    with xr.open_dataset(local_files[0], **_open_kw) as probe:
        if "lat" in probe.dims:
            lat_dim = "lat"
        elif "latitude" in probe.dims:
            lat_dim = "latitude"
        if "lon" in probe.dims:
            lon_dim = "lon"
        elif "longitude" in probe.dims:
            lon_dim = "longitude"
    crops = []
    for f in local_files:
        try:
            d = xr.open_dataset(f, **_open_kw)
        except Exception as exc:
            return f"{model} {variable} {experiment}: failed to open downloaded file — {exc}"
        if variable not in d.data_vars:
            d.close()
            continue
        if len(d[lon_dim]) and float(d[lon_dim].values.max()) > 180:
            # 0-360 convention: convert domain bounds and roll to -180..180
            d = d.assign_coords({lon_dim: ((d[lon_dim] + 180) % 360) - 180})
            d = d.sortby(lon_dim)
        d = d.sel({
            lat_dim: slice(south, north), lon_dim: slice(west, east),
            "time": slice(f"{start_year}-01-01", f"{end_year}-12-31"),
        })
        if d.sizes[lat_dim] and d.sizes[lon_dim] and d.sizes["time"]:
            crops.append(d[[variable]])
        else:
            d.close()
    if not crops:
        return f"{model} {variable} {experiment}: crop empty over {area}"

    ds = xr.concat(crops, dim="time")
    for d in crops:
        d.close()

    if keep_global:
        global_dir = target.parent / "global"
        global_dir.mkdir(exist_ok=True)
        for f in local_files:
            f.rename(global_dir / f.name)

    tmp_out = target.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp_out)
    tmp_out.rename(target)
    print(f"[done] {target}")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="src/configs/data/cmip6.yaml")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--area", nargs=4, type=float, default=None,
                        metavar=("N", "W", "S", "E"),
                        help="Override crop area [N, W, S, E] for all models")
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

    # Scenarios: accept `scenarios` (list) or `scenario` (str) in the config.
    scenarios = cfg.get("scenarios", cfg.get("scenario", "ssp370"))
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    spans = experiment_spans(cfg["years"]["start"], cfg["years"]["end"], scenarios)

    # Known GCM gaps (e.g. "sfcWind:ssp370") downgraded to warnings.
    ignored: set[tuple[str, str, str] | tuple[str, str]] = set()
    for entry in cfg.get("ignore_missing", []):
        parts = entry.split(":")
        if len(parts) == 3:
            ignored.add((parts[0], parts[1], parts[2]))
        elif len(parts) == 2:
            ignored.add((parts[0], parts[1]))

    # Member IDs: config > DEFAULT_MEMBERS
    cfg_members = cfg.get("members", {})
    cfg_areas = cfg.get("areas", {})

    tasks = []
    skipped = 0
    for model in models:
        if model not in SOURCE_IDS:
            print(f"WARNING: unknown model {model}, skipping")
            continue
        source_id = SOURCE_IDS[model]
        member = cfg_members.get(model, DEFAULT_MEMBERS.get(model, "r1i1p1f1"))
        area = args.area or cfg_areas.get(model, cfg["area"])
        model_dir = output_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        for variable in variables:
            for experiment, start, end in spans:
                target = model_dir / f"{variable}_{experiment}_{start}-{end}.nc"
                if target.exists():
                    skipped += 1
                    continue
                tasks.append((model, source_id, variable, experiment,
                              start, end, member, area,
                              target, args.keep_global))

    if skipped:
        print(f"{skipped} file(s) already present, skipped")
    if not tasks:
        print("Nothing to download.")
        return 0
    print(f"{len(tasks)} task(s) with {args.jobs} parallel job(s)\n")

    all_failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        for result in executor.map(_download_stage, tasks):
            if isinstance(result, str):
                all_failures.append(result)
                continue
            task, tmp = result
            try:
                err = _crop_stage(task, tmp)
                if err:
                    all_failures.append(err)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    warnings, failures = [], []
    for err in all_failures:
        key = tuple(err.split(":")[0].split())  # (model, variable, experiment)
        if key[:3] in ignored or key[1:3] in ignored:
            warnings.append(err)
        else:
            failures.append(err)

    if warnings:
        print(f"\n{len(warnings)} expected missing (ignored):")
        for line in warnings:
            print(f"  {line}")
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
