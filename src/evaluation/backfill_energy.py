#!/usr/bin/env python3
"""Backfill day-of-year cycles and energy-event stats into existing results.

`plot_data.json` normally gets both artefacts from evaluate.py:

  * `plot_data[var]["doy"]` — 365-value daily climatology of the region mean
    and of each configured energy site (ERA5 / GCM / prediction), used by the
    dashboard's *Line plots → doy* view;
  * `plot_data["_events"]`  — Dunkelflaute and cold-and-calm spell counts and
    mean durations (ERA5 / GCM / prediction, shared ERA5 thresholds), used by
    the *Inter-variable → Energy events* bar charts.

For results generated before this feature existed, recompute only those two
artefacts, reusing:

  * local `predictions.pt` when present, or an inline registry method
    (bicubic / cdft) otherwise;
  * the source GCM (bicubically regridded onto the ERA5 grid and re-stamped
    with the ERA5 time axis, since bicubic_regrid keeps the GCM's cftime axis).

Usage:
    python src/evaluation/backfill_energy.py                        # all regions
    python src/evaluation/backfill_energy.py --region australia
    python src/evaluation/backfill_energy.py --method cdft
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (data/, results/)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for data_processing / energy_views

from data_processing import load_era5, load_gcm, bicubic_regrid, align_time, load_predictions
from baselines import METHOD_REGISTRY
from energy_views import (
    build_doy_block, domain_series, energy_events, location_cycles, resolve_sites,
)

VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl", "rsds"]
EVENT_VARS = ("sfcWind", "rsds", "tas")
START, END = 2000, 2020

# Inline fallback is only safe for stateless methods (bicubic, cdft, dotc). The rest
# need calibration data / SBCK / checkpoints — without a local predictions.pt
# they must come from a cluster re-run of evaluate.py.
INLINE_OK = {"bicubic", "cdft", "dotc"}

# Factorial couplings — (region, gcm_key, scenario, gcm_dir_rel). Mirrors
# tools/gen_factorial_jobs.py. ceSM2 couples use 5 vars (no sfcWindmax); the
# others use 6 (ENERGY soon uses EVENT_VARIABLES which never needs sfcWindmax).
FACTORIAL_COUPLINGS = [
    ("france",       "cesm2",       "ssp245",
     "cmip6/_factorial/cmip6_cesm2_r4/france/cesm2"),
    ("france",       "cnrm_cm6_1",  "ssp370",
     "cmip6/_factorial/cmip6_factorial/france/cnrm_cm6_1"),
    ("australia",    "ipsl_cm6a_lr","ssp370",
     "cmip6/_factorial/cmip6_factorial/australia/ipsl_cm6a_lr"),
    ("australia",    "cesm2",       "ssp245",
     "cmip6/_factorial/cmip6_cesm2_r4/australia/cesm2"),
    ("us_central",   "ipsl_cm6a_lr","ssp370",
     "cmip6/_factorial/cmip6_factorial/us_central/ipsl_cm6a_lr"),
    ("us_central",   "cnrm_cm6_1",  "ssp370",
     "cmip6/_factorial/cmip6_factorial/us_central/cnrm_cm6_1"),
]


def _pick_era5_dir(region: str) -> Path:
    """ERA5 root differs per cluster: try jz layout, cleps layout, flat."""
    for cand in (ROOT / "data" / "era5" / region,
                 ROOT / "data" / "era5_orig" / region,
                 ROOT / "data" / "era5"):
        if cand.is_dir():
            return cand
    raise RuntimeError(f"{region}: no ERA5 dir found (data/era5/..., "
                       f"data/era5_orig/..., data/era5)")


def _align_pair(a: xr.DataArray, b: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    aligned = align_time({"a": a, "b": b})
    return aligned["a"], aligned["b"]


def _regions() -> dict[str, dict]:
    """region → {era5_dir, gcm_dir, gcm, scenario} from regions.yaml."""
    path = ROOT / "src" / "configs" / "data" / "regions.yaml"
    return (yaml.safe_load(path.read_text()) or {}).get("regions", {})


def _dump_plot_data(obj: dict, path: Path) -> None:
    """Compact JSON — these files are machine-written and size-sensitive."""
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))


def _resolve_preds(method: str, out_dir: Path):
    """('preds', dict) | ('inline', None) | (None, reason-skip)."""
    if (out_dir / "predictions.pt").exists():
        try:
            return "preds", load_predictions(out_dir, VARIABLES, mmap=True)
        except Exception as e:
            print(f"    predictions.pt unusable ({e}); trying inline")
    if method in INLINE_OK and method in METHOD_REGISTRY:
        return "inline", None
    return None, "no predictions.pt and inline not allowed"


def load_region_refs(region: str, cfg: dict) -> dict:
    """Native layout loader (results/{region}/{method}) — one GCM per region."""
    return _load_refs(
        region,
        _pick_era5_dir(region),
        ROOT / cfg["gcm_dir"] / cfg["gcm"],
        cfg.get("scenario", "ssp370"),
    )


def load_coupling_refs(region: str, coupling: tuple) -> dict:
    """Factorial layout loader (results/{region}/{gcm}/{method})."""
    _, _gcm_key, scenario, gcm_rel = coupling
    return _load_refs(region, _pick_era5_dir(region),
                      ROOT / "data" / gcm_rel, scenario)


def _load_refs(region: str, era5_dir: Path, gcm_dir: Path, scenario: str) -> dict:
    """Load ERA5 + GCM once per region/coupling; compute cycles and series.

    Returns {"sites", "era5", "gcm_up", "gcm_nat", "doy_era5", "doy_gcm",
    "dom_era5", "dom_gcm"} keyed by variable where applicable.
    """
    refs: dict = {"sites": None, "era5": {}, "gcm_up": {}, "gcm_nat": {},
                  "doy_era5": {}, "doy_gcm": {}, "dom_era5": {}, "dom_gcm": {}}
    for variable in VARIABLES:
        try:
            obs = load_era5(era5_dir, variable, START, END)
        except FileNotFoundError as e:
            print(f"  SKIP {region}/{variable}: ERA5 {e}")
            continue
        try:
            gcm_nat = load_gcm(gcm_dir, variable, START, END, scenario)
        except FileNotFoundError as e:
            print(f"  SKIP {region}/{variable}: GCM {e}")
            continue

        obs, gcm_nat = _align_pair(obs, gcm_nat)
        gcm_up = bicubic_regrid(gcm_nat, obs.lat.values, obs.lon.values)
        gcm_up = gcm_up.assign_coords(time=obs.time)

        if refs["sites"] is None:
            refs["sites"] = resolve_sites(region, obs.lat.values, obs.lon.values)
            print(f"  sites: {len(refs['sites'])} in-domain "
                  f"({', '.join(s['name'] for s in refs['sites']) or 'none'})")

        refs["era5"][variable] = obs
        refs["gcm_up"][variable] = gcm_up
        refs["gcm_nat"][variable] = gcm_nat
        refs["doy_era5"][variable] = location_cycles(obs, refs["sites"])
        refs["doy_gcm"][variable] = location_cycles(gcm_up, refs["sites"])
        refs["dom_era5"][variable] = domain_series(obs)
        refs["dom_gcm"][variable] = domain_series(gcm_up)
    if not refs["era5"]:
        raise RuntimeError(f"{region}: no ERA5/GCM data available")
    if refs["sites"] is None:
        raise RuntimeError(f"{region}: no grid loaded")
    return refs


def backfill_method(label: str, method: str, out_dir: Path, refs: dict) -> str:
    """Compute doy + events for one method and merge them into plot_data.json."""
    mode, payload = _resolve_preds(method, out_dir)
    if mode is None:
        return f"skip ({payload})"
    preds = payload if mode == "preds" else None

    sites = refs["sites"]
    doy_blocks: dict[str, dict] = {}
    domains: dict[str, dict] = {}
    for variable in VARIABLES:
        if variable not in refs["era5"]:
            continue
        obs = refs["era5"][variable]
        gcm_up = refs["gcm_up"].get(variable)

        if preds is not None:
            if variable not in preds:
                print(f"    no prediction for {variable}")
                continue
            _, pred = _align_pair(obs, preds[variable])
        else:
            try:
                pred = METHOD_REGISTRY[method](refs["gcm_nat"][variable], obs, variable)
            except Exception as e:
                print(f"    inline {method} failed for {variable}: {e}")
                continue

        block = build_doy_block(obs, gcm_up, pred, sites)
        if block:
            doy_blocks[variable] = block
        domains[variable] = {
            "era5": refs["dom_era5"][variable],
            "gcm": refs["dom_gcm"].get(variable),
            "pred": domain_series(pred),
        }

    events = None
    if all(v in domains for v in EVENT_VARS):
        events = energy_events(domains, period=(START, END))
    else:
        missing = [v for v in EVENT_VARS if v not in domains]
        if domains:
            print(f"    events skipped, missing {missing}")

    if not doy_blocks and events is None:
        return "no data (predictions unavailable)"

    pf = out_dir / "plot_data.json"
    plot = {}
    if pf.exists():
        with open(pf) as f:
            plot = json.load(f)
    for variable, block in doy_blocks.items():
        plot.setdefault(variable, {})["doy"] = block
    if events is not None:
        plot["_events"] = events
    _dump_plot_data(plot, pf)

    n_sites = len(next(iter(doy_blocks.values()))) if doy_blocks else 0
    counts = events["dunkelflaute"]["count"] if events else None
    return (f"ok — doy for {len(doy_blocks)}/{len(VARIABLES)} vars "
            f"({n_sites} locations), dunkelflaute era5/gcm/pred = "
            f"{_fmt_triple(counts)}")


def _fmt_triple(d: dict | None) -> str:
    if not d:
        return "n/a"
    return "/".join(str(d.get(k, "-")) for k in ("era5", "gcm", "pred"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", default=None,
                    help="One of france, australia, us_central (default: all)")
    ap.add_argument("--method", default=None, help="Limit to one method name")
    ap.add_argument("--factorial", action="store_true",
                    help="Also backfill factorial couplings "
                         "(results/{region}/{gcm}/{method})")
    ap.add_argument("--variables", nargs="*", default=VARIABLES)
    args = ap.parse_args()

    # CDF-t / DOTC inline needs SBCK — drop it from the inline list when absent
    # instead of failing once per variable.
    global INLINE_OK
    if "cdft" in INLINE_OK and importlib.util.find_spec("sbck") is None:
        INLINE_OK = INLINE_OK - {"cdft"}
        print("note: sbck not installed → inline cdft skipped")
    if "dotc" in INLINE_OK and importlib.util.find_spec("sbck") is None:
        INLINE_OK = INLINE_OK - {"dotc"}
        print("note: sbck not installed → inline dotc skipped")

    regions = _regions()
    todo = [args.region] if args.region else list(regions)
    for region in todo:
        if region not in regions:
            print(f"Unknown region {region}; choose from {list(regions)}")
            return 1
        cfg = regions[region]

        # Scope: native layout (results/{region}/{method}) and/or each factorial
        # coupling (results/{region}/{gcm}/{method}) with its own GCM + scenario.
        if args.factorial:
            couplings = [c for c in FACTORIAL_COUPLINGS if c[0] == region]
            scopes: list[tuple | None] = couplings or [None]
        else:
            scopes = [None]
        for c in scopes:
            if c is None:
                label, gcm_desc = region, f"gcm: {cfg['gcm']}, scenario: {cfg.get('scenario')}"
                coupling_dir = ROOT / "results" / region
                try:
                    refs = load_region_refs(region, cfg)
                except RuntimeError as e:
                    print(f"  SKIP {region}: {e}")
                    continue
            else:
                r, gcm_key, scenario, gcm_rel = c
                label = f"{r}/{gcm_key}"
                gcm_desc = f"gcm: {gcm_key}, scenario: {scenario}"
                coupling_dir = ROOT / "results" / r / gcm_key
                if not coupling_dir.is_dir():
                    print(f"  SKIP {label}: {coupling_dir} missing")
                    continue
                try:
                    refs = load_coupling_refs(r, c)
                except RuntimeError as e:
                    print(f"  SKIP {label}: {e}")
                    continue
            print(f"\n=== {label}  ({gcm_desc}) ===")
            if not refs["era5"]:
                print(f"  SKIP {label}: no ERA5/GCM data available")
                continue
            if args.variables != VARIABLES:
                # keep only the requested variables in the loaded refs
                for key in ("era5", "gcm_up", "gcm_nat", "doy_era5", "doy_gcm",
                            "dom_era5", "dom_gcm"):
                    refs[key] = {k: v for k, v in refs[key].items() if k in args.variables}

            methods = [args.method] if args.method else sorted(
                d.name for d in coupling_dir.iterdir()
                if d.is_dir() and (d / "metrics.json").exists())
            if not methods:
                print(f"  {label}: no methods with metrics.json")
            for m in methods:
                out_dir = coupling_dir / m
                print(f"  {label}/{m}: ", end="", flush=True)
                try:
                    status = backfill_method(label, m, out_dir, refs)
                except Exception as e:
                    status = f"ERROR {type(e).__name__}: {e}"
                print(status)
            for k in ("era5", "gcm_up"):
                refs[k].clear()
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
