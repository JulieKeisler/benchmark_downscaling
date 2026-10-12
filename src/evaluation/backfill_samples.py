#!/usr/bin/env python3
"""Backfill `samples_data_{variable}.npz` for existing results.

The samples npz (dashboard "Sample days" plot) is normally written by
evaluate.py. For results generated before that feature existed, this script
regenerates exactly that one artefact per variable, reusing:

  * local `predictions.pt` when present, or an inline registry method
    (bicubic / cdft) otherwise;
  * the source GCM (regridded bicubically onto the ERA5 grid, re-stamped with
    the ERA5 time axis — AGENTS.md gotcha #1).

Usage:
    python src/evaluation/backfill_samples.py                      # all regions
    python src/evaluation/backfill_samples.py --region australia
    python src/evaluation/backfill_samples.py --method cdft
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root (data/, results/)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for data_processing

from data_processing import load_era5, load_gcm, bicubic_regrid, align_time, load_predictions
from baselines import METHOD_REGISTRY
from plots.spatial_plots import plot_sample_days

REGIONS = {
    "france":     ("data/era5",               "data/cmip6/ipsl_cm6a_lr"),
    "australia":  ("data/era5/australia",     "data/cmip6/australia/cnrm_cm6_1"),
    "us_central": ("data/era5/us_central",    "data/cmip6/us_central/cesm2"),
}
VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
START, END = 2000, 2020

# Cluster-only methods (no local predictions.pt) and the corrdiff stub (raises):
SKIP = {"corrdiff_qm", "cm", "SerpentflowClim", "serpentflowrml_eval",
        "bicubic_eval"}

# Inline fallback is only safe for stateless methods (bicubic, cdft). bcsd/dotc
# need calibration data / SBCK; without a local predictions.pt their samples
# must come from the cluster re-run of evaluate.py.
INLINE_OK = {"bicubic", "cdft"}


def _align_pair(obs: xr.DataArray, gcm: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    aligned = align_time({"obs": obs, "gcm": gcm})
    return aligned["obs"], aligned["gcm"]


def _era5_dir(region: str) -> Path:
    return ROOT / REGIONS[region][0]


def _gcm_dir(region: str) -> Path:
    return ROOT / REGIONS[region][1]


def backfill_region(region: str, era5_dir: Path, gcm_dir: Path, method: str, label: str) -> None:
    results_root = ROOT / "results" / region
    out_dir      = results_root / method
    plot_dir     = out_dir / "plots"
    pred_dir     = out_dir

    if not plot_dir.exists():
        print(f"  SKIP {region}/{method}: no plots dir")
        return

    precomputed: dict[str, xr.DataArray] = {}
    use_inline = False
    if (pred_dir / "predictions.pt").exists():
        print(f"  {region}/{method}: loading predictions.pt ...")
        try:
            precomputed = load_predictions(pred_dir, VARIABLES)
        except Exception as e:
            print(f"  {region}/{method}: predictions.pt unusable ({e}); "
                  f"falling back to inline registry")
            use_inline = True
    elif method in METHOD_REGISTRY:
        use_inline = True
    else:
        print(f"  SKIP {region}/{method}: no predictions.pt and not inline")
        return
    if use_inline and method not in INLINE_OK:
        print(f"  SKIP {region}/{method}: no predictions.pt and "
              f"inline not allowed for {method}")
        return

    for variable in VARIABLES:
        var_plot_dir = plot_dir / variable
        if not var_plot_dir.exists():
            continue
        try:
            obs = load_era5(era5_dir, variable, START, END)
        except FileNotFoundError as e:
            print(f"  SKIP {region}/{method}/{variable}: ERA5 {e}")
            continue
        try:
            gcm_era5 = load_gcm(gcm_dir, variable, START, END)
        except FileNotFoundError as e:
            print(f"  SKIP {region}/{method}/{variable}: GCM {e}")
            continue

        # GCM on the ERA5 grid with the ERA5 time axis (sample-days reference)
        obs, gcm_era5 = _align_pair(obs, gcm_era5)
        gcm_on_era5 = bicubic_regrid(gcm_era5, obs.lat.values, obs.lon.values)
        gcm_on_era5 = gcm_on_era5.assign_coords(time=obs.time)

        if precomputed and not use_inline:
            if variable not in precomputed:
                print(f"  SKIP {region}/{method}/{variable}: not in predictions.pt")
                continue
            _, pred_obs = _align_pair(obs, precomputed[variable])
        else:
            pred_obs = METHOD_REGISTRY[method](gcm_era5, obs, variable)

        plot_sample_days(obs, pred_obs, gcm_on_era5, variable, label, var_plot_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=None,
                    help="One of france, australia, us_central (default: all)")
    ap.add_argument("--method", default=None, help="Limit to one method name")
    ap.add_argument("--label", default=None,
                    help="Display label recorded in nothing (informational)")
    args = ap.parse_args()

    regions = [args.region] if args.region else list(REGIONS)
    for region in regions:
        if region not in REGIONS:
            print(f"Unknown region {region}; choose from {list(REGIONS)}")
            return 1
        era5_dir, gcm_dir = _era5_dir(region), _gcm_dir(region)
        print(f"\n=== {region}  (era5: {era5_dir}, gcm: {gcm_dir}) ===")
        results_root = ROOT / "results" / region
        methods = [args.method] if args.method else sorted(
            d.name for d in results_root.iterdir() if d.is_dir())
        for m in methods:
            if m in SKIP:
                continue
            label = args.label or m
            backfill_region(region, era5_dir, gcm_dir, m, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())