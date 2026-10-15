#!/usr/bin/env python3
"""Bicubic baseline inference — cubic convolution (Keys) interpolation to the ERA5 grid.

No training required. Produces predictions.pt in physical units covering the
full GCM evaluation period (default 2000–2100).

Usage:
    python src/baselines/inference_bicubic.py \
        --era5-dir /scratch/jkeisler/benchmark_downscaling/era5 \
        --gcm-dir  /scratch/jkeisler/benchmark_downscaling/cmip6/ipsl_cm6a_lr \
        --out-dir  results/bicubic \
        --eval-start 2000 --eval-end 2100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from data_processing import (
    DEFAULT_VARIABLES, load_era5, load_gcm, bicubic_regrid, save_predictions,
)


def run(era5_dir: Path, gcm_dir: Path, out_dir: Path,
        variables: list[str], eval_start: int, eval_end: int,
        scenario: str = "ssp370") -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    # ERA5 needed only for target lat/lon — load one year
    era5_ref = load_era5(era5_dir, variables[0], 2000, 2000)
    target_lat = era5_ref.lat.values
    target_lon = era5_ref.lon.values
    print(f"  Target grid: {len(target_lat)}×{len(target_lon)}", flush=True)

    arrays = []
    ref_da = None
    for var in variables:
        print(f"  {var}: loading GCM {eval_start}–{eval_end} ...", flush=True)
        gcm = load_gcm(gcm_dir, var, eval_start, eval_end, scenario)
        pred = bicubic_regrid(gcm, target_lat, target_lon)
        if var == "pr":
            pred.values[:] = np.clip(pred.values, 0.0, None)
        if var == "rsds":
            pred.values[:] = np.clip(pred.values, 0.0, None)
        if ref_da is None:
            ref_da = pred
        arrays.append(torch.tensor(pred.values, dtype=torch.float32))
        print(f"  {var}: done — {dict(pred.sizes)}", flush=True)

    tensor = torch.stack(arrays, dim=1)
    save_predictions(tensor, variables, out_dir, ref_da)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",    required=True)
    p.add_argument("--gcm-dir",     required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--variables",   nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--eval-start",  type=int, default=2000)
    p.add_argument("--eval-end",    type=int, default=2100)
    p.add_argument("--scenario",    default="ssp370")
    args = p.parse_args()
    run(Path(args.era5_dir), Path(args.gcm_dir), Path(args.out_dir),
        args.variables, args.eval_start, args.eval_end, args.scenario)


if __name__ == "__main__":
    main()
