#!/usr/bin/env python3
"""Consistency model inference — stroke-guidance downscaling.

Loads the checkpoint from train_consistency.py and applies stroke guidance
to downscale GCM fields to the ERA5 grid for the full eval period (2000-2100).

Stroke guidance: add Gaussian noise at stroke_time to the upsampled GCM,
then apply a single consistency-model denoising step (one-step generation).

Output is saved in the standard predictions.pt format.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from utils import (
    build_unet, _n_in, encode, decode,
    neural_predict, month_channels, EMA,
)
from data_processing import (
    load_era5, load_gcm, bicubic_regrid, align_time, save_predictions, qdm_correct,
)

VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS    = len(VARIABLES)
MODE      = "consistency"
ARCH      = "diffusers"
PHYS_MIN  = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0}


def _load_model(ckpt_path: Path):
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    n_in  = ckpt.get("in_channels", _n_in(MODE, N_VARS))
    arch  = ckpt.get("arch", ARCH)
    model = build_unet(arch, n_in, N_VARS)
    if "ema" in ckpt:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
    else:
        model.load_state_dict(ckpt.get("model", ckpt.get("net", {})))
    return model, stats


def run(
    era5_dir:    Path,
    gcm_dir:     Path,
    ckpt_dir:    Path,
    out_dir:     Path,
    calib_start: int   = 1980,
    eval_start:  int   = 2000,
    eval_end:    int   = 2100,
    scenario:    str   = "ssp585",
    stroke_time: float = 0.5,
    n_members:   int   = 1,
    batch_size:  int   = 32,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run train_consistency.py first.")

    model, stats = _load_model(ckpt_path)

    ref        = load_era5(era5_dir, VARIABLES[0], calib_start, calib_start)
    target_lat = ref.lat.values
    target_lon = ref.lon.values

    print(f"  [CM] loading ERA5+GCM calibration {calib_start}–{calib_start+19} ...", flush=True)
    era5_calib = {v: load_era5(era5_dir, v, calib_start, calib_start + 19)
                  for v in VARIABLES}
    gcm_calib  = align_time({v: load_gcm(gcm_dir, v, calib_start, calib_start + 19, "historical")
                              for v in VARIABLES})
    # Align ERA5 and GCM calibration to common dates
    _all = {f"e_{v}": era5_calib[v] for v in VARIABLES}
    _all.update({f"g_{v}": gcm_calib[v] for v in VARIABLES})
    _aligned  = align_time(_all)
    era5_cal  = {v: _aligned[f"e_{v}"] for v in VARIABLES}
    gcm_cal   = {v: _aligned[f"g_{v}"] for v in VARIABLES}

    print(f"  [CM] loading GCM {eval_start}–{eval_end} ...", flush=True)
    gcm_eval = align_time({v: load_gcm(gcm_dir, v, eval_start, eval_end, scenario)
                            for v in VARIABLES})
    gcm_time = next(iter(gcm_eval.values())).time
    T, H, W  = gcm_time.size, len(target_lat), len(target_lon)

    print("  [CM] bicubic regrid + QDM + normalize ...", flush=True)
    x_in_norm = np.empty((T, N_VARS, H, W), dtype=np.float32)
    for i, var in enumerate(VARIABLES):
        # 1. Bicubic regrid to ERA5 grid (calibration GCM and eval GCM)
        gcm_cal_up  = bicubic_regrid(gcm_cal[var],  target_lat, target_lon)
        era5_cal_up = era5_cal[var]   # already at ERA5 grid
        gcm_eval_up = bicubic_regrid(gcm_eval[var], target_lat, target_lon)
        # 2. QDM at ERA5 grid resolution (per cell, per calendar month)
        gcm_qdm     = qdm_correct(gcm_eval_up, gcm_cal_up, era5_cal_up, var)
        x_in_norm[:, i] = encode(gcm_qdm.values.astype(np.float32), stats, var)

    months = gcm_time.dt.month.values.astype(np.float32)
    mc     = month_channels(months, (T, H, W))

    print(f"  [CM] stroke guidance (t*={stroke_time}, members={n_members}) ...", flush=True)
    pred_norm = neural_predict(MODE, model, x_in_norm, mc, device,
                               stroke_time=stroke_time, n_members=n_members,
                               n_vars=N_VARS, batch_size=batch_size)

    arrays = []
    for i, var in enumerate(VARIABLES):
        arr = decode(pred_norm[:, i], stats, var)
        if var in PHYS_MIN:
            arr = np.clip(arr, PHYS_MIN[var], None)
        arrays.append(torch.tensor(arr, dtype=torch.float32))

    ref_da = xr.DataArray(
        dims=["time", "lat", "lon"],
        coords={"time": gcm_time, "lat": target_lat, "lon": target_lon},
    )
    save_predictions(torch.stack(arrays, dim=1), VARIABLES, out_dir, ref_da)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",       required=True)
    p.add_argument("--gcm-dir",        required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--calib-start",    type=int, default=1980)
    p.add_argument("--eval-start",     type=int, default=2000)
    p.add_argument("--eval-end",       type=int, default=2100)
    p.add_argument("--scenario",       default="ssp585")
    p.add_argument("--stroke-time",    type=float, default=1.0)
    p.add_argument("--n-members",      type=int,   default=1)
    p.add_argument("--batch-size",     type=int,   default=32)
    args = p.parse_args()
    run(
        era5_dir    = Path(args.era5_dir),
        gcm_dir     = Path(args.gcm_dir),
        ckpt_dir    = Path(args.checkpoint_dir),
        out_dir     = Path(args.out_dir),
        calib_start = args.calib_start,
        eval_start  = args.eval_start,
        eval_end    = args.eval_end,
        scenario    = args.scenario,
        stroke_time = args.stroke_time,
        n_members   = args.n_members,
        batch_size  = args.batch_size,
    )


if __name__ == "__main__":
    main()
