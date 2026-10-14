#!/usr/bin/env python3
"""Consistency model inference — stroke-guidance downscaling.

Loads the checkpoint from train_consistency.py and applies stroke guidance
to downscale GCM fields to the ERA5 grid for the full eval period (2000-2100).

Stroke guidance: add Gaussian noise at stroke_time to the upsampled GCM,
then apply a single consistency-model denoising step (one-step generation).

t* is taken from the PSD sweep (src/evaluation/psd_sweep.py, --stroke-time-json)
when given; each run is one seeded draw.

Output is saved in the standard predictions.pt format.
"""
from __future__ import annotations

import argparse
import json
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
    load_era5, load_gcm, bicubic_regrid, align_time, save_predictions, qdm_windowed,
)
from inference_cdft import _sliding_windows

VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS    = len(VARIABLES)
MODE      = "consistency"
ARCH      = "diffusers"
PHYS_MIN  = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0, "rsds": 0.0}


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
    scenario:    str   = "ssp370",
    stroke_time: float = 0.5,
    n_members:   int   = 1,
    batch_size:  int   = 32,
    seed:        int   = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run train_consistency.py first.")

    model, stats = _load_model(ckpt_path)

    calib_end = calib_start + 19
    print(f"  [CM] loading ERA5+GCM calibration {calib_start}–{calib_end} ...", flush=True)
    _all = {}
    for v in VARIABLES:
        _all[f"e_{v}"] = load_era5(era5_dir, v, calib_start, calib_end)
        _all[f"g_{v}"] = load_gcm(gcm_dir, v, calib_start, calib_end, "historical")
    _aligned = align_time(_all)
    era5_cal = {v: _aligned[f"e_{v}"] for v in VARIABLES}
    gcm_cal  = {v: _aligned[f"g_{v}"] for v in VARIABLES}

    ref_v      = era5_cal[VARIABLES[0]]
    target_lat = ref_v.lat.values
    target_lon = ref_v.lon.values

    # QDM in 20-y moving windows (Cannon et al. 2015): the GCM is
    # read from eval_start - 5 so the first window (e.g. 1995-2014) is full.
    windows = _sliding_windows(eval_start, eval_end, 20, 10, 5)
    load_start = min(w[0] for w in windows)
    print(f"  [CM] loading GCM {load_start}–{eval_end} ...", flush=True)
    gcm_eval = align_time({v: load_gcm(gcm_dir, v, load_start, eval_end, scenario)
                            for v in VARIABLES})

    print("  [CM] bicubic regrid + windowed QDM + normalize ...", flush=True)
    x_in_norm, gcm_time = None, None
    for i, var in enumerate(VARIABLES):
        # 1. Bicubic regrid to ERA5 grid (calibration GCM and eval GCM)
        gcm_cal_up  = bicubic_regrid(gcm_cal[var],  target_lat, target_lon)
        era5_cal_up = era5_cal[var]   # already at ERA5 grid
        gcm_eval_up = bicubic_regrid(gcm_eval[var], target_lat, target_lon)
        # 2. QDM per cell, calendar month and window; keeps eval_start..eval_end
        gcm_qdm     = qdm_windowed(gcm_eval_up, gcm_cal_up, era5_cal_up, var, windows)
        if x_in_norm is None:
            gcm_time = gcm_qdm.time
            x_in_norm = np.empty((gcm_time.size, N_VARS, len(target_lat), len(target_lon)),
                                 dtype=np.float32)
        x_in_norm[:, i] = encode(gcm_qdm.values.astype(np.float32), stats, var)
    T, H, W = gcm_time.size, len(target_lat), len(target_lon)

    months = gcm_time.dt.month.values.astype(np.float32)
    mc     = month_channels(months, (T, H, W))

    print(f"  [CM] stroke guidance (t*={stroke_time:.4f}, members={n_members}, "
          f"seed={seed}) ...", flush=True)
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
    p.add_argument("--scenario",       default="ssp370")
    p.add_argument("--stroke-time",    type=float, default=1.0)
    p.add_argument("--stroke-time-json", default=None,
                   help="psd_sweep.py output; its t_star overrides --stroke-time")
    p.add_argument("--seed",           type=int,   default=0)
    p.add_argument("--n-members",      type=int,   default=1)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--variables",      nargs="+",  default=None,
                   help="Override variable set (e.g. drop sfcWindmax when the GCM lacks it).")
    args = p.parse_args()
    global VARIABLES, N_VARS  # noqa: PLW0603
    if args.variables:
        VARIABLES = list(args.variables)
        N_VARS    = len(VARIABLES)
    stroke_time = args.stroke_time
    if args.stroke_time_json:
        import hashlib  # noqa: PLC0415
        sweep = json.loads(Path(args.stroke_time_json).read_text())
        sha = hashlib.sha256((Path(args.checkpoint_dir) / "best.pt").read_bytes()).hexdigest()
        if sweep.get("checkpoint_sha256") != sha:
            raise SystemExit("t* was swept on another checkpoint (normalisation differs)")
        stroke_time = float(sweep["t_star"])
    run(
        era5_dir    = Path(args.era5_dir),
        gcm_dir     = Path(args.gcm_dir),
        ckpt_dir    = Path(args.checkpoint_dir),
        out_dir     = Path(args.out_dir),
        calib_start = args.calib_start,
        eval_start  = args.eval_start,
        eval_end    = args.eval_end,
        scenario    = args.scenario,
        stroke_time = stroke_time,
        n_members   = args.n_members,
        batch_size  = args.batch_size,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
