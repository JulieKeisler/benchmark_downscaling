#!/usr/bin/env python3
"""iDownscale — CERFACS perfect-prognosis U-Net downscaling.

Train on: ERA5 coarsened (4× subsample → bicubic up) → ERA5 fine.
Infer on: GCM (after coarse-scale CDF-t bias correction) → fine ERA5 grid.

Architecture: plain UNet(7→5), deterministic regression. No t conditioning.
This matches the original idownscale / CERFACS approach (Soares et al. 2024).
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
    build_unet, _n_in, encode, decode, compute_stats,
    ERA5DownscalingDataset, train_neural_model, neural_predict,
    month_channels,
)
from inference_cdft import _cdft_field_sliding
from data_processing import (
    load_era5, load_gcm, bicubic_regrid, align_time, save_predictions,
)

VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS    = len(VARIABLES)
MODE      = "idownscale"
ARCH      = "unet"      # plain U-Net — faithful to CERFACS iDownscale paper
STRIDE    = 4
PHYS_MIN  = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0}


def _load_era5_raw(era5_dir, calib_start, calib_end):
    das     = {v: load_era5(era5_dir, v, calib_start, calib_end) for v in VARIABLES}
    aligned = align_time(das)
    ref     = next(iter(aligned.values()))
    months  = ref.time.dt.month.values.astype(np.float32)
    raw     = np.stack([aligned[v].values.astype(np.float32) for v in VARIABLES], axis=1)
    return raw, months, ref.lat.values, ref.lon.values


def _cdft_coarse(gcm_eval, gcm_calib, era5_calib_da, variable, coarse_lat, coarse_lon):
    """CDF-t at coarse ERA5 grid to remove GCM bias before the U-Net pass.

    Uses 20-10 sliding windows matching the iDownscale inference run."""
    gcm_e_c = bicubic_regrid(gcm_eval,  coarse_lat, coarse_lon)
    gcm_h_c = bicubic_regrid(gcm_calib, coarse_lat, coarse_lon)
    era5_c  = era5_calib_da.interp(lat=coarse_lat, lon=coarse_lon, method="linear")
    era5_c  = era5_c.assign_coords(time=gcm_h_c.time)
    return _cdft_field_sliding(gcm_h_c, era5_c, gcm_e_c, variable)


def run(
    out_dir:     Path,
    era5_dir:    Path | None = None,
    gcm_dir:     Path | None = None,
    calib_start: int  = 1980,
    calib_end:   int  = 1999,
    eval_start:  int  = 2000,
    eval_end:    int  = 2100,
    scenario:    str  = "ssp370",
    do_train:    bool = True,
    do_predict:  bool = True,
    epochs:      int  = 30,
    batch_size:  int  = 32,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "idownscale.pt"
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Training ──────────────────────────────────────────────────────────────
    if do_train and not ckpt_path.exists():
        if era5_dir is None:
            raise ValueError("--era5-dir required for training")
        raw, months, lat, lon = _load_era5_raw(era5_dir, calib_start, calib_end)
        stats   = compute_stats(raw, VARIABLES)
        dataset = ERA5DownscalingDataset(raw, months, stats, VARIABLES,
                                         mode=MODE, stride=STRIDE)
        model   = build_unet(ARCH, _n_in(MODE, N_VARS), N_VARS)
        train_neural_model(MODE, dataset, model, ckpt_path,
                           n_epochs=epochs, batch_size=batch_size, device=device)
        # Augment checkpoint with metadata
        obj = torch.load(ckpt_path, weights_only=False)
        obj.update({"stats": stats, "variables": VARIABLES,
                    "mode": MODE, "arch": ARCH,
                    "in_channels": _n_in(MODE, N_VARS), "out_channels": N_VARS,
                    "stride": STRIDE})
        torch.save(obj, ckpt_path)
    elif do_train:
        print(f"  [iDownscale] skipping train — checkpoint exists: {ckpt_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    if do_predict:
        if gcm_dir is None or era5_dir is None:
            raise ValueError("--era5-dir and --gcm-dir required for inference")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run --train first.")

        ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stats  = ckpt["stats"]
        stride = ckpt.get("stride", STRIDE)
        model  = build_unet(ckpt.get("arch", ARCH),
                            ckpt.get("in_channels", _n_in(MODE, N_VARS)), N_VARS)
        model.load_state_dict(ckpt["model"])

        ref        = load_era5(era5_dir, VARIABLES[0], calib_start, calib_start)
        target_lat = ref.lat.values
        target_lon = ref.lon.values
        coarse_lat = target_lat[::stride]
        coarse_lon = target_lon[::stride]

        print(f"  [iDownscale] loading ERA5+GCM calibration {calib_start}–{calib_end} ...", flush=True)
        era5_calib = align_time({v: load_era5(era5_dir, v, calib_start, calib_end) for v in VARIABLES})
        gcm_calib  = align_time({v: load_gcm(gcm_dir, v, calib_start, calib_end, "historical") for v in VARIABLES})
        gcm_eval   = align_time({v: load_gcm(gcm_dir, v, eval_start,  eval_end,   scenario)    for v in VARIABLES})

        # Align ERA5 and GCM calib to same time axis
        all_c    = {f"e_{v}": era5_calib[v] for v in VARIABLES}
        all_c.update({f"g_{v}": gcm_calib[v] for v in VARIABLES})
        aligned  = align_time(all_c)
        era5_cal = {v: aligned[f"e_{v}"] for v in VARIABLES}
        gcm_cal  = {v: aligned[f"g_{v}"] for v in VARIABLES}

        gcm_time = next(iter(gcm_eval.values())).time
        T, H, W  = gcm_time.size, len(target_lat), len(target_lon)

        print("  [iDownscale] CDF-t coarse correction + normalize ...", flush=True)
        x_in_norm = np.empty((T, N_VARS, H, W), dtype=np.float32)
        for i, var in enumerate(VARIABLES):
            bc_coarse = _cdft_coarse(gcm_eval[var], gcm_cal[var], era5_cal[var],
                                     var, coarse_lat, coarse_lon)
            bc_up     = bicubic_regrid(bc_coarse, target_lat, target_lon)
            x_in_norm[:, i] = encode(bc_up.values.astype(np.float32), stats, var)

        months   = gcm_time.dt.month.values.astype(np.float32)
        mc       = month_channels(months, (T, H, W))

        print("  [iDownscale] running inference ...", flush=True)
        pred_norm = neural_predict(MODE, model, x_in_norm, mc, device)

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
    p.add_argument("--era5-dir",    default=None)
    p.add_argument("--gcm-dir",     default=None)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--calib-start", type=int, default=1980)
    p.add_argument("--calib-end",   type=int, default=1999)
    p.add_argument("--eval-start",  type=int, default=2000)
    p.add_argument("--eval-end",    type=int, default=2100)
    p.add_argument("--scenario",    default="ssp370")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--train",       action="store_true")
    p.add_argument("--predict",     action="store_true")
    args = p.parse_args()
    do_train   = args.train   or (not args.train and not args.predict)
    do_predict = args.predict or (not args.train and not args.predict)
    run(
        out_dir=Path(args.out_dir),
        era5_dir=Path(args.era5_dir) if args.era5_dir else None,
        gcm_dir=Path(args.gcm_dir)   if args.gcm_dir  else None,
        calib_start=args.calib_start, calib_end=args.calib_end,
        eval_start=args.eval_start,   eval_end=args.eval_end,
        scenario=args.scenario, do_train=do_train, do_predict=do_predict,
        epochs=args.epochs, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
