#!/usr/bin/env python3
"""SerpentFlow — self-contained conditional flow-matching downscaling.

Fully self-contained: no external SF source tree required.

Architecture: UNetAttention(8→5) — U-Net with self-attention at the two
lowest spatial resolutions. Faithful to the original UNetFlow which uses
attention in the bottleneck. Time t is broadcast as a constant spatial
channel (simpler than sinusoidal embedding; unified with CM).

Training objective: conditional linear flow matching.
  x_0 = spectral low-pass of ERA5 (shared structure with GCM, cutoff r_cut)
  x_1 = full ERA5
  x_t = (1-t)*x_0 + t*x_1,  target velocity v = x_1 - x_0

Inference: Euler ODE from t=0 (spectral-lowpass GCM) to t=1 (fine prediction).
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
    spectral_lowpass, month_channels, EMA,
)
from data_processing import (
    load_era5, load_gcm, bicubic_regrid, align_time, save_predictions,
)

VARIABLES    = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS       = len(VARIABLES)
MODE         = "serpentflow"
ARCH         = "unet_attn"   # U-Net with attention — faithful to original SerpentFlow
R_CUT        = 7             # Fourier cutoff radius (pixels); matches original SF default
HF_NOISE_STD = 0.1           # std of HF noise added to low-pass source during training+inference
PHYS_MIN     = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0}


def _load_era5_raw(era5_dir, calib_start, calib_end):
    das     = {v: load_era5(era5_dir, v, calib_start, calib_end) for v in VARIABLES}
    aligned = align_time(das)
    ref     = next(iter(aligned.values()))
    months  = ref.time.dt.month.values.astype(np.float32)
    raw     = np.stack([aligned[v].values.astype(np.float32) for v in VARIABLES], axis=1)
    return raw, months, ref.lat.values, ref.lon.values


def run(
    out_dir:      Path,
    era5_dir:     Path | None = None,
    gcm_dir:      Path | None = None,
    calib_start:  int   = 1980,
    calib_end:    int   = 1999,
    eval_start:   int   = 2000,
    eval_end:     int   = 2100,
    scenario:     str   = "ssp585",
    do_train:     bool  = True,
    do_predict:   bool  = True,
    epochs:       int   = 50,
    batch_size:   int   = 16,
    r_cut:        int   = R_CUT,
    hf_noise_std: float = HF_NOISE_STD,
    n_ode_steps:  int   = 50,
    rtol:         float = 1e-4,
    atol:         float = 1e-4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir  = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "serpentflow.pt"
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Training ──────────────────────────────────────────────────────────────
    if do_train and not ckpt_path.exists():
        if era5_dir is None:
            raise ValueError("--era5-dir required for training")
        raw, months, lat, lon = _load_era5_raw(era5_dir, calib_start, calib_end)
        stats   = compute_stats(raw, VARIABLES)
        dataset = ERA5DownscalingDataset(raw, months, stats, VARIABLES,
                                         mode=MODE, r_cut=r_cut)
        model   = build_unet(ARCH, _n_in(MODE, N_VARS), N_VARS)
        train_neural_model(MODE, dataset, model, ckpt_path,
                           n_epochs=epochs, batch_size=batch_size, device=device,
                           hf_noise_std=hf_noise_std)
        obj = torch.load(ckpt_path, weights_only=False)
        obj.update({"stats": stats, "variables": VARIABLES,
                    "mode": MODE, "arch": ARCH,
                    "in_channels": _n_in(MODE, N_VARS), "out_channels": N_VARS,
                    "r_cut": r_cut, "hf_noise_std": hf_noise_std})
        torch.save(obj, ckpt_path)
    elif do_train:
        print(f"  [SF] skipping train — checkpoint exists: {ckpt_path}")

    # ── Inference ─────────────────────────────────────────────────────────────
    if do_predict:
        if gcm_dir is None or era5_dir is None:
            raise ValueError("--era5-dir and --gcm-dir required for inference")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run --train first.")

        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stats        = ckpt["stats"]
        r            = ckpt.get("r_cut", r_cut)
        noise_std    = ckpt.get("hf_noise_std", hf_noise_std)
        model = build_unet(ckpt.get("arch", ARCH),
                           ckpt.get("in_channels", _n_in(MODE, N_VARS)), N_VARS)
        if "ema" in ckpt:
            ema = EMA(model)
            ema.load_state_dict(ckpt["ema"])
            ema.apply_shadow()
        else:
            model.load_state_dict(ckpt["model"])

        ref        = load_era5(era5_dir, VARIABLES[0], calib_start, calib_start)
        target_lat = ref.lat.values
        target_lon = ref.lon.values

        print(f"  [SF] loading GCM {eval_start}–{eval_end} ...", flush=True)
        gcm_eval = align_time({v: load_gcm(gcm_dir, v, eval_start, eval_end, scenario)
                                for v in VARIABLES})
        gcm_time = next(iter(gcm_eval.values())).time
        T, H, W  = gcm_time.size, len(target_lat), len(target_lon)

        print(f"  [SF] bicubic regrid + spectral lowpass (r_cut={r}) + HF noise ...", flush=True)
        x_in_norm = np.empty((T, N_VARS, H, W), dtype=np.float32)
        for i, var in enumerate(VARIABLES):
            gcm_up          = bicubic_regrid(gcm_eval[var], target_lat, target_lon)
            gcm_up_lp       = spectral_lowpass(gcm_up.values.astype(np.float32)[None], r)[0]
            x_in_norm[:, i] = encode(gcm_up_lp, stats, var)
        # Add the same HF noise as in training so the model is conditioned correctly
        x_in_norm += np.random.randn(*x_in_norm.shape).astype(np.float32) * noise_std

        months = gcm_time.dt.month.values.astype(np.float32)
        mc     = month_channels(months, (T, H, W))

        print(f"  [SF] dopri5 ODE integration ...", flush=True)
        pred_norm = neural_predict(MODE, model, x_in_norm, mc, device,
                                   n_steps=n_ode_steps, rtol=rtol, atol=atol)

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
    p.add_argument("--era5-dir",     default=None)
    p.add_argument("--gcm-dir",      default=None)
    p.add_argument("--out-dir",      required=True)
    p.add_argument("--calib-start",  type=int, default=1980)
    p.add_argument("--calib-end",    type=int, default=1999)
    p.add_argument("--eval-start",   type=int, default=2000)
    p.add_argument("--eval-end",     type=int, default=2100)
    p.add_argument("--scenario",     default="ssp585")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--r-cut",         type=int,   default=R_CUT)
    p.add_argument("--hf-noise-std",  type=float, default=HF_NOISE_STD)
    p.add_argument("--n-ode-steps",   type=int,   default=50)
    p.add_argument("--rtol",          type=float, default=1e-4)
    p.add_argument("--atol",          type=float, default=1e-4)
    p.add_argument("--train",         action="store_true")
    p.add_argument("--predict",       action="store_true")
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
        r_cut=args.r_cut, hf_noise_std=args.hf_noise_std,
        n_ode_steps=args.n_ode_steps, rtol=args.rtol, atol=args.atol,
    )


if __name__ == "__main__":
    main()
