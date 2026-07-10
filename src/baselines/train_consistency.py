#!/usr/bin/env python3
"""Consistency model training — two-phase pipeline following Hess et al. 2025.

Phase 1 — Score network (EDM denoising score matching, Karras et al. 2022):
    Trains UNet2DModel (diffusers, 128/128/256/256 channels, SiLU, GroupNorm,
    FiLM time conditioning) on ERA5 with log-normal σ sampling.
    Checkpoint: <output-dir>/score.pt

Phase 2 — Consistency distillation (Song et al. 2023):
    Initialises a second UNet2DModel from the score checkpoint, then trains
    with the CD loss: one Euler step of the PF-ODE via the score network gives
    x_n from x_n+1, and we enforce f_θ(x_n+1, t_n+1) ≈ f_θ̄(x_n, t_n).
    Checkpoint: <output-dir>/best.pt  (used by inference_cm.py)

Usage:
    python train_consistency.py \\
        --era5-dir /path/to/era5 \\
        --output-dir /path/to/checkpoints/consistency \\
        --score-epochs 200 --n-epochs 150 --batch-size 32
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from utils import (
    build_unet, _n_in, compute_stats, EMA,
    ERA5DownscalingDataset, train_neural_model, train_score_model,
)
from data_processing import load_era5, align_time

VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS    = len(VARIABLES)
ARCH      = "diffusers"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",       required=True)
    p.add_argument("--output-dir",     default="models/consistency")
    p.add_argument("--train-start",    type=int, default=1980)
    p.add_argument("--train-end",      type=int, default=1999)
    p.add_argument("--score-epochs",   type=int, default=200,
                   help="Epochs for score network pre-training (Phase 1)")
    p.add_argument("--n-epochs",       type=int, default=150,
                   help="Epochs for consistency distillation (Phase 2)")
    p.add_argument("--batch-size",     type=int, default=32)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--skip-score",     action="store_true",
                   help="Skip Phase 1 if score.pt already exists")
    args = p.parse_args()

    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    score_path  = out_dir / "score.pt"
    cm_path     = out_dir / "best.pt"
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    n_in        = _n_in("consistency", N_VARS)   # N_VARS + 2 (no t channel)

    # ── Load ERA5 calibration data ─────────────────────────────────────────────
    era5_das = {v: load_era5(Path(args.era5_dir), v, args.train_start, args.train_end)
                for v in VARIABLES}
    aligned  = align_time(era5_das)
    ref      = next(iter(aligned.values()))
    months   = ref.time.dt.month.values.astype(np.float32)
    raw      = np.stack([aligned[v].values.astype(np.float32) for v in VARIABLES], axis=1)
    stats    = compute_stats(raw, VARIABLES)
    dataset  = ERA5DownscalingDataset(raw, months, stats, VARIABLES, mode="consistency")

    # ── Phase 1: score network ─────────────────────────────────────────────────
    if args.skip_score and score_path.exists():
        print(f"  Skipping Phase 1: loading score from {score_path}", flush=True)
    else:
        print("=== Phase 1: score network (EDM) ===", flush=True)
        score_model = build_unet(ARCH, n_in, N_VARS)
        train_score_model(
            dataset, score_model, score_path,
            n_epochs=args.score_epochs, batch_size=args.batch_size,
            lr=args.lr, device=device,
        )
        obj = torch.load(score_path, weights_only=False)
        obj.update({"stats": stats, "variables": VARIABLES,
                    "mode": "score", "arch": ARCH,
                    "in_channels": n_in, "out_channels": N_VARS})
        torch.save(obj, score_path)
        print(f"  Score checkpoint → {score_path}", flush=True)

    # ── Phase 2: consistency distillation ─────────────────────────────────────
    print("=== Phase 2: consistency distillation ===", flush=True)
    score_model = build_unet(ARCH, n_in, N_VARS)
    score_ckpt  = torch.load(score_path, map_location="cpu", weights_only=False)
    from utils.ema import EMA
    score_ema = EMA(score_model)
    score_ema.load_state_dict(score_ckpt["ema"])
    score_ema.apply_shadow()   # load EMA weights into score_model

    # Init CM from score weights (Song et al. 2023 §4.3)
    cm_model = build_unet(ARCH, n_in, N_VARS)
    cm_model.load_state_dict(score_model.state_dict())

    train_neural_model(
        "consistency", dataset, cm_model, cm_path,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        lr=args.lr, device=device,
        score_model=score_model,
    )

    # Augment checkpoint with metadata
    obj = torch.load(cm_path, weights_only=False)
    obj.update({"stats": stats, "variables": VARIABLES,
                "mode": "consistency", "arch": ARCH,
                "in_channels": n_in, "out_channels": N_VARS})
    torch.save(obj, cm_path)
    print(f"  CM checkpoint → {cm_path}", flush=True)


if __name__ == "__main__":
    main()
