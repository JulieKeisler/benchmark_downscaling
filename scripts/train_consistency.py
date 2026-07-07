#!/usr/bin/env python3
"""Consistency model training for climate downscaling — multivariate.

Based on: "Fast, scale-adaptive and uncertainty-aware downscaling of Earth
system model fields with generative machine learning" (Nature 2025).
https://github.com/p-hss/consistency-climate-downscaling

One joint model is trained on all variables simultaneously (5-channel field),
so the model learns the joint distribution and preserves inter-variable
correlations at inference time.

Training: unconditional consistency model on stacked ERA5 daily fields.
Inference: stroke guidance — upsample each GCM field to ERA5 grid, stack
           into 5-channel tensor, add noise at t*, one-step denoising.

Usage:
    python scripts/train_consistency.py \
        --era5-dir /scratch/jkeisler/era5 \
        --output-dir /scratch/jkeisler/consistency_models \
        --n-epochs 50 --batch-size 16
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from diffusers import UNet2DModel
from torch.utils.data import DataLoader, Dataset

# ── Variables (fixed canonical order — must match evaluate.py) ────────────────
VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS    = len(VARIABLES)
LOG_VARS  = {"pr"}
LOG_EPS   = 1e-6

# ── Consistency-model hyper-parameters (from paper) ───────────────────────────
BINS_MIN    = 2
BINS_MAX    = 150
BINS_RHO    = 7.0
DATA_STD    = 0.5
T_MIN       = 0.002
T_MAX       = 80.0
INITIAL_EMA = 0.9

# UNet architecture — 5 input + 5 output channels, same interior
CHANNELS    = (128, 128, 256, 256)
DOWN_BLOCKS = ("DownBlock2D",) * 4
UP_BLOCKS   = ("UpBlock2D",)   * 4

# ERA5 grid: 57 lat × 65 lon → pad to 64 × 80 (divisible by 2^4 = 16)
PAD      = (0, 15, 0, 7)   # F.pad: (left, right, top, bottom)
H_OBS, W_OBS = 57, 65
H_PAD, W_PAD = 64, 80


# ── Per-variable transforms ───────────────────────────────────────────────────

def _to_log(x: np.ndarray) -> np.ndarray:
    return np.log(x + LOG_EPS) - np.log(LOG_EPS)


def _from_log(x: np.ndarray) -> np.ndarray:
    return np.exp(x + np.log(LOG_EPS)) - LOG_EPS


def norm(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0


def inv_norm(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    return (x + 1.0) / 2.0 * (xmax - xmin) + xmin


def encode(x: np.ndarray, stats: dict, variable: str) -> np.ndarray:
    if variable in LOG_VARS:
        x = _to_log(x)
    return norm(x, stats[variable]["xmin"], stats[variable]["xmax"])


def decode(x: np.ndarray, stats: dict, variable: str) -> np.ndarray:
    x = inv_norm(x, stats[variable]["xmin"], stats[variable]["xmax"])
    if variable in LOG_VARS:
        x = _from_log(x)
    return x


# ── Data loading ─────────────────────────────────────────────────────────────

def load_era5_multivar(era5_dir: Path, year_start: int, year_end: int
                        ) -> tuple[np.ndarray, dict]:
    """Returns stacked array [T, N_VARS, H, W] and per-variable stats."""
    channels = []
    time_ref = None

    for var in VARIABLES:
        files = sorted(era5_dir.glob(f"{var}_*.nc"))
        files = [f for f in files
                 if year_start <= int(f.stem.split("_")[-1]) <= year_end]
        if not files:
            sys.exit(f"No ERA5 files for {var} in {era5_dir} "
                     f"(years {year_start}–{year_end})")

        parts = []
        for f in files:
            ds = xr.open_dataset(f)
            ds = ds.rename({k: v for k, v in {"latitude": "lat",
                                                "longitude": "lon"}.items()
                            if k in ds.dims})
            parts.append(ds[var])
        da = xr.concat(parts, dim="time")

        if time_ref is None:
            time_ref = da.time.values
        else:
            # Align to common times
            common = np.intersect1d(time_ref, da.time.values)
            da = da.sel(time=common)
            time_ref = common

        arr = da.values.astype(np.float32)
        if var == "pr":
            # ERA5 pr is in m/day from the CDS daily-stats pipeline;
            # convert to mm/day to match the CMIP6 unit at inference.
            # Clip tiny negatives (numerical noise in ERA5) before conversion.
            arr = np.clip(arr, 0.0, None) * 1000.0
        channels.append(arr)  # [T, H, W]

    # Trim all channels to common time length (in case of late alignment)
    T = min(c.shape[0] for c in channels)
    channels = [c[:T] for c in channels]

    stacked = np.stack(channels, axis=1)  # [T, N_VARS, H, W]
    print(f"Loaded ERA5: {stacked.shape} ({T} days, {N_VARS} variables)", flush=True)

    # Per-variable normalisation stats computed on the training split
    stats = {}
    for i, var in enumerate(VARIABLES):
        raw = stacked[:, i]                          # [T, H, W]
        transformed = _to_log(raw) if var in LOG_VARS else raw
        stats[var] = {
            "xmin": float(transformed.min()),
            "xmax": float(transformed.max()),
            "log_transform": var in LOG_VARS,
        }
        print(f"  {var}: xmin={stats[var]['xmin']:.4f}  "
              f"xmax={stats[var]['xmax']:.4f}", flush=True)

    return stacked, stats


class ERA5MultivariateDataset(Dataset):
    def __init__(self, raw: np.ndarray, stats: dict):
        # raw: [T, N_VARS, H, W] — physical units, un-normalised
        T, C, H, W = raw.shape
        data = np.empty_like(raw)
        for i, var in enumerate(VARIABLES):
            data[:, i] = encode(raw[:, i], stats, var)
        self.data = torch.from_numpy(data).float()  # [T, N_VARS, H, W]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i]                             # [N_VARS, H, W]
        return F.pad(x, PAD)                         # [N_VARS, H_PAD, W_PAD]


# ── Consistency model maths ───────────────────────────────────────────────────

def timestep_to_time(n: torch.Tensor, N: int, device) -> torch.Tensor:
    n = n.to(device).float()
    return (
        T_MIN ** (1.0 / BINS_RHO)
        + n / (N - 1) * (T_MAX ** (1.0 / BINS_RHO) - T_MIN ** (1.0 / BINS_RHO))
    ).pow(BINS_RHO).clamp(0.0, T_MAX)


def consistency_forward(net, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    t4       = t[:, None, None, None]
    skip     = DATA_STD ** 2 / ((t4 - T_MIN) ** 2 + DATA_STD ** 2)
    out_coef = DATA_STD * t4 / (t4 ** 2 + DATA_STD ** 2) ** 0.5
    pred     = net(x, t).sample
    return skip * x + out_coef * pred


@torch.no_grad()
def ema_update(online, ema, N: int):
    decay = math.exp(BINS_MIN * math.log(INITIAL_EMA) / N)
    for p, p_ema in zip(online.parameters(), ema.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def current_bins(global_step: int, total_steps: int) -> int:
    frac = global_step / max(total_steps, 1)
    return math.ceil(
        math.sqrt(frac * (BINS_MAX ** 2 - BINS_MIN ** 2) + BINS_MIN ** 2)
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, stats = load_era5_multivar(Path(args.era5_dir),
                                    args.train_start, args.train_end)

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    dataset = ERA5MultivariateDataset(raw, stats)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=4, pin_memory=True,
                         persistent_workers=True)

    total_steps = len(loader) * args.n_epochs
    print(f"Training: {len(dataset)} samples, "
          f"{total_steps} total steps, {args.n_epochs} epochs", flush=True)

    net = UNet2DModel(
        in_channels=N_VARS, out_channels=N_VARS,
        block_out_channels=CHANNELS,
        down_block_types=DOWN_BLOCKS,
        up_block_types=UP_BLOCKS,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"UNet params: {n_params/1e6:.1f}M  "
          f"(in_channels={N_VARS}, out_channels={N_VARS})", flush=True)

    net_ema = copy.deepcopy(net)
    net_ema.requires_grad_(False)

    opt = torch.optim.RAdam(net.parameters(), lr=args.lr)

    best_loss = float("inf")
    global_step = 0

    for epoch in range(args.n_epochs):
        net.train()
        epoch_loss = 0.0

        for batch in loader:
            x = batch.to(device)                      # [B, N_VARS, H_PAD, W_PAD]
            B = x.shape[0]
            N = current_bins(global_step, total_steps)

            n_idx = torch.randint(0, N - 1, (B,), device=device)
            t_n   = timestep_to_time(n_idx,     N, device)
            t_n1  = timestep_to_time(n_idx + 1, N, device)

            noise = torch.randn_like(x)
            x_n   = x + noise * t_n[:,  None, None, None]
            x_n1  = x + noise * t_n1[:, None, None, None]

            with torch.no_grad():
                target = consistency_forward(net_ema, x_n,  t_n)
            pred = consistency_forward(net,     x_n1, t_n1)
            loss = F.mse_loss(pred, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema_update(net, net_ema, N)

            epoch_loss  += loss.item()
            global_step += 1

        avg = epoch_loss / len(loader)
        print(f"  epoch {epoch+1:3d}/{args.n_epochs}  "
              f"loss={avg:.5f}  bins={N}", flush=True)

        if avg < best_loss:
            best_loss = avg
            torch.save({"net": net.state_dict(), "ema": net_ema.state_dict(),
                        "stats": stats, "variables": VARIABLES},
                       out_dir / "best.pt")

    torch.save({"net": net.state_dict(), "ema": net_ema.state_dict(),
                "stats": stats, "variables": VARIABLES},
               out_dir / "last.pt")
    print(f"Checkpoint saved → {out_dir}/best.pt", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",    required=True)
    p.add_argument("--output-dir",  default="models/consistency")
    p.add_argument("--train-start", type=int, default=2000)
    p.add_argument("--train-end",   type=int, default=2015)
    p.add_argument("--n-epochs",    type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=16)
    p.add_argument("--lr",          type=float, default=2e-4)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
