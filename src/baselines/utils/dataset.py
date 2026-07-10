"""Unified ERA5 dataset for all three neural baselines.

mode="idownscale"
    x_in  = coarsened ERA5 (stride× subsample + bicubic upsample)
    x_target = full ERA5

mode="serpentflow"
    x_in  = spectral low-pass of ERA5 at r_cut (shared GCM structure)
    x_target = full ERA5

mode="consistency"
    x_in  = x_target = full ERA5  (training loop injects noise per batch)

All modes: month_ch = [sin(2π·m/12), cos(2π·m/12)] broadcast to [2, H, W].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset

from .fourier import spectral_lowpass
from .month import month_channels
from .normalization import encode

VALID_MODES = ("idownscale", "serpentflow", "consistency")


def _coarsen_upsample(data: np.ndarray, stride: int) -> np.ndarray:
    """Subsample by stride then bicubic upsample. data: [T, C, H, W] float32."""
    T, C, H, W = data.shape
    H_c, W_c = H // stride, W // stride
    x = torch.from_numpy(data).view(T * C, 1, H, W)
    down = TF.interpolate(x, (H_c, W_c), mode="bilinear",  align_corners=True)
    up   = TF.interpolate(down, (H, W),  mode="bicubic",   align_corners=True)
    return up.view(T, C, H, W).numpy()


class ERA5DownscalingDataset(Dataset):
    """Paired (x_in, x_target, month_ch) tensors for training.

    Args:
        raw:       [T, N_VARS, H, W] — physical-unit ERA5.
        months:    [T] — calendar months (1–12).
        stats:     Per-variable dict from ``normalization.compute_stats``.
        variables: Ordered list of variable names.
        mode:      One of "idownscale", "serpentflow", "consistency".
        r_cut:     Fourier cutoff radius (used for mode="serpentflow").
        stride:    Coarsening stride   (used for mode="idownscale").
    """

    def __init__(
        self,
        raw:       np.ndarray,
        months:    np.ndarray,
        stats:     dict,
        variables: list[str],
        mode:      str = "serpentflow",
        r_cut:     int = 7,
        stride:    int = 4,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        T, C, H, W = raw.shape
        data_norm = np.empty((T, C, H, W), dtype=np.float32)
        for i, var in enumerate(variables):
            data_norm[:, i] = encode(raw[:, i], stats, var)

        if mode == "idownscale":
            x_in = _coarsen_upsample(data_norm, stride)
        elif mode == "serpentflow":
            x_in = spectral_lowpass(data_norm, r_cut)
        else:   # consistency — noise added by training loop
            x_in = data_norm.copy()

        self.x_in     = torch.from_numpy(x_in)
        self.x_target = torch.from_numpy(data_norm)
        self.month_ch = torch.from_numpy(month_channels(months, (T, H, W)))

    def __len__(self) -> int:
        return len(self.x_target)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "x_in":     self.x_in[i],      # [N_VARS, H, W]
            "x_target": self.x_target[i],  # [N_VARS, H, W]
            "month_ch": self.month_ch[i],  # [2, H, W]
        }
