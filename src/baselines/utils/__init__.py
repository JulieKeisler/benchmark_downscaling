"""Shared utilities for iDownscale, SerpentFlow, and ConsistencyModel.

Architecture defaults per baseline:
    idownscale  → UNet          ("unet",      7 in / 5 out)
    serpentflow → UNetAttention ("unet_attn", 8 in / 5 out, t as spatial channel)
    consistency → DiffusersUNet ("diffusers", 7 in / 5 out, t via FiLM)
    score       → DiffusersUNet ("diffusers", 7 in / 5 out, σ via FiLM)
"""
from .unet            import UNet, UNetAttention, DiffusersUNet, build_unet, PAD, H_OUT, W_OUT
from .month           import month_channels
from .fourier         import spectral_lowpass
from .ema             import EMA
from .normalization   import encode, decode, compute_stats, LOG_VARS
from .dataset         import ERA5DownscalingDataset
from .training        import (
    train_neural_model, train_score_model,
    consistency_forward, score_forward, _n_in,
)
from .inference_utils import neural_predict

__all__ = [
    "UNet", "UNetAttention", "DiffusersUNet", "build_unet", "PAD", "H_OUT", "W_OUT",
    "month_channels", "spectral_lowpass", "EMA",
    "encode", "decode", "compute_stats", "LOG_VARS",
    "ERA5DownscalingDataset",
    "train_neural_model", "train_score_model",
    "consistency_forward", "score_forward", "_n_in",
    "neural_predict",
]
