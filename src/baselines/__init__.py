"""Baseline method registry.

Each method is a callable with signature:
    method(gcm, obs, variable) -> xr.DataArray

Pre-compute methods (bcsd, cdft, idownscale, consistency, serpentflow) produce
predictions.pt via their inference scripts and are loaded via --pred-dir.
Inline methods (bicubic) run directly inside evaluate.py.

Adding a new per-variable method:
    1. Implement it and register in METHOD_REGISTRY below.
    2. Add an inference_<name>.py script using save_predictions.
"""
from .methods import bicubic_downscale
from .inference_cdft import cdft_downscale

METHOD_REGISTRY: dict = {
    "bicubic": bicubic_downscale,
    "cdft":    cdft_downscale,
}

METHOD_LABELS: dict[str, str] = {
    "bicubic":     "Bicubic",
    "bcsd":        "BCSD",
    "cdft":        "CDF-t",
    "idownscale":  "iDownscale",
    "consistency": "CM",
    "cm":          "CM",
    "serpentflow": "SF",
    "sf":          "SF",
}

__all__ = ["METHOD_REGISTRY", "METHOD_LABELS",
           "bicubic_downscale", "cdft_downscale"]
