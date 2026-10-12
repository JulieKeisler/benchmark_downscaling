"""Baseline method registry.

Each method is a callable with signature:
    method(gcm, obs, variable) -> xr.DataArray

Pre-compute methods (bcsd, cdft, idownscale, consistency, dotc) produce
predictions.pt via their inference scripts and are loaded via --pred-dir.
Inline methods (bicubic) run directly inside evaluate.py.

Adding a new per-variable method:
    1. Implement it and register in METHOD_REGISTRY below.
    2. Add an inference_<name>.py script using save_predictions.
"""
from .methods import bicubic_downscale
from .inference_cdft import cdft_downscale
from .inference_dotc import dotc_downscale
from .inference_bcsd import bcsd_downscale


def _corrdiff_qm_stub(gcm, obs, variable):
    """Registry safety-net: CorrDiff-QM is a pre-computed (checkpoint) method.

    Eager import of inference_corrdiff would pull in torch/diffusers at module
    level, breaking lightweight registry lookups — see AGENTS.md gotcha #5.
    """
    raise RuntimeError(
        "CorrDiff-QM is a checkpoint-based, pre-computed method. "
        "Run `python src/baselines/inference_corrdiff.py --train --predict "
        "--out-dir results/corrdiff_qm --era5-dir ... --gcm-dir ...` first, "
        "then evaluate with `--method corrdiff_qm --pred-dir results/corrdiff_qm`."
    )


METHOD_REGISTRY: dict = {
    "bicubic":     bicubic_downscale,
    "bcsd":        bcsd_downscale,
    "cdft":        cdft_downscale,
    "dotc":        dotc_downscale,
    "corrdiff_qm": _corrdiff_qm_stub,
    "cordiff":     bicubic_downscale,
}

METHOD_LABELS: dict[str, str] = {
    "bicubic":            "Bicubic",
    "bcsd":               "BCSD",
    "cdft":               "CDF-t",
    "dotc":               "DOTC",
    "corrdiff_qm":        "CorrDiff-QM",
    "cordiff":            "CorDiff",
    "idownscale":         "iDownscale",
    "consistency":        "CM",
    "cm":                 "CM",
    "automated_degradation":  "Automated Degradation",
    # Discovered-mu alignment + conditional flow matching (this project).
    # mu is a searched DAG of interpretable climate operators rather than the
    # fixed Fourier low-pass SerpentFlow uses -- hence "chimera", an assembly
    # of unlike parts. Supersedes the "automated_degradation" entry, which was
    # produced before the GCM latitude-orientation fix and is excluded from the
    # dashboard.
    "ChimeraFlow":        "ChimeraFlow",
    "chimeraflow":        "ChimeraFlow",
    "SerpentflowClim":    "SerpentFlow-CLIM",
    "serpentflow":        "SerpentFlow",
    "serpentflow_clim":   "SerpentFlow-CLIM",
    "serpentflowclim":    "SerpentFlow-CLIM",
}
__all__ = ["METHOD_REGISTRY", "METHOD_LABELS",
           "bicubic_downscale", "cdft_downscale", "dotc_downscale"]
