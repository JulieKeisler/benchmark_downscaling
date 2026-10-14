"""Baseline method registry.

Inline methods are callables with signature:
    method(gcm, obs, variable) -> xr.DataArray

Inline methods (bicubic, bcsd, cdft, dotc) run directly inside evaluate.py.
Pre-computed methods (iDownscale, CorrDiff, ConsistencyModel, SerpentFlow-CLIM)
produce predictions.pt via their inference scripts and are loaded via --pred-dir;
``corrdiff_qm`` is registered only as a safety-net stub.

The heavy import chain (torch / xarray / sbck via the inference modules) is
deferred to first access of ``METHOD_REGISTRY`` (module ``__getattr__``) so that
lightweight consumers — notably the dashboard, which only needs display labels —
can ``import baselines`` without installing torch. See ``labels.py``.

Adding a new per-variable method:
    1. Implement it and register in ``_build_registry`` below.
    2. Add an inference_<name>.py script using save_predictions.
"""
from .labels import METHOD_LABELS, is_unpublished

__all__ = ["METHOD_REGISTRY", "METHOD_LABELS", "is_unpublished"]


def _build_registry() -> dict:
    from .methods import bicubic_downscale
    from .inference_cdft import cdft_downscale
    from .inference_dotc import dotc_downscale
    from .inference_bcsd import bcsd_downscale

    def _corrdiff_qm_stub(gcm, obs, variable):
        """Registry safety-net: CorrDiff-QM is a pre-computed (checkpoint) method.

        Eager import of inference_corrdiff would pull in torch/diffusers —
        see AGENTS.md gotcha #5.
        """
        raise RuntimeError(
            "CorrDiff-QM is a checkpoint-based, pre-computed method. "
            "Run `python src/baselines/inference_corrdiff.py --train --predict "
            "--out-dir results/corrdiff_qm --era5-dir ... --gcm-dir ...` first, "
            "then evaluate with `--method corrdiff_qm --pred-dir results/corrdiff_qm`."
        )

    from functools import partial
    return {
        "bicubic":     bicubic_downscale,
        "bcsd":        bcsd_downscale,
        "eqm":         partial(bcsd_downscale, detrend=False),
        "cdft":        cdft_downscale,
        "dotc":        dotc_downscale,
        "corrdiff_qm": _corrdiff_qm_stub,
        "cordiff":     bicubic_downscale,
    }


def __getattr__(name):
    if name == "METHOD_REGISTRY":
        global METHOD_REGISTRY
        METHOD_REGISTRY = _build_registry()
        return METHOD_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
