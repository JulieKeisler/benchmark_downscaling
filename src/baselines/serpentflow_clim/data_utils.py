"""Tensor <-> xarray conversion, normalization, and cross-variable alignment."""

import numpy as np
import torch
import xarray as xr

from .time_utils import assert_time_axes_aligned


def maybe_to_device(x, device):
    """`x.to(device)` that passes `None` through -- for optional dataloader fields."""
    return x.to(device) if x is not None else None


def ensure_batched(x: torch.Tensor, expected_channels: int, what: str = "tensor") -> torch.Tensor:
    """Add a leading batch dim if missing, and check the channel count.

    Shared by `load_static_conditioning` and `model.UNetFlow.set_static`,
    which both need the same "(C, H, W) or (1, C, H, W) -> (1, C, H, W)"
    normalization for the static-conditioning tensor.
    """
    if x.ndim == 3:
        x = x.unsqueeze(0)
    assert x.shape[1] == expected_channels, f"{what} has {x.shape[1]} channels, expected {expected_channels}"
    return x


def normalize_column(x: torch.Tensor) -> torch.Tensor:
    """In-place per-channel z-score, ignoring non-finite values."""
    mask = torch.isfinite(x)
    if not mask.any():
        return x
    mean_val = x[mask].mean()
    std_val = x[mask].std()
    x[mask] = (x[mask] - mean_val) / (std_val + 1e-8)
    return x


def denormalize(data: xr.Dataset, ref: xr.Dataset, var: str, eps: float = 1e-6,
                 mean_mode: str = "local", std_mode: str = "local") -> xr.Dataset:
    """Map a model's normalized output back to physical units (paper Eq. denorm).

    BUG (confirmed present in the original notebook and in the submitted
    paper's Tables B1-B3, not just this port): `mean_data`/`std_data` here
    are recomputed from `data` itself -- whatever field is passed in at call
    time -- rather than from the training target, as the paper's Eq. (denorm)
    and its own prose ("the denormalization statistics are computed from the
    training target") both claim. Because the rescale is `(data - mean_data)
    / std_data * std_ref + mean_ref`, self-referencing `mean_data`/`std_data`
    to `data` makes the output's mean/std converge to `ref`'s almost
    regardless of what `data` contains -- verified empirically: pure noise,
    a constant field, and CDF-t's own real (non-noise) output all score
    within rounding error of the real trained model once run through this
    function. See `exact_global_inverse` for the corrected version; kept
    here unchanged only for backward-compatible reproduction of the
    as-submitted (bugged) paper numbers.

    mean_mode/std_mode: "local" (per grid point, the default used
    throughout the project) or "global" (single scalar over the whole
    domain).
    """
    data_adjusted = data.copy()

    def _stat(ds, mode, op):
        return getattr(ds[var], op)() if mode == "global" else getattr(ds[var], op)(dim="time", keep_attrs=True)

    mean_data = _stat(data, mean_mode, "mean")
    mean_ref = _stat(ref, mean_mode, "mean")
    std_data = _stat(data, std_mode, "std")
    std_ref = _stat(ref, std_mode, "std")

    if mean_mode == "global" and std_mode == "global":
        data_adjusted[var] = ((data[var] - mean_data) / (std_data + eps)) * std_ref + mean_ref
    else:
        data_adjusted[var].values = (
            (data[var].values - mean_data.values) / (std_data.values + eps)
        ) * std_ref.values + mean_ref.values
    return data_adjusted


def exact_global_inverse(data: xr.Dataset, var: str, train_mean: float, train_std: float) -> xr.Dataset:
    """Corrected inverse of the training-time z-score normalization.

    The model's raw output is z-scored per channel with a single GLOBAL
    scalar mean/std (see `normalize_column`, applied once to the whole
    training target tensor -- confirmed by reading the original
    Training-Inference.ipynb dataset class directly, not just this port).
    The faithful inverse is therefore an exact affine undo with those same
    fixed scalars, computed once from the training target and passed in
    here -- no per-pixel remap onto a reference field (that remap, self-
    referential to `data` as in `denormalize` above, is what destroys
    discriminating power). Validated: noise/constant fields score badly
    under this inverse while the real trained model scores in a sane,
    CDF-t-comparable range (see fix_validation*.py in the audit).
    """
    data_adjusted = data.copy()
    data_adjusted[var].values = data[var].values * train_std + train_mean
    return data_adjusted


def train_global_stats(train_target: xr.Dataset, var: str) -> tuple[float, float]:
    """Global scalar (mean, std) of a training-target field, matching what
    `normalize_column` computes once over the whole training tensor -- the
    fixed constants `exact_global_inverse` needs as `train_mean`/`train_std`.
    """
    return float(train_target[var].mean().values), float(train_target[var].std().values)


def xr_to_tensor(ds: xr.Dataset, var_name: str) -> torch.Tensor:
    """(T, H, W) DataArray -> (T, 1, H, W) float tensor, NaN -> 0."""
    arr = np.nan_to_num(ds[var_name].to_numpy(), nan=0.0)
    return torch.from_numpy(arr).float().unsqueeze(1)


def stack_variables_as_channels(named_arrays: "dict[str, xr.DataArray]", time_dim: str = "time") -> torch.Tensor:
    """Stack same-length per-variable fields into one (T, C, H, W) tensor.

    Replaces the notebooks' pattern of independently converting each
    variable to a tensor and combining them with a bare
    `torch.cat([...], dim=1)`: that assumed, without checking, that every
    variable's time axis was identical sample-for-sample. Here every
    variable is required to share the exact same time coordinate (after
    Feb-29 removal) before stacking, so a missing/duplicated day in any one
    variable's source file raises immediately instead of silently
    misaligning channels.
    """
    assert_time_axes_aligned(named_arrays, time_dim=time_dim)
    tensors = [torch.from_numpy(np.nan_to_num(named_arrays[n].to_numpy(), nan=0.0)).float().unsqueeze(1)
               for n in named_arrays]
    return torch.cat(tensors, dim=1)


def load_static_conditioning(path: str, standardize: bool = False) -> torch.Tensor:
    """Static predictors (1, 4, H, W): land-sea mask, sdor, slor, isor."""
    static = ensure_batched(torch.load(path, map_location="cpu").float(), expected_channels=4, what="static")
    if standardize:
        for c in range(1, 4):
            ch = static[:, c]
            m, s = ch.mean(), ch.std().clamp_min(1e-8)
            static[:, c] = (ch - m) / s
    return static
