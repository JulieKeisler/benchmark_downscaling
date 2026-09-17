from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import ks_2samp, pearsonr

# ------------------------------------------------------------------
# Spatial spectrum metric (isotropic Fourier-spectrum comparison)
# ------------------------------------------------------------------

@functools.lru_cache(maxsize=8)
def _k_grid(ny: int, nx: int, dx: float, dy: float) -> np.ndarray:
    """Cached radial wavenumber grid (ravelled) for a given grid size + spacing."""
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kx2d, ky2d = np.meshgrid(kx, ky)
    return np.sqrt(kx2d ** 2 + ky2d ** 2).ravel()


def _isotropic_psd_one(field2d: np.ndarray, dx: float, dy: float,
                        k_bins: np.ndarray) -> np.ndarray:
    """Radially-averaged isotropic PSD for one 2D field.

    dx, dy: grid spacing in meters.
    k_bins: wavenumber bin edges in cycles/meter.
    Returns mean PSD per bin.
    """
    ny, nx = field2d.shape
    psd2 = (np.abs(np.fft.fft2(field2d - field2d.mean())) ** 2).ravel()
    k    = _k_grid(ny, nx, dx, dy)
    nbins = len(k_bins) - 1
    bin_idx = np.searchsorted(k_bins, k, side="right") - 1
    valid   = (bin_idx >= 0) & (bin_idx < nbins)
    sums   = np.bincount(bin_idx[valid], weights=psd2[valid], minlength=nbins)
    counts = np.bincount(bin_idx[valid],                      minlength=nbins)
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def spatial_spectrum_metric(obs: xr.DataArray, pred: xr.DataArray,
                            nbins: int = 40) -> tuple[float, dict]:
    """Normalised RMSE of isotropic spatial power spectra (obs vs pred).

      - Fixed k_bins in [1e-6, 1e-4] cycles/m (physical, domain-independent)
      - No cos(lat) correction on dx (matches original)
      - Only large-scale bins used (bottom 30%): avoids penalising fine scales
        where the model has no information (r_cut conditioning)
      - Score = RMSE(pred_PSD, obs_PSD) / mean(obs_PSD) over large-scale bins

    Returns (scalar score, dict with k_centers_km/era5/pred arrays for plotting).
    Lower score = spectra closer to ERA5.
    """
    lat = obs.lat.values
    lon = obs.lon.values
    dx = abs(float(np.mean(np.diff(lon)))) * 111_000   # no cos(lat), matching original
    dy = abs(float(np.mean(np.diff(lat)))) * 111_000

    k_bins    = np.logspace(-6, -4, nbins + 1)          # fixed physical range
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # Large-scale restriction: bottom 30% of bins (matches original kmax_metric)
    kmax_idx  = int(nbins * 0.3)
    mask_k    = np.arange(nbins) <= kmax_idx

    obs_np  = np.nan_to_num(obs.values,  copy=False)
    pred_np = np.nan_to_num(pred.values, copy=False)
    nt = obs_np.shape[0]

    obs_spectra  = np.stack([_isotropic_psd_one(obs_np[t],  dx, dy, k_bins) for t in range(nt)])
    pred_spectra = np.stack([_isotropic_psd_one(pred_np[t], dx, dy, k_bins) for t in range(nt)])

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obs_mean  = np.nanmean(obs_spectra,  axis=0)
        pred_mean = np.nanmean(pred_spectra, axis=0)

    valid = np.isfinite(obs_mean) & np.isfinite(pred_mean) & (obs_mean > 0) & mask_k
    if valid.sum() == 0:
        return float("nan"), {}

    ref_mean = float(np.nanmean(obs_mean[valid]))
    score    = float(np.sqrt(np.nanmean((pred_mean[valid] - obs_mean[valid]) ** 2)) / ref_mean)

    plot_data = {
        "k_centers_km": (1.0 / k_centers[valid] / 1000.0).tolist(),
        "era5": obs_mean[valid].tolist(),
        "pred": pred_mean[valid].tolist(),
    }
    return score, plot_data
