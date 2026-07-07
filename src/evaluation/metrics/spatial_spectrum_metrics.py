from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import ks_2samp, pearsonr

# ------------------------------------------------------------------
# Spatial spectrum metric (adapted from serpentflow/generate_plot.py)
# ------------------------------------------------------------------

def _isotropic_psd_one(field2d: np.ndarray, dx: float, dy: float,
                        k_bins: np.ndarray) -> np.ndarray:
    """Radially-averaged isotropic PSD for one 2D field.

    dx, dy: grid spacing in meters.
    k_bins: wavenumber bin edges in cycles/meter.
    Returns mean PSD per bin.
    """
    ny, nx = field2d.shape
    fft2 = np.fft.fft2(field2d - field2d.mean())
    psd2 = np.abs(fft2) ** 2
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dy)
    kx2d, ky2d = np.meshgrid(kx, ky)
    k = np.sqrt(kx2d ** 2 + ky2d ** 2).ravel()
    psd2 = psd2.ravel()
    nbins = len(k_bins) - 1
    spec = np.full(nbins, np.nan)
    for i in range(nbins):
        mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
        if mask.sum() > 0:
            spec[i] = psd2[mask].mean()
    return spec


def spatial_spectrum_metric(obs: xr.DataArray, pred: xr.DataArray,
                            nbins: int = 40) -> tuple[float, dict]:
    """Normalised RMSE of isotropic spatial power spectra (obs vs pred).

    Returns (scalar score, dict with k_centers_km/era5/pred arrays for plotting).
    Lower score = spectra closer to ERA5.
    """
    lat = obs.lat.values
    lon = obs.lon.values
    dx = abs(float(np.mean(np.diff(lon)))) * 111_000 * np.cos(np.radians(float(np.mean(lat))))
    dy = abs(float(np.mean(np.diff(lat)))) * 111_000

    k_max = min(0.5 / dx, 0.5 / dy)
    k_bins = np.logspace(np.log10(1 / (max(obs.lat.size, obs.lon.size) * max(dx, dy))),
                         np.log10(k_max), nbins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    obs_np  = obs.values
    pred_np = pred.values
    nt = obs_np.shape[0]

    obs_spectra  = np.stack([_isotropic_psd_one(obs_np[t],  dx, dy, k_bins) for t in range(nt)])
    pred_spectra = np.stack([_isotropic_psd_one(pred_np[t], dx, dy, k_bins) for t in range(nt)])

    obs_mean  = np.nanmean(obs_spectra,  axis=0)
    pred_mean = np.nanmean(pred_spectra, axis=0)

    valid = np.isfinite(obs_mean) & np.isfinite(pred_mean) & (obs_mean > 0)
    if valid.sum() == 0:
        return float("nan"), {}

    score = float(np.sqrt(np.nanmean(
        ((pred_mean[valid] - obs_mean[valid]) / obs_mean[valid]) ** 2
    )))

    k_km = (1.0 / k_centers / 1000.0).tolist()  # wavelength in km
    plot_data = {
        "k_centers_km": k_km,
        "era5": obs_mean[valid].tolist(),
        "pred": pred_mean[valid].tolist(),
        "k_centers_km_valid": (1.0 / k_centers[valid] / 1000.0).tolist(),
    }
    return score, plot_data
