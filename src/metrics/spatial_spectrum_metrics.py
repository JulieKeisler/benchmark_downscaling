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


# ------------------------------------------------------------------
# Fine-scale spectrum metric (wavelengths below ~200 km)
# ------------------------------------------------------------------

def _tapered_power(fields: np.ndarray) -> np.ndarray:
    """Time-mean 2D power of [T, H, W] fields with a Hann taper.

    The domain is not periodic: without a taper the edge jump leaks power to
    every wavenumber and masks the fine-scale differences this metric targets.
    """
    T, H, W = fields.shape
    taper = np.outer(np.hanning(H), np.hanning(W))
    norm = H * W * float(np.mean(taper ** 2))
    power = np.zeros((H, W))
    for t in range(T):
        f = fields[t] - fields[t].mean()
        power += np.abs(np.fft.fft2(f * taper)) ** 2 / norm
    return power / T


def fine_spectrum_metric(obs: xr.DataArray, pred: xr.DataArray,
                         max_wavelength_km: float = 200.0,
                         nbins: int = 10, variable: str | None = None,
                         ) -> tuple[float, float, dict]:
    """Log-ratio of fine-scale power spectra, prediction vs ERA5.

    Grid spacing in km uses cos(mean latitude) for longitude (over a 14°
    domain cos(lat) varies by about ±10 %, so the 200 km cut-off moves by as
    much; the pred/ERA5 ratio is unaffected). pr is compared as log1p(pr):
    raw precipitation spectra are dominated by a few convective days.
    NaN in either field raises — a zero-filled hole would add spurious power. Power is binned
    by radial wavenumber on log-spaced bins from 1/max_wavelength_km to the
    Nyquist wavenumber of the coarser direction, and averaged over days.

    Returns (score, bias, plot_data):
      score = mean over bins of |log10(P_pred / P_ERA5)|  (0 = same power;
              0.3 = a factor 2 off), lower is better;
      bias  = mean over bins of log10(P_pred / P_ERA5)    (< 0: too smooth);
      plot_data = full-range spectra (domain scale to Nyquist) for plotting.
    """
    lat = obs.lat.values
    lon = obs.lon.values
    dy = abs(float(np.mean(np.diff(lat)))) * 111.0
    dx = abs(float(np.mean(np.diff(lon)))) * 111.0 * float(np.cos(np.deg2rad(lat.mean())))
    o, p = obs.values, pred.values
    if np.isnan(o).any() or np.isnan(p).any():
        raise ValueError("fine_spectrum_metric: NaN in the fields (contract §E4)")
    if variable == "pr":
        o, p = np.log1p(np.clip(o, 0.0, None)), np.log1p(np.clip(p, 0.0, None))
    p_obs = _tapered_power(o)
    p_pred = _tapered_power(p)
    ny, nx = p_obs.shape
    kx, ky = np.meshgrid(np.fft.fftfreq(nx, dx), np.fft.fftfreq(ny, dy))
    k = np.sqrt(kx ** 2 + ky ** 2)                       # cycles / km
    k_nyq = min(0.5 / dx, 0.5 / dy)

    def binned(edges):
        idx = np.digitize(k.ravel(), edges) - 1
        ok = (idx >= 0) & (idx < len(edges) - 1) & (k.ravel() > 0)
        cnt = np.bincount(idx[ok], minlength=len(edges) - 1)
        po = np.bincount(idx[ok], weights=p_obs.ravel()[ok], minlength=len(edges) - 1)
        pp = np.bincount(idx[ok], weights=p_pred.ravel()[ok], minlength=len(edges) - 1)
        keep = cnt > 0
        centres = np.sqrt(edges[:-1] * edges[1:])[keep]
        return centres, po[keep] / cnt[keep], pp[keep] / cnt[keep]

    _, po, pp = binned(np.logspace(np.log10(1.0 / max_wavelength_km), np.log10(k_nyq), nbins + 1))
    valid = (po > 0) & (pp > 0)
    if not valid.any():
        return float("nan"), float("nan"), {}
    lr = np.log10(pp[valid] / po[valid])
    k_dom = 1.0 / (max(ny * dy, nx * dx))
    kc, fo, fp = binned(np.logspace(np.log10(k_dom), np.log10(k_nyq), 31))
    plot_data = {"wavelength_km": (1.0 / kc).tolist(), "era5": fo.tolist(), "pred": fp.tolist(),
                 "fine_cutoff_km": max_wavelength_km}
    return float(np.mean(np.abs(lr))), float(np.mean(lr)), plot_data
