"""Benchmark metrics and plots for downscaling evaluation.

Two evaluation axes:
  1. vs ERA5 (obs)  — statistical quality of the downscaled output
  2. vs GCM (input) — temporal consistency / climate-change signal preservation

All functions accept xarray DataArrays with dims (time, lat, lon).
Returned scalar metrics are plain Python floats; maps are DataArrays.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.stats import ks_2samp, pearsonr


# ------------------------------------------------------------------
# Scalar metrics: obs vs pred
# ------------------------------------------------------------------

def bias(obs: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Temporal-mean bias at each grid point (pred − obs)."""
    return pred.mean("time") - obs.mean("time")


def clim_rmse(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """RMSE of the time-mean spatial field."""
    return float(np.sqrt(((pred.mean("time") - obs.mean("time")) ** 2).mean().values))


def spatial_mean_abs_bias(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """Mean absolute temporal-mean bias across all grid points."""
    return float(np.abs(pred.mean("time") - obs.mean("time")).mean().values)


def std_error(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """Mean absolute error of temporal std across all grid points."""
    return float(np.abs(pred.std("time") - obs.std("time")).mean().values)


def ks_stat(obs: xr.DataArray, pred: xr.DataArray, quantile: float | None = None) -> float:
    """KS distance between flattened distributions (or tail above/below quantile)."""
    o = obs.values.ravel()
    p = pred.values.ravel()
    o = o[np.isfinite(o)]
    p = p[np.isfinite(p)]
    if quantile is not None:
        if quantile >= 0.5:
            thr = np.quantile(o, quantile)
            o = o[o > thr]
            p = p[p > thr]
        else:
            thr = np.quantile(o, quantile)
            o = o[o <= thr]
            p = p[p <= thr]
    if len(o) == 0 or len(p) == 0:
        return float("nan")
    return float(ks_2samp(o, p).statistic)


def seasonal_mae(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """MAE of monthly climatology averaged over space."""
    obs_clim = obs.groupby("time.month").mean("time")
    pred_clim = pred.groupby("time.month").mean("time")
    return float(np.abs(pred_clim - obs_clim).mean().values)


def extreme_freq_bias(obs: xr.DataArray, pred: xr.DataArray, q: float = 0.95) -> float:
    """Spatial mean of bias in fraction of days above obs q-quantile."""
    thr = float(obs.quantile(q))
    obs_freq = (obs > thr).mean("time")
    pred_freq = (pred > thr).mean("time")
    return float((pred_freq - obs_freq).mean().values)


def extreme_intensity_bias(obs: xr.DataArray, pred: xr.DataArray, q: float = 0.95) -> float:
    """Spatial mean of bias in mean intensity above obs q-quantile."""
    thr = float(obs.quantile(q))
    obs_int = obs.where(obs > thr).mean("time")
    pred_int = pred.where(pred > thr).mean("time")
    return float((pred_int - obs_int).mean().values)


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


# ------------------------------------------------------------------
# Scalar metrics: GCM vs pred (input consistency)
# ------------------------------------------------------------------

def temporal_corr_map(gcm: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Pearson correlation over time at each grid point (daily)."""
    return xr.corr(gcm, pred, dim="time")


def mean_temporal_corr(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Spatial mean of daily temporal Pearson correlation."""
    return float(temporal_corr_map(gcm, pred).mean().values)


def temporal_corr_annual_map(gcm: xr.DataArray, pred: xr.DataArray) -> xr.DataArray:
    """Pearson correlation of annual-mean anomalies at each grid point.

    More informative than daily correlation for free-running GCMs: daily
    temporal_corr is trivially ~1 for any deterministic function of the GCM input.
    Annual correlation tests whether the method preserves interannual variability.
    """
    gcm_ann  = gcm.groupby("time.year").mean("time")
    pred_ann = pred.groupby("time.year").mean("time")
    return xr.corr(gcm_ann, pred_ann, dim="year")


def mean_temporal_corr_annual(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Spatial mean of annual temporal Pearson correlation."""
    return float(temporal_corr_annual_map(gcm, pred).mean().values)


def lag1_autocorr(da: xr.DataArray) -> xr.DataArray:
    """Lag-1 temporal autocorrelation at each grid point."""
    da0 = da.isel(time=slice(None, -1))
    da1 = da.isel(time=slice(1, None))
    da0 = da0.assign_coords(time=da1.time)
    return xr.corr(da0, da1, dim="time")


def lag1_autocorr_mae(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """Mean absolute error of lag-1 autocorrelation maps."""
    return float(np.abs(lag1_autocorr(pred) - lag1_autocorr(gcm)).mean().values)


def annual_anomaly_bias(gcm: xr.DataArray, pred: xr.DataArray) -> float:
    """MAE of annual-mean anomalies (climate trend preservation)."""
    gcm_anom  = gcm.groupby("time.year").mean("time")  - gcm.mean("time")
    pred_anom = pred.groupby("time.year").mean("time") - pred.mean("time")
    return float(np.abs(pred_anom - gcm_anom).mean().values)


# ------------------------------------------------------------------
# Inter-variable correlation
# ------------------------------------------------------------------

def _pointwise_corr(a: xr.DataArray, b: xr.DataArray) -> np.ndarray:
    """Pearson r along time at each grid cell, returned as a plain [lat, lon] array."""
    a_np = a.values.astype(float)
    b_np = b.values.astype(float)
    a_c  = a_np - a_np.mean(axis=0, keepdims=True)
    b_c  = b_np - b_np.mean(axis=0, keepdims=True)
    num   = (a_c * b_c).mean(axis=0)
    denom = a_np.std(axis=0) * b_np.std(axis=0)
    return np.where(denom > 1e-10, num / denom, np.nan)


def intervar_corr_mae(obs1: xr.DataArray, obs2: xr.DataArray,
                      pred1: xr.DataArray, pred2: xr.DataArray) -> float:
    """Mean absolute difference in pointwise temporal correlation between two variable pairs."""
    obs_corr  = _pointwise_corr(obs1, obs2)
    pred_corr = _pointwise_corr(pred1, pred2)
    return float(np.nanmean(np.abs(obs_corr - pred_corr)))


# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError:
        return None, None


def _add_geo(ax, lon, lat, ccrs, cfeature):
    if ccrs is not None:
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
        ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.5, linestyle=":")
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())


def plot_bias_map(obs: xr.DataArray, pred: xr.DataArray, variable: str,
                  method: str, plot_dir: Path,
                  bias_vmax: float | None = None,
                  std_vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    mean_bias = pred.mean("time").values - obs.mean("time").values
    std_bias  = pred.std("time").values  - obs.std("time").values
    lon = obs.lon.values
    lat = obs.lat.values

    vm_mean = bias_vmax if bias_vmax is not None else float(np.nanpercentile(np.abs(mean_bias), 98))
    vm_std  = std_vmax  if std_vmax  is not None else float(np.nanpercentile(np.abs(std_bias),  98))

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5),
        subplot_kw={"projection": proj} if proj else {},
    )
    for ax, (data, title, cmap, vm) in zip(axes, [
        (mean_bias, f"Mean bias — {variable}", "RdBu_r", vm_mean),
        (std_bias,  f"Std bias — {variable}",  "PuOr",   vm_std),
    ]):
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=-vm, vmax=vm,
                           transform=ccrs_mod.PlateCarree() if ccrs else None)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

    fig.suptitle(f"{method} — {variable}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / f"bias_map_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_cdf(obs: xr.DataArray, pred: xr.DataArray, variable: str,
             method: str, plot_dir: Path,
             xlim: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [
        ("Full distribution", None),
        ("Upper tail (q>0.95)", 0.95),
        ("Lower tail (q<0.05)", 0.05),
    ]
    o_all = obs.values.ravel();  o_all = o_all[np.isfinite(o_all)]
    p_all = pred.values.ravel(); p_all = p_all[np.isfinite(p_all)]

    for ax, (title, q) in zip(axes, configs):
        o, p = o_all.copy(), p_all.copy()
        if q is not None:
            thr = np.quantile(o, q)
            if q >= 0.5:
                o = o[o > thr]; p = p[p > thr]
            else:
                o = o[o < thr]; p = p[p < thr]
        for data, label, color in [(o, "ERA5", "black"), (p, method, "tab:blue")]:
            xs = np.sort(data)
            ys = np.linspace(0, 1, len(xs))
            ax.plot(xs, ys, label=label, color=color, lw=1.5)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel(variable)
        if q is None and xlim is not None:
            ax.set_xlim(xlim)

    fig.suptitle(f"CDF — {variable} — {method}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / f"cdf_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_seasonal_cycle(obs: xr.DataArray, pred: xr.DataArray, variable: str,
                        method: str, plot_dir: Path,
                        ylim: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    obs_cycle  = obs.groupby("time.month").mean("time").mean(["lat", "lon"])
    pred_cycle = pred.groupby("time.month").mean("time").mean(["lat", "lon"])
    months = np.arange(1, 13)

    # Expand ylim to include both ERA5 and pred values (caller ylim may be ERA5-only)
    all_vals = np.concatenate([obs_cycle.values, pred_cycle.values])
    val_range = float(np.nanmax(all_vals) - np.nanmin(all_vals))
    margin = 0.15 * max(val_range, 1e-6)
    auto_ylim = (float(np.nanmin(all_vals)) - margin, float(np.nanmax(all_vals)) + margin)
    if ylim is not None:
        ylim = (min(ylim[0], auto_ylim[0]), max(ylim[1], auto_ylim[1]))
    else:
        ylim = auto_ylim

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, obs_cycle.values,  "k-o", label="ERA5", lw=2, ms=5)
    ax.plot(months, pred_cycle.values, "b-s", label=method, lw=2, ms=5)
    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylim(ylim)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Seasonal cycle — {variable} — {method}", fontsize=12)
    plt.tight_layout()
    out = plot_dir / f"seasonal_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_temporal_corr(gcm: xr.DataArray, pred: xr.DataArray, variable: str,
                       method: str, plot_dir: Path) -> None:
    """Plot annual-mean temporal correlation (daily corr is trivially ≈1 for deterministic methods)."""
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    corr = temporal_corr_annual_map(gcm, pred).values
    lon = gcm.lon.values
    lat = gcm.lat.values

    # Auto-scale around the actual data range for spatial variation
    vmin = max(-1.0, float(np.nanpercentile(corr, 2)))
    vmax = min( 1.0, float(np.nanpercentile(corr, 98)))
    if vmax - vmin < 0.05:   # nearly flat — center on mean
        vmid = float(np.nanmean(corr))
        vmin, vmax = max(-1, vmid - 0.1), min(1, vmid + 0.1)

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, ax = plt.subplots(
        figsize=(7, 5),
        subplot_kw={"projection": proj} if proj else {},
    )
    im = ax.pcolormesh(lon, lat, corr, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       transform=ccrs_mod.PlateCarree() if ccrs else None)
    _add_geo(ax, lon, lat, ccrs, cfeature)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (annual)")
    ax.set_title(f"Annual temporal corr vs GCM — {variable} — {method}", fontsize=11)
    plt.tight_layout()
    out = plot_dir / f"temporal_corr_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_autocorr_comparison(gcm: xr.DataArray, pred: xr.DataArray, variable: str,
                             method: str, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    ac_gcm  = lag1_autocorr(gcm).values
    ac_pred = lag1_autocorr(pred).values
    diff = ac_pred - ac_gcm
    lon = gcm.lon.values
    lat = gcm.lat.values

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.5),
        subplot_kw={"projection": proj} if proj else {},
    )
    for ax, data, title, cmap, sym in [
        (axes[0], ac_gcm,  "GCM lag-1 autocorr",      "RdYlGn", False),
        (axes[1], ac_pred, f"{method} lag-1 autocorr", "RdYlGn", False),
        (axes[2], diff,    "Difference (pred − GCM)",  "RdBu_r", True),
    ]:
        vm = np.nanpercentile(np.abs(data), 98) if sym else None
        kw = {"cmap": cmap, "transform": ccrs_mod.PlateCarree() if ccrs else None}
        kw.update({"vmin": -vm, "vmax": vm} if sym else {"vmin": -1, "vmax": 1})
        im = ax.pcolormesh(lon, lat, data, **kw)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Lag-1 autocorrelation — {variable}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / f"autocorr_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_annual_anomalies(gcm: xr.DataArray, pred: xr.DataArray, variable: str,
                          method: str, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    gcm_ann  = gcm.mean(["lat", "lon"]).groupby("time.year").mean("time")
    pred_ann = pred.mean(["lat", "lon"]).groupby("time.year").mean("time")

    fig, ax = plt.subplots(figsize=(9, 4))
    # Use each array's own year axis to avoid length mismatches (e.g. consistency
    # prediction time axis may differ slightly from per-variable gcm time axis)
    ax.plot(gcm_ann.year.values,  gcm_ann.values,  "k-",  label="GCM",    lw=1.5)
    ax.plot(pred_ann.year.values, pred_ann.values,  "b--", label=method,   lw=1.5)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlabel("Year"); ax.set_ylabel(variable)
    ax.set_title(f"Annual means — {variable} — {method}", fontsize=12)
    plt.tight_layout()
    out = plot_dir / f"annual_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_intervar_corr(obs1: xr.DataArray, obs2: xr.DataArray,
                       pred1: xr.DataArray, pred2: xr.DataArray,
                       var1: str, var2: str,
                       method: str, plot_dir: Path,
                       diff_vmax: float = 0.5) -> None:
    """Three-panel map (ERA5 corr / pred corr / bias) + separate ERA5 and pred PNGs.

    Separate files allow the dashboard to show ERA5 once with all methods alongside.
    """
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    obs_corr  = _pointwise_corr(obs1, obs2)
    pred_corr = _pointwise_corr(pred1, pred2)
    diff      = pred_corr - obs_corr
    lon = obs1.lon.values
    lat = obs1.lat.values

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None

    # Combined 3-panel figure (backward compat)
    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.5),
        subplot_kw={"projection": proj} if proj else {},
    )
    for ax, data, title, cmap, vmin, vmax in [
        (axes[0], obs_corr,  f"ERA5 r({var1},{var2})",     "RdBu_r", -1,         1),
        (axes[1], pred_corr, f"{method} r({var1},{var2})", "RdBu_r", -1,         1),
        (axes[2], diff,      "Bias (pred − ERA5)",         "PuOr",   -diff_vmax, diff_vmax),
    ]:
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs_mod.PlateCarree() if ccrs else None)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Inter-variable correlation — {var1} × {var2} — {method}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / f"intervar_corr_{var1}_{var2}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Save separate panels so the dashboard can overlay all methods
    for data, suffix, cmap, vmin, vmax, title in [
        (obs_corr,  "era5", "RdBu_r", -1,         1,         f"ERA5 r({var1},{var2})"),
        (pred_corr, "pred", "RdBu_r", -1,         1,         f"{method} r({var1},{var2})"),
        (diff,      "bias", "PuOr",   -diff_vmax, diff_vmax, f"Bias pred−ERA5"),
    ]:
        fig2, ax2 = plt.subplots(
            figsize=(6, 4.5),
            subplot_kw={"projection": proj} if proj else {},
        )
        im2 = ax2.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs_mod.PlateCarree() if ccrs else None)
        _add_geo(ax2, lon, lat, ccrs, cfeature)
        ax2.set_title(title, fontsize=10)
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        plt.tight_layout()
        fig2.savefig(plot_dir / f"intervar_corr_{var1}_{var2}_{suffix}.png",
                     dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  saved intervar_corr_{var1}_{var2}*.png")


def plot_radar(metrics_vs_obs: dict, metrics_vs_gcm: dict,
               method: str, plot_dir: Path) -> None:
    """Radar plot with available metrics (lower = better, except temporal_corr*)."""
    import matplotlib.pyplot as plt

    all_metrics: dict[str, float] = {}
    for k, v in metrics_vs_obs.items():
        all_metrics[k] = float(v) if np.isfinite(v) else np.nan
    for k, v in metrics_vs_gcm.items():
        all_metrics[k] = float(1 - v) if k.startswith("temporal_corr") else float(v)

    labels = list(all_metrics.keys())
    vals = np.array([all_metrics[k] for k in labels], dtype=float)

    if len(labels) < 3:
        print("  too few metrics for radar plot, skipping")
        return

    norm = np.nanmax(np.abs(vals)) or 1.0
    vals_norm = vals / norm

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_plot = np.concatenate([vals_norm, vals_norm[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles, vals_plot, "b-o", lw=2)
    ax.fill(angles, vals_plot, alpha=0.15, color="blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"Benchmark summary — {method}", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    out = plot_dir / "radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")
