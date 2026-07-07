from pathlib import Path


from src.evaluation.metrics.inter_variables_metrics import _pointwise_corr
from src.evaluation.metrics.scalar_metrics_gcm import lag1_autocorr, temporal_corr_annual_map
from src.plots.utils import _add_geo, _try_cartopy
import xarray as xr
import numpy as np

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
