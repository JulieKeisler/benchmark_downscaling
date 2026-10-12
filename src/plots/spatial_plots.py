import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics.inter_variables_metrics import _pointwise_corr
from metrics.scalar_metrics_gcm import lag1_autocorr, temporal_corr_annual_map
from plots.utils import _add_geo, _try_cartopy
import xarray as xr
import numpy as np


def _sm(cmap, vmin, vmax):
    """Return a ScalarMappable for a shared colorbar (no data needed)."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=plt.get_cmap(cmap), norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    return sm


def plot_bias_map(obs: xr.DataArray, pred: xr.DataArray, variable: str,
                  method: str, plot_dir: Path,
                  bias_vmax: float | None = None,
                  std_vmax: float | None = None) -> None:
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    obs_mean  = obs.mean("time").values
    obs_std   = obs.std("time").values
    mean_bias = pred.mean("time").values - obs_mean
    std_bias  = pred.std("time").values  - obs_std
    lon = obs.lon.values
    lat = obs.lat.values

    vm_mean = bias_vmax if bias_vmax is not None else float(np.nanpercentile(np.abs(mean_bias), 98))
    vm_std  = std_vmax  if std_vmax  is not None else float(np.nanpercentile(np.abs(std_bias),  98))

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5),
        subplot_kw={"projection": proj} if proj else {},
        layout="constrained",
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
    out = plot_dir / f"bias_map_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")

    np.savez_compressed(
        plot_dir / f"bias_data_{variable}.npz",
        mean_bias=mean_bias, std_bias=std_bias,
        obs_mean=obs_mean, obs_std=obs_std, lat=lat, lon=lon,
    )


def plot_spatial_comparison(
    method_data: "dict[str, dict]",
    variable: str,
    out_path: "Path",
) -> None:
    """One figure per variable: ERA5 reference + all-method bias maps, shared colorbars."""
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    methods = list(method_data.keys())
    N = len(methods)
    ncols = 1 + N

    all_mean_bias = np.stack([method_data[m]["mean_bias"] for m in methods])
    all_std_bias  = np.stack([method_data[m]["std_bias"]  for m in methods])
    vm_mean = max(float(np.nanpercentile(np.abs(all_mean_bias), 98)), 1e-9)
    vm_std  = max(float(np.nanpercentile(np.abs(all_std_bias),  98)), 1e-9)

    ref      = next(iter(method_data.values()))
    obs_mean = ref["obs_mean"]
    obs_std  = ref["obs_std"]
    lat      = ref["lat"]
    lon      = ref["lon"]

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(
        2, ncols, figsize=(4.2 * ncols, 7.5),
        subplot_kw={"projection": proj} if proj else {},
        squeeze=False, layout="constrained",
    )

    def _panel(ax, data, cmap, vmin, vmax, title):
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs_mod.PlateCarree() if ccrs else None)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=9)
        return im

    _panel(axes[0, 0], obs_mean, "viridis",
           float(np.nanmin(obs_mean)), float(np.nanmax(obs_mean)), "ERA5 mean")
    _panel(axes[1, 0], obs_std,  "plasma",
           0.0, float(np.nanmax(obs_std)), "ERA5 std")
    fig.colorbar(_sm("viridis", np.nanmin(obs_mean), np.nanmax(obs_mean)),
                 ax=axes[0, 0], shrink=0.85, pad=0.03, format="%.3g")
    fig.colorbar(_sm("plasma", 0.0, np.nanmax(obs_std)),
                 ax=axes[1, 0], shrink=0.85, pad=0.03, format="%.3g")

    for col, m in enumerate(methods, start=1):
        d = method_data[m]
        _panel(axes[0, col], d["mean_bias"], "RdBu_r", -vm_mean, vm_mean, f"{m}\nmean bias")
        _panel(axes[1, col], d["std_bias"],  "PuOr",   -vm_std,  vm_std,  f"{m}\nstd bias")

    fig.colorbar(_sm("RdBu_r", -vm_mean, vm_mean),
                 ax=axes[0, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Mean bias (pred − ERA5)", format="%.3g")
    fig.colorbar(_sm("PuOr", -vm_std, vm_std),
                 ax=axes[1, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Std bias (pred − ERA5)", format="%.3g")

    fig.suptitle(f"Spatial bias — {variable}", fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def plot_temporal_corr(gcm: xr.DataArray, pred: xr.DataArray, variable: str,
                       method: str, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    corr = temporal_corr_annual_map(gcm, pred).values
    lon = gcm.lon.values
    lat = gcm.lat.values

    np.savez_compressed(
        plot_dir / f"temporal_corr_data_{variable}.npz",
        corr=corr, lat=lat, lon=lon,
    )

    vmin, vmax = 0.0, 1.0

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, ax = plt.subplots(figsize=(7, 5),
                           subplot_kw={"projection": proj} if proj else {},
                           layout="constrained")
    im = ax.pcolormesh(lon, lat, corr, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       transform=ccrs_mod.PlateCarree() if ccrs else None)
    _add_geo(ax, lon, lat, ccrs, cfeature)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r (annual)")
    ax.set_title(f"Annual temporal corr vs GCM — {variable} — {method}", fontsize=11)
    out = plot_dir / f"temporal_corr_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_autocorr_comparison(gcm: xr.DataArray, pred: xr.DataArray, variable: str,
                             method: str, plot_dir: Path) -> None:
    """3-panel map: GCM reference autocorr | method autocorr | bias (pred - GCM)."""
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    ac_gcm  = lag1_autocorr(gcm).values
    ac_pred = lag1_autocorr(pred).values
    diff    = ac_pred - ac_gcm
    lon = gcm.lon.values
    lat = gcm.lat.values

    np.savez_compressed(
        plot_dir / f"autocorr_data_{variable}.npz",
        ac_gcm=ac_gcm, ac_pred=ac_pred, lat=lat, lon=lon,
    )

    vm = max(float(np.nanpercentile(np.abs(diff), 98)), 1e-9)
    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             subplot_kw={"projection": proj} if proj else {},
                             squeeze=False, layout="constrained")
    tr = ccrs_mod.PlateCarree() if ccrs else None

    for ax, data, title, cmap, vmin, vmax in [
        (axes[0, 0], ac_gcm,  f"GCM lag-1 autocorr", "RdYlGn", -1, 1),
        (axes[0, 1], ac_pred, f"{method} lag-1 autocorr", "RdYlGn", -1, 1),
        (axes[0, 2], diff,    f"{method} bias (pred−GCM)", "RdBu_r", -vm, vm),
    ]:
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=tr)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Lag-1 autocorrelation — {variable} — {method}",
                 fontsize=11, fontweight="bold")
    out = plot_dir / f"autocorr_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_intervar_corr(obs1: xr.DataArray, obs2: xr.DataArray,
                       pred1: xr.DataArray, pred2: xr.DataArray,
                       var1: str, var2: str,
                       method: str, plot_dir: Path,
                       diff_vmax: float = 0.5) -> None:
    """1-panel map: method inter-var correlation."""
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    era5_corr = _pointwise_corr(obs1, obs2)
    pred_corr = _pointwise_corr(pred1, pred2)
    diff      = pred_corr - era5_corr
    lon = obs1.lon.values
    lat = obs1.lat.values

    np.savez_compressed(
        plot_dir / f"intervar_corr_data_{var1}_{var2}.npz",
        era5_corr=era5_corr, pred_corr=pred_corr, diff=diff, lat=lat, lon=lon,
    )

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, ax = plt.subplots(figsize=(7, 5),
                           subplot_kw={"projection": proj} if proj else {},
                           layout="constrained")
    im = ax.pcolormesh(lon, lat, pred_corr, cmap="RdBu_r", vmin=-1, vmax=1,
                       transform=ccrs_mod.PlateCarree() if ccrs else None)
    _add_geo(ax, lon, lat, ccrs, cfeature)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title(f"{method} r({var1},{var2})", fontsize=11)
    out = plot_dir / f"intervar_corr_{var1}_{var2}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    # Save ERA5 reference map as well for comparison basis
    fig_era5, ax_era5 = plt.subplots(figsize=(7, 5),
                                   subplot_kw={"projection": proj} if proj else {},
                                   layout="constrained")
    im_era5 = ax_era5.pcolormesh(lon, lat, era5_corr, cmap="RdBu_r", vmin=-1, vmax=1,
                                 transform=ccrs_mod.PlateCarree() if ccrs else None)
    _add_geo(ax_era5, lon, lat, ccrs, cfeature)
    plt.colorbar(im_era5, ax=ax_era5, shrink=0.8, label="Pearson r")
    ax_era5.set_title(f"ERA5 reference r({var1},{var2})", fontsize=11)
    out_era5 = plot_dir / f"intervar_corr_{var1}_{var2}_era5.png"
    plt.savefig(out_era5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved intervar_corr_{var1}_{var2}.png and _era5.png")


def plot_delta_map(
    pred_delta: np.ndarray,
    gcm_delta:  np.ndarray,
    bias:       np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    variable: str,
    method: str,
    period_label: str,
    plot_dir: Path,
) -> None:
    """3-panel map: GCM warming signal | method warming signal | bias (pred - GCM)."""
    np.savez_compressed(
        plot_dir / f"delta_data_{variable}_{period_label}.npz",
        pred_delta=pred_delta, gcm_delta=gcm_delta, bias=bias, lat=lat, lon=lon,
    )

    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod

    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                             subplot_kw={"projection": proj} if proj else {},
                             squeeze=False, layout="constrained")

    vm_sig = max(float(np.nanpercentile(np.abs(np.concatenate([pred_delta.ravel(), gcm_delta.ravel()])), 98)), 1e-9)
    vm_bias = max(float(np.nanpercentile(np.abs(bias), 98)), 1e-9)
    tr = ccrs_mod.PlateCarree() if ccrs else None

    for ax, data, title, vm in [
        (axes[0, 0], gcm_delta,  f"GCM Δ{variable} {period_label.replace('_','–')}", vm_sig),
        (axes[0, 1], pred_delta, f"{method} Δ{variable}", vm_sig),
        (axes[0, 2], bias,       f"{method} bias (pred−GCM)", vm_bias),
    ]:
        im = ax.pcolormesh(lon, lat, data, cmap="RdBu_r", vmin=-vm, vmax=vm,
                           transform=tr)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Climate change signal — {variable} {period_label.replace('_','–')}",
                 fontsize=11, fontweight="bold")
    out = plot_dir / f"delta_{period_label}_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved delta_{period_label}_{variable}.png")


def plot_sample_days(
    obs: xr.DataArray,
    pred: xr.DataArray,
    gcm: xr.DataArray | None,
    variable: str,
    method: str,
    plot_dir: Path,
    n_days: int = 5,
    seed: int | None = None,
) -> None:
    """Select a few random days and save the daily fields for the 'samples' plot.

    The same days are used across all methods (seed derived from the variable
    name by default) so the dashboard can compare GCM / ERA5 / methods
    column-wise on identical dates.

    Saves ``samples_data_{variable}.npz`` with the raw daily fields:
        era5  [n_days, H, W],  gcm  [n_days, H, W],  pred  [n_days, H, W],
        dates [n_days] (str),  lat, lon
    """
    import zlib

    if obs.time.size < n_days:
        print(f"  skip sample-days {variable}: only {obs.time.size} timestamps")
        return

    rng_seed = zlib.crc32(variable.encode("utf-8")) if seed is None else seed
    rng = np.random.default_rng(rng_seed)
    idx = np.sort(rng.choice(obs.time.size, size=n_days, replace=False))
    dates = [str(obs.time.values[i])[:10] for i in idx]

    era5_a = obs.values[idx]
    pred_a = pred.values[idx]
    gcm_a  = gcm.values[idx] if gcm is not None else None

    np.savez_compressed(
        plot_dir / f"samples_data_{variable}.npz",
        era5=era5_a, gcm=gcm_a, pred=pred_a,
        dates=np.array(dates),
        lat=obs.lat.values, lon=obs.lon.values,
    )
    print(f"  saved samples_data_{variable}.npz ({', '.join(dates)})")
