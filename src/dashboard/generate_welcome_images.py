#!/usr/bin/env python3
"""Generate pedagogic images for the dashboard welcome page.

Run via the TAM venv:
    /home/jkeisler/Documents/Code/TAM_downscaling/.venv/bin/python \
        src/dashboard/generate_welcome_images.py
"""
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT = Path(__file__).resolve().parent.parent.parent
ERA5_DIR = ROOT / "data" / "era5"
GCM_DIR = ROOT / "data" / "cmip6" / "ipsl_cm6a_lr"
RESULTS_DIR = ROOT / "results"
OUT = Path(__file__).resolve().parent / "static"
OUT.mkdir(exist_ok=True)

# France/Europe box covered by the 57×65 ERA5 grid (lat 41–55, lon −7..11)
MAP_EXTENT = [-7.5, 11.5, 40.5, 55.5]

def _style_geo_axes(ax, add_coastlines=True):
    """Set cartopy Plate-Carrée axes with gridlines and (optional) coastlines."""
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                      linewidth=0.4, color="0.6", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    if add_coastlines:
        ax.coastlines(resolution="10m", linewidth=0.7, color="0.15")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="0.25")
    return ax

# ── Image 1: Resolution gap ─────────────────────────────────────────────────
def make_resolution_gap():
    """GCM raw field vs ERA5 field for a single date (tas, winter), with coastlines."""
    date = "2005-01-15"
    era5 = xr.open_dataset(ERA5_DIR / "tas_2005.nc")
    gcm_hist = xr.open_dataset(GCM_DIR / "tas_historical_1960-2014.nc")
    gcm_ssp = xr.open_dataset(GCM_DIR / "tas_ssp370_2015-2100.nc")
    # Concat historical + ssp for year 2005
    gcm_all = xr.concat([gcm_hist, gcm_ssp], dim="time")
    gcm_day = gcm_all["tas"].sel(time=date).squeeze() - 273.15  # K → °C
    era5_day = era5["tas"].sel(time=date) - 273.15      # K → °C

    vmin = min(float(gcm_day.min()), float(era5_day.min()))
    vmax = max(float(gcm_day.max()), float(era5_day.max()))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmin+vmax)/2, vmax=vmax)

    crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 2, figsize=(8, 3.2), subplot_kw={"projection": crs}, constrained_layout=True,
    )
    # ERA5
    im0 = axes[0].pcolormesh(
        era5.longitude.values, era5.latitude.values, era5_day.values,
        cmap="RdBu_r", norm=norm, shading="auto", transform=crs,
    )
    _style_geo_axes(axes[0])
    axes[0].set_title("ERA5 (≈0.25°, 57×65)", fontsize=11, fontweight="bold")
    # GCM — coarsen to show raw grid cells
    im1 = axes[1].pcolormesh(
        gcm_all["lon"].values, gcm_all["lat"].values,
        gcm_day.values, cmap="RdBu_r", norm=norm, shading="nearest",
        transform=crs,
    )
    _style_geo_axes(axes[1])
    axes[1].set_title("IPSL-CM6A-LR (≈2.5°, 11×7)", fontsize=11, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=10)

    fig.suptitle(f"Resolution gap — {date}", fontsize=11, fontweight="bold")
    fig.savefig(OUT / "resolution_gap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT / 'resolution_gap.png'}")


# ── Image 2: Bias comparison across methods ──────────────────────────────────
def _load_bias_map(method, variable, plot_type="bias_map"):
    """Load a per-method spatial field produced by evaluate.py.

    Sources the same npz files as the dashboard 'Spatial maps' tab, so the
    welcome-page figure is guaranteed to show data whenever the maps tab can.
    The evaluated outputs (plots/) always live under results/<method>, even for
    pre-computed methods whose raw predictions sit in results/consistency.
    Returns (lat, lon, array) or None.
    """
    m_dir = RESULTS_DIR / method / "plots" / variable
    npz_file = m_dir / f"bias_data_{variable}.npz"
    try:
        d = np.load(npz_file)
        return d["lat"], d["lon"], d["mean_bias"]
    except (FileNotFoundError, KeyError, OSError):
        return None


def make_bias_comparison():
    """Side-by-side bias maps for bicubic vs DOTC vs CM, variable=tas (cartopy)."""
    variable = "tas"
    methods = [("bicubic", "Bicubic"), ("dotc", "DOTC"), ("cm", "CM")]
    crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5), subplot_kw={"projection": crs}, constrained_layout=True,
    )
    vmin, vmax = -3, 3
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = None
    for ax, (method, label) in zip(axes, methods):
        try:
            data = _load_bias_map(method, variable)
            if data is None:
                raise FileNotFoundError(method)
            lat, lon, bias = data
            if bias is None or np.all(np.isnan(bias)):
                raise ValueError("empty bias field")
            im = ax.pcolormesh(lon, lat, bias, cmap="RdBu_r", norm=norm, transform=crs)
        except Exception as e:
            ax.text(0.5, 0.5, f"No data\n({method})", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
        ax.set_title(label, fontsize=11, fontweight="bold")
        _style_geo_axes(ax)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
        cbar.set_label(f"{variable.title()} bias (°C)", fontsize=10)
    fig.suptitle("Temperature bias (Prediction − ERA5) — climatological mean", fontsize=13, fontweight="bold")
    fig.savefig(OUT / "bias_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT / 'bias_comparison.png'}")


# ── Image 3: Method spectrum (simple vs complex) ─────────────────────────────
def make_method_spectrum():
    """Bar chart showing method complexity: simple → complex."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    methods = ["Bicubic", "BCSD", "CDF-t", "DOTC", "iDownscale", "CorrDiff-QM", "CM"]
    complexity = [0, 1, 2, 2, 5, 8, 7]  # relative complexity score
    colors = ["#4C72B0", "#55A868", "#55A868", "#55A868", "#C44E52", "#8172B2", "#8172B2"]
    categories = ["None", "Statistical", "Statistical", "Statistical", "Neural", "Neural", "Neural"]
    bars = ax.barh(methods, complexity, color=colors, edgecolor="white", height=0.6)
    for bar, cat in zip(bars, categories):
        w = bar.get_width()
        ax.text(w + 0.15, bar.get_y() + bar.get_height()/2, cat,
                va="center", fontsize=9, color="gray")
    ax.set_xlabel("Relative complexity", fontsize=10)
    ax.set_xlim(0, 11)
    ax.set_title("Method spectrum: statistical → neural", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.savefig(OUT / "method_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {OUT / 'method_spectrum.png'}")


if __name__ == "__main__":
    print("Generating welcome page images...")
    make_resolution_gap()
    make_bias_comparison()
    make_method_spectrum()
    print("Done.")
