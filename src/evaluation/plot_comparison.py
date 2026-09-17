#!/usr/bin/env python3
"""Generate combined comparison figures across all evaluated methods.

After running evaluate.py for each method, call this script once to produce:
  - spatial_{variable}.png       : ERA5 ref + all method bias maps (shared colorbar)
  - temporal_corr_{variable}.png : all method annual correlation maps (shared colorbar)
  - autocorr_{variable}.png      : GCM ref + all method autocorr diffs (shared colorbar)
  - intervar_corr_{v1}_{v2}.png  : ERA5 ref + all method inter-var corr (shared colorbar)
  - delta_heatmap.png            : delta change scores per variable × method

Usage:
    python plot_comparison.py --results-dir /path/to/results \\
        [--variables tas pr sfcWind sfcWindmax psl] \\
        [--methods bicubic bcsd cdft ...] \\
        [--out-dir /path/to/comparison_plots]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from plots import plot_spatial_comparison
from plots.utils import _add_geo, _try_cartopy
from plots.spatial_plots import _sm

DEFAULT_VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
VAR_PAIRS = [
    ("tas", "pr"), ("tas", "sfcWind"), ("tas", "sfcWindmax"),
    ("pr", "sfcWind"), ("pr", "psl"), ("sfcWind", "sfcWindmax"),
]
METHOD_LABELS = {
    "bicubic":     "Bicubic",
    "bcsd":        "BCSD",
    "cdft":        "CDF-t",
    "idownscale":  "iDownscale",
    "consistency": "CM",
    "cm":          "CM",
}


# ── Data loaders ─────────────────────────────────────────────────────────────

def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}

def _load_bias_data(method_dir: Path, variable: str) -> dict | None:
    return _load(method_dir / "plots" / variable / f"bias_data_{variable}.npz")

def _load_temporal_corr_data(method_dir: Path, variable: str) -> dict | None:
    return _load(method_dir / "plots" / variable / f"temporal_corr_data_{variable}.npz")

def _load_autocorr_data(method_dir: Path, variable: str) -> dict | None:
    return _load(method_dir / "plots" / variable / f"autocorr_data_{variable}.npz")

def _load_intervar_corr_data(method_dir: Path, v1: str, v2: str) -> dict | None:
    return _load(method_dir / "plots" / f"intervar_corr_data_{v1}_{v2}.npz")

def _load_delta_data(method_dir: Path, variable: str, period: str) -> dict | None:
    return _load(method_dir / "plots" / variable / f"delta_data_{variable}_{period}.npz")


# ── Comparison plot helpers ───────────────────────────────────────────────────

def _geo_panels(nrows, ncols, figsize):
    import matplotlib.pyplot as plt
    ccrs, cfeature = _try_cartopy()
    import cartopy.crs as ccrs_mod
    proj = ccrs_mod.PlateCarree() if ccrs is not None else None
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        subplot_kw={"projection": proj} if proj else {},
        squeeze=False, layout="constrained",
    )
    return fig, axes, ccrs, cfeature


def _pmesh(ax, lon, lat, data, cmap, vmin, vmax, title, ccrs, cfeature):
    import cartopy.crs as ccrs_mod
    im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                       transform=ccrs_mod.PlateCarree() if ccrs else None)
    _add_geo(ax, lon, lat, ccrs, cfeature)
    ax.set_title(title, fontsize=9)
    return im


def _save(fig, out_path: Path):
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ── Temporal correlation comparison (2 rows) ───────────────────────────────────

def plot_temporal_corr_comparison(method_data: dict, variable: str, out_path: Path) -> None:
    """2-row figure:
       Row 0: Annual temporal correlation vs GCM (0-1)
       Row 1: Bias = 1 - correlation (0 = perfect)
    Shared colorbar per row."""
    methods = list(method_data.keys())
    N = len(methods)
    ref = next(iter(method_data.values()))
    lat, lon = ref["lat"], ref["lon"]

    # Row 0: correlation [0, 1]
    corr_vmin, corr_vmax = 0.0, 1.0

    # Row 1: bias = 1 - correlation [0, 1] (0 = perfect)
    bias_vmin, bias_vmax = 0.0, 1.0

    fig, axes, ccrs, cfeature = _geo_panels(2, N, (5 * N, 8.5))

    # Row 0: Correlation
    for col, m in enumerate(methods):
        _pmesh(axes[0, col], lon, lat, method_data[m]["corr"],
               "RdYlGn", corr_vmin, corr_vmax, f"{m}\nannual corr vs GCM", ccrs, cfeature)
    fig.colorbar(_sm("RdYlGn", corr_vmin, corr_vmax),
                 ax=axes[0, :].tolist(), shrink=0.8, pad=0.02,
                 label="Pearson r (annual)")

    # Row 1: Bias (1 - corr, 0 = perfect)
    for col, m in enumerate(methods):
        bias = 1.0 - method_data[m]["corr"]
        _pmesh(axes[1, col], lon, lat, bias,
               "RdBu_r", bias_vmin, bias_vmax, f"{m}\nbias (1 - r)", ccrs, cfeature)
    fig.colorbar(_sm("RdBu_r", bias_vmin, bias_vmax),
                 ax=axes[1, :].tolist(), shrink=0.8, pad=0.02,
                 label="Bias (1 - r, lower = better)")

    fig.suptitle(f"Annual temporal correlation vs GCM — {variable}",
                 fontsize=12, fontweight="bold")
    _save(fig, out_path)


# ── 4-row Spatial bias comparison ────────────────────────────────────────────

def plot_spatial_bias_4row(method_data: dict, variable: str, out_path: Path) -> None:
    """
    4-row figure per variable:
      Row 0: Mean (ERA5 + methods)
      Row 1: Std (ERA5 + methods)
      Row 2: Mean bias (pred - ERA5)
      Row 3: Std bias (pred - ERA5)
    Shared colorbar per row.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs_mod
    methods = list(method_data.keys())
    N = len(methods)
    ncols = 1 + N
    ref = next(iter(method_data.values()))
    lat, lon = ref["lat"], ref["lon"]

    obs_mean = ref["obs_mean"]
    obs_std  = ref["obs_std"]

    all_mean_bias = np.stack([method_data[m]["mean_bias"] for m in methods])
    all_std_bias  = np.stack([method_data[m]["std_bias"]  for m in methods])

    # Compute method mean and std from bias + obs
    all_pred_mean = np.stack([obs_mean + method_data[m]["mean_bias"] for m in methods])
    all_pred_std  = np.stack([obs_std  + method_data[m]["std_bias"]  for m in methods])

    vm_mean = max(float(np.nanpercentile(np.abs(all_mean_bias), 98)), 1e-9)
    vm_std  = max(float(np.nanpercentile(np.abs(all_std_bias),  98)), 1e-9)

    # Range for mean/std values (absolute, not bias)
    all_mean_vals = np.concatenate([obs_mean.ravel()] + [all_pred_mean[i].ravel() for i in range(N)])
    all_std_vals  = np.concatenate([obs_std.ravel()]  + [all_pred_std[i].ravel()  for i in range(N)])
    obs_mean_min, obs_mean_max = float(np.nanmin(all_mean_vals)), float(np.nanmax(all_mean_vals))
    obs_std_min, obs_std_max   = 0.0, float(np.nanmax(all_std_vals))

    fig, axes, ccrs, cfeature = _geo_panels(4, ncols, (4.2 * ncols, 14.0))

    def _panel(ax, data, cmap, vmin, vmax, title):
        im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax,
                           transform=ccrs_mod.PlateCarree() if ccrs else None)
        _add_geo(ax, lon, lat, ccrs, cfeature)
        ax.set_title(title, fontsize=9)
        return im

    # Row 0: Mean (ERA5 + methods)
    _panel(axes[0, 0], obs_mean, "viridis", obs_mean_min, obs_mean_max, "ERA5 mean")
    fig.colorbar(_sm("viridis", obs_mean_min, obs_mean_max),
                 ax=axes[0, 0], shrink=0.85, pad=0.03, format="%.3g")
    for col, m in enumerate(methods, start=1):
        pred_mean = obs_mean + method_data[m]["mean_bias"]
        _panel(axes[0, col], pred_mean, "viridis", obs_mean_min, obs_mean_max, f"{m}\nmean")
    fig.colorbar(_sm("viridis", obs_mean_min, obs_mean_max),
                 ax=axes[0, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Mean", format="%.3g")

    # Row 1: Std (ERA5 + methods)
    _panel(axes[1, 0], obs_std, "plasma", obs_std_min, obs_std_max, "ERA5 std")
    fig.colorbar(_sm("plasma", obs_std_min, obs_std_max),
                 ax=axes[1, 0], shrink=0.85, pad=0.03, format="%.3g")
    for col, m in enumerate(methods, start=1):
        pred_std = obs_std + method_data[m]["std_bias"]
        _panel(axes[1, col], pred_std, "plasma", obs_std_min, obs_std_max, f"{m}\nstd")
    fig.colorbar(_sm("plasma", obs_std_min, obs_std_max),
                 ax=axes[1, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Std", format="%.3g")

    # Row 2: Mean bias
    for col, m in enumerate(methods, start=1):
        _panel(axes[2, col], method_data[m]["mean_bias"], "RdBu_r", -vm_mean, vm_mean, f"{m}\nmean bias")
    axes[2, 0].set_visible(False)
    fig.colorbar(_sm("RdBu_r", -vm_mean, vm_mean),
                 ax=axes[2, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Mean bias (pred − ERA5)", format="%.3g")

    # Row 3: Std bias
    for col, m in enumerate(methods, start=1):
        _panel(axes[3, col], method_data[m]["std_bias"], "PuOr", -vm_std, vm_std, f"{m}\nstd bias")
    axes[3, 0].set_visible(False)
    fig.colorbar(_sm("PuOr", -vm_std, vm_std),
                 ax=axes[3, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Std bias (pred − ERA5)", format="%.3g")

    fig.suptitle(f"Spatial bias (4-row) — {variable}", fontsize=12, fontweight="bold")
    _save(fig, out_path)


# ── Autocorrelation comparison (2 rows) ────────────────────────────────────────

def plot_autocorr_all(method_data: dict, variable: str, out_path: Path) -> None:
    """2-row figure:
       Row 0: GCM autocorr + method autocorr
       Row 1: Method bias (pred - GCM)
    Shared colorbar per row."""
    import matplotlib.pyplot as plt
    methods = list(method_data.keys())
    N = len(methods)
    ref = next(iter(method_data.values()))
    lat, lon = ref["lat"], ref["lon"]
    ac_gcm = ref["ac_gcm"]

    all_pred = np.stack([method_data[m]["ac_pred"] for m in methods])
    vm_pred = max(float(np.nanpercentile(np.abs(all_pred), 98)), 1e-9)

    all_diff = np.stack([method_data[m]["ac_pred"] - method_data[m]["ac_gcm"] for m in methods])
    vm_diff = max(float(np.nanpercentile(np.abs(all_diff), 98)), 1e-9)

    ncols = 1 + N
    fig, axes, ccrs, cfeature = _geo_panels(2, ncols, (4.5 * ncols, 9.0))

    # Row 0: GCM autocorr + method autocorr
    _pmesh(axes[0, 0], lon, lat, ac_gcm, "RdYlGn", -1, 1, "GCM lag-1 autocorr", ccrs, cfeature)
    fig.colorbar(_sm("RdYlGn", -1, 1), ax=axes[0, 0], shrink=0.85, pad=0.03,
                 label="lag-1 autocorr")
    for col, m in enumerate(methods, start=1):
        _pmesh(axes[0, col], lon, lat, method_data[m]["ac_pred"], "RdYlGn", -vm_pred, vm_pred,
               f"{m}\nlag-1 autocorr", ccrs, cfeature)
    fig.colorbar(_sm("RdYlGn", -vm_pred, vm_pred),
                 ax=axes[0, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="lag-1 autocorr")

    # Row 1: Bias (pred - GCM)
    axes[1, 0].set_visible(False)
    for col, m in enumerate(methods, start=1):
        diff = method_data[m]["ac_pred"] - method_data[m]["ac_gcm"]
        _pmesh(axes[1, col], lon, lat, diff, "RdBu_r", -vm_diff, vm_diff,
               f"{m}\nbias vs GCM", ccrs, cfeature)
    fig.colorbar(_sm("RdBu_r", -vm_diff, vm_diff),
                 ax=axes[1, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="Δ lag-1 autocorr (pred − GCM)")

    fig.suptitle(f"Lag-1 autocorrelation — {variable}", fontsize=12, fontweight="bold")
    _save(fig, out_path)


# ── Inter-variable correlation comparison ────────────────────────────────────

def plot_intervar_corr_comparison(method_data: dict, v1: str, v2: str, out_path: Path) -> None:
    """Two-row figure: top = ERA5 + all method corr, bottom = per-method bias (pred - ERA5)."""
    methods = list(method_data.keys())
    N = len(methods)
    ref = next(iter(method_data.values()))
    lat, lon = ref["lat"], ref["lon"]

    # shared bias range across all methods
    all_diffs = np.concatenate([method_data[m]["diff"].ravel() for m in methods])
    vm_diff = max(float(np.nanpercentile(np.abs(all_diffs), 98)), 1e-4)

    ncols = 1 + N
    fig, axes, ccrs, cfeature = _geo_panels(2, ncols, (4.5 * ncols, 9.0))

    # Row 0: ERA5 corr | method corr maps
    _pmesh(axes[0, 0], lon, lat, ref["era5_corr"], "RdBu_r", -1, 1,
           f"ERA5 r({v1},{v2})", ccrs, cfeature)
    for col, m in enumerate(methods, start=1):
        _pmesh(axes[0, col], lon, lat, method_data[m]["pred_corr"], "RdBu_r", -1, 1,
               f"{m} r({v1},{v2})", ccrs, cfeature)
    fig.colorbar(_sm("RdBu_r", -1, 1),
                 ax=axes[0, :].tolist(), shrink=0.8, pad=0.02, label="Pearson r")

    # Row 1: blank ERA5 slot | method bias maps (pred - ERA5)
    axes[1, 0].set_visible(False)
    for col, m in enumerate(methods, start=1):
        _pmesh(axes[1, col], lon, lat, method_data[m]["diff"], "RdBu_r", -vm_diff, vm_diff,
               f"{m} bias r({v1},{v2})", ccrs, cfeature)
    fig.colorbar(_sm("RdBu_r", -vm_diff, vm_diff),
                 ax=axes[1, 1:].tolist(), shrink=0.8, pad=0.02, label="Δr (pred − ERA5)")

    fig.suptitle(f"Inter-variable correlation — {v1} × {v2}",
                 fontsize=12, fontweight="bold")
    _save(fig, out_path)


# ── Delta spatial comparison ─────────────────────────────────────────────────

def plot_delta_comparison(method_data: dict, variable: str, period: str, out_path: Path) -> None:
    """Two-row figure: top = GCM signal + all method signals, bottom = per-method biases."""
    methods = list(method_data.keys())
    N = len(methods)
    ref = next(iter(method_data.values()))
    lat, lon = ref["lat"], ref["lon"]

    all_signals = np.concatenate(
        [ref["gcm_delta"].ravel()] + [method_data[m]["pred_delta"].ravel() for m in methods]
    )
    all_biases = np.concatenate([method_data[m]["bias"].ravel() for m in methods])
    vm_sig  = max(float(np.nanpercentile(np.abs(all_signals), 98)), 1e-9)
    vm_bias = max(float(np.nanpercentile(np.abs(all_biases), 98)), 1e-9)

    ncols = 1 + N
    fig, axes, ccrs, cfeature = _geo_panels(2, ncols, (4.5 * ncols, 9.0))
    plabel = period.replace("_", "–")

    # Row 0: GCM signal | method signals
    _pmesh(axes[0, 0], lon, lat, ref["gcm_delta"], "RdBu_r", -vm_sig, vm_sig,
           f"GCM Δ{variable} {plabel}", ccrs, cfeature)
    for col, m in enumerate(methods, start=1):
        _pmesh(axes[0, col], lon, lat, method_data[m]["pred_delta"], "RdBu_r", -vm_sig, vm_sig,
               f"{m} Δ{variable}", ccrs, cfeature)
    fig.colorbar(_sm("RdBu_r", -vm_sig, vm_sig),
                 ax=axes[0, :].tolist(), shrink=0.8, pad=0.02,
                 label=f"Δ{variable} (future − hist)")

    # Row 1: blank | method biases (pred - GCM)
    axes[1, 0].set_visible(False)
    for col, m in enumerate(methods, start=1):
        _pmesh(axes[1, col], lon, lat, method_data[m]["bias"], "RdBu_r", -vm_bias, vm_bias,
               f"{m} bias vs GCM", ccrs, cfeature)
    fig.colorbar(_sm("RdBu_r", -vm_bias, vm_bias),
                 ax=axes[1, 1:].tolist(), shrink=0.8, pad=0.02,
                 label="pred Δ − GCM Δ")

    fig.suptitle(f"Climate change signal — {variable} {plabel}",
                 fontsize=12, fontweight="bold")
    _save(fig, out_path)


# ── Delta change score heatmap ───────────────────────────────────────────────

def plot_delta_heatmap(methods_delta: dict, variables: list[str], out_path: Path) -> None:
    """Heatmap: rows = methods, columns = variables, two panels for 2040-2060 and 2080-2100."""
    import matplotlib.pyplot as plt

    periods = ["2040_2060", "2080_2100"]
    period_labels = ["2040–2060", "2080–2100"]
    methods = list(methods_delta.keys())
    vars_present = [v for v in variables if any(v in methods_delta[m] for m in methods)]

    if not vars_present or not methods:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(6, 2 * len(vars_present)), max(3, 1.2 * len(methods))))
    for ax, period, plabel in zip(axes, periods, period_labels):
        mat = np.full((len(methods), len(vars_present)), np.nan)
        for i, m in enumerate(methods):
            for j, v in enumerate(vars_present):
                val = methods_delta.get(m, {}).get(v, {}).get(period)
                if val is not None:
                    mat[i, j] = val

        vm = float(np.nanmax(np.abs(mat[np.isfinite(mat)]))) if np.any(np.isfinite(mat)) else 1.0
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=vm, aspect="auto")
        ax.set_xticks(range(len(vars_present)))
        ax.set_xticklabels(vars_present, fontsize=9)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_title(f"Delta score — {plabel}", fontsize=11)
        for i in range(len(methods)):
            for j in range(len(vars_present)):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if v > 0.6 * vm else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Delta score (lower = better)")

    fig.suptitle("Delta change score (climate signal preservation)", fontsize=12, fontweight="bold")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results-dir", required=True)
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    EXCLUDED_METHODS = {"automated_degradation"}
    if args.methods:
        method_dirs = {m: results_dir / m for m in args.methods}
    else:
        method_dirs = {
            d.name: d for d in sorted(results_dir.iterdir())
            if d.is_dir() and d.name != "comparison" and d.name not in EXCLUDED_METHODS
        }

    if not method_dirs:
        print("No method directories found.")
        return 1

    # ── Spatial bias ─────────────────────────────────────────────────────────
    print("\n=== Spatial bias ===")
    for variable in args.variables:
        print(f"\n--- {variable} ---")
        method_data = {}
        for name, mdir in method_dirs.items():
            data = _load_bias_data(mdir, variable)
            if data is None:
                print(f"  {name}: no bias_data_{variable}.npz — skipping")
                continue
            method_data[METHOD_LABELS.get(name, name)] = data
            print(f"  {name}: loaded")
        if method_data:
            plot_spatial_bias_4row(method_data, variable, out_dir / f"spatial_{variable}.png")

    # ── Temporal correlation ──────────────────────────────────────────────────
    print("\n=== Temporal correlation ===")
    for variable in args.variables:
        print(f"\n--- {variable} ---")
        method_data = {}
        for name, mdir in method_dirs.items():
            data = _load_temporal_corr_data(mdir, variable)
            if data is None:
                print(f"  {name}: no temporal_corr_data — skipping")
                continue
            method_data[METHOD_LABELS.get(name, name)] = data
            print(f"  {name}: loaded")
        if method_data:
            plot_temporal_corr_comparison(method_data, variable, out_dir / f"temporal_corr_{variable}.png")

    # ── Autocorrelation ───────────────────────────────────────────────────────
    print("\n=== Autocorrelation ===")
    for variable in args.variables:
        print(f"\n--- {variable} ---")
        method_data = {}
        for name, mdir in method_dirs.items():
            data = _load_autocorr_data(mdir, variable)
            if data is None:
                print(f"  {name}: no autocorr_data — skipping")
                continue
            method_data[METHOD_LABELS.get(name, name)] = data
            print(f"  {name}: loaded")
        if method_data:
            plot_autocorr_all(method_data, variable, out_dir / f"autocorr_{variable}.png")

    # ── Inter-variable correlation ────────────────────────────────────────────
    print("\n=== Inter-variable correlation ===")
    for v1, v2 in VAR_PAIRS:
        if v1 not in args.variables or v2 not in args.variables:
            continue
        print(f"\n--- {v1} × {v2} ---")
        method_data = {}
        for name, mdir in method_dirs.items():
            data = _load_intervar_corr_data(mdir, v1, v2)
            if data is None:
                print(f"  {name}: no intervar_corr_data — skipping")
                continue
            method_data[METHOD_LABELS.get(name, name)] = data
            print(f"  {name}: loaded")
        if method_data:
            plot_intervar_corr_comparison(method_data, v1, v2, out_dir / f"intervar_corr_{v1}_{v2}.png")

    # ── Delta spatial maps ────────────────────────────────────────────────────
    print("\n=== Delta spatial maps ===")
    for period in ["2040_2060", "2080_2100"]:
        for variable in args.variables:
            print(f"\n--- {variable} {period} ---")
            method_data = {}
            for name, mdir in method_dirs.items():
                data = _load_delta_data(mdir, variable, period)
                if data is None:
                    print(f"  {name}: no delta_data — skipping")
                    continue
                method_data[METHOD_LABELS.get(name, name)] = data
                print(f"  {name}: loaded")
            if method_data:
                plot_delta_comparison(method_data, variable, period,
                                      out_dir / f"delta_{period}_{variable}.png")

    # ── Delta change scores heatmap ───────────────────────────────────────────
    print("\n=== Delta change scores ===")
    methods_delta = {}
    for name, mdir in method_dirs.items():
        mf = mdir / "metrics.json"
        if not mf.exists():
            continue
        d = json.load(open(mf))
        if "_delta" in d:
            methods_delta[METHOD_LABELS.get(name, name)] = d["_delta"]
    if methods_delta:
        plot_delta_heatmap(methods_delta, args.variables, out_dir / "delta_heatmap.png")
    else:
        print("  No delta scores found (only pre-compute methods with --gcm-end > --end produce them).")

    print(f"\nComparison figures saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
