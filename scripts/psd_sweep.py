#!/usr/bin/env python3
"""PSD sweep to calibrate t* for consistency model downscaling.

Computes radially-averaged 2D power spectral densities of ERA5 and
upsampled CMIP6 fields, finds their intersection wavenumber k*, and
derives t* = N * sqrt(PSD(k*)) for the CM variance schedule (eq. 11 of
Hess et al. 2025, Nature Machine Intelligence).

Usage:
    python scripts/psd_sweep.py \
        --era5-dir /scratch/jkeisler/era5 \
        --cmip6-dir /scratch/jkeisler/cmip6/IPSL-CM6A-LR \
        --output-dir results/psd \
        --variable pr \
        --year-start 2000 --year-end 2014
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

LOG_EPS = 1e-6
LOG_VARS = {"pr"}

# Padded grid dimensions used in training (must match train_consistency.py)
H_PAD, W_PAD = 64, 80  # grid size N used in t* = N * sqrt(PSD)


# ── PSD utilities ─────────────────────────────────────────────────────────────

def mean_2d_psd(fields: np.ndarray) -> np.ndarray:
    """Mean 2D power spectral density over time.

    fields: [T, H, W] — already detrended/normalized
    Returns: [H, W] PSD (power per grid cell per frequency bin)
    """
    T, H, W = fields.shape
    psds = []
    for t in range(T):
        f = fields[t] - fields[t].mean()
        f2d = np.fft.fft2(f)
        power = np.abs(f2d) ** 2 / (H * W)
        psds.append(power)
    return np.mean(psds, axis=0)


def radial_profile(power_2d: np.ndarray, H: int, W: int,
                   n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged PSD from 2D power spectrum.

    Returns k_centers [n_bins] and psd_radial [n_bins].
    k is in cycles per grid point (0 to 0.5 = Nyquist).
    """
    kx = np.fft.fftfreq(W)   # cycles per grid point
    ky = np.fft.fftfreq(H)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = min(0.5, K.max())
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    psd_radial = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i + 1])
        if mask.sum() > 0:
            psd_radial[i] = power_2d[mask].mean()

    return k_centers, psd_radial


# ── Data loading ──────────────────────────────────────────────────────────────

def _log_transform(x: np.ndarray) -> np.ndarray:
    return np.log(x + LOG_EPS) - np.log(LOG_EPS)


def load_era5(era5_dir: Path, variable: str,
              year_start: int, year_end: int) -> np.ndarray:
    files = sorted(era5_dir.glob(f"{variable}_*.nc"))
    files = [f for f in files
             if year_start <= int(f.stem.split("_")[-1]) <= year_end]
    if not files:
        sys.exit(f"No ERA5 files for {variable} in {era5_dir}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        ds = ds.rename({k: v for k, v in {"latitude": "lat",
                                           "longitude": "lon"}.items()
                        if k in ds.dims})
        parts.append(ds[variable])
    da = xr.concat(parts, dim="time")
    vals = da.values.astype(np.float32)
    if variable == "pr":
        # ERA5 pr is in m/day; convert to mm/day to match CMIP6 units.
        # Clip tiny negatives (ERA5 numerical noise) before log transform.
        vals = np.clip(vals, 0.0, None) * 1000.0
    if variable in LOG_VARS:
        vals = _log_transform(vals)
    print(f"ERA5 loaded: {vals.shape}  "
          f"[{vals.min():.3f}, {vals.max():.3f}]")
    return vals, da.lat.values, da.lon.values


def load_cmip6_regridded(cmip6_dir: Path, variable: str,
                          year_start: int, year_end: int,
                          target_lat: np.ndarray,
                          target_lon: np.ndarray) -> np.ndarray:
    """Load CMIP6 files and bilinearly interpolate to ERA5 grid."""
    files = sorted(cmip6_dir.glob(f"{variable}_*.nc"))
    if not files:
        sys.exit(f"No CMIP6 files for {variable} in {cmip6_dir}")

    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        # Select time range
        if "time" in ds.dims:
            ds = ds.sel(time=slice(str(year_start), str(year_end)))
        if ds.sizes.get("time", 0) == 0:
            continue
        da = ds[variable]

        # Handle coordinate names
        lat_name = "lat" if "lat" in da.dims else "latitude"
        lon_name = "lon" if "lon" in da.dims else "longitude"
        if lat_name != "lat":
            da = da.rename({lat_name: "lat", lon_name: "lon"})

        # Bilinear interpolation to ERA5 grid
        da_regrid = da.interp(lat=target_lat, lon=target_lon,
                              method="linear",
                              kwargs={"fill_value": "extrapolate"})
        parts.append(da_regrid)

    if not parts:
        sys.exit(f"No CMIP6 data for {variable} in {year_start}–{year_end}")

    da_all = xr.concat(parts, dim="time")
    vals = da_all.values.astype(np.float32)

    # Unit conversion for precipitation: kg m-2 s-1 → mm/day
    if variable == "pr":
        vals = vals * 86400.0

    # Clip negatives (interpolation artefacts)
    if variable in LOG_VARS:
        vals = np.maximum(vals, 0.0)
        vals = _log_transform(vals)

    print(f"CMIP6 loaded+regridded: {vals.shape}  "
          f"[{vals.min():.3f}, {vals.max():.3f}]")
    return vals


# ── Intersection and t* ───────────────────────────────────────────────────────

def find_intersection(k: np.ndarray, psd1: np.ndarray,
                      psd2: np.ndarray) -> float:
    """Return k where psd1 and psd2 cross (psd1 transitions from < to > psd2).

    At small k (large scales) ESM ≥ ERA5; at large k (small scales) ERA5 > ESM.
    Returns k* at the first crossing where ERA5 PSD equals ESM PSD.
    """
    diff = psd1 - psd2   # ERA5 - ESM
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        # No clean crossing — return k at minimum |diff|
        k_star = k[np.nanargmin(np.abs(diff))]
        print("  Warning: no clean PSD crossing found; using minimum |diff|")
    else:
        idx = sign_changes[0]
        # Linear interpolation for precise crossing
        k0, k1 = k[idx], k[idx + 1]
        d0, d1 = diff[idx], diff[idx + 1]
        k_star = k0 - d0 * (k1 - k0) / (d1 - d0)
    return float(k_star)


def compute_t_star(k_star: float, psd_era5: np.ndarray, k: np.ndarray,
                   N: int) -> float:
    """t* = N * sqrt(PSD_ERA5(k*))  from Hess et al. eq. (11)."""
    psd_at_kstar = float(np.interp(k_star, k, psd_era5))
    return N * np.sqrt(psd_at_kstar)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",   required=True, help="ERA5 directory")
    p.add_argument("--cmip6-dir",  required=True, help="CMIP6 model directory")
    p.add_argument("--output-dir", default="results/psd")
    p.add_argument("--variable",   default="pr",
                   choices=["pr", "tas", "sfcWind", "sfcWindmax", "psl"])
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end",   type=int, default=2014)
    p.add_argument("--n-bins",     type=int, default=50,
                   help="Number of radial wavenumber bins")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    era5, lat, lon = load_era5(
        Path(args.era5_dir), args.variable, args.year_start, args.year_end)

    cmip6 = load_cmip6_regridded(
        Path(args.cmip6_dir), args.variable, args.year_start, args.year_end,
        lat, lon)

    # Align lengths
    T = min(len(era5), len(cmip6))
    era5  = era5[:T]
    cmip6 = cmip6[:T]

    H, W = era5.shape[1], era5.shape[2]
    print(f"Grid: {H}×{W}  (padded training grid: {H_PAD}×{W_PAD})")

    # ── Compute PSDs ───────────────────────────────────────────────────────────
    print("Computing ERA5 PSD...")
    psd2d_era5  = mean_2d_psd(era5)
    k, psd_era5 = radial_profile(psd2d_era5, H, W, args.n_bins)

    print("Computing CMIP6 PSD...")
    psd2d_cmip6  = mean_2d_psd(cmip6)
    _,  psd_cmip6 = radial_profile(psd2d_cmip6, H, W, args.n_bins)

    # ── Find k* and t* ────────────────────────────────────────────────────────
    # Use only valid (non-NaN) bins
    valid = np.isfinite(psd_era5) & np.isfinite(psd_cmip6) & (k > 0)
    k_v, era5_v, cmip6_v = k[valid], psd_era5[valid], psd_cmip6[valid]

    k_star = find_intersection(k_v, era5_v, cmip6_v)

    # Use the padded training grid dimension as N
    N = max(H_PAD, W_PAD)
    t_star = compute_t_star(k_star, era5_v, k_v, N)

    print(f"\nResults for {args.variable}  ({args.year_start}–{args.year_end}):")
    print(f"  k* = {k_star:.4f} cycles/grid-point "
          f"(wavelength ≈ {1/k_star:.1f} grid pts)")
    print(f"  N  = {N} (max padded grid dim)")
    print(f"  t* = {t_star:.4f}  (paper value: 0.468 for global pr)")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "variable": args.variable,
        "year_start": args.year_start,
        "year_end": args.year_end,
        "H": H, "W": W,
        "H_pad": H_PAD, "W_pad": W_PAD,
        "N": N,
        "k_star": k_star,
        "t_star": t_star,
        "k": k_v.tolist(),
        "psd_era5": era5_v.tolist(),
        "psd_cmip6": cmip6_v.tolist(),
    }
    import json
    with open(out_dir / f"psd_{args.variable}.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: PSD vs wavenumber (linear scale)
    ax = axes[0]
    ax.semilogy(k_v, era5_v,  color="black",   label="ERA5")
    ax.semilogy(k_v, cmip6_v, color="tab:blue", label="IPSL-CM6A-LR (regridded)")
    ax.axvline(k_star, color="red", linestyle="--",
               label=f"k* = {k_star:.4f}")
    ax.set_xlabel("Wavenumber k (cycles / grid point)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title(f"Radially-averaged PSD — {args.variable}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Right: PSD vs wavelength (in grid points)
    ax = axes[1]
    wavelength = 1.0 / k_v
    ax.semilogy(wavelength, era5_v,  color="black",   label="ERA5")
    ax.semilogy(wavelength, cmip6_v, color="tab:blue", label="IPSL-CM6A-LR (regridded)")
    ax.axvline(1.0 / k_star, color="red", linestyle="--",
               label=f"λ* = {1/k_star:.1f} grid pts")
    ax.set_xlabel("Wavelength (grid points)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_title(f"t* = {t_star:.3f}  (paper: 0.468)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle(
        f"PSD sweep — {args.variable} — France domain {H}×{W} grid\n"
        f"k* = {k_star:.4f},  N = {N},  t* = {t_star:.4f}", y=1.02)
    fig.tight_layout()
    plot_path = out_dir / f"psd_{args.variable}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
