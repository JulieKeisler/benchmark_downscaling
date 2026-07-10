#!/usr/bin/env python3
"""PSD sweep — calibrate t* for the consistency model.

Computes radially-averaged 2D power spectral densities of ERA5 and
upsampled GCM fields, finds their crossing wavenumber k*, and derives
t* = N × √PSD(k*) for the CM variance schedule (Hess et al. 2025, eq. 11).

Usage:
    python evaluation/psd_sweep.py \
        --era5-dir /scratch/jkeisler/era5 \
        --gcm-dir  /scratch/jkeisler/cmip6/ipsl_cm6a_lr \
        --out-dir  results/psd \
        --variable pr --year-start 2000 --year-end 2014
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Padded grid dimensions used in CM training (must match train_consistency.py)
H_PAD, W_PAD = 64, 80

LOG_EPS  = 1e-6
LOG_VARS = {"pr"}


# ── PSD utilities ─────────────────────────────────────────────────────────────

def mean_2d_psd(fields: np.ndarray) -> np.ndarray:
    """Mean 2D power spectral density over time. fields: [T, H, W]."""
    T, H, W = fields.shape
    psds = [np.abs(np.fft.fft2(fields[t] - fields[t].mean())) ** 2 / (H * W)
            for t in range(T)]
    return np.mean(psds, axis=0)


def radial_profile(power_2d: np.ndarray, H: int, W: int,
                   n_bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged PSD. Returns k_centers and psd_radial."""
    KX, KY = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H))
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_bins = np.linspace(0, min(0.5, K.max()), n_bins + 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    psd = np.array([power_2d[(K >= k_bins[i]) & (K < k_bins[i + 1])].mean()
                    if ((K >= k_bins[i]) & (K < k_bins[i + 1])).any() else np.nan
                    for i in range(n_bins)])
    return k_centers, psd


# ── Intersection and t* ───────────────────────────────────────────────────────

def find_intersection(k: np.ndarray, psd_era5: np.ndarray,
                      psd_gcm: np.ndarray) -> float:
    """k where ERA5 PSD crosses GCM PSD (ERA5 < GCM at large scales)."""
    diff = psd_era5 - psd_gcm
    crossings = np.where(np.diff(np.sign(diff)))[0]
    if not crossings.size:
        print("  Warning: no clean PSD crossing; using minimum |diff|.")
        return float(k[np.nanargmin(np.abs(diff))])
    i = crossings[0]
    return float(k[i] - diff[i] * (k[i + 1] - k[i]) / (diff[i + 1] - diff[i]))


def compute_t_star(k_star: float, psd_era5: np.ndarray,
                   k: np.ndarray, N: int) -> float:
    """t* = N × √PSD_ERA5(k*) — Hess et al. eq. (11)."""
    return N * np.sqrt(float(np.interp(k_star, k, psd_era5)))


# ── Data prep ─────────────────────────────────────────────────────────────────

def _to_psd_array(da, variable: str) -> np.ndarray:
    """Convert xarray DataArray to float32 array with PSD-specific transforms."""
    vals = da.values.astype(np.float32)
    if variable == "pr":
        vals = np.clip(vals, 0.0, None) * 1000.0  # → mm/day for ERA5
    if variable in LOG_VARS:
        vals = np.log(vals + LOG_EPS) - np.log(LOG_EPS)
    return vals


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",   required=True)
    p.add_argument("--gcm-dir",    required=True)
    p.add_argument("--out-dir",    default="results/psd")
    p.add_argument("--variable",   default="pr",
                   choices=["pr", "tas", "sfcWind", "sfcWindmax", "psl"])
    p.add_argument("--year-start", type=int, default=2000)
    p.add_argument("--year-end",   type=int, default=2014)
    p.add_argument("--n-bins",     type=int, default=50)
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_processing import load_era5, load_gcm, bicubic_regrid

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load ERA5
    era5_da = load_era5(Path(args.era5_dir), args.variable,
                        args.year_start, args.year_end)
    era5 = _to_psd_array(era5_da, args.variable)
    lat, lon = era5_da.lat.values, era5_da.lon.values
    print(f"ERA5: {era5.shape}  [{era5.min():.3f}, {era5.max():.3f}]")

    # Load GCM and regrid
    gcm_da = load_gcm(Path(args.gcm_dir), args.variable,
                      args.year_start, args.year_end)
    gcm_da = bicubic_regrid(gcm_da, lat, lon)
    gcm = _to_psd_array(gcm_da, args.variable)
    if args.variable == "pr":
        gcm = gcm * 86400  # GCM pr already in mm/day after unit conv in load_gcm
    print(f"GCM:  {gcm.shape}  [{gcm.min():.3f}, {gcm.max():.3f}]")

    # Align lengths
    T = min(len(era5), len(gcm))
    era5, gcm = era5[:T], gcm[:T]
    H, W = era5.shape[1], era5.shape[2]

    # Compute PSDs
    print("Computing PSDs...")
    k, psd_era5 = radial_profile(mean_2d_psd(era5), H, W, args.n_bins)
    _, psd_gcm  = radial_profile(mean_2d_psd(gcm),  H, W, args.n_bins)

    # k* and t*
    valid = np.isfinite(psd_era5) & np.isfinite(psd_gcm) & (k > 0)
    k_v, e_v, g_v = k[valid], psd_era5[valid], psd_gcm[valid]
    k_star = find_intersection(k_v, e_v, g_v)
    N      = max(H_PAD, W_PAD)
    t_star = compute_t_star(k_star, e_v, k_v, N)

    print(f"\n{args.variable}  ({args.year_start}–{args.year_end})")
    print(f"  k* = {k_star:.4f} cycles/grid-pt  "
          f"(λ* ≈ {1/k_star:.1f} grid pts)")
    print(f"  t* = {t_star:.4f}  (paper: 0.468 for global pr)")

    # Save
    with open(out_dir / f"psd_{args.variable}.json", "w") as f:
        json.dump({"variable": args.variable, "k_star": k_star, "t_star": t_star,
                   "N": N, "H": H, "W": W, "k": k_v.tolist(),
                   "psd_era5": e_v.tolist(), "psd_gcm": g_v.tolist()}, f, indent=2)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, x, xlabel in [
        (ax1, k_v,       "Wavenumber k (cycles / grid point)"),
        (ax2, 1.0 / k_v, "Wavelength (grid points)"),
    ]:
        ax.semilogy(x, e_v, color="black",   label="ERA5")
        ax.semilogy(x, g_v, color="tab:blue", label="GCM (regridded)")
        ax.axvline(k_star if ax is ax1 else 1 / k_star,
                   color="red", linestyle="--",
                   label=f"{'k' if ax is ax1 else 'λ'}* = "
                         f"{k_star:.4f if ax is ax1 else 1/k_star:.1f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PSD (a.u.)")
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
    if ax2:
        ax2.invert_xaxis()
        ax2.set_title(f"t* = {t_star:.3f}")
    ax1.set_title(f"PSD — {args.variable}")
    fig.suptitle(f"k* = {k_star:.4f},  N = {N},  t* = {t_star:.4f}", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / f"psd_{args.variable}.png", dpi=150,
                bbox_inches="tight")
    print(f"Plot → {out_dir}/psd_{args.variable}.png")


if __name__ == "__main__":
    main()
