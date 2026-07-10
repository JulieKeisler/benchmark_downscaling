#!/usr/bin/env python3
"""Unified data pipeline — prepare ERA5 (training) and GCM (evaluation) tensors.

Outputs (all saved to --out-dir):
  ERA5_train.pt      [T_train, C, H, W]  float32, physical units
  GCM_eval.pt        [T_eval,  C, H, W]  float32, physical units, on ERA5 grid
  norm_stats.json    ERA5 channel-wise mean/std (for SF denormalisation)
  eval_dates.npy     ISO date strings for the eval time axis
  lat.npy, lon.npy   ERA5 grid coordinates

Variable order is set by --variables (default: tas pr sfcWind sfcWindmax psl).

Interpolation modes:
  bicubic   — standard bicubic upsampling (default; use for CM, BCSD, bicubic)
  spectral  — bicubic + Fourier low-pass at --r-cut pixels (use for SerpentFlow)

Run on Cleps (CPU node):
  python evaluation/data_processing.py \
      --era5-dir /scratch/jkeisler/era5 \
      --gcm-dir  /scratch/jkeisler/cmip6/ipsl_cm6a_lr \
      --out-dir  /scratch/jkeisler/serpenflow/data \
      --interpolation spectral --r-cut 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.interpolate import CubicSpline


DEFAULT_VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]

CMIP6_UNIT_CONV: dict = {
    "pr": lambda da: da * 86400,  # kg m⁻² s⁻¹ → mm/day
}


# ── Unit conversions ──────────────────────────────────────────────────────────

def _era5_pr_convert(da: xr.DataArray) -> xr.DataArray:
    """Auto-detect ERA5 pr units (annual-total m vs daily-mean m) → mm/day."""
    ref = float(np.nanmean(da.isel(time=0).values))
    if ref > 0.05:
        return da * 1000 / 365.25  # annual total in m → mm/day
    return da * 1000               # daily value in m → mm/day


def _rename_coords(ds: xr.Dataset) -> xr.Dataset:
    renames = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
               if k in ds.dims}
    return ds.rename(renames) if renames else ds


# ── Data loading ──────────────────────────────────────────────────────────────

def load_era5(era5_dir: Path, variable: str, start: int, end: int) -> xr.DataArray:
    """Load and concatenate annual ERA5 NetCDF files for one variable."""
    files = sorted(era5_dir.glob(f"{variable}_*.nc"))
    files = [f for f in files if start <= int(f.stem.split("_")[-1]) <= end]
    if not files:
        raise FileNotFoundError(
            f"No ERA5 files for {variable} {start}–{end} in {era5_dir}")
    parts = [_rename_coords(xr.open_dataset(f))[variable] for f in files]
    da = xr.concat(parts, dim="time")
    if variable == "pr":
        da = _era5_pr_convert(da)
    return da.load()


def load_gcm(gcm_dir: Path, variable: str, start: int, end: int,
             scenario: str = "ssp585") -> xr.DataArray:
    """Load historical + scenario GCM files, select [start, end]."""
    candidates = [
        gcm_dir / f"{variable}_historical_1960-2014.nc",
        gcm_dir / f"{variable}_{scenario}_2015-2100.nc",
    ]
    das = []
    for f in candidates:
        if not f.exists():
            continue
        ds = xr.open_dataset(f, use_cftime=True)
        ds = _rename_coords(ds)
        da = ds[variable].sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
        if da.sizes["time"] > 0:
            das.append(da)
    if not das:
        raise FileNotFoundError(
            f"No GCM files for {variable} {start}–{end} in {gcm_dir}")
    da = xr.concat(das, dim="time") if len(das) > 1 else das[0]
    if variable in CMIP6_UNIT_CONV:
        da = CMIP6_UNIT_CONV[variable](da)
    return da.load()


# ── Regridding ────────────────────────────────────────────────────────────────

def _cubic_kernel(x, a=-0.5):
    """
    Keys cubic convolution kernel.
    Equivalent to the common bicubic interpolation kernel.
    """
    x = np.abs(x)

    w = np.zeros_like(x)

    m1 = x <= 1
    m2 = (x > 1) & (x < 2)

    w[m1] = (
        (a + 2) * x[m1]**3
        - (a + 3) * x[m1]**2
        + 1
    )

    w[m2] = (
        a * x[m2]**3
        - 5 * a * x[m2]**2
        + 8 * a * x[m2]
        - 4 * a
    )

    return w


def bicubic_regrid(
    gcm: xr.DataArray,
    target_lat,
    target_lon,
) -> xr.DataArray:
    """
    Bicubic interpolation similar to CDO remapbic.

    Parameters
    ----------
    gcm : xr.DataArray
        Dimensions must be (time, lat, lon)

    target_lat : array
        Target latitude coordinates

    target_lon : array
        Target longitude coordinates
    """

    src_lat = gcm.lat.values
    src_lon = gcm.lon.values
    data = gcm.values

    target_lat = np.asarray(target_lat)
    target_lon = np.asarray(target_lon)

    nt = data.shape[0]

    out = np.empty(
        (nt, len(target_lat), len(target_lon)),
        dtype=data.dtype,
    )

    # assume regular grids
    dlat = src_lat[1] - src_lat[0]
    dlon = src_lon[1] - src_lon[0]

    for i, lat in enumerate(target_lat):

        # fractional source index
        y = (lat - src_lat[0]) / dlat
        iy = int(np.floor(y))
        fy = y - iy

        iy0 = np.clip(
            np.arange(iy - 1, iy + 3),
            0,
            len(src_lat) - 1,
        )

        wy = _cubic_kernel(
            np.arange(-1, 3) - fy
        )
        wy /= wy.sum()

        for j, lon in enumerate(target_lon):

            x = (lon - src_lon[0]) / dlon
            ix = int(np.floor(x))
            fx = x - ix

            ix0 = np.clip(
                np.arange(ix - 1, ix + 3),
                0,
                len(src_lon) - 1,
            )

            wx = _cubic_kernel(
                np.arange(-1, 3) - fx
            )
            wx /= wx.sum()

            # 4x4 bicubic stencil
            patch = data[:, iy0[:, None], ix0[None, :]]

            # tensor product weights
            out[:, i, j] = np.einsum(
                "tij,i,j->t",
                patch,
                wy,
                wx,
            )

    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={
            "time": gcm.time,
            "lat": target_lat,
            "lon": target_lon,
        },
        name=gcm.name,
        attrs=gcm.attrs,
    )

def _fourier_lowpass(x: torch.Tensor, r_cut: int) -> torch.Tensor:
    """Circular low-pass filter in Fourier space. x: [T, H, W]."""
    T, H, W = x.shape
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cy, cx = H // 2, W // 2
    mask = ((X - cx).float() ** 2 + (Y - cy).float() ** 2 <= r_cut ** 2)
    mask = mask[None].float()  # [1, H, W]
    x4 = x.unsqueeze(1)       # [T, 1, H, W]
    fft = torch.fft.fftshift(torch.fft.fft2(x4, dim=(-2, -1)), dim=(-2, -1))
    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(fft * mask, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return filtered.squeeze(1)  # [T, H, W]


def spectral_regrid(gcm: xr.DataArray, target_lat, target_lon,
                    r_cut: int = 12) -> xr.DataArray:
    """Bicubic upsample then Fourier low-pass at r_cut pixels.

    Removes bicubic ringing artefacts and keeps only frequencies the GCM
    genuinely contains. r_cut ≈ min(GCM_H, GCM_W) / 2 is a sensible default.
    For SerpentFlow use r_cut=1 (only shared low-freq content).
    """
    upsampled = bicubic_regrid(gcm, target_lat, target_lon)
    arr = torch.tensor(upsampled.values, dtype=torch.float32)  # [T, H, W]
    filtered = _fourier_lowpass(arr, r_cut).numpy()
    return upsampled.copy(data=filtered)


# ── Quantile Delta Mapping ────────────────────────────────────────────────────

_MULT_VARS = {"pr", "sfcWind", "sfcWindmax"}  # multiplicative delta variables


def qdm_correct(
    gcm_eval:  xr.DataArray,
    gcm_hist:  xr.DataArray,
    era5_hist: xr.DataArray,
    variable:  str,
    n_quantiles: int = 100,
) -> xr.DataArray:
    """Quantile Delta Mapping (QDM) bias correction, per calendar month, per cell.

    Maps the GCM evaluation distribution to the ERA5 reference distribution while
    preserving the GCM's additive (or multiplicative) delta above the historical
    envelope — crucial for out-of-sample future values.

    For additive variables (tas, psl):
        corrected = Q_era5(τ) + (x_eval - Q_gcm_hist(τ))
    For multiplicative variables (pr, sfcWind, sfcWindmax):
        corrected = Q_era5(τ) × clip(x_eval / Q_gcm_hist(τ), 0, 10)

    where τ = F_gcm_hist(x_eval) is the quantile of the evaluation value in the
    historical GCM distribution. All three inputs must be on the same spatial grid.
    """
    multiplicative = variable in _MULT_VARS
    eps = 1e-6

    gcm_e_val  = gcm_eval.values.astype(np.float64)    # [T_eval, H, W]
    gcm_h_val  = gcm_hist.values.astype(np.float64)    # [T_hist, H, W]
    era5_h_val = era5_hist.values.astype(np.float64)   # [T_hist, H, W]

    months_eval = gcm_eval.time.dt.month.values
    months_hist = gcm_hist.time.dt.month.values

    T_eval, H, W = gcm_e_val.shape
    N_cells      = H * W
    result       = gcm_e_val.copy()

    qs = np.linspace(0.0, 100.0, n_quantiles)   # percentile levels

    for m in range(1, 13):
        idx_e = np.where(months_eval == m)[0]
        idx_h = np.where(months_hist == m)[0]
        if len(idx_e) == 0 or len(idx_h) < 2:
            continue

        x_e = gcm_e_val[idx_e].reshape(-1, N_cells)     # [T_e, N_cells]
        x_h = gcm_h_val[idx_h].reshape(-1, N_cells)     # [T_h, N_cells]
        y_h = era5_h_val[idx_h].reshape(-1, N_cells)    # [T_h, N_cells]

        # Quantile functions: [n_quantiles, N_cells]
        Q_gcm  = np.percentile(x_h, qs, axis=0)
        Q_era5 = np.percentile(y_h, qs, axis=0)

        out = np.empty_like(x_e)
        for t in range(x_e.shape[0]):
            x = x_e[t]  # [N_cells]
            # τ for each cell: interpolate x into Q_gcm
            tau = np.array(
                [np.interp(x[c], Q_gcm[:, c], qs) for c in range(N_cells)]
            )  # [N_cells], percentile in [0, 100]

            # Map τ back through Q_gcm and Q_era5
            q_g = np.array(
                [np.interp(tau[c], qs, Q_gcm[:, c]) for c in range(N_cells)]
            )
            q_e = np.array(
                [np.interp(tau[c], qs, Q_era5[:, c]) for c in range(N_cells)]
            )

            if multiplicative:
                ratio = np.clip(x / np.maximum(q_g, eps), 0.0, 10.0)
                out[t] = q_e * ratio
            else:
                out[t] = q_e + (x - q_g)

        result[idx_e] = out.reshape(-1, H, W)

    return xr.DataArray(
        result.astype(np.float32),
        dims=gcm_eval.dims,
        coords=gcm_eval.coords,
    )


# ── Tensor utilities ──────────────────────────────────────────────────────────

def align_time(das: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Subset each DataArray to the common date intersection."""
    date_sets = [set(da.time.dt.strftime("%Y-%m-%d").values) for da in das.values()]
    common = sorted(set.intersection(*date_sets))
    if not common:
        raise ValueError(
            "align_time: no common dates found across the supplied DataArrays. "
            "Check that the ERA5 and GCM periods overlap and that both directories contain data."
        )
    print(f"  Common timesteps: {len(common)}")
    result = {}
    for key, da in das.items():
        dates = da.time.dt.strftime("%Y-%m-%d").values
        idx = [i for i, d in enumerate(dates) if d in set(common)]
        result[key] = da.isel(time=idx)
    return result


def stack_to_tensor(das: dict[str, xr.DataArray],
                    variables: list[str]) -> torch.Tensor:
    """Stack [T,H,W] DataArrays → [T, C, H, W] float32."""
    arrays = [torch.tensor(das[v].values, dtype=torch.float32) for v in variables]
    return torch.stack(arrays, dim=1)


def compute_norm_stats(tensor: torch.Tensor, variables: list[str]) -> dict:
    """Per-channel mean/std over all T×H×W (for SF denormalisation)."""
    stats = {}
    for c, var in enumerate(variables):
        ch = tensor[:, c]
        mask = torch.isfinite(ch)
        stats[var] = {"mean": ch[mask].mean().item(),
                      "std":  ch[mask].std().item()}
    return stats


# ── Prediction I/O ────────────────────────────────────────────────────────────

def save_predictions(
    tensor: torch.Tensor,
    variables: list[str],
    out_dir: Path,
    ref_da: xr.DataArray,
) -> None:
    """Save inference output in the standard benchmark format.

    Files written to out_dir:
      predictions.pt  — [T, C, H, W] float32, physical units, C = len(variables)
      variables.npy   — channel names in order
      eval_dates.npy  — ISO date strings
      lat.npy, lon.npy — ERA5 grid coordinates
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_dir / "predictions.pt")
    np.save(out_dir / "variables.npy", np.array(variables))
    np.save(out_dir / "eval_dates.npy", ref_da.time.dt.strftime("%Y-%m-%d").values)
    np.save(out_dir / "lat.npy", ref_da.lat.values)
    np.save(out_dir / "lon.npy", ref_da.lon.values)
    print(f"Saved predictions → {out_dir}  {list(tensor.shape)}")


def load_predictions(
    pred_dir: Path,
    variables: list[str],
) -> dict[str, "xr.DataArray"]:
    """Load standard-format predictions.pt produced by any inference_*.py script.

    Returns dict variable → DataArray on the saved ERA5 grid and time axis.
    Physical units (same as ERA5), no normalisation applied.
    """
    import pandas as pd

    for fname in ["predictions.pt", "eval_dates.npy", "lat.npy", "lon.npy"]:
        p = pred_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing prediction file: {p}")

    tensor = torch.load(pred_dir / "predictions.pt", map_location="cpu")  # [T, C, H, W]
    vars_file = pred_dir / "variables.npy"
    saved_vars = list(np.load(vars_file, allow_pickle=True)) if vars_file.exists() \
                 else variables
    dates = np.load(pred_dir / "eval_dates.npy", allow_pickle=True)
    lat   = np.load(pred_dir / "lat.npy")
    lon   = np.load(pred_dir / "lon.npy")
    time_coord = pd.to_datetime(dates)

    results = {}
    for var in variables:
        if var not in saved_vars:
            continue
        i = saved_vars.index(var)
        results[var] = xr.DataArray(
            tensor[:, i].numpy(),
            coords={"time": time_coord, "lat": lat, "lon": lon},
            dims=["time", "lat", "lon"],
        )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir", required=True)
    p.add_argument("--gcm-dir",  required=True)
    p.add_argument("--out-dir",  required=True)
    p.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    p.add_argument("--train-start", type=int, default=1979)
    p.add_argument("--train-end",   type=int, default=2003)
    p.add_argument("--eval-start",  type=int, default=2004)
    p.add_argument("--eval-end",    type=int, default=2014)
    p.add_argument("--interpolation", choices=["bicubic", "spectral"],
                   default="bicubic")
    p.add_argument("--r-cut", type=int, default=12,
                   help="Fourier low-pass radius in pixels (spectral mode only)")
    p.add_argument("--scenario", default="ssp585",
                   help="GCM scenario for files beyond 2014")
    args = p.parse_args()

    era5_dir = Path(args.era5_dir)
    gcm_dir  = Path(args.gcm_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variables = args.variables

    regrid_fn = (lambda gcm, lat, lon: spectral_regrid(gcm, lat, lon, args.r_cut)
                 if args.interpolation == "spectral"
                 else bicubic_regrid(gcm, lat, lon))

    # ── ERA5 training tensor ───────────────────────────────────────────────────
    print(f"\n=== ERA5 training {args.train_start}–{args.train_end} ===")
    era5_train = {}
    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        era5_train[var] = load_era5(era5_dir, var, args.train_start, args.train_end)
        print(f"    shape={era5_train[var].shape}", flush=True)

    era5_train = align_time(era5_train)
    era5_tensor = stack_to_tensor(era5_train, variables)
    print(f"ERA5 training tensor: {era5_tensor.shape}")

    torch.save(era5_tensor, out_dir / "ERA5_train.pt")
    print(f"Saved ERA5_train.pt")

    norm_stats = compute_norm_stats(era5_tensor, variables)
    for var, s in norm_stats.items():
        print(f"  {var}: mean={s['mean']:.4g}, std={s['std']:.4g}")
    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print("Saved norm_stats.json")

    # ── GCM evaluation tensor ──────────────────────────────────────────────────
    print(f"\n=== GCM eval {args.eval_start}–{args.eval_end} "
          f"(interpolation={args.interpolation}) ===")
    era5_eval, gcm_eval = {}, {}
    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        era5_eval[var] = load_era5(era5_dir, var, args.eval_start, args.eval_end)
        gcm_eval[var]  = load_gcm(gcm_dir, var, args.eval_start, args.eval_end,
                                  args.scenario)

    all_das = {f"era5_{v}": era5_eval[v] for v in variables}
    all_das.update({f"gcm_{v}": gcm_eval[v] for v in variables})
    aligned = align_time(all_das)
    era5_eval = {v: aligned[f"era5_{v}"] for v in variables}
    gcm_eval  = {v: aligned[f"gcm_{v}"]  for v in variables}

    ref = next(iter(era5_eval.values()))
    target_lat, target_lon = ref.lat.values, ref.lon.values

    gcm_on_era5 = {}
    for var in variables:
        print(f"  Regridding {var} ({args.interpolation}) ...", flush=True)
        da = regrid_fn(gcm_eval[var], target_lat, target_lon)
        da = da.assign_coords(time=ref.time)
        if var == "pr":
            da.values[:] = np.clip(da.values, 0.0, None)
        gcm_on_era5[var] = da

    gcm_tensor = stack_to_tensor(gcm_on_era5, variables)
    print(f"GCM eval tensor: {gcm_tensor.shape}")

    torch.save(gcm_tensor, out_dir / "GCM_eval.pt")
    print("Saved GCM_eval.pt")

    dates = ref.time.dt.strftime("%Y-%m-%d").values
    np.save(out_dir / "eval_dates.npy", dates)
    np.save(out_dir / "lat.npy", target_lat)
    np.save(out_dir / "lon.npy", target_lon)
    print(f"Saved eval_dates.npy ({len(dates)} dates), lat.npy, lon.npy")
    print("Done.")


if __name__ == "__main__":
    main()
