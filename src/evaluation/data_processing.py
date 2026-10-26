#!/usr/bin/env python3
"""Unified data pipeline — prepare ERA5 (training) and GCM (evaluation) tensors.

Outputs (all saved to --out-dir):
  ERA5_train.pt      [T_train, C, H, W]  float32, physical units
  GCM_eval.pt        [T_eval,  C, H, W]  float32, physical units, on ERA5 grid
  norm_stats.json    ERA5 channel-wise mean/std (normalisation stats)
  eval_dates.npy     ISO date strings for the eval time axis
  lat.npy, lon.npy   ERA5 grid coordinates

Variable order is set by --variables (default: tas pr sfcWind sfcWindmax psl).

Interpolation modes:
  bicubic   — standard bicubic upsampling (default)

Run on Cleps (CPU node):
  python evaluation/data_processing.py \
      --era5-dir /scratch/jkeisler/era5 \
      --gcm-dir  /scratch/jkeisler/cmip6/ipsl_cm6a_lr \
      --out-dir  /scratch/jkeisler/benchmark/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from scipy.interpolate import CubicSpline


DEFAULT_VARIABLES = ["tas", "pr", "sfcWind", "sfcWindmax", "psl", "rsds"]

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


def _lon_to_180(ds: xr.Dataset) -> xr.Dataset:
    # ERA5 crops use -180..180; a 0..360 GCM crop would otherwise be placed by
    # index and collapse onto one column in bicubic_regrid.
    if "lon" in ds.coords and float(ds.lon.max()) > 180.0:
        ds = ds.assign_coords(lon=((ds.lon + 180.0) % 360.0) - 180.0).sortby("lon")
    return ds


def check_same_member(files: list[Path]) -> str | None:
    """Raise if the files of one GCM variable come from different realisations.

    An r1 historical spliced to an r4 scenario changes every
    2000-2020 and future-change number. Files without a variant_label (derived
    products) are skipped. Returns the common label.
    """
    labels = {}
    for f in files:
        with xr.open_dataset(f, decode_times=False) as ds:
            lab = ds.attrs.get("variant_label")
            var = f.name.split("_")[0]
            if lab is None and var in ds:      # some downloads label the variable, not the file
                lab = ds[var].attrs.get("variant_label")
        labels[f.name] = lab
    known = {k: v for k, v in labels.items() if v is not None}
    if len(set(known.values())) > 1:
        raise ValueError(f"GCM files mix realisations: {labels}")
    if known and len(known) < len(labels):
        raise ValueError(f"GCM files mix labelled and unlabelled realisations: {labels}")
    labels = known
    return next(iter(labels.values()), None)


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
        # ERA5 precipitation has small negative values; left in, they put a floor
        # under the KS distance of every product clipped at zero.
        da = _era5_pr_convert(da).clip(min=0.0)
    return da.load()


def load_gcm(gcm_dir: Path, variable: str, start: int, end: int,
             scenario: str = "ssp370") -> xr.DataArray:
    """Load historical + scenario GCM files, select [start, end]."""
    candidates = [
        gcm_dir / f"{variable}_historical_1960-2014.nc",
        gcm_dir / f"{variable}_{scenario}_2015-2100.nc",
    ]
    check_same_member([f for f in candidates if f.exists()])
    das = []
    for f in candidates:
        if not f.exists():
            continue
        ds = xr.open_dataset(f, use_cftime=True)
        ds = _lon_to_180(_rename_coords(ds))
        da = ds[variable].sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
        if da.sizes["time"] > 0:
            das.append(da)
    if not das:
        raise FileNotFoundError(
            f"No GCM files for {variable} {start}–{end} in {gcm_dir}")
    da = xr.concat(das, dim="time")
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

    src_lat    = gcm.lat.values
    src_lon    = gcm.lon.values
    data       = gcm.values                     # [T, H_s, W_s]
    target_lat = np.asarray(target_lat)
    target_lon = np.asarray(target_lon)

    nt   = data.shape[0]
    H_s  = len(src_lat)
    W_s  = len(src_lon)
    H_t  = len(target_lat)
    W_t  = len(target_lon)
    dlat = src_lat[1] - src_lat[0]
    dlon = src_lon[1] - src_lon[0]
    # The index arithmetic below assumes uniform spacing (Gaussian grids are
    # uniform to ~1e-6 over our crops); refuse anything coarser than 1 %.
    for name, src in (("lat", src_lat), ("lon", src_lon)):
        dv = np.diff(src)
        if np.abs(dv / dv.mean() - 1.0).max() > 0.01:
            raise ValueError(f"bicubic_regrid: source {name} spacing is not uniform")
    # The source grid must cover the target (to one source cell);
    # mismatched lon conventions otherwise give a silent one-column field.
    for name, src, tgt, d in (("lat", src_lat, target_lat, dlat),
                              ("lon", src_lon, target_lon, dlon)):
        if (src.min() > tgt.min() + abs(d)) or (src.max() < tgt.max() - abs(d)):
            raise ValueError(
                f"bicubic_regrid: source {name} [{src.min():.2f}, {src.max():.2f}] "
                f"does not cover target [{tgt.min():.2f}, {tgt.max():.2f}]")

    offsets = np.array([-1, 0, 1, 2])

    # Vectorise over all target latitudes
    y   = (target_lat - src_lat[0]) / dlat      # [H_t]
    iy  = np.floor(y).astype(int)
    fy  = y - iy
    iy0 = np.clip(iy[:, None] + offsets, 0, H_s - 1)   # [H_t, 4]
    wy  = _cubic_kernel(offsets - fy[:, None])           # [H_t, 4]
    wy /= wy.sum(axis=1, keepdims=True)

    # Precompute longitude weights once
    x   = (target_lon - src_lon[0]) / dlon      # [W_t]
    ix  = np.floor(x).astype(int)
    fx  = x - ix
    ix0 = np.clip(ix[:, None] + offsets, 0, W_s - 1)   # [W_t, 4]
    wx  = _cubic_kernel(offsets - fx[:, None])           # [W_t, 4]
    wx /= wx.sum(axis=1, keepdims=True)

    out = np.empty((nt, H_t, W_t), dtype=data.dtype)
    for i in range(H_t):
        rows   = data[:, iy0[i], :]                      # [T, 4, W_s]
        rows_w = np.einsum("p,tpw->tw", wy[i], rows)    # [T, W_s]
        cols   = rows_w[:, ix0]                          # [T, W_t, 4]
        out[:, i, :] = np.einsum("jq,tjq->tj", wx, cols)  # [T, W_t]

    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": gcm.time, "lat": target_lat, "lon": target_lon},
        name=gcm.name,
        attrs=gcm.attrs,
    )

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

def _quantile_at(sorted_cols: np.ndarray, tau: np.ndarray) -> np.ndarray:
    # Linear-interpolated quantile of each sorted column [n, C] at probabilities tau [m, C].
    pos = tau * (sorted_cols.shape[0] - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, sorted_cols.shape[0] - 1)
    w = pos - lo
    return (np.take_along_axis(sorted_cols, lo, 0) * (1.0 - w)
            + np.take_along_axis(sorted_cols, hi, 0) * w)


def qdm_windowed(
    gcm_proj:  xr.DataArray,
    gcm_hist:  xr.DataArray,
    era5_hist: xr.DataArray,
    variable:  str,
    windows:   list[tuple[int, int, int, int]],
) -> xr.DataArray:
    """Quantile delta mapping (Cannon et al. 2015) in moving projection windows.

    Per cell, calendar month and window (win_start, win_end, keep_start,
    keep_end), each GCM value x gets its quantile tau in the window's own GCM
    distribution; the GCM change at that quantile is added to (or, for pr and
    wind, multiplies) the ERA5 1980-1999 value at tau:
        additive        y = Q_era5(tau) + (x - Q_gcm_hist(tau))
        multiplicative  y = Q_era5(tau) * clip(x / Q_gcm_hist(tau), 0, 10)
    Only the centre years of each window are kept. A single projection window
    over 2000-2100 would map late-century days through the upper tail of the
    bias and lose part of the warming (Australia tas: 0.94 kept, against 1.00
    with 20-year windows). All inputs on the same grid; returns float32 for
    the years covered by the keep ranges.
    """
    multiplicative = variable in _MULT_VARS
    eps = 1e-6
    x_all = gcm_proj.values.astype(np.float64)
    T, H, W = x_all.shape
    x_all = x_all.reshape(T, H * W)
    h_all = gcm_hist.values.astype(np.float64).reshape(-1, H * W)
    o_all = era5_hist.values.astype(np.float64).reshape(-1, H * W)
    m_p = gcm_proj.time.dt.month.values
    y_p = gcm_proj.time.dt.year.values
    m_h = gcm_hist.time.dt.month.values
    m_o = era5_hist.time.dt.month.values
    out = np.full_like(x_all, np.nan)
    for m in range(1, 13):
        Sh = np.sort(h_all[m_h == m], axis=0)
        So = np.sort(o_all[m_o == m], axis=0)
        for ws, we, ks, ke in windows:
            win = (m_p == m) & (y_p >= ws) & (y_p <= we)
            if not win.any():
                continue
            xw = x_all[win]
            tau = (np.argsort(np.argsort(xw, axis=0), axis=0) + 0.5) / xw.shape[0]
            qh, qo = _quantile_at(Sh, tau), _quantile_at(So, tau)
            if multiplicative:
                yw = qo * np.clip(xw / np.maximum(qh, eps), 0.0, 10.0)
            else:
                yw = qo + (xw - qh)
            idx = np.flatnonzero(win)
            keep = (y_p[idx] >= ks) & (y_p[idx] <= ke)
            out[idx[keep]] = yw[keep]
    covered = ~np.isnan(out).any(axis=1)
    res = xr.DataArray(out.reshape(T, H, W).astype(np.float32), dims=gcm_proj.dims,
                       coords=gcm_proj.coords)
    return res.isel(time=np.flatnonzero(covered))


def align_time(das: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Subset each DataArray to the common date intersection.

    Every returned DataArray is re-stamped to a date-only numpy datetime64
    axis so downstream xarray ops (xr.corr, xr.align, save_predictions) align
    on identical values. Without this, a cftime GCM axis (e.g. midday 12:00)
    and an np.datetime64 prediction axis (midnight 00:00) share the same dates
    yet yield all-NaN xr.corr inner-joins.
    """
    # Compute date strings once per DataArray (cftime strftime is slow at 36k+ steps)
    keys = list(das.keys())
    all_dates = [np.array([str(t)[:10] for t in das[k].time.values]) for k in keys]
    date_sets = [set(d) for d in all_dates]
    common = sorted(set.intersection(*date_sets))
    if not common:
        raise ValueError(
            "align_time: no common dates found across the supplied DataArrays. "
            "Check that the ERA5 and GCM periods overlap and that both directories contain data."
        )
    print(f"  Common timesteps: {len(common)}")
    common64 = np.array(common, dtype="datetime64[ns]")
    result = {}
    for k, dates in zip(keys, all_dates):
        idx = np.where(np.isin(dates, common))[0]
        da = das[k].isel(time=idx)
        result[k] = da.assign_coords(time=common64)
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
    import pandas as pd
    tvals = ref_da.time.values
    if hasattr(tvals[0], "strftime"):   # cftime_index (GCM native calendar)
        pd_times = pd.to_datetime([t.strftime("%Y-%m-%d") for t in tvals])
    else:
        pd_times = pd.to_datetime(tvals)
    dates = pd_times.strftime("%Y-%m-%d").to_numpy()
    np.save(out_dir / "eval_dates.npy", dates)
    np.save(out_dir / "lat.npy", ref_da.lat.values)
    np.save(out_dir / "lon.npy", ref_da.lon.values)
    print(f"Saved predictions → {out_dir}  {list(tensor.shape)}")


def load_predictions(
    pred_dir: Path,
    variables: list[str],
    mmap: bool = False,
) -> dict[str, "xr.DataArray"]:
    """Load standard-format predictions.pt produced by any inference_*.py script.

    Returns dict variable → DataArray on the saved ERA5 grid and time axis.
    Physical units (same as ERA5), no normalisation applied.

    `mmap=True` maps the tensor from disk instead of copying into RAM (useful
    for multi-GB predictions when only reductions like region means are needed).
    """
    import pandas as pd

    for fname in ["predictions.pt", "eval_dates.npy", "lat.npy", "lon.npy"]:
        p = pred_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing prediction file: {p}")

    try:
        tensor = torch.load(pred_dir / "predictions.pt", map_location="cpu",
                            mmap=mmap)  # [T, C, H, W]
    except TypeError:
        tensor = torch.load(pred_dir / "predictions.pt", map_location="cpu")
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
    p.add_argument("--train-start", type=int, default=1960)
    p.add_argument("--train-end",   type=int, default=2003)
    p.add_argument("--eval-start",  type=int, default=2004)
    p.add_argument("--eval-end",    type=int, default=2014)
    p.add_argument("--interpolation", choices=["bicubic"],
                   default="bicubic")
    p.add_argument("--scenario", default="ssp370",
                   help="GCM scenario for files beyond 2014")
    args = p.parse_args()

    era5_dir = Path(args.era5_dir)
    gcm_dir  = Path(args.gcm_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variables = args.variables

    regrid_fn = bicubic_regrid

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
