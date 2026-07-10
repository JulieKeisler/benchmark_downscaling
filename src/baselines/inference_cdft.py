#!/usr/bin/env python3
"""CDF-t baseline inference.

True CDF-t (Michelangeli et al. 2009): for each grid cell and calendar month,
applies the mapping

    F_Rf(x) = F_Rp( F_Gp^{-1}( F_Gf(x) ) )

where Rp = ERA5 calibration period, Gp = GCM calibration period,
      Gf = GCM evaluation period (may be future).

This explicitly accounts for the shift in GCM distribution between calibration
and evaluation, unlike plain empirical QM.

Per-variable hyperparameters (scale, pui, npas, etype, dry-day handling) were
optimised with Optuna on the France domain to minimise per-month KS statistic
(see serpentflow/test_notebooks/climatology_cdft.ipynb).

Usage:
    python inference_cdft.py \
        --era5-dir /path/to/era5 \
        --gcm-dir  /path/to/cmip6/ipsl_cm6a_lr \
        --out-dir  results/cdft \
        --calib-start 1979 --calib-end 2003 \
        --eval-start  2004 --eval-end  2014
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

# Per-variable CDF-t hyperparameters.
# Keys not listed in a var's entry fall back to DEFAULT_PARAMS.
_P = {
    "tas": {
        "scale": 1, "dev": 2, "npas": 100, "pui": 0.5, "etype": 4,
    },
    "pr": {
        "scale": 0, "pui": 0.5, "npas": 100, "etype": 4, "use_dry_day_correction": True, "wet_threshold": 1.0
    },
    "sfcWind": {
        "scale": 0, "dev": 2, "npas": 100, "pui": 0.5, "etype": 4,
        "clip_min": 0.0,
    },
    "sfcWindmax": {
        "scale": 0, "dev": 2, "npas": 100, "pui": 0.5, "etype": 4,
        "clip_min": 0.0,
    },
    "rsds": {
        "scale": 0, "dev": 2, "npas": 100, "pui": 1, "etype": 4,
        "clip_min": 0.0,
    },
    "psl": {
        "scale": 1, "dev": 2, "npas": 100, "pui": 1.0, "etype": 4,
    },
}
_DEFAULT = {"scale": 1, "dev": 2, "npas": 100, "pui": 1.0, "etype": 4}


def _params(variable: str) -> dict:
    base = dict(_DEFAULT)
    base.update(_P.get(variable, {}))
    return base


# ── Core CDF-t maths ─────────────────────────────────────────────────────────

def _ecdf_t(x: np.ndarray, etype: int = 4):
    """Step-function ECDF (matches R ecdfT with approxfun method='constant')."""
    xs = np.sort(x.ravel())
    n = len(xs)
    vals, inv = np.unique(xs, return_inverse=True)
    cc = np.cumsum(np.bincount(inv))
    if etype == 4:
        y = cc / n
    elif etype == 5:
        y = (cc - 0.5) / n
    elif etype == 6:
        y = cc / (n + 1)
    else:                    # etype == 7
        y = (cc - 1) / (n - 1)

    def F(v):
        v = np.atleast_1d(np.asarray(v, dtype=float))
        idx = np.clip(np.searchsorted(vals, v, side="right") - 1, 0, len(y) - 1)
        out = y[idx].copy()
        out[v < vals[0]] = 0.0
        out[v > vals[-1]] = 1.0
        return out

    return F


def _cdft_1d(DataGp: np.ndarray, ObsRp: np.ndarray, DataGf: np.ndarray,
             npas: int = 100, dev: int = 2, scale: int = 1,
             pui: float = 1.0, etype: int = 4) -> np.ndarray:
    """CDF-t transform for one grid cell, one month.

    F_Rf(x) = F_Rp( F_Gp^{-1}( F_Gf(x) ) )
    with tail corrections for extrapolation and support expansion.
    """
    add = pui >= 1

    mO, mGp = np.mean(ObsRp), np.mean(DataGp)
    Gp2, Gf2 = DataGp.copy(), DataGf.copy()
    if scale == 1:
        Gp2 = DataGp + (mO - mGp)
        Gf2 = DataGf + (mO - mGp)
    elif scale == 2:
        Gp2 = DataGp * mO / mGp
        Gf2 = DataGf * mO / mGp

    FRp = _ecdf_t(ObsRp, etype)
    FGp = _ecdf_t(Gp2, etype)
    FGf = _ecdf_t(Gf2, etype)

    m = min(np.min(ObsRp), np.min(DataGp), np.min(DataGf))
    M = max(np.max(ObsRp), np.max(DataGp), np.max(DataGf))
    if add:
        a = abs(np.mean(DataGf) - np.mean(DataGp))
        m -= dev * a
        M += dev * a
    else:
        denom = np.mean(DataGp)
        if abs(denom) < 1e-12:
            a = 1.0
        else:
            a = abs(np.mean(DataGf) / denom)
        m = m * a / dev
        M = M * dev * a

    # Guard: fractional pui requires non-negative base
    if pui < 1.0:
        m = max(m, 0.0)
        M = max(M, 1e-30)

    x = np.linspace(m ** pui, M ** pui, npas) ** (1.0 / pui)
    n = len(x)

    gf2s, gp2s, obs_s = np.sort(Gf2), np.sort(Gp2), np.sort(ObsRp)
    FRP = FRp(x)
    FGF = np.clip(np.interp(x, gf2s, FGf(gf2s)), 0.0, 1.0)
    FGPm1_FGF = np.quantile(Gp2, FGF)
    FRF = np.interp(FGPm1_FGF, obs_s, FRp(obs_s))

    # Lower tail correction
    if np.min(ObsRp) < np.min(Gf2):
        thresh = np.quantile(ObsRp, FRF[0])
        i = 0
        while i < n and x[i] <= thresh:
            i += 1
        j = 0
        while j < n and x[j] < np.min(Gf2):
            j += 1
        k = i
        while j >= 0 and k >= 0:
            FRF[j] = FRP[k]
            j -= 1
            k -= 1
        if j >= 0:
            FRF[: j + 1] = 0.0

    # Upper tail correction
    if FRF[-1] < 1.0:
        QQ = np.quantile(ObsRp, FRF[-1])
        i = n - 1
        while i >= 0 and x[i] >= QQ:
            i -= 1
        i += 1
        j = n - 2
        while j >= 0 and FRF[j] == FRF[-1]:
            j -= 1
        if j < 0:
            raise ValueError("CDF-t: dev is too small, increase it")
        dif = min(n - 1 - j, n - 1 - i)
        FRF[j: j + dif + 1] = FRP[i: i + dif + 1]
        k = j + dif
        if k < n - 1:
            FRF[k:] = 1.0

    qntl = FGf(Gf2)
    return np.interp(qntl, FRF, x, left=x[0], right=x[-1])


def _ssr_perturb(x: np.ndarray, threshold: float, rng: np.random.Generator) -> np.ndarray:
    """Singularity Stochastic Removal: replace values below threshold with U(0, threshold)."""
    x = x.copy()
    mask = x < threshold
    if mask.any():
        x[mask] = rng.uniform(0.0, threshold, size=int(mask.sum()))
    return x


def _cdft_1d_precip(DataGp: np.ndarray, ObsRp: np.ndarray, DataGf: np.ndarray,
                    wet_threshold: float = 1.0, clip_min: float | None = None,
                    use_ssr: bool = False, ssr_rezero: bool = False,
                    ssr_seed: int = 0, **cdft_kwargs) -> np.ndarray:
    """CDF-t with dry-day handling for precipitation.

    use_ssr=False: hard frequency adaptation — zero values below GCM threshold
        matched to ERA5 wet-day frequency, fit CDF-t on wet days only.
    use_ssr=True: Singularity Stochastic Removal (Vrac et al. 2016) — replace
        sub-threshold values with U(0, wet_threshold) noise so CDF-t operates
        on continuous distributions; use a small wet_threshold (e.g. 0.25 mm/day).
    """
    DataGp = np.asarray(DataGp, dtype=float).ravel()
    ObsRp  = np.asarray(ObsRp,  dtype=float).ravel()
    DataGf = np.asarray(DataGf, dtype=float).ravel()

    if use_ssr:
        rng = np.random.default_rng(ssr_seed)
        corrected = _cdft_1d(
            _ssr_perturb(DataGp, wet_threshold, rng),
            _ssr_perturb(ObsRp,  wet_threshold, rng),
            _ssr_perturb(DataGf, wet_threshold, rng),
            **cdft_kwargs,
        )
        if ssr_rezero:
            corrected = np.where(corrected < wet_threshold, 0.0, corrected)
        return np.maximum(corrected, 0.0)

    if clip_min is None:
        clip_min = wet_threshold

    obs_wet_freq = float(np.mean(ObsRp >= wet_threshold))
    if obs_wet_freq == 0.0:
        return np.zeros_like(DataGf)
    gcm_thresh = 0.0 if obs_wet_freq >= 1.0 else float(np.quantile(DataGp, 1.0 - obs_wet_freq))

    gp_wet  = DataGp >= gcm_thresh
    gf_wet  = DataGf >= gcm_thresh
    obs_wet = ObsRp  >= wet_threshold

    if gp_wet.sum() < 2 or gf_wet.sum() < 2 or obs_wet.sum() < 2:
        out = np.zeros_like(DataGf)
        out[gf_wet] = np.maximum(DataGf[gf_wet], clip_min)
        return out

    corrected_wet = _cdft_1d(DataGp[gp_wet], ObsRp[obs_wet], DataGf[gf_wet], **cdft_kwargs)
    out = np.zeros_like(DataGf)
    out[gf_wet] = np.maximum(corrected_wet, clip_min)
    return out


def _cdft_field(gcm_calib: xr.DataArray, obs_calib: xr.DataArray,
                gcm_eval: xr.DataArray, params: dict) -> xr.DataArray:
    """Apply CDF-t month-by-month to a single variable field on the ERA5 grid."""
    use_precip = params.get("use_dry_day_correction", False)
    wet_threshold = params.get("wet_threshold", 1.0)
    clip_min  = params.get("clip_min")
    use_ssr   = params.get("use_ssr", False)
    ssr_rezero = params.get("ssr_rezero", False)
    ssr_seed  = params.get("ssr_seed", 0)
    core_kw   = {k: params[k] for k in ("npas", "dev", "scale", "pui", "etype")
                 if k in params}

    def _apply(gp, obs, gf):
        if use_precip:
            return _cdft_1d_precip(gp, obs, gf,
                                   wet_threshold=wet_threshold,
                                   clip_min=clip_min,
                                   use_ssr=use_ssr,
                                   ssr_rezero=ssr_rezero,
                                   ssr_seed=ssr_seed,
                                   **core_kw)
        out = _cdft_1d(gp, obs, gf, **core_kw)
        if clip_min is not None:
            out = np.maximum(out, clip_min)
        return out

    results = []
    for m in tqdm(range(1, 13), desc="month"):
        gp  = gcm_calib.isel(time=gcm_calib.time.dt.month == m).drop_vars("time")
        obs = obs_calib.isel(time=obs_calib.time.dt.month  == m).drop_vars("time")
        gf  = gcm_eval.isel(time=gcm_eval.time.dt.month   == m)

        if gp.sizes["time"] == 0 or obs.sizes["time"] == 0 or gf.sizes["time"] == 0:
            continue

        corrected = xr.apply_ufunc(
            _apply,
            gp, obs, gf.drop_vars("time"),
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True,
            output_dtypes=[gf.dtype],
            exclude_dims={"time"},
        )
        corrected = corrected.assign_coords(time=gf.time)
        results.append(corrected)

    return xr.concat(results, dim="time").sortby("time").transpose("time", "lat", "lon")


# ── Public callable for METHOD_REGISTRY ──────────────────────────────────────

def cdft_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                   variable: str) -> xr.DataArray:
    """CDF-t downscaling: bicubic regrid then true CDF-t transform.

    Calibration period equals the evaluation period (degenerate case),
    which reduces to a properly-tailed empirical QM. For a separate
    calibration period use inference_cdft.py with --calib-start/--calib-end.
    """
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent / "evaluation"))
    from data_processing import bicubic_regrid  # noqa: PLC0415

    gcm_fine = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
    gcm_fine = gcm_fine.assign_coords(time=obs.time)

    p = _params(variable)
    return _cdft_field(gcm_fine, obs, gcm_fine, p)


# ── Standalone inference ──────────────────────────────────────────────────────

def run(era5_dir: Path, gcm_dir: Path, out_dir: Path,
        variables: list[str],
        calib_start: int, calib_end: int,
        eval_start: int, eval_end: int,
        scenario: str = "ssp585") -> None:

    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent / "evaluation"))
    from data_processing import (  # noqa: PLC0415
        DEFAULT_VARIABLES, load_era5, load_gcm, align_time,
        save_predictions, bicubic_regrid,
    )
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Calibration: {calib_start}–{calib_end}  |  Evaluation: {eval_start}–{eval_end}")

    # CDF-t only needs ERA5 + GCM for calibration, and GCM for evaluation.
    # ERA5 is NOT loaded for the eval period — so eval can extend past ERA5 coverage
    # (e.g. 2000-2100 while ERA5 only exists through 2020).
    era5_calib, gcm_calib, gcm_eval = {}, {}, {}

    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        era5_calib[var] = load_era5(era5_dir, var, calib_start, calib_end)
        gcm_calib[var]  = load_gcm(gcm_dir, var, calib_start, calib_end, "historical")
        gcm_eval[var]   = load_gcm(gcm_dir, var, eval_start,  eval_end,   scenario)

    # Align calibration period (ERA5 + GCM calib must share a common time axis)
    all_calib = {f"era5_{v}": era5_calib[v] for v in variables}
    all_calib.update({f"gcm_{v}": gcm_calib[v] for v in variables})
    aligned_calib = align_time(all_calib)
    era5_calib = {v: aligned_calib[f"era5_{v}"] for v in variables}
    gcm_calib  = {v: aligned_calib[f"gcm_{v}"]  for v in variables}

    # Use ERA5 calib grid as target (all variables share the same ERA5 grid)
    ref_era5 = next(iter(era5_calib.values()))
    target_lat, target_lon = ref_era5.lat.values, ref_era5.lon.values

    arrays = []
    ref_da = None
    for var in variables:
        print(f"  CDF-t {var} ...", flush=True)
        gc_c = bicubic_regrid(gcm_calib[var], target_lat, target_lon)
        gc_c = gc_c.assign_coords(time=era5_calib[var].time)

        # Eval: bicubic-regrid GCM to ERA5 grid; keep GCM's own time axis
        gc_e = bicubic_regrid(gcm_eval[var], target_lat, target_lon)
        # Assign a time coordinate compatible with _cdft_field (month groupby)
        # cftime coordinates work fine for groupby — keep them as-is
        gc_e = gc_e.assign_coords(time=gcm_eval[var].time)

        p = _params(var)
        corrected = _cdft_field(gc_c, era5_calib[var], gc_e, p)
        if ref_da is None:
            ref_da = corrected
        arrays.append(torch.tensor(corrected.values, dtype=torch.float32))

    tensor = torch.stack(arrays, dim=1)
    save_predictions(tensor, variables, out_dir, ref_da)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",    required=True)
    p.add_argument("--gcm-dir",     required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--variables",   nargs="+", default=None)
    p.add_argument("--calib-start", type=int, default=None,
                   help="Start of calibration period (default: same as --eval-start)")
    p.add_argument("--calib-end",   type=int, default=None,
                   help="End of calibration period (default: same as --eval-end)")
    p.add_argument("--eval-start",  type=int, default=2004)
    p.add_argument("--eval-end",    type=int, default=2014)
    p.add_argument("--scenario",    default="ssp585")
    args = p.parse_args()

    # Resolve default variables here to avoid importing data_processing at module level
    _HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(_HERE.parent / "evaluation"))
    from data_processing import DEFAULT_VARIABLES  # noqa: PLC0415

    variables    = args.variables or DEFAULT_VARIABLES
    calib_start  = args.calib_start if args.calib_start is not None else args.eval_start
    calib_end    = args.calib_end   if args.calib_end   is not None else args.eval_end

    run(Path(args.era5_dir), Path(args.gcm_dir), Path(args.out_dir),
        variables, calib_start, calib_end,
        args.eval_start, args.eval_end, args.scenario)


if __name__ == "__main__":
    main()
