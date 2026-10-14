#!/usr/bin/env python3
"""dOTC baseline — multivariate dynamical optimal-transport correction (SBCK.dOTC).

dOTC (Robin et al. 2019, HESS 23:773) corrects all variables jointly. For each
ERA5 grid cell and calendar month it transports the GCM's evolution between the
calibration period (X0) and a projection window (X1) onto the ERA5 calibration
distribution (Y0), so the inter-variable dependence of ERA5 is kept and the
GCM's change is transferred (cov_factor="std" rescales the change by the
ERA5/GCM spread ratio).

Pipeline:
    1. Bicubic-regrid the GCM (calibration and evaluation) to the ERA5 grid.
    2. Per cell and calendar month, fit SBCK.dOTC on (Y0=ERA5 1980-1999,
       X0=GCM 1980-1999, X1=GCM window) for every 20-year sliding window, and
       keep the centre decade of each window (same windows as CDF-t).
    3. Each variable is standardised by the ERA5 calibration mean and sd of
       that cell and month before the transport (the OT cost is a Euclidean
       distance: in physical units psl in Pa decides every match; on test
       cells the output then correlates 0.41 with the GCM day for tas), with
       bins of 0.01 sd so outputs stay close to the transported values.
    4. Precipitation dry days use SSR (Vrac et al. 2016) at 0.1 mm/day
       (Julie, 2026-10-13): values <= 0.1 are replaced by uniform draws in
       (0.001, 0.1) before the transport and outputs <= 0.1 are set to 0.
       Calibration-only split (fit 1980-89, test 1990-99, Australia): pr KS
       0.22 against 0.32 with SBCK's default threshold.
    5. pr, sfcWind, sfcWindmax and rsds are clipped at 0 after correction.

Cells are corrected independently, jointly across variables (Julie,
2026-10-13), so neighbouring cells can draw on different ERA5 days: the
fields are less spatially coherent than ERA5, which the spatial metrics show.

dOTC samples inside its histogram bins, so each fit is seeded from
(cell, month, window) and the output is reproducible.

Usage:
    python inference_dotc.py \
        --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
        --out-dir results/dotc --calib-start 1980 --calib-end 1999 \
        --eval-start 2000 --eval-end 2100 --workers 64
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from inference_cdft import _sliding_windows  # noqa: E402

CLIP_MIN: dict[str, float] = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0, "rsds": 0.0}
PR_DRY = 0.1      # mm/day: SSR threshold for precipitation (contract §C)
BIN_WIDTH = 0.01  # dOTC histogram bin width in ERA5 standard deviations

# Filled before the pool forks; workers read them copy-on-write.
_G: dict = {}


def _dotc_fit_predict(Y0: np.ndarray, X0: np.ndarray, X1: np.ndarray,
                      pr_col: int | None, seed: int) -> np.ndarray:
    """One dOTC correction: Y0/X0 [n0, V] calibration, X1 [n1, V] window → [n1, V].

    SSR on pr at PR_DRY, then every variable standardised by the ERA5 (Y0)
    mean and sd, transported with SBCK.dOTC, and mapped back.
    """
    import SBCK  # noqa: PLC0415 — heavy optional dependency
    np.random.seed(seed)                      # SBCK draws from the global RNG
    rng = np.random.default_rng(seed)
    Y0, X0, X1 = (np.asarray(a, dtype=np.float64).copy() for a in (Y0, X0, X1))
    if pr_col is not None:
        for A in (Y0, X0, X1):
            dry = A[:, pr_col] <= PR_DRY
            A[dry, pr_col] = rng.uniform(PR_DRY / 100, PR_DRY, int(dry.sum()))
    mu, sd = Y0.mean(0), Y0.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    bc = SBCK.dOTC(bin_width=np.full(Y0.shape[1], BIN_WIDTH))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bc.fit((Y0 - mu) / sd, (X0 - mu) / sd, (X1 - mu) / sd)
        Z = np.asarray(bc.predict((X1 - mu) / sd)) * sd + mu
    if pr_col is not None:
        Z[:, pr_col] = np.where(Z[:, pr_col] > PR_DRY, Z[:, pr_col], 0.0)
    return Z.astype(np.float32)


def _correct_row(i: int) -> np.ndarray:
    """dOTC for every cell of ERA5 row i, every month and window → [T_eval, V, W]."""
    obs, gcm_c, gcm_e = _G["obs"], _G["gcm_c"], _G["gcm_e"]
    cal_mon, ev_mon, ev_yr = _G["cal_mon"], _G["ev_mon"], _G["ev_yr"]
    windows, pr_col = _G["windows"], _G["pr_col"]
    T_e, V, _, W = gcm_e.shape
    out = np.full((T_e, V, W), np.nan, dtype=np.float32)
    for j in range(W):
        for m in range(1, 13):
            c_sel = cal_mon == m
            Y0 = obs[c_sel, :, i, j]
            X0 = gcm_c[c_sel, :, i, j]
            for w, (win_start, win_end, keep_start, keep_end) in enumerate(windows):
                win = (ev_mon == m) & (ev_yr >= win_start) & (ev_yr <= win_end)
                if not win.any():
                    continue
                seed = ((i * W + j) * 12 + (m - 1)) * len(windows) + w
                Z1 = _dotc_fit_predict(Y0, X0, gcm_e[win, :, i, j], pr_col, seed)
                idx = np.flatnonzero(win)
                keep = (ev_yr[idx] >= keep_start) & (ev_yr[idx] <= keep_end)
                out[idx[keep], :, j] = Z1[keep]
    return out


def dotc_field(obs: np.ndarray, gcm_c: np.ndarray, gcm_e: np.ndarray,
               cal_mon: np.ndarray, ev_mon: np.ndarray, ev_yr: np.ndarray,
               variables: list[str], windows: list[tuple[int, int, int, int]],
               workers: int = 1) -> np.ndarray:
    """dOTC on [T, V, H, W] arrays already on the ERA5 grid → [T_eval, V, H, W].

    obs / gcm_c: calibration ERA5 / GCM (time-aligned); gcm_e: GCM to correct.
    """
    _G.update(obs=obs, gcm_c=gcm_c, gcm_e=gcm_e, cal_mon=cal_mon, ev_mon=ev_mon,
              ev_yr=ev_yr, windows=windows,
              pr_col=variables.index("pr") if "pr" in variables else None)
    H = gcm_e.shape[2]
    t0 = time.time()
    rows: list[np.ndarray] = [None] * H  # type: ignore[list-item]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            for n, (i, row) in enumerate(zip(range(H), pool.imap(_correct_row, range(H)))):
                rows[i] = row
                print(f"    row {n + 1}/{H} ({time.time() - t0:.0f}s)", flush=True)
    else:
        for i in range(H):
            rows[i] = _correct_row(i)
    out = np.stack(rows, axis=2)  # [T_eval, V, H, W]
    for c, var in enumerate(variables):
        if var in CLIP_MIN:
            np.maximum(out[:, c], CLIP_MIN[var], out=out[:, c])
    return out


# ── Convenience wrapper (for inline METHOD_REGISTRY use) ─────────────────────

def dotc_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                   variable: str) -> xr.DataArray:
    """Univariate dOTC on the ERA5 grid (degenerate: calib = eval period)."""
    from data_processing import bicubic_regrid  # noqa: PLC0415

    gcm_fine = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
    gcm_fine = gcm_fine.assign_coords(time=obs.time)
    years = obs.time.dt.year.values
    window = [(int(years.min()), int(years.max()), int(years.min()), int(years.max()))]
    out = dotc_field(obs.values[:, None], gcm_fine.values[:, None], gcm_fine.values[:, None],
                     obs.time.dt.month.values, obs.time.dt.month.values, years,
                     [variable], window)
    return obs.copy(data=out[:, 0])


# ── Standalone inference ──────────────────────────────────────────────────────

def run(era5_dir: Path, gcm_dir: Path, out_dir: Path, variables: list[str],
        calib_start: int, calib_end: int, eval_start: int, eval_end: int,
        scenario: str = "ssp370", win_size: int = 20, step: int = 10, trim: int = 5,
        workers: int = 1) -> None:
    from data_processing import (  # noqa: PLC0415
        load_era5, load_gcm, align_time, save_predictions, bicubic_regrid,
    )
    import torch  # noqa: PLC0415

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"dOTC | calib {calib_start}–{calib_end} | eval {eval_start}–{eval_end} | "
          f"window {win_size}y step {step}y trim {trim}y | {workers} workers", flush=True)

    calib = {}
    gcm_eval = {}
    for var in variables:
        print(f"  Loading {var} ...", flush=True)
        calib[f"era5_{var}"] = load_era5(era5_dir, var, calib_start, calib_end)
        calib[f"gcm_{var}"] = load_gcm(gcm_dir, var, calib_start, calib_end, "historical")
        # From eval_start - trim, so the first window (e.g. 1995-2014) is full.
        gcm_eval[var] = load_gcm(gcm_dir, var, eval_start - trim, eval_end, scenario)
    calib = align_time(calib)
    gcm_eval = align_time(gcm_eval)

    ref = calib[f"era5_{variables[0]}"]
    lat, lon = ref.lat.values, ref.lon.values
    obs = np.stack([calib[f"era5_{v}"].values for v in variables], 1).astype(np.float32)
    gcm_c = np.stack([bicubic_regrid(calib[f"gcm_{v}"], lat, lon).values
                      for v in variables], 1).astype(np.float32)
    ev_time = gcm_eval[variables[0]].time
    gcm_e = np.stack([bicubic_regrid(gcm_eval[v], lat, lon).values
                      for v in variables], 1).astype(np.float32)
    print(f"  calib {obs.shape}, eval {gcm_e.shape}", flush=True)

    windows = _sliding_windows(eval_start, eval_end, win_size, step, trim)
    out = dotc_field(obs, gcm_c, gcm_e, ref.time.dt.month.values,
                     ev_time.dt.month.values, ev_time.dt.year.values,
                     variables, windows, workers)
    in_eval = ev_time.dt.year.values >= eval_start
    out, ev_time = out[in_eval], ev_time[in_eval]
    n_nan = int(np.isnan(out).sum())
    if n_nan:
        raise RuntimeError(f"dOTC left {n_nan} values unfilled — check the windows")

    ref_da = xr.DataArray(dims=["time", "lat", "lon"],
                          coords={"time": ev_time.values, "lat": lat, "lon": lon})
    save_predictions(torch.from_numpy(out), variables, out_dir, ref_da)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",    required=True)
    p.add_argument("--gcm-dir",     required=True)
    p.add_argument("--out-dir",     required=True)
    p.add_argument("--variables",   nargs="+", default=None)
    p.add_argument("--calib-start", type=int, default=1980)
    p.add_argument("--calib-end",   type=int, default=1999)
    p.add_argument("--eval-start",  type=int, default=2000)
    p.add_argument("--eval-end",    type=int, default=2100)
    p.add_argument("--scenario",    default="ssp370")
    p.add_argument("--win-size",    type=int, default=20)
    p.add_argument("--step",        type=int, default=10)
    p.add_argument("--trim",        type=int, default=5)
    p.add_argument("--workers",     type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = p.parse_args()

    from data_processing import DEFAULT_VARIABLES  # noqa: PLC0415
    run(Path(args.era5_dir), Path(args.gcm_dir), Path(args.out_dir),
        args.variables or DEFAULT_VARIABLES,
        args.calib_start, args.calib_end, args.eval_start, args.eval_end,
        args.scenario, args.win_size, args.step, args.trim, args.workers)


if __name__ == "__main__":
    main()
