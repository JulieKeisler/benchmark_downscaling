#!/usr/bin/env python3
"""Add the daily ACC vs the GCM, the extreme maps and the precipitation SAL scores to finished runs.

These metrics (2026-11-19) are computed by evaluate.py since they were added. For a run evaluated
before, this tool loads the same ERA5 years, GCM files and predictions, aligns them as evaluate.py
does, and computes them with the functions evaluate.py calls. The run dir is only read. Output:
    <out>/<cell>/metrics_extra.json       {var: {acc_daily, acc_daily_fullclim, [sal_*]}, _meta}
    <out>/<cell>/plots/<var>/extremes_data_<var>.npz, acc_daily_data_<var>.npz
    <out>/<cell>/plots/pr/sal_data_pr.npz, sal_daily_pr.npz (daily SAL quantities, for later analysis)

Tripwires (a mismatch stops the tool before anything is written for the cell):
- seasonal_mae recomputed from ERA5 and the predictions equals the stored value: same ERA5 dir,
  years, predictions and alignment;
- temporal_corr_annual recomputed from the GCM and the predictions equals the stored value: same GCM
  files, scenario, regridding and last year.

Usage (cluster job; one call per region x GCM pair, whose cells share ERA5, GCM and scenario):
    python tools/backfill_extra_metrics.py --root results_v2 --out results_extra/results_v2 \
        --era5-dir <dir> --gcm-dir <dir> --scenario ssp370 france/cm france/bcsd ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "src" / "evaluation"))

from data_processing import align_time, bicubic_regrid, load_era5, load_gcm, load_predictions  # noqa: E402
from metrics import (  # noqa: E402
    acc_daily_map, extreme_maps, grid_geometry, mean_acc_daily, mean_temporal_corr_annual,
    sal_cell_maps, sal_daily, sal_distribution_scores, seasonal_mae,
)
from metrics.object_metrics import THR_FACTOR, THR_QUANTILE, WET_MM  # noqa: E402
from plots import plot_acc_daily, plot_extreme_maps, plot_sal_maps  # noqa: E402

REL_TOL = 1e-6   # same functions, same float64 inputs: only summation order may differ


def _check(name: str, value: float, stored: float, where: str) -> None:
    if not abs(value - stored) <= REL_TOL * max(1.0, abs(stored)):   # also stops on NaN
        raise SystemExit(f"{where}: {name} {value!r} != stored {stored!r}: inputs differ")


def _pair(a, b):
    aligned = align_time({"a": a, "b": b})
    return aligned["a"], aligned["b"]


def backfill_run(run: Path, out: Path, era5_dir: Path, gcm_dir: Path, scenario: str,
                 start: int, end: int, gcm_end: int) -> dict:
    """All new metrics for one run dir; maps written under out/plots/<var>/."""
    raw = (run / "metrics.json").read_bytes()
    stored = json.loads(raw)
    variables = [v for v in stored if not v.startswith("_")]
    preds = load_predictions(run, variables, mmap=True)
    res = {"_meta": {"metrics_json_md5": hashlib.md5(raw).hexdigest(), "era5_dir": str(era5_dir),
                     "gcm_dir": str(gcm_dir), "scenario": scenario, "period": [start, end],
                     "gcm_end": gcm_end, "acc_block_years": 20,
                     "sal": {"thr_factor": THR_FACTOR, "thr_quantile": THR_QUANTILE, "wet_mm": WET_MM}}}
    maps_out: dict[str, list] = {}
    for var in variables:
        pred32 = preds[var]
        n_keep = int((pred32.time.dt.year <= gcm_end).sum())        # as evaluate.py
        pred_full = pred32.isel(time=slice(0, n_keep)).astype("float64")
        obs = load_era5(era5_dir, var, start, end).astype("float64")
        obs, pred_obs = _pair(obs, pred_full)
        _check("seasonal_mae", seasonal_mae(obs, pred_obs), stored[var]["vs_obs"]["seasonal_mae"],
               f"{run} {var}")
        gcm_full = load_gcm(gcm_dir, var, start, gcm_end, scenario).astype("float64")
        gcm_up = bicubic_regrid(gcm_full, obs.lat.values, obs.lon.values)
        pred_gcm, gcm_m = _pair(pred_full, gcm_up)
        _check("temporal_corr_annual", mean_temporal_corr_annual(gcm_m, pred_gcm),
               stored[var]["vs_gcm"]["temporal_corr_annual"], f"{run} {var}")

        lat, lon = obs.lat.values, obs.lon.values
        acc_map = acc_daily_map(gcm_m, pred_gcm).values
        entry = {"acc_daily": float(np.nanmean(acc_map)),
                 "acc_daily_fullclim": mean_acc_daily(gcm_m, pred_gcm, block_years=None),
                 "n_days_era5": int(obs.sizes["time"]), "n_days_gcm": int(pred_gcm.sizes["time"])}
        maps_out[var] = [("acc", acc_map),
                         ("extremes", extreme_maps(obs, pred_obs, q_low=None if var == "pr" else 0.01))]
        if var == "pr":
            q_obs, q_pred = sal_daily(obs), sal_daily(pred_obs)
            entry.update(sal_distribution_scores(q_obs, q_pred, grid_geometry(lat, lon)[3]))
            sal_maps, sal_scores = sal_cell_maps(q_obs, q_pred)
            entry.update(sal_scores)
            daily = {f"{tag}_{k}": q[k] for tag, q in (("obs", q_obs), ("pred", q_pred))
                     for k in ("D", "xc", "yc", "r", "V", "n_obj", "total", "obj_area", "obj_flat", "obj_day")}
            maps_out[var] += [("sal", sal_maps), ("sal_daily", daily)]
        maps_out[var].append(("latlon", (lat, lon)))
        res[var] = entry
        print(f"  {var}: acc_daily {entry['acc_daily']:.4f} (one climatology {entry['acc_daily_fullclim']:.4f})"
              + (f"  SAL S {entry['sal_S']:+.3f} A {entry['sal_A']:+.3f} L {entry['sal_L']:.3f}"
                 if var == "pr" else "") + "  (seasonal_mae, temporal_corr_annual = stored)", flush=True)

    # Written only once every variable passed both tripwires.
    for var, items in maps_out.items():
        d = dict(items)
        lat, lon = d["latlon"]
        vdir = out / "plots" / var
        vdir.mkdir(parents=True, exist_ok=True)
        plot_acc_daily(d["acc"], lat, lon, var, run.name, vdir, save_png=False)
        plot_extreme_maps(d["extremes"], lat, lon, var, run.name, vdir, save_png=False)
        if "sal" in d:
            plot_sal_maps(d["sal"], lat, lon, run.name, vdir, save_png=False)
            np.savez_compressed(vdir / "sal_daily_pr.npz", **d["sal_daily"])
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="results root holding the run dirs")
    p.add_argument("--out", required=True, help="output root (the runs are not written)")
    p.add_argument("--era5-dir", required=True)
    p.add_argument("--gcm-dir", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2020)
    p.add_argument("--gcm-end", type=int, default=2099)
    p.add_argument("cells", nargs="+", help="<region>[/<gcm>]/<method dir> relative to --root")
    args = p.parse_args()
    for cell in args.cells:
        out = Path(args.out) / cell
        dst = out / "metrics_extra.json"
        if dst.exists():
            print(f"{cell}: done already", flush=True)
            continue
        print(f"{cell}", flush=True)
        res = backfill_run(Path(args.root) / cell, out, Path(args.era5_dir), Path(args.gcm_dir),
                           args.scenario, args.start, args.end, args.gcm_end)
        dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
