#!/usr/bin/env python3
"""Add seasonal_corr and lag1_autocorr_anom_mae to finished runs without re-running their evaluation.

Both metrics are vs-ERA5 accuracy metrics that evaluate.py computes since they were added. For a run
evaluated before, this tool loads the same ERA5 years and the same predictions, aligns them as
evaluate.py does, and computes the two metrics with the functions evaluate.py calls. The run's
metrics.json is only read: the output is <out>/<cell>/metrics_acc2.json.

Tripwire: seasonal_mae is recomputed from the same inputs and must equal the value stored in the run's
metrics.json, which shows the inputs are the evaluation's own (ERA5 directory, years, predictions,
alignment). A mismatch stops the tool.

Usage (cluster job; a 2000-2100 France tensor is ~3.3 GB, read by memory map):
    python tools/backfill_accuracy_metrics.py --root results_v2 --out results_acc2/results_v2 \
        --era5 france=<dir> --era5 australia=<dir> --era5 us_central=<dir> \
        france/cm australia/cesm2/cdft ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE.parent / "src" / "evaluation"))

from data_processing import align_time, load_era5, load_predictions  # noqa: E402
from metrics import lag1_autocorr_anom_mae, seasonal_corr, seasonal_mae  # noqa: E402

REL_TOL = 1e-6   # same functions, same float64 inputs: only summation order may differ


def backfill_run(run: Path, era5_dir: Path, start: int, end: int) -> dict:
    """Both metrics per variable for one run dir, with the seasonal_mae check."""
    raw = (run / "metrics.json").read_bytes()
    stored = json.loads(raw)
    variables = [v for v in stored if not v.startswith("_")]
    preds = load_predictions(run, variables, mmap=True)
    out = {"_meta": {"metrics_json_md5": hashlib.md5(raw).hexdigest(), "era5_dir": str(era5_dir),
                     "period": [start, end]}}
    for var in variables:
        obs = load_era5(era5_dir, var, start, end).astype("float64")
        pred = preds[var]
        years = pred.time.dt.year
        pred = pred.isel(time=((years >= start) & (years <= end)).values).astype("float64")
        aligned = align_time({"obs": obs, "pred": pred})
        obs, pred = aligned["obs"], aligned["pred"]
        check = seasonal_mae(obs, pred)
        ref = stored[var]["vs_obs"]["seasonal_mae"]
        if not abs(check - ref) <= REL_TOL * max(1.0, abs(ref)):   # also stops on NaN
            raise SystemExit(f"{run} {var}: seasonal_mae {check!r} != stored {ref!r}: inputs differ")
        out[var] = {"seasonal_corr": seasonal_corr(obs, pred),
                    "lag1_autocorr_anom_mae": lag1_autocorr_anom_mae(obs, pred),
                    "seasonal_mae_check": check, "n_days": int(obs.sizes["time"])}
        print(f"  {var}: seasonal_corr {out[var]['seasonal_corr']:.4f}  "
              f"lag1_autocorr_anom_mae {out[var]['lag1_autocorr_anom_mae']:.4f}  "
              f"(seasonal_mae {check:.6g} = stored)", flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True, help="results root holding the run dirs")
    p.add_argument("--out", required=True, help="output root (the runs are not written)")
    p.add_argument("--era5", action="append", required=True, help="region=ERA5 dir, once per region")
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2020)
    p.add_argument("cells", nargs="+", help="<region>[/<gcm>]/<method dir> relative to --root")
    args = p.parse_args()
    era5 = dict(e.split("=", 1) for e in args.era5)
    for cell in args.cells:
        dst = Path(args.out) / cell / "metrics_acc2.json"
        if dst.exists():
            print(f"{cell}: done already", flush=True)
            continue
        print(f"{cell}", flush=True)
        res = backfill_run(Path(args.root) / cell, Path(era5[cell.split("/")[0]]), args.start, args.end)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
