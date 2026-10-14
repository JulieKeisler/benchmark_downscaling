#!/usr/bin/env python3
"""Tripwire: a finished method run is complete and finite (contract §E4).

Checks `<method_dir>/metrics.json`: every expected variable has finite values
for every vs_obs and vs_gcm metric and both change periods; and the
predictions cover 2000-01-01..2100-12-31 with the expected variables.
Exits non-zero on any failure.

Usage:
    python tools/check_results.py results_v2/france/dotc --variables tas pr psl sfcWind sfcWindmax rsds
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

VS_OBS = ["bias_mae", "std_error", "ks_full", "ks_q95", "seasonal_mae",
          "extreme_freq", "extreme_intensity", "spectral_mae",
          "spectral_fine", "spectral_fine_bias"]
VS_GCM = ["temporal_corr", "temporal_corr_annual", "annual_anomaly_bias", "lag1_autocorr_mae"]
DELTA_PERIODS = ["2040_2060", "2080_2100"]


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def check(method_dir: Path, variables: list[str], pred_dir: Path | None = None) -> list[str]:
    fails = []
    mfile = method_dir / "metrics.json"
    if not mfile.exists():
        return [f"missing {mfile}"]
    m = json.loads(mfile.read_text())
    for var in variables:
        if var not in m:
            fails.append(f"{var}: absent from metrics.json")
            continue
        block = m[var]
        for grp, keys in (("vs_obs", VS_OBS), ("vs_gcm", VS_GCM)):
            for k in keys:
                if not _finite(block.get(grp, {}).get(k)):
                    fails.append(f"{var}.{grp}.{k} = {block.get(grp, {}).get(k)!r}")
        delta = m.get("_delta", {}).get(var, {})
        for per in DELTA_PERIODS:
            if not _finite(delta.get(per)):
                fails.append(f"_delta.{var}.{per} = {delta.get(per)!r}")
    pred_dir = pred_dir or method_dir
    dates = pred_dir / "eval_dates.npy"
    if dates.exists():
        d = np.load(dates, allow_pickle=True)
        if str(d[0])[:10] != "2000-01-01" or str(d[-1])[:10] != "2100-12-31":
            fails.append(f"predictions cover {str(d[0])[:10]}..{str(d[-1])[:10]}, not 2000-01-01..2100-12-31")
        # contiguous for any GCM calendar: consecutive days, Feb 29/30 may be absent
        ymd = np.array([[int(x) for x in str(t)[:10].split("-")] for t in d])
        day = ymd[:, 0] * 372 + (ymd[:, 1] - 1) * 31 + ymd[:, 2]
        if (np.diff(day) <= 0).any() or (np.diff(day) > 4).any():
            fails.append("eval_dates not strictly increasing day by day (gap or duplicate)")
        pt = pred_dir / "predictions.pt"
        if pt.exists():
            import torch  # noqa: PLC0415
            t = torch.load(pt, map_location="cpu", mmap=True)
            if t.shape[0] != len(d):
                fails.append(f"predictions.pt has {t.shape[0]} days, eval_dates {len(d)}")
            n_bad = sum(int((~torch.isfinite(t[i:i + 1000])).sum()) for i in range(0, t.shape[0], 1000))
            if n_bad:
                fails.append(f"predictions.pt has {n_bad} non-finite values")
        names = [str(v) for v in np.load(pred_dir / "variables.npy", allow_pickle=True)]
        if sorted(names) != sorted(variables):
            fails.append(f"predictions hold {names}, expected {variables}")
    else:
        fails.append("eval_dates.npy missing")
    return fails


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("method_dir")
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--pred-dir", default=None,
                   help="where predictions live when evaluated into another dir")
    args = p.parse_args()
    fails = check(Path(args.method_dir), args.variables,
                  Path(args.pred_dir) if args.pred_dir else None)
    if fails:
        print(f"TRIPWIRE FAILED for {args.method_dir}:\n  " + "\n  ".join(fails[:40]))
        sys.exit(1)
    print(f"{args.method_dir}: complete and finite")


if __name__ == "__main__":
    main()
