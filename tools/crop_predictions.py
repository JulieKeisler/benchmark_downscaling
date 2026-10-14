#!/usr/bin/env python3
"""Copy method dirs with predictions cut to a year range (for the net-load lens).

For each <sub>/<method> given, writes <dst>/<sub>/<method>/ with
predictions.pt and eval_dates.npy restricted to start..end, plus lat.npy,
lon.npy, variables.npy and metrics.json unchanged. A 2000-2100 tensor is
~3.3 GB; 2000-2022 is ~0.75 GB.

Usage (cluster job):
    python tools/crop_predictions.py --root results_v2 --dst results_v2_tam \
        --start 2000 --end 2022 france/cm france/cnrm_cm6_1/cdft ...
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--root", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2022)
    p.add_argument("cells", nargs="+", help="<sub>/<method> relative to --root")
    args = p.parse_args()
    for cell in args.cells:
        src, dst = Path(args.root) / cell, Path(args.dst) / cell
        dates = np.load(src / "eval_dates.npy", allow_pickle=True)
        years = np.array([int(str(d)[:4]) for d in dates])
        keep = (years >= args.start) & (years <= args.end)
        pred = torch.load(src / "predictions.pt", map_location="cpu", weights_only=True)
        if pred.shape[0] != len(dates):
            raise SystemExit(f"{cell}: {pred.shape[0]} days in predictions, {len(dates)} dates")
        dst.mkdir(parents=True, exist_ok=True)
        idx = torch.from_numpy(np.flatnonzero(keep))
        torch.save(pred.index_select(0, idx).clone(), dst / "predictions.pt.tmp")
        (dst / "predictions.pt.tmp").rename(dst / "predictions.pt")
        np.save(dst / "eval_dates.npy", dates[keep])
        for f in ("lat.npy", "lon.npy", "variables.npy", "metrics.json"):
            shutil.copy2(src / f, dst / f)
        print(f"{cell}: {int(keep.sum())} days {dates[keep][0]}..{dates[keep][-1]}", flush=True)


if __name__ == "__main__":
    main()
