#!/usr/bin/env python3
"""Inter-variable dependence in the tails, per grid cell, for the dashboard.

For every pair of variables (X, Y) and every choice of part for each (low / high / full),
over 2000-2020:
  * tail x tail: co-occurrence ratio P(X in part and Y in part) / (P(X part) P(Y part));
    1 = independent, < 1 = rarely together, > 1 = often together;
  * tail x full: mean percentile (0-100, within the source's own distribution at that cell)
    of the full variable on the days the other is in its tail; 50 = no relation;
  * full x full: Spearman rank correlation.
"Low" = at or below the q-th percentile of ERA5 at that grid cell (2000-2020), "high" =
above the (100-q)-th; q = 10 or 20. The same ERA5 thresholds are applied to every source,
so a tail is the same physical event everywhere. For pr, "low" is a dry day (< 1 mm/day,
whatever q): the ERA5 10th percentile is exactly 0 in most cells and dry days are 59-86%
of the days, so daily low pr is not a tail, and an exact-zero threshold would depend on
how each method stores drizzle.

Sources: ERA5 and the 9 methods (bicubic = the interpolated GCM), read from the 2000-2022
crops in results/v2_tam (tools/crop_predictions.py). Writes, per method dir of
results/v2, plots/tail_dependence.npz, and the ERA5 maps once per pair as
plots/tail_dependence_era5.npz in the bicubic dir. Keys: "v1|v2|q|part1|part2" (float16
maps) and "v1|v2|spearman".

Usage: python tools/tail_dependence.py [region:gcm ...]
"""
from __future__ import annotations

import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "tools"))
from data_processing import load_era5  # noqa: E402
from fetch_v2_results import PAIRS, RESULT_DIRS as METHODS, era5_dir, sub  # noqa: E402

CROPS = ROOT / "results" / "v2_tam"
OUT = ROOT / "results" / "v2"
VARIABLES = ["tas", "pr", "psl", "sfcWind", "sfcWindmax", "rsds"]   # key order of the maps
QS = (10, 20)
PARTS = ("low", "high", "full")
YEARS = (2000, 2020)
DRY_MM = 1.0        # ETCCDI wet-day threshold, mm/day


def load_source(reg: str, gcm: str, src: str, vars_: list[str]) -> tuple[dict, np.ndarray, np.ndarray]:
    """{var: [T, H, W] float32}, lat, lon over YEARS."""
    if src == "era5":
        das = {v: load_era5(era5_dir(reg), v, *YEARS) for v in vars_}
        lat, lon = das[vars_[0]].lat.values, das[vars_[0]].lon.values
        return {v: da.values.astype(np.float32) for v, da in das.items()}, lat, lon
    d = CROPS / sub(reg, gcm) / src
    saved = list(np.load(d / "variables.npy", allow_pickle=True))
    dates = pd.to_datetime(np.load(d / "eval_dates.npy", allow_pickle=True))
    p = torch.load(d / "predictions.pt", map_location="cpu", weights_only=True).numpy()
    n = min(len(p), len(dates))
    keep = (dates[:n].year >= YEARS[0]) & (dates[:n].year <= YEARS[1])
    p = p[:n][keep]
    return ({v: p[:, saved.index(v)].astype(np.float32) for v in vars_},
            np.load(d / "lat.npy"), np.load(d / "lon.npy"))


def tail_mask(x: np.ndarray, var: str, part: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if part == "low":
        return x < DRY_MM if var == "pr" else x <= lo
    return x > hi


def maps_for(fields: dict, thresholds: dict, vars_: list[str]) -> dict[str, np.ndarray]:
    ranks = {v: (rankdata(f, axis=0) / f.shape[0] * 100).astype(np.float32) for v, f in fields.items()}
    out = {}
    for v1, v2 in combinations(vars_, 2):
        r1, r2 = ranks[v1] - ranks[v1].mean(0), ranks[v2] - ranks[v2].mean(0)
        rho = (r1 * r2).mean(0) / (r1.std(0) * r2.std(0))
        out[f"{v1}|{v2}|spearman"] = rho.astype(np.float16)
        for q in QS:
            masks = {(v, p): tail_mask(fields[v], v, p, *thresholds[(v, q)])
                     for v in (v1, v2) for p in ("low", "high")}
            for p1 in PARTS:
                for p2 in PARTS:
                    if p1 == p2 == "full":
                        continue
                    key = f"{v1}|{v2}|{q}|{p1}|{p2}"
                    if p1 != "full" and p2 != "full":
                        a, b = masks[(v1, p1)], masks[(v2, p2)]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            val = (a & b).mean(0) / (a.mean(0) * b.mean(0))
                    else:   # mean percentile of the full variable on the other's tail days
                        tail, full_rank = ((masks[(v1, p1)], ranks[v2]) if p2 == "full"
                                           else (masks[(v2, p2)], ranks[v1]))
                        cnt = tail.sum(0)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            val = np.where(cnt > 0, (full_rank * tail).sum(0) / cnt, np.nan)
                    out[key] = val.astype(np.float16)
    return out


def save(path: Path, maps: dict) -> None:
    # the app may read while this runs: never leave a half-written zip under the final name
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **maps)
    os.replace(tmp, path)


def main() -> None:
    wanted = [tuple(a.split(":")) for a in sys.argv[1:]] or PAIRS
    for reg, gcm in wanted:
        saved = set(np.load(CROPS / sub(reg, gcm) / "bicubic" / "variables.npy", allow_pickle=True))
        vars_ = [v for v in VARIABLES if v in saved]      # CESM2 has no sfcWindmax
        era5, lat, lon = load_source(reg, gcm, "era5", vars_)
        thresholds = {(v, q): (np.percentile(era5[v], q, axis=0), np.percentile(era5[v], 100 - q, axis=0))
                      for v in vars_ for q in QS}
        bic = OUT / sub(reg, gcm) / "bicubic" / "plots"
        bic.mkdir(parents=True, exist_ok=True)
        save(bic / "tail_dependence_era5.npz", maps_for(era5, thresholds, vars_))
        del era5
        for m in METHODS:
            fields, la, lo_ = load_source(reg, gcm, m, vars_)
            assert np.allclose(la, lat) and np.allclose(lo_, lon), f"{m}: grid differs from ERA5"
            d = OUT / sub(reg, gcm) / m / "plots"
            d.mkdir(parents=True, exist_ok=True)
            save(d / "tail_dependence.npz", maps_for(fields, thresholds, vars_))
            del fields
        print(f"{reg} x {gcm} done", flush=True)


if __name__ == "__main__":
    main()
