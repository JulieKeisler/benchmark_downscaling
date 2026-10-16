#!/usr/bin/env python3
"""Why a product has more or fewer energy events than ERA5: the factors of the count.

An event (evaluate.py, energy_views.energy_events) is a spell of >= 2 days on which the
region-mean daily series of two variables are both below their ERA5 20th percentiles,
2000-2020: Dunkelflaute = sfcWind and rsds, cold and calm = tas and sfcWind. Spell days are
joint days, so the count splits exactly into

    events = N x P(A low) x P(B low) x R x s / d

with R = P(both) / (P(A) P(B)) the dependence (1 = independent), s the share of the joint
days that fall in spells of >= 2 days and d the mean spell length.

For every pair and method, the tool rebuilds the region-mean series as evaluate.py does
(ERA5 on the product's dates), asserts that energy_events() gives back the counts and mean
lengths stored in the method's plot_data.json, and writes into the bicubic dir of results/v2:
  plots/energy_event_factors.json  the factors of ERA5 and of each product, per event;
  plots/energy_event_series.npz    the region-mean daily series, for the dashboard's scatter.
Reads the 2000-2022 crops in results/v2_tam (tools/crop_predictions.py).

Usage: python tools/energy_event_factors.py [region:gcm ...]
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "tools"))
from data_processing import load_era5  # noqa: E402
from energy_views import (EVENT_DEFS, EVENT_VARIABLES, domain_series, energy_events,  # noqa: E402
                          spell_stats)
from fetch_v2_results import PAIRS, RESULT_DIRS as METHODS, era5_dir, sub  # noqa: E402

CROPS = ROOT / "results" / "v2_tam"
OUT = ROOT / "results" / "v2"
YEARS = (2000, 2020)


def era5_series(reg: str) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    # float64 before the mean, as evaluate.py
    das = {v: load_era5(era5_dir(reg), v, *YEARS).astype("float64") for v in EVENT_VARIABLES}
    dates = pd.to_datetime(das["tas"].time.values).normalize()
    for v, da in das.items():
        assert pd.to_datetime(da.time.values).normalize().equals(dates), f"{reg}: ERA5 {v} dates differ"
    return dates, {v: domain_series(da) for v, da in das.items()}


def product_series(reg: str, gcm: str, m: str) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    d = CROPS / sub(reg, gcm) / m
    saved = list(np.load(d / "variables.npy", allow_pickle=True))
    dates = pd.to_datetime(np.load(d / "eval_dates.npy", allow_pickle=True)).normalize()
    p = torch.load(d / "predictions.pt", map_location="cpu", weights_only=True).numpy()
    n = min(len(p), len(dates))
    keep = np.flatnonzero((dates[:n].year >= YEARS[0]) & (dates[:n].year <= YEARS[1]))
    # float64 before the mean, as evaluate.py (domain_series = nanmean over the grid)
    return dates[keep], {v: np.nanmean(p[keep, saved.index(v)].astype(np.float64), axis=(1, 2))
                         for v in EVENT_VARIABLES}


def _num(x: float) -> float | None:
    return None if x is None or not math.isfinite(x) else float(x)


def factors(a: np.ndarray, b: np.ndarray, ta: float, tb: float, min_days: int) -> dict:
    la, lb = a < ta, b < tb
    both = la & lb
    edges = np.diff(np.concatenate(([False], both, [False])).astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    count, mean_len = spell_stats(both, min_days)
    n_both = int(both.sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        dep = both.mean() / (la.mean() * lb.mean())
    return dict(n_days=int(both.size), p_a=_num(la.mean()), p_b=_num(lb.mean()), n_both=n_both,
                dependence=_num(dep),
                spell_share=_num(lengths[lengths >= min_days].sum() / n_both) if n_both else None,
                events=count, mean_length=_num(mean_len) if count else None)


def main() -> None:
    wanted = [tuple(a.split(":")) for a in sys.argv[1:]] or PAIRS
    for reg, gcm in wanted:
        e_dates, e_ser = era5_series(reg)
        out = {"meta": None, "events": {ev: {"era5": None, "methods": {}} for ev in EVENT_DEFS}}
        arrays = {"era5|date": e_dates.strftime("%Y%m%d").astype(int).values,
                  **{f"era5|{v}": s for v, s in e_ser.items()}}
        for m in METHODS:
            pd_file = OUT / sub(reg, gcm) / m / "plot_data.json"
            if not (CROPS / sub(reg, gcm) / m / "predictions.pt").exists() or not pd_file.exists():
                print(f"  WARNING {reg} {gcm} {m}: no crop or no plot_data.json, skipped", flush=True)
                continue
            p_dates, p_ser = product_series(reg, gcm, m)
            common = e_dates.intersection(p_dates)
            ie, ip = e_dates.get_indexer(common), p_dates.get_indexer(common)
            series = {v: {"era5": e_ser[v][ie], "pred": p_ser[v][ip]} for v in EVENT_VARIABLES}
            ev = energy_events(series, period=YEARS)
            stored = json.loads(pd_file.read_text())["_events"]
            assert ev["meta"]["thresholds"] == stored["meta"]["thresholds"], \
                f"{reg} {gcm} {m}: thresholds {ev['meta']['thresholds']} != plot_data {stored['meta']['thresholds']}"
            for name in EVENT_DEFS:
                for src in ("era5", "pred"):
                    got = (ev[name]["count"][src], ev[name]["mean_duration"][src])
                    want = (stored[name]["count"][src], stored[name]["mean_duration"][src])
                    assert got == want, f"{reg} {gcm} {m} {name} {src}: {got} != plot_data {want}"
            thr, min_days = ev["meta"]["thresholds"], ev["meta"]["min_days"]
            if m == "bicubic":
                out["meta"] = {**ev["meta"], "events": {k: list(v) for k, v in EVENT_DEFS.items()}}
            for name, (va, vb) in EVENT_DEFS.items():
                f_e = factors(series[va]["era5"], series[vb]["era5"], thr[va], thr[vb], min_days)
                f_p = factors(series[va]["pred"], series[vb]["pred"], thr[va], thr[vb], min_days)
                # thresholds of this method's own count (ERA5 on its dates: SFC and CESM2
                # have no 29 February)
                out["events"][name]["methods"][m] = {"pred": f_p, "era5": f_e,
                                                     "thresholds": [thr[va], thr[vb]]}
                if m == "bicubic":
                    out["events"][name]["era5"] = f_e
            arrays[f"{m}|date"] = p_dates.strftime("%Y%m%d").astype(int).values
            arrays.update({f"{m}|{v}": s for v, s in p_ser.items()})
        d = OUT / sub(reg, gcm) / "bicubic" / "plots"
        d.mkdir(parents=True, exist_ok=True)
        # the app may read while this runs: never leave a half-written file under the final name
        tmp = d / "energy_event_factors.tmp.json"
        tmp.write_text(json.dumps(out, indent=1))
        os.replace(tmp, d / "energy_event_factors.json")
        tmp = d / "energy_event_series.tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, d / "energy_event_series.npz")
        print(f"{reg} x {gcm} done (counts match plot_data.json for {len(METHODS)} methods)", flush=True)


if __name__ == "__main__":
    main()
