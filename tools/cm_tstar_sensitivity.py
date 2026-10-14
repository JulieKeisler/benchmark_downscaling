#!/usr/bin/env python3
"""CM t* sensitivity: warming kept and fine-scale realism for several noise levels.

For each t* (the sweep's min / median / max over variables, and the hand-set
1.0 of v1), the consistency model is run on 2000-2001 and 2099-2100 only
(seed 0). The QDM pre-correction works day by day (each value's quantile in
the 1980-1999 GCM distribution), so these short runs see exactly the input of
the full 2000-2100 run.

Reported per t*:
  tau_tas  = domain-mean tas change (2099-2100 minus 2000-2001) of the CM
             output over the same change of the bicubic GCM; 1 = all kept;
  tau_qdm  = the same for the pre-corrected input alone (before the CM), so
             the loss can be split between the pre-correction and the noise;
  spectral_fine (tas, pr, sfcWind, rsds) of the 2000-2001 output vs ERA5.

Usage (JZ job): python tools/cm_tstar_sensitivity.py --region france \
    --era5-dir ... --gcm-dir ... --scenario ssp370 --variables ... \
    --checkpoint-dir ... --t-json results_v2/france/psd/t_star.json --out-json ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "baselines"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "src"))

WINDOWS = {"early": (2000, 2001), "late": (2099, 2100)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir", required=True)
    p.add_argument("--gcm-dir", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--t-json", required=True)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    import inference_cm as cm  # noqa: PLC0415
    from data_processing import (  # noqa: PLC0415
        load_era5, load_gcm, bicubic_regrid, align_time, qdm_correct, load_predictions,
    )
    from metrics import fine_spectrum_metric  # noqa: PLC0415

    cm.VARIABLES = list(args.variables)
    cm.N_VARS = len(cm.VARIABLES)
    sweep = json.loads(Path(args.t_json).read_text())
    lo, hi = sweep["t_star_range"]
    t_values = {"min": lo, "median": sweep["t_star"], "max": hi, "v1_hand_set": 1.0}

    era5_dir, gcm_dir = Path(args.era5_dir), Path(args.gcm_dir)
    e_cal = load_era5(era5_dir, "tas", 1980, 1999)
    g_cal = load_gcm(gcm_dir, "tas", 1980, 1999, "historical")
    cal = align_time({"e": e_cal, "g": g_cal})
    lat, lon = cal["e"].lat.values, cal["e"].lon.values
    g_cal_up = bicubic_regrid(cal["g"], lat, lon)
    gcm_mean, qdm_mean = {}, {}
    for w, (a, b) in WINDOWS.items():
        g = bicubic_regrid(load_gcm(gcm_dir, "tas", a, b, args.scenario), lat, lon)
        # float64 means: float32 reductions of ~1e6 values drift by kelvins on JZ
        gcm_mean[w] = float(g.values.astype(np.float64).mean())
        qdm_mean[w] = float(qdm_correct(g, g_cal_up, cal["e"], "tas").values.astype(np.float64).mean())
        print(f"  window {w} {a}-{b}: {g.sizes['time']} days, {str(g.time.values[0])[:10]}.."
              f"{str(g.time.values[-1])[:10]}, GCM mean {gcm_mean[w]:.3f} K, QDM mean {qdm_mean[w]:.3f} K",
              flush=True)
    d_gcm = gcm_mean["late"] - gcm_mean["early"]
    out = {"gcm_dtas": d_gcm, "tau_qdm": (qdm_mean["late"] - qdm_mean["early"]) / d_gcm,
           "runs": {}}
    print(f"GCM dtas {d_gcm:+.3f} K; tau of the QDM input alone {out['tau_qdm']:.3f}", flush=True)

    era5_early = {v: load_era5(era5_dir, v, *WINDOWS["early"]) for v in ("tas", "pr", "sfcWind", "rsds")
                  if v in args.variables}
    # Compute-node /tmp is small: keep the short runs next to the output, one at a time.
    with tempfile.TemporaryDirectory(dir=Path(args.out_json).parent) as tmp:
        for name, t in t_values.items():
            means, fine = {}, {}
            for w, (a, b) in WINDOWS.items():
                d = Path(tmp) / f"{name}_{w}"
                cm.run(era5_dir, gcm_dir, Path(args.checkpoint_dir), d, calib_start=1980,
                       eval_start=a, eval_end=b, scenario=args.scenario, stroke_time=float(t),
                       seed=0)
                pred = {v: x.load() for v, x in load_predictions(d, args.variables).items()}
                shutil.rmtree(d)
                means[w] = float(pred["tas"].values.astype(np.float64).mean())
                if w == "early":
                    for v, e in era5_early.items():
                        pv = pred[v].assign_coords(time=pred[v].time.values)
                        common = np.intersect1d(e.time.values, pv.time.values)
                        fine[v] = fine_spectrum_metric(e.sel(time=common), pv.sel(time=common),
                                                       variable=v)[0]
            tau = (means["late"] - means["early"]) / d_gcm
            out["runs"][name] = {"t_star": float(t), "tau_tas": tau, "spectral_fine": fine}
            print(f"{name:12s} t*={t:.4f}  tau_tas={tau:.3f}  spectral_fine="
                  + ", ".join(f"{k} {v:.3f}" for k, v in fine.items()), flush=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
