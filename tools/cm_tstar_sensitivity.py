#!/usr/bin/env python3
"""CM t* sensitivity: warming kept and fine-scale realism for several noise levels.

For each t* (the sweep's min / median / max over variables, and the hand-set
1.0 of v1), the consistency model is run on 2000-2001 and 2099-2100 only
(seed 0). The input is the benchmark's: windowed QDM with the same 20-year
windows as the full 2000-2100 run, only the kept years cut to the short
periods. It is prepared once per period and denoised at each t*, with the
RNG seeded as in a full run, so the runs differ only by t*.

Reported per t*:
  tau_tas  = domain-mean tas change (2099-2100 minus 2000-2001) of the CM
             output over the same change of the bicubic GCM; 1 = all kept;
  tau_qdm  = the same for the pre-corrected input alone (before the CM), so
             the loss can be split between the pre-correction and the noise;
  spectral_fine (tas, pr, sfcWind, rsds) of the 2000-2001 output vs ERA5.
Check: with --reference-pred-dir (the benchmark's seed-0 CM run at the median
t*), the 2000-2001 output at the median t* must reproduce it.

Usage (JZ job): python tools/cm_tstar_sensitivity.py --region france \
    --era5-dir ... --gcm-dir ... --scenario ssp370 --variables ... \
    --checkpoint-dir ... --t-json results_v2/france/psd/t_star.json --out-json ... \
    [--reference-pred-dir results_v2/france/cm]
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
EVAL = (2000, 2100)


def short_windows(a: int, b: int, full: list[tuple[int, int, int, int]]):
    """The full run's windows that correct years a..b, with their keep ranges cut to a..b."""
    return [(ws, min(we, EVAL[1]), max(ks, a), min(ke, b))
            for ws, we, ks, ke in full if ks <= b and ke >= a]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir", required=True)
    p.add_argument("--gcm-dir", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--t-json", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--reference-pred-dir", default=None)
    args = p.parse_args()

    import torch  # noqa: PLC0415
    import inference_cm as cm  # noqa: PLC0415
    from inference_cdft import _sliding_windows  # noqa: PLC0415
    from data_processing import load_era5, load_gcm, bicubic_regrid, load_predictions  # noqa: PLC0415
    from utils import decode  # noqa: PLC0415
    from metrics import fine_spectrum_metric  # noqa: PLC0415

    cm.VARIABLES = list(args.variables)
    cm.N_VARS = len(cm.VARIABLES)
    sweep = json.loads(Path(args.t_json).read_text())
    lo, hi = sweep["t_star_range"]
    t_values = {"min": lo, "median": sweep["t_star"], "max": hi, "v1_hand_set": 1.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint_dir) / "best.pt"
    _, stats = cm._load_model(ckpt)
    full = _sliding_windows(*EVAL, 20, 10, 5)

    era5_dir, gcm_dir = Path(args.era5_dir), Path(args.gcm_dir)
    lat = lon = None
    inputs, gcm_mean, qdm_mean = {}, {}, {}
    i_tas = cm.VARIABLES.index("tas")
    for w, (a, b) in WINDOWS.items():
        win = short_windows(a, b, full)
        x, t, lat, lon = cm.prepare_inputs(era5_dir, gcm_dir, stats, win, max(v[1] for v in win),
                                           1980, args.scenario)
        inputs[w] = (x, t)
        g = bicubic_regrid(load_gcm(gcm_dir, "tas", a, b, args.scenario), lat, lon)
        # float64 means: float32 reductions of ~1e6 values drift by kelvins on JZ
        gcm_mean[w] = float(g.values.astype(np.float64).mean())
        qdm_mean[w] = float(decode(x[:, i_tas], stats, "tas").astype(np.float64).mean())
        print(f"  window {w} {a}-{b}: {t.size} days ({str(t.values[0])[:10]}..{str(t.values[-1])[:10]}), "
              f"QDM windows {win}, GCM mean {gcm_mean[w]:.3f} K, QDM mean {qdm_mean[w]:.3f} K",
              flush=True)
        assert t.size == g.sizes["time"], (t.size, g.sizes["time"])
    d_gcm = gcm_mean["late"] - gcm_mean["early"]
    out = {"gcm_dtas": d_gcm, "tau_qdm": (qdm_mean["late"] - qdm_mean["early"]) / d_gcm,
           "pre_correction": "qdm_windowed (benchmark windows)", "runs": {}}
    print(f"GCM dtas {d_gcm:+.3f} K; tau of the QDM input alone {out['tau_qdm']:.3f}", flush=True)

    era5_early = {v: load_era5(era5_dir, v, *WINDOWS["early"]) for v in ("tas", "pr", "sfcWind", "rsds")
                  if v in args.variables}
    # Compute-node /tmp is small: keep the short runs next to the output, one at a time.
    with tempfile.TemporaryDirectory(dir=Path(args.out_json).parent) as tmp:
        for name, t_star in t_values.items():
            means, fine = {}, {}
            for w in WINDOWS:
                d = Path(tmp) / f"{name}_{w}"
                torch.manual_seed(0)   # as cm.run: seed, then build the model, then sample
                np.random.seed(0)
                model, _ = cm._load_model(ckpt)
                x, t = inputs[w]
                cm.denoise(model, stats, x, t, lat, lon, d, device, float(t_star), 1, 32, 0)
                pred = {v: xx.load() for v, xx in load_predictions(d, args.variables).items()}
                shutil.rmtree(d)
                means[w] = float(pred["tas"].values.astype(np.float64).mean())
                if w == "early":
                    for v, e in era5_early.items():
                        pv = pred[v].assign_coords(time=pred[v].time.values)
                        common = np.intersect1d(e.time.values, pv.time.values)
                        fine[v] = fine_spectrum_metric(e.sel(time=common), pv.sel(time=common),
                                                       variable=v)[0]
                    if name == "median" and args.reference_pred_dir:
                        ref = load_predictions(Path(args.reference_pred_dir), args.variables)
                        diff = {v: float(np.abs(pred[v].values - ref[v].values[:pred[v].sizes["time"]]).max())
                                for v in args.variables}
                        out["check_vs_benchmark_run"] = diff
                        print("  check vs benchmark CM run, max|diff| 2000-2001: "
                              + ", ".join(f"{k} {v:.2e}" for k, v in diff.items()), flush=True)
            tau = (means["late"] - means["early"]) / d_gcm
            out["runs"][name] = {"t_star": float(t_star), "tau_tas": tau, "spectral_fine": fine}
            print(f"{name:12s} t*={t_star:.4f}  tau_tas={tau:.3f}  spectral_fine="
                  + ", ".join(f"{k} {v:.3f}" for k, v in fine.items()), flush=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
