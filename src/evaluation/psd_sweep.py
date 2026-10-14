#!/usr/bin/env python3
"""PSD sweep — calibrate the consistency model's stroke time t* from spectra.

Following Hess et al. (2025), t* is the noise level whose (white) power equals
the ERA5 power at the wavenumber k* beyond which the GCM input no longer
carries ERA5-like spatial structure; noising at t* drowns those scales and
keeps the larger ones.

Both fields are encoded with the checkpoint's normalisation (min-max to
[-1, 1], log for pr), the space in which the CM adds its noise:
  * GCM: bicubic-regridded to the ERA5 grid (calibration period);
  * ERA5: same period.
The GCM is compared before QDM: the per-cell quantile map that inference_cm.py
applies injects grid-scale noise (on Australian test data the QDM-corrected
psl has 55x the ERA5 power at the grid scale), which hides the crossing. The
crossing of the bicubic GCM is its effective resolution (about 17 grid
points, 450 km, for CNRM-CM6-1 tas/psl/sfcWind/rsds over Australia).

PSD convention: P(k) = |FFT(w x)|^2 / (H W mean(w^2)) with a 2-D Hann taper w
(the domain is not periodic), radially averaged, the k = 0 mode excluded. White noise of standard deviation s then has P(k) = s^2 at every k,
so the noise matches the ERA5 power at k* when t* = sqrt(P_ERA5(k*)).

k*: the last crossing of the two spectra — the smallest k beyond which the
GCM power stays below the ERA5 power. If the GCM is below ERA5 at every
resolved k, k* is set to the lowest k and flagged (no crossing).

One t* per variable is written; the CM adds one noise level to all channels,
so the region's t* is the median over the variables with a proper crossing.
A crossing is degenerate when the GCM is below ERA5 at every k (k* at the
lowest bin: t* would erase the GCM) or never drops below it (k* at the
Nyquist bin, e.g. pr in log space); those variables are flagged and left out.
As the CM actually sees the QDM-corrected field, the JSON also stores, per
variable, max over k >= k* of P_QDM(k) / t*^2: values above 1 mean QDM grid-
scale artefacts survive the noise.

Usage:
    python src/evaluation/psd_sweep.py --era5-dir ... --gcm-dir ... \
        --checkpoint-dir checkpoints/consistency_rsds/france \
        --variables tas pr psl sfcWind sfcWindmax rsds \
        --calib-start 1980 --calib-end 1999 --out-json results/psd/france.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "baselines"))


def radial_psd(fields: np.ndarray, n_bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Time-mean radially averaged PSD of [T, H, W] fields (k in cycles/grid point).

    P = |FFT(w x)|^2 / (H W mean(w^2)) with a Hann taper w, so white noise of
    std s has P = s^2; the k = 0 mode (domain mean) is excluded.
    """
    T, H, W = fields.shape
    # Hann taper: the domain is not periodic, and the edge jump would otherwise
    # leak power to every k and hide the fine-scale deficit. Dividing by the
    # taper's mean square keeps white noise of std s at P = s^2.
    taper = np.outer(np.hanning(H), np.hanning(W))
    norm = H * W * float(np.mean(taper ** 2))
    power = np.zeros((H, W))
    for t in range(T):
        f = (fields[t] - fields[t].mean()) * taper
        power += np.abs(np.fft.fft2(f)) ** 2 / norm
    power /= T
    KX, KY = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(H))
    K = np.sqrt(KX ** 2 + KY ** 2)
    k_min = 1.0 / max(H, W)
    edges = np.logspace(np.log10(k_min), np.log10(0.5), n_bins + 1)   # resolves low k
    idx = np.digitize(K.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < n_bins) & (K.ravel() > 0)
    sums = np.bincount(idx[ok], weights=power.ravel()[ok], minlength=n_bins)
    counts = np.bincount(idx[ok], minlength=n_bins)
    keep = counts > 0
    return 0.5 * (edges[:-1] + edges[1:])[keep], (sums[keep] / counts[keep])


def last_crossing(k: np.ndarray, p_ref: np.ndarray, p_gcm: np.ndarray) -> tuple[float, bool]:
    """Smallest k beyond which p_gcm < p_ref; (k*, found). Interpolated in log space."""
    above = p_gcm >= p_ref
    if not above.any():
        return float(k[0]), False
    i = int(np.flatnonzero(above)[-1])
    if i == len(k) - 1:
        return float(k[-1]), True
    d0 = np.log(p_gcm[i] / p_ref[i])
    d1 = np.log(p_gcm[i + 1] / p_ref[i + 1])
    return float(k[i] + d0 / (d0 - d1) * (k[i + 1] - k[i])), True


def t_star_at(k_star: float, k: np.ndarray, p_ref: np.ndarray) -> float:
    """sqrt of the ERA5 power at k*, interpolated in log-power."""
    return float(np.sqrt(np.exp(np.interp(k_star, k, np.log(p_ref)))))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir", required=True)
    p.add_argument("--gcm-dir", required=True)
    p.add_argument("--checkpoint-dir", required=True,
                   help="dir with the CM best.pt (its normalisation stats are used)")
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--calib-start", type=int, default=1980)
    p.add_argument("--calib-end", type=int, default=1999)
    p.add_argument("--n-bins", type=int, default=30)
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    import torch  # noqa: PLC0415
    from data_processing import load_era5, load_gcm, bicubic_regrid, align_time, qdm_correct  # noqa: PLC0415
    from utils.normalization import encode  # noqa: PLC0415

    import hashlib  # noqa: PLC0415
    ckpt = Path(args.checkpoint_dir) / "best.pt"
    stats = torch.load(ckpt, map_location="cpu", weights_only=False)["stats"]
    out = {"calib": [args.calib_start, args.calib_end],
           "psd_norm": "|FFT(hann*x)|^2/(H*W*mean(hann^2))",
           "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
           "variables": {}}
    for var in args.variables:
        da = align_time({"e": load_era5(Path(args.era5_dir), var, args.calib_start, args.calib_end),
                         "g": load_gcm(Path(args.gcm_dir), var, args.calib_start, args.calib_end,
                                       "historical")})
        era5 = da["e"]
        gcm_up = bicubic_regrid(da["g"], era5.lat.values, era5.lon.values)
        e = encode(era5.values.astype(np.float32), stats, var)
        g = encode(gcm_up.values.astype(np.float32), stats, var)
        k, pe = radial_psd(e, args.n_bins)
        _, pg = radial_psd(g, args.n_bins)
        k_star, found = last_crossing(k, pe, pg)
        degenerate = bool((not found) or k_star >= k[-1] or k_star <= k[0])
        t_v = t_star_at(k_star, k, pe)
        gq = encode(qdm_correct(gcm_up, gcm_up, era5, var).values.astype(np.float32), stats, var)
        _, pq = radial_psd(gq, args.n_bins)
        qdm_excess = float((pq[k >= k_star] / t_v ** 2).max()) if (k >= k_star).any() else float("nan")
        H, W = e.shape[1:]
        print(f"{var:10s} k*={k_star:.4f} cyc/pt (λ*≈{1 / k_star:.1f} pts)  t*={t_v:.4f}  "
              f"max P_QDM/t*²(k≥k*)={qdm_excess:.3g}{'  [DEGENERATE: excluded]' if degenerate else ''}",
              flush=True)
        out["variables"][var] = {"k_star": float(k_star), "t_star": float(t_v), "degenerate": degenerate,
                                 "qdm_excess": qdm_excess, "H": int(H), "W": int(W),
                                 "k": k.tolist(), "psd_era5": pe.tolist(),
                                 "psd_gcm": pg.tolist(), "psd_gcm_qdm": pq.tolist()}
    ts = [v["t_star"] for v in out["variables"].values() if not v["degenerate"]]
    if not ts:
        raise SystemExit("psd_sweep: no variable has a proper PSD crossing — no t*")
    out["t_star"] = float(np.median(ts))
    out["t_star_rule"] = "median over variables with a non-degenerate crossing"
    out["t_star_range"] = [float(min(ts)), float(max(ts))]
    print(f"t* (median over {len(ts)} variables) = {out['t_star']:.4f}  "
          f"[range {min(ts):.4f}-{max(ts):.4f}]", flush=True)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
