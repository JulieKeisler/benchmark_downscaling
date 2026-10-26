#!/bin/bash
# Net-load lens counterfactual for France x CNRM-CM6-1: CDF-t and iDownscale with tas shifted
# by minus the GCM-ERA5 warming difference from 1980-1999 to 2000-2020
# (paper/data/calib_eval_shift.csv), i.e. as if the GCM had warmed as ERA5 did.
# Inputs: results/v2_tam (tools/crop_predictions.py). Output: results/tam_v2_shift/.
# Usage: bash tools/run_tam_shift.sh
set -euo pipefail
BM=/home/jkeisler/Documents/Code/benchmark_downscaling
TAM=/home/jkeisler/Documents/Code/TAM_downscaling
OUT=$BM/results/tam_v2_shift
ROOT=$OUT/predictions
mkdir -p "$OUT"
python3 - "$BM" "$ROOT" <<'PY'
import csv, shutil, sys
from pathlib import Path
import numpy as np, torch
BM, OUT = Path(sys.argv[1]), Path(sys.argv[2])
row = next(r for r in csv.DictReader(open(BM / "paper/data/calib_eval_shift.csv"))
           if (r["region"], r["gcm"], r["variable"]) == ("france", "cnrm_cm6_1", "tas"))
shift = -float(row["gcm_minus_era5"])
print(f"tas shift: {shift:+.4f} K")
for m in ("cdft", "idownscale"):
    src, dst = BM / "results/v2_tam/france/cnrm_cm6_1" / m, OUT / "france/cnrm_cm6_1" / m
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("eval_dates.npy", "lat.npy", "lon.npy", "metrics.json", "variables.npy"):
        shutil.copy(src / f, dst / f)
    v = list(np.load(src / "variables.npy", allow_pickle=True))
    p = torch.load(src / "predictions.pt", map_location="cpu", weights_only=True)
    p[:, v.index("tas")] += shift
    torch.save(p, dst / "predictions.pt")
(OUT.parent / "tas_shift_K.txt").write_text(f"{shift}\n")
PY
cd "$TAM"
.venv/bin/python net_lens_region.py --region france --gcm cnrm_cm6_1 --results-root "$ROOT" \
  --gcm-dir "$BM/data/cmip6/_v2_inputs/france/cnrm_cm6_1" --out-dir "$OUT" --freeze-trend > "$OUT/lens.log" 2>&1
grep -E "KS=" "$OUT/lens.log"
