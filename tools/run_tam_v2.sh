#!/bin/bash
# Net-load (TAM) lens on the v2 predictions, all 9 GCM x region pairs, run locally.
# Inputs: results/v2_tam (predictions cut to 2000-2022 by tools/crop_predictions.py),
# data/cmip6/_v2_inputs/<region>/<gcm> (the GCM files the v2 runs read).
# Output: results/tam_v2/{metrics_results,cdf_results,tam_forecasts_results}_<region>[_<gcm>].*
# Usage: bash tools/run_tam_v2.sh [region:gcm ...]
set -euo pipefail
BM=/home/jkeisler/Documents/Code/benchmark_downscaling
TAM=/home/jkeisler/Documents/Code/TAM_downscaling
OUT=$BM/results/tam_v2
mkdir -p "$OUT/logs"
PAIRS=("$@")
[ ${#PAIRS[@]} -gt 0 ] || PAIRS=(france:ipsl_cm6a_lr france:cnrm_cm6_1 france:cesm2
  australia:cnrm_cm6_1 australia:ipsl_cm6a_lr australia:cesm2
  us_central:cesm2 us_central:ipsl_cm6a_lr us_central:cnrm_cm6_1)
cd "$TAM"
for p in "${PAIRS[@]}"; do
  r=${p%%:*}; g=${p##*:}
  echo "=== $r x $g ($(date +%H:%M))"
  .venv/bin/python net_lens_region.py --region "$r" --gcm "$g" \
    --results-root "$BM/results/v2_tam" --gcm-dir "$BM/data/cmip6/_v2_inputs/$r/$g" \
    --out-dir "$OUT" > "$OUT/logs/${r}_${g}.log" 2>&1
  grep -E "Methods with local predictions|KS=" "$OUT/logs/${r}_${g}.log" | sed 's/^/  /'
done
