#!/bin/bash
# Net-load lens on the change (2000-2020 -> 2080-2100, against the bicubic GCM), 9 pairs, local.
# Inputs: results/v2_tam (2000-2022 crops), results/v2_tam2100 (2080-2100 crops, tools/crop_predictions.py),
# results/tam_v2 (the published lens, for the same-TAM check).
# Output: results/tam_v2_change/metrics_change_<region>[_<gcm>].json
# Usage: [OUT=<dir>] [FUTURE=<root>] bash tools/run_tam_change.sh [region:gcm ...]
# (a FUTURE root holding only some methods' crops scores only those methods; see
# tools/merge_change_lens.py to fold such a partial run into the full one)
set -euo pipefail
BM=/home/jkeisler/Documents/Code/benchmark_downscaling
TAM=/home/jkeisler/Documents/Code/TAM_downscaling
OUT=${OUT:-$BM/results/tam_v2_change}
FUTURE=${FUTURE:-$BM/results/v2_tam2100}
mkdir -p "$OUT/logs"
declare -A NATIVE=([france]=ipsl_cm6a_lr [australia]=cnrm_cm6_1 [us_central]=cesm2)
PAIRS=("$@")
[ ${#PAIRS[@]} -gt 0 ] || PAIRS=(france:ipsl_cm6a_lr france:cnrm_cm6_1 france:cesm2
  australia:cnrm_cm6_1 australia:ipsl_cm6a_lr australia:cesm2
  us_central:cesm2 us_central:ipsl_cm6a_lr us_central:cnrm_cm6_1)
cd "$TAM"
for p in "${PAIRS[@]}"; do
  r=${p%%:*}; g=${p##*:}
  if [ "$g" = "${NATIVE[$r]}" ]; then sfx=$r; else sfx=${r}_$g; fi
  echo "=== $r x $g ($(date +%H:%M))"
  .venv/bin/python net_lens_change.py --region "$r" --gcm "$g" \
    --past-root "$BM/results/v2_tam" --future-root "$FUTURE" \
    --gcm-dir "$BM/data/cmip6/_v2_inputs/$r/$g" \
    --lens-json "$BM/results/tam_v2/metrics_results_$sfx.json" --out-dir "$OUT" \
    > "$OUT/logs/${r}_${g}.log" 2>&1
  grep -E "change W1" "$OUT/logs/${r}_${g}.log" | sed 's/^/  /'
done
