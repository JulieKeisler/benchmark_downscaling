#!/bin/bash
# v2 reruns on cleps. Code from ~/benchmark_downscaling_v2;
# data read-only from the shared dirs; outputs to /scratch/.../results_v2 only ($HOME quota is 100 G).
set -euo pipefail
REGION="$1"
METHOD="$2"
export V2=/home/jkeisler/benchmark_downscaling_v2
BM=/home/jkeisler/benchmark_downscaling
S=/scratch/jkeisler/benchmark_downscaling
# GCM pair: the region's native GCM unless GCM=<key> is exported (non-native pairs).
case "$REGION" in
  france)     NATIVE=ipsl_cm6a_lr; ERA5_DIR=$S/era5 ;;
  australia)  NATIVE=cnrm_cm6_1;   ERA5_DIR=$BM/data/era5_orig/australia ;;
  us_central) NATIVE=cesm2;        ERA5_DIR=$BM/data/era5_orig/us_central ;;
  *) echo "unknown REGION=$REGION" >&2; exit 1 ;;
esac
GCM="${GCM:-$NATIVE}"
F=$BM/data/cmip6/_factorial
case "$REGION/$GCM" in
  france/ipsl_cm6a_lr)  GCM_DIR=$S/cmip6/ipsl_cm6a_lr ;;
  australia/cnrm_cm6_1) GCM_DIR=$BM/data/cmip6/australia/cnrm_cm6_1 ;;
  us_central/cesm2)     GCM_DIR=$S/data_v2/cmip6/us_central/cesm2_r4 ;;          # r4, copied from JZ (md5)
  */cnrm_cm6_1)         GCM_DIR=$BM/data/cmip6/_v2_pairs/$REGION/cnrm_cm6_1 ;;    # derived sfcWind both periods
  */cesm2)              GCM_DIR=$F/cmip6_cesm2_r4/$REGION/cesm2 ;;
  */ipsl_cm6a_lr)       GCM_DIR=$F/cmip6_factorial/$REGION/ipsl_cm6a_lr ;;
  *) echo "unknown GCM=$GCM" >&2; exit 1 ;;
esac
if [ "$GCM" = cesm2 ]; then SCENARIO=ssp245; VARS="tas pr psl sfcWind rsds"
else SCENARIO=ssp370; VARS="tas pr psl sfcWind sfcWindmax rsds"; fi
# Derived products without a variant_label: CNRM sfcWind over Australia.
ALLOW_UNLABELLED=""; [ "$REGION/$GCM" = australia/cnrm_cm6_1 ] && ALLOW_UNLABELLED="sfcWind"
# CDS downloads without metadata (Julie, 2026-10-14): 9/12 bit-identical to the expected
# member on Pangeo; ssp370 rsds unverified (ESGF down). Re-check when ESGF answers.
case "$REGION/$GCM" in
  australia/ipsl_cm6a_lr|us_central/ipsl_cm6a_lr|us_central/cnrm_cm6_1) ALLOW_UNLABELLED="rsds sfcWindmax" ;;
esac
if [ "$GCM" = "$NATIVE" ]; then SUB=$REGION; else SUB=$REGION/$GCM; fi
OUT_BASE=$S/results_v2/$SUB
OLD_BASE=$S/results_rsds/$SUB     # v1 predictions (re-evaluation only)
OUT_DIR=$OUT_BASE/$METHOD
mkdir -p "$OUT_DIR" $S/logs_v2
export ERA5_DIR GCM_DIR SCENARIO VARS OUT_BASE OLD_BASE OUT_DIR GCM NATIVE SUB
export PYTHONPATH=$V2/src/evaluation:$V2/src
cd "$V2"
echo "=== v2 $METHOD $REGION x $GCM | $SCENARIO | $VARS | GCM $GCM_DIR | job ${SLURM_JOB_ID:-} | code $(cat $V2/CODE_VERSION 2>/dev/null) ==="

check_inputs() {
  python "$V2/tools/check_gcm_inputs.py" --gcm-dir "$GCM_DIR" --era5-dir "$ERA5_DIR" \
      --scenario "$SCENARIO" --variables $VARS --allow-unlabelled $ALLOW_UNLABELLED
}
evaluate() {   # evaluate <method-name> <pred-dir> <output-base>
  local name="$1" pred="$2" base="$3"
  python "$V2/src/evaluation/evaluate.py" --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" \
      --scenario "$SCENARIO" --method "$name" --pred-dir "$pred" --output-dir "$base" \
      --region "$REGION" --variables $VARS --start 2000 --end 2020 --gcm-end 2100
  python "$V2/tools/check_results.py" "$base/$name" --pred-dir "$pred" --variables $VARS
}
