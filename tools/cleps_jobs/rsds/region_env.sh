#!/bin/bash
# Per-region data paths, scenario and variable set for the rsds (radiation) reruns on cleps.
# Usage (inside a job): source ~/benchmark_downscaling/tools/cleps_jobs/rsds/region_env.sh "$REGION" <method>
set -e
REGION="$1"
METHOD="$2"
BM=/home/jkeisler/benchmark_downscaling
cd "$BM"
export PYTHONPATH=$BM/src/evaluation:$BM/src

case "$REGION" in
  france)
    ERA5_DIR=/scratch/jkeisler/benchmark_downscaling/era5
    GCM_DIR=/scratch/jkeisler/benchmark_downscaling/cmip6/ipsl_cm6a_lr
    SCENARIO=ssp370
    VARS="tas pr psl sfcWind sfcWindmax rsds" ;;
  australia)
    ERA5_DIR=$BM/data/era5/australia
    GCM_DIR=$BM/data/cmip6/australia/cnrm_cm6_1
    SCENARIO=ssp370
    VARS="tas pr psl sfcWind sfcWindmax rsds" ;;
  us_central)
    # CESM2: no sfcWindmax at all; ssp245 because ssp370 has no daily rsds.
    ERA5_DIR=$BM/data/era5/us_central
    GCM_DIR=$BM/data/cmip6/us_central/cesm2
    SCENARIO=ssp245
    VARS="tas pr psl sfcWind rsds" ;;
  *) echo "unknown REGION=$REGION" >&2; exit 1 ;;
esac

# Results on /scratch: $HOME has a 100G hard quota (a 6-var France predictions.pt is ~3.3G).
OUT_BASE=/scratch/jkeisler/benchmark_downscaling/results_rsds/$REGION
OUT_DIR=$OUT_BASE/$METHOD
mkdir -p "$OUT_DIR" $BM/logs
echo "=== $METHOD $REGION | $SCENARIO | vars: $VARS | job $SLURM_JOB_ID ==="

evaluate() {
  python src/evaluation/evaluate.py \
      --era5-dir   "$ERA5_DIR" \
      --gcm-dir    "$GCM_DIR" \
      --scenario   "$SCENARIO" \
      --method     "$METHOD" \
      --pred-dir   "$OUT_DIR" \
      --output-dir "$OUT_BASE" \
      --variables  $VARS \
      --start      2000 \
      --end        2020 \
      --gcm-end    2100
}
