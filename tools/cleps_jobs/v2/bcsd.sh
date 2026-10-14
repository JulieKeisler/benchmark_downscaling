#!/bin/bash
#SBATCH --partition=cpu_devel
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# MODE=bcsd (Wood 2004) or eqm (--no-detrend); CPU only. Same code as jz/jobs/v2/bcsd.sh.
# Submit: sbatch --job-name=v2_${MODE}_<tag> --export=ALL,REGION=<region>,GCM=<gcm>,MODE=<bcsd|eqm> bcsd.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" $MODE
check_inputs
FLAG=""; [ "$MODE" = eqm ] && FLAG="--no-detrend"
python "$V2/src/baselines/inference_bcsd.py" --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" \
    --out-dir "$OUT_DIR" --calib-start 1980 --calib-end 1999 --eval-start 2000 --eval-end 2100 \
    --scenario "$SCENARIO" --win-size 20 --step 10 --trim 5 --variables $VARS $FLAG
evaluate "$MODE" "$OUT_DIR" "$OUT_BASE"
