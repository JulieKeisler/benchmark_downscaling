#!/bin/bash
#SBATCH --partition=cpu_homogen
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# CDF-t (US r4 rerun; v1 code path, GCM from 2000 as in v1).
# Submit: sbatch --job-name=v2_cdft_$REGION --export=ALL,REGION=us_central cdft.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" cdft
check_inputs
python "$V2/src/baselines/inference_cdft.py" --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" \
    --out-dir "$OUT_DIR" --calib-start 1980 --calib-end 1999 --eval-start 2000 --eval-end 2100 \
    --scenario "$SCENARIO" --win-size 20 --step 10 --trim 5 --variables $VARS
evaluate cdft "$OUT_DIR" "$OUT_BASE"
