#!/bin/bash
#SBATCH --partition=cpu_devel
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# Bicubic; CPU only. Submit: sbatch --job-name=v2_bicubic_<tag> --export=ALL,REGION=<region>,GCM=<gcm> bicubic.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" bicubic
check_inputs
python "$V2/src/baselines/inference_bicubic.py" --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" \
    --out-dir "$OUT_DIR" --eval-start 2000 --eval-end 2100 --scenario "$SCENARIO" --variables $VARS
evaluate bicubic "$OUT_DIR" "$OUT_BASE"
