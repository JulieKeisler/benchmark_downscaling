#!/bin/bash
#SBATCH --partition=cpu_homogen
#SBATCH --account=inria
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# SBCK.dOTC jointly over variables. ~60 CPU-h per region (0.45 s per fit).
# Submit: sbatch --job-name=v2_dotc_$REGION --export=ALL,REGION=<region> dotc.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" dotc
check_inputs
python "$V2/src/baselines/inference_dotc.py" --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" \
    --out-dir "$OUT_DIR" --calib-start 1980 --calib-end 1999 --eval-start 2000 --eval-end 2100 \
    --scenario "$SCENARIO" --win-size 20 --step 10 --trim 5 --workers 64 --variables $VARS
evaluate dotc "$OUT_DIR" "$OUT_BASE"
