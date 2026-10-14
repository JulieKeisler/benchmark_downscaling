#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=arches
#SBATCH --exclude=gpu004
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# gpu004 excluded: GTX 1080 Ti, no kernels in the env's torch build.
# iDownscale (US r4 rerun) in a fresh dir: inference_idownscale.py reuses any checkpoint it finds.
# Submit: sbatch --job-name=v2_idownscale_$REGION --export=ALL,REGION=us_central idownscale.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" idownscale
check_inputs
test ! -e "$OUT_DIR/checkpoints/idownscale.pt" || { echo "stale checkpoint in $OUT_DIR" >&2; exit 1; }
python "$V2/src/baselines/inference_idownscale.py" --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" \
    --out-dir "$OUT_DIR" --calib-start 1980 --calib-end 1999 --eval-start 2000 --eval-end 2100 \
    --scenario "$SCENARIO" --epochs 30 --batch-size 16 --variables $VARS
evaluate idownscale "$OUT_DIR" "$OUT_BASE"
