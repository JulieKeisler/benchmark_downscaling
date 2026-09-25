#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/home/jkeisler/benchmark_downscaling/logs/%x_%j.out
#SBATCH --error=/home/jkeisler/benchmark_downscaling/logs/%x_%j.err
# rsds rerun. Submit: sbatch --job-name=idownscale_<region> --export=ALL,REGION=<france|australia|us_central> idownscale.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling/cleps_jobs/rsds/region_env.sh "$REGION" idownscale
# Fresh OUT_DIR required: inference_idownscale.py reuses any checkpoints/idownscale.pt it finds.
test ! -e "$OUT_DIR/checkpoints/idownscale.pt" || { echo "stale checkpoint in $OUT_DIR" >&2; exit 1; }
python src/baselines/inference_idownscale.py \
    --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" --out-dir "$OUT_DIR" \
    --calib-start 1980 --calib-end 1999 --eval-start 2000 --eval-end 2100 \
    --scenario "$SCENARIO" --epochs 30 --batch-size 16 --variables $VARS
evaluate
