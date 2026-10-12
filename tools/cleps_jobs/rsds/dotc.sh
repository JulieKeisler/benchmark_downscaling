#!/bin/bash
#SBATCH --partition=cpu_homogen
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/jkeisler/benchmark_downscaling/logs/%x_%j.out
#SBATCH --error=/home/jkeisler/benchmark_downscaling/logs/%x_%j.err
# rsds rerun. Submit: sbatch --job-name=dotc_<region> --export=ALL,REGION=<france|australia|us_central> dotc.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
source /home/jkeisler/benchmark_downscaling/tools/cleps_jobs/rsds/region_env.sh "$REGION" dotc
python src/baselines/inference_dotc.py \
    --era5-dir "$ERA5_DIR" --gcm-dir "$GCM_DIR" --out-dir "$OUT_DIR" \
    --calib-start 1980 --calib-end 1999 --eval-start 2000 --eval-end 2100 \
    --scenario "$SCENARIO" --win-size 20 --step 10 --trim 5 --variables $VARS
evaluate
