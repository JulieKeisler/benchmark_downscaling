#!/bin/bash
#SBATCH --job-name=idownscale_australia
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=$HOME/benchmark_downscaling/logs/idownscale_australia_%j.out
#SBATCH --error=$HOME/benchmark_downscaling/logs/idownscale_australia_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
cd $HOME/benchmark_downscaling
export PYTHONPATH=$HOME/benchmark_downscaling/src/evaluation:$HOME/benchmark_downscaling/src
mkdir -p $HOME/benchmark_downscaling/logs

echo "=== iDownscale australia (GPU + SBCK on cleps, calib 1980-1999, eval 2000-2100, ssp370) ==="
pip install --quiet diffusers torchdiffeq netCDF4 sbck 2>/dev/null || true

python src/baselines/inference_idownscale.py \
    --era5-dir    data/era5/australia \
    --gcm-dir     data/cmip6/australia/cnrm_cm6_1 \
    --out-dir     results/australia/idownscale \
    --calib-start 1980 \
    --calib-end   1999 \
    --eval-start  2000 \
    --eval-end    2100 \
    --scenario    ssp370 \
    --epochs      30 \
    --batch-size  16 \
    --variables   tas pr psl sfcWind sfcWindmax

echo "--- Evaluate ---"
python src/evaluation/evaluate.py \
    --era5-dir   data/era5/australia \
    --gcm-dir    data/cmip6/australia/cnrm_cm6_1 \
    --method     idownscale \
    --pred-dir   results/australia/idownscale \
    --output-dir results/australia \
    --variables  tas pr psl sfcWind sfcWindmax \
    --start      2000 \
    --end        2020 \
    --gcm-end    2100

echo "Done — results in results/australia/idownscale"
