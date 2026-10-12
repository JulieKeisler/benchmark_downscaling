#!/bin/bash
#SBATCH --job-name=dotc_australia
#SBATCH --partition=cpu_homogen
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=$HOME/benchmark_downscaling/logs/dotc_australia_%j.out
#SBATCH --error=$HOME/benchmark_downscaling/logs/dotc_australia_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
cd $HOME/benchmark_downscaling
export PYTHONPATH=$HOME/benchmark_downscaling/src/evaluation:$HOME/benchmark_downscaling/src
mkdir -p $HOME/benchmark_downscaling/logs

echo "=== DOTC australia ==="
python src/baselines/inference_dotc.py \
    --era5-dir    data/era5/australia \
    --gcm-dir     data/cmip6/australia/cnrm_cm6_1 \
    --out-dir     results/australia/dotc \
    --calib-start 1980 \
    --calib-end   1999 \
    --eval-start  2000 \
    --eval-end    2100 \
    --scenario    ssp370 \
    --win-size    20 \
    --step        10 \
    --trim        5 \
    --variables   tas pr psl sfcWind sfcWindmax
echo "--- Evaluate ---"
python src/evaluation/evaluate.py \
    --era5-dir   data/era5/australia \
    --gcm-dir    data/cmip6/australia/cnrm_cm6_1 \
    --method     dotc \
    --pred-dir   results/australia/dotc \
    --output-dir results/australia \
    --variables  tas pr psl sfcWind sfcWindmax \
    --start      2000 \
    --end        2020 \
    --gcm-end    2100
