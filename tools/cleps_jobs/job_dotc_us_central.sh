#!/bin/bash
#SBATCH --job-name=dotc_us_central
#SBATCH --partition=cpu_homogen
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=$HOME/benchmark_downscaling/logs/dotc_us_central_%j.out
#SBATCH --error=$HOME/benchmark_downscaling/logs/dotc_us_central_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
cd $HOME/benchmark_downscaling
export PYTHONPATH=$HOME/benchmark_downscaling/src/evaluation:$HOME/benchmark_downscaling/src
mkdir -p $HOME/benchmark_downscaling/logs

echo "=== DOTC us_central ==="
python src/baselines/inference_dotc.py \
    --era5-dir    data/era5/us_central \
    --gcm-dir     data/cmip6/us_central/cesm2 \
    --out-dir     results/us_central/dotc \
    --calib-start 1980 \
    --calib-end   1999 \
    --eval-start  2000 \
    --eval-end    2100 \
    --scenario    ssp245 \
    --win-size    20 \
    --step        10 \
    --trim        5 \
    --variables   tas pr psl sfcWind
echo "--- Evaluate ---"
python src/evaluation/evaluate.py \
    --era5-dir   data/era5/us_central \
    --gcm-dir    data/cmip6/us_central/cesm2 \
    --scenario   ssp245 \
    --method     dotc \
    --pred-dir   results/us_central/dotc \
    --output-dir results/us_central \
    --variables  tas pr psl sfcWind \
    --start      2000 \
    --end        2020 \
    --gcm-end    2100
