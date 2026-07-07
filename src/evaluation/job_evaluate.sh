#!/bin/bash
#SBATCH --job-name=benchmark_eval
#SBATCH --partition=cpu_homogen
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/jkeisler/benchmark_results/bicubic/eval_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_results/bicubic/eval_%j.err

echo "=========================================="
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "=========================================="

mkdir -p /scratch/jkeisler/benchmark_results/bicubic

cd /scratch/jkeisler/benchmark_scripts

conda run -n sbck-env python scripts/evaluate.py \
    --config configs/benchmark.yaml \
    --era5-dir /scratch/jkeisler/era5 \
    --ipsl-dir /scratch/jkeisler/cmip6/ipsl_cm6a_lr \
    --output-dir /scratch/jkeisler/benchmark_results
