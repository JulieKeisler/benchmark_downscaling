#!/bin/bash
# CLEPS: backfill energy events for factorial couplings of one region.
#SBATCH --job-name=fbk_${R}
#SBATCH --partition=cpu_homogen
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/jkeisler/benchmark_downscaling/logs/backfill_{J}_%j.out
#SBATCH --error=/home/jkeisler/benchmark_downscaling/logs/backfill_{J}_%j.err
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
cd /home/jkeisler/benchmark_downscaling
python3 src/evaluation/backfill_energy.py --factorial --region {R}
echo "=== done {J} ==="