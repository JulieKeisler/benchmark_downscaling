#!/bin/bash
# JZ: backfill energy events for factorial couplings of one region.
#SBATCH --job-name=fbk_${R}
#SBATCH --partition=gpu_p13
#SBATCH --account=bqu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --output=/gpfswork/rech/bqu/uyv87oi/benchmark_downscaling/logs/backfill_{J}_%j.out
#SBATCH --error=/gpfswork/rech/bqu/uyv87oi/benchmark_downscaling/logs/backfill_{J}_%j.err
set -e
source /etc/profile
module purge
module load pytorch-gpu/py3/2.7.0
cd /gpfswork/rech/bqu/uyv87oi/benchmark_downscaling
python3 src/evaluation/backfill_energy.py --factorial --region {R}
echo "=== done {J} ==="