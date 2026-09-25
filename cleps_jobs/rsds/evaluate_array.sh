#!/bin/bash
#SBATCH --job-name=rsds_eval
#SBATCH --partition=cpu_devel
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/jkeisler/benchmark_downscaling/logs/%x_%A_%a.out
#SBATCH --error=/home/jkeisler/benchmark_downscaling/logs/%x_%A_%a.err
# Re-evaluate existing predictions only (e.g. after adding metrics / variable pairs).
# Task id -> (region, method): 3 methods x 3 regions.
METHODS=(cdft dotc idownscale)
REGIONS=(france australia us_central)
METHOD=${METHODS[$((SLURM_ARRAY_TASK_ID / 3))]}
REGION=${REGIONS[$((SLURM_ARRAY_TASK_ID % 3))]}
source ~/miniconda3/etc/profile.d/conda.sh
conda activate py311
source /home/jkeisler/benchmark_downscaling/cleps_jobs/rsds/region_env.sh "$REGION" "$METHOD"
evaluate
