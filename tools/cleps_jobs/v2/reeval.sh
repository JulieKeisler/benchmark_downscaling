#!/bin/bash
#SBATCH --partition=cpu_devel
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# Re-evaluate unchanged v1 predictions (France/Australia: same GCM files) with the v2 metrics.
# Submit: sbatch --job-name=v2_reeval_${METHOD}_$REGION --export=ALL,REGION=<france|australia>,METHOD=<cdft|idownscale> reeval.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" $METHOD
[ "$REGION" != us_central ] || { echo "US v1 predictions are r1+r4 spliced: rerun" >&2; exit 1; }
evaluate "$METHOD" "$OLD_BASE/$METHOD" "$OUT_BASE"
