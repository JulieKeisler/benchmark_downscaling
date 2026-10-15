#!/bin/bash
#SBATCH --partition=cpu_devel
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# 2080-2100 crops of the cleps cells for the net-load lens on the change (results_v2_tam2100).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
S=/scratch/jkeisler/benchmark_downscaling
python $S/tools_tstar/crop_predictions.py --root $S/results_v2 --dst $S/results_v2_tam2100 \
    --start 2080 --end 2100 australia/bicubic australia/cdft australia/cesm2/bcsd australia/cesm2/bicubic australia/cesm2/cdft australia/cesm2/dotc australia/cesm2/eqm australia/cesm2/idownscale australia/dotc australia/eqm australia/idownscale australia/ipsl_cm6a_lr/bcsd australia/ipsl_cm6a_lr/bicubic australia/ipsl_cm6a_lr/cdft australia/ipsl_cm6a_lr/dotc australia/ipsl_cm6a_lr/eqm australia/ipsl_cm6a_lr/idownscale france/bicubic france/cdft france/cesm2/bcsd france/cesm2/bicubic france/cesm2/cdft france/cesm2/dotc france/cesm2/eqm france/cesm2/idownscale france/cnrm_cm6_1/bcsd france/cnrm_cm6_1/bicubic france/cnrm_cm6_1/cdft france/cnrm_cm6_1/dotc france/cnrm_cm6_1/eqm france/cnrm_cm6_1/idownscale france/dotc france/eqm france/idownscale us_central/cdft us_central/cnrm_cm6_1/bcsd us_central/cnrm_cm6_1/bicubic us_central/cnrm_cm6_1/cdft us_central/cnrm_cm6_1/dotc us_central/cnrm_cm6_1/eqm us_central/cnrm_cm6_1/idownscale us_central/dotc us_central/idownscale us_central/ipsl_cm6a_lr/bcsd us_central/ipsl_cm6a_lr/bicubic us_central/ipsl_cm6a_lr/cdft us_central/ipsl_cm6a_lr/dotc us_central/ipsl_cm6a_lr/eqm us_central/ipsl_cm6a_lr/idownscale
