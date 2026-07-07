#!/bin/bash
#SBATCH --job-name=download_era5
#SBATCH --time=02:00:00
#SBATCH --partition=archive
#SBATCH -A idl@cpu
#SBATCH --cpus-per-task=8
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --chdir=/lustre/fswork/projects/rech/idl/uzm26qx/code/archesweathersr

# Load the modules
module purge

module load pytorch-gpu

set -x

srun python scripts/dl_era.py --res 0.25 --compress --folder $SCRATCH/era5/1440x721/weatherbench2/yearly --years ${SLURM_ARRAY_TASK_ID}
