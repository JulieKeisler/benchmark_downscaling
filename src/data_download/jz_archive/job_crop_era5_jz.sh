#!/bin/bash
#SBATCH --job-name=crop_era5
#SBATCH --partition=compil
#SBATCH -A bqu@v100
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/bqu/uyv87oi/crop_era5_%j.out
#SBATCH --error=/lustre/fswork/projects/rech/bqu/uyv87oi/crop_era5_%j.err

source /etc/profile        # makes `module` available in SLURM
module purge
module load pytorch-gpu   # provides h5py, numpy, pandas, xarray, hdf5plugin

set -x

ERA5_OUT=/lustre/fswork/projects/rech/bqu/uyv87oi/era5_france

python /lustre/fswork/projects/rech/bqu/uyv87oi/crop_era5_jz.py \
    --output-dir ${ERA5_OUT} \
    --variables tas psl pr sfcWind sfcWindmax
