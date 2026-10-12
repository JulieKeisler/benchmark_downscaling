#!/bin/bash
# Backfill doy + energy events for factual couplings on the clusters.
# Usage (on-cluster): backfill_energy_cluster.sh <cluster> <region>
#   cluster = jz | cleps ; region = france | australia | us_central | all
# Runs backfill_energy.py --factorial, which only touches dirs that have
# metrics.json + predictions.pt (methods present on that cluster).
set -e

CLUSTER="$1"; REGION="$2"
SRC=/home/jkeisler/Documents/Code/benchmark_downscaling/src/evaluation/backfill_energy.py

if [ "$CLUSTER" = "jz" ]; then
    cd /gpfswork/rech/bqu/uyv87oi/benchmark_downscaling
    source /etc/profile; module purge; module load pytorch-gpu/py3/2.7.0
    REGION_ARG="--region $REGION"
    if [ "$REGION" = "all" ]; then REGION_ARG=""; fi
    python3 src/evaluation/backfill_energy.py --factorial $REGION_ARG
elif [ "$CLUSTER" = "cleps" ]; then
    cd /home/jkeisler/benchmark_downscaling
    source ~/miniconda3/etc/profile.d/conda.sh; conda activate sbck-gpu
    REGION_ARG="--region $REGION"
    if [ "$REGION" = "all" ]; then REGION_ARG=""; fi
    python3 src/evaluation/backfill_energy.py --factorial $REGION_ARG
else
    echo "unknown cluster $CLUSTER" >&2; exit 2
fi
echo "=== done $CLUSTER $REGION ==="