#!/bin/bash
# Sync updated scripts to Cleps and re-run all evaluations.
# Run from your LOCAL machine:
#   bash scripts/submit_eval_all.sh

set -e
CLEPS=cleps
BM=/scratch/jkeisler/downscaling_benchmark

echo "=== Syncing scripts to Cleps ==="
rsync -az scripts/ "$CLEPS:$BM/scripts/"
rsync -az configs/ "$CLEPS:$BM/configs/"

echo "=== Submitting eval jobs ==="
for method in bicubic bcsd psm cm; do
    ssh "$CLEPS" "sbatch $BM/jobs/eval_${method}.sh"
done
ssh "$CLEPS" "squeue -u jkeisler"
