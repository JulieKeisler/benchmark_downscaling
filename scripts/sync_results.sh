#!/bin/bash
# Sync evaluation results from Cleps to local.
# Run after all eval jobs complete:
#   bash scripts/sync_results.sh

set -e
CLEPS=cleps
REMOTE=/scratch/jkeisler/downscaling_benchmark/results
LOCAL=results

for method in bicubic bcsd gnn consistency serpentflow; do
    echo "=== Syncing $method ==="
    mkdir -p "$LOCAL/$method"
    scp "$CLEPS:$REMOTE/$method/metrics.json"   "$LOCAL/$method/" 2>/dev/null || echo "  metrics.json not found for $method"
    scp "$CLEPS:$REMOTE/$method/plot_data.json" "$LOCAL/$method/" 2>/dev/null || echo "  plot_data.json not found for $method"
    rsync -az --progress \
        "$CLEPS:$REMOTE/$method/plots/" \
        "$LOCAL/$method/plots/" 2>/dev/null || true
done
echo "=== Done. Dashboard should now show updated plots ==="
