#!/bin/bash
# Flatten double-nested eval output in factorial result dirs.
# Layout produced by jobs: results/{r}/{g}/{method}/{method}/{metrics.json,...}
# Layout expected by dashboard: results/{r}/{g}/{method}/metrics.json
# Usage: flatten_factorial_results.sh <results_root>
set -e
RESULTS_ROOT="${1:?usage: flatten_factorial_results.sh <results_root>}"
cd "$RESULTS_ROOT" || exit 1
for r in australia france us_central; do
  [ -d "$r" ] || continue
  for g in "$r"/*; do
    [ -d "$g" ] || continue
    for m in "$g"/*; do
      [ -d "$m" ] || continue
      nested="$m/$(basename "$m")"
      if [ -d "$nested" ]; then
        echo "flattening: $m"
        # move nested eval artifacts up one level (only if parent lacks metrics.json)
        if [ ! -f "$m/metrics.json" ]; then
          mv "$nested/metrics.json" "$m/" 2>/dev/null || true
        fi
        mv "$nested/plot_data.json" "$m/" 2>/dev/null || true
        mv -n "$nested/plots" "$m/" 2>/dev/null || true
        rmdir "$nested" 2>/dev/null || true
      fi
    done
  done
done
echo "=== done ==="