#!/bin/bash
# Fetch remote method dirs one at a time, crop them locally with crop_predictions.py, and
# delete the full copy (escape when the cluster queue is blocked; ~3.3 GB in flight).
# Usage: bash tools/crop_remote_cells.sh <host:root> <local dst> <start> <end> <tmp dir> cell...
set -euo pipefail
SRC=$1 DST=$2 START=$3 END=$4 TMP=$5; shift 5
BM=$(cd "$(dirname "$0")/.." && pwd)
for c in "$@"; do
  if [ -f "$DST/$c/predictions.pt" ]; then echo "$c: already cropped"; continue; fi
  mkdir -p "$TMP/$c"
  rsync -a "$SRC/$c/"{predictions.pt,eval_dates.npy,lat.npy,lon.npy,variables.npy,metrics.json} "$TMP/$c/"
  python3 "$BM/tools/crop_predictions.py" --root "$TMP" --dst "$DST" --start "$START" --end "$END" "$c"
  rm -rf "$TMP/$c"
done
