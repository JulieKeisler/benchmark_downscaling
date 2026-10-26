#!/bin/bash
#SBATCH --partition=cpu_devel
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.out
#SBATCH --error=/scratch/jkeisler/benchmark_downscaling/logs_v2/%x_%j.err
# Re-evaluate one GCM x region pair in place with ERA5 pr clipped at 0 (load_era5, 2026-10-26).
# Predictions are not touched. The previous outputs are kept once as metrics_prneg.json,
# plot_data_prneg.json and plots_prneg/, so the earlier numbers stay reproducible.
# Submit: sbatch --job-name=v2_prclip_$REGION_$GCM \
#   --export=ALL,REGION=<region>,GCM=<gcm>,METHODS="cdft dotc idownscale serpentflowclim",MEMBERS="serpentflowclim_m2 serpentflowclim_m3" \
#   reeval_prclip.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
source /home/jkeisler/benchmark_downscaling_v2/tools/cleps_jobs/v2/region_env.sh "$REGION" prclip
rmdir "$OUT_DIR" 2>/dev/null || true          # region_env creates <base>/<method>; none here
# $HOME is over quota on cleps: the v2 code with the patched loader is a copy in /scratch
export V2=/scratch/jkeisler/benchmark_downscaling/code_v2_prclip
export PYTHONPATH=$V2/src/evaluation:$V2/src
cd "$V2"
grep -q "clip(min=0.0)" "$V2/src/evaluation/data_processing.py" \
  || { echo "TRIPWIRE: load_era5 does not clip ERA5 pr in $V2" >&2; exit 1; }
MEMBERS_BASE=/scratch/jkeisler/benchmark_downscaling/results_v2_members/$SUB

keep() {   # keep the pre-clip outputs once
  local d="$1"
  [ -f "$d/metrics_prneg.json" ] || cp -p "$d/metrics.json" "$d/metrics_prneg.json"
  [ -f "$d/plot_data_prneg.json" ] || { [ ! -f "$d/plot_data.json" ] || cp -p "$d/plot_data.json" "$d/plot_data_prneg.json"; }
  [ -d "$d/plots_prneg" ] || { [ ! -d "$d/plots" ] || cp -a "$d/plots" "$d/plots_prneg"; }
}

fail=0
for m in ${METHODS:-}; do
  keep "$OUT_BASE/$m"
  evaluate "$m" "$OUT_BASE/$m" "$OUT_BASE" || { echo "FAILED $m" >&2; fail=$((fail + 1)); }
done
for m in ${MEMBERS:-}; do
  keep "$MEMBERS_BASE/$m"
  evaluate "$m" "$MEMBERS_BASE/$m" "$MEMBERS_BASE" || { echo "FAILED $m" >&2; fail=$((fail + 1)); }
done
echo "PRCLIP DONE $REGION $GCM failed=$fail"
exit $fail
