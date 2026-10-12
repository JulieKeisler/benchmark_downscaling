"""Generate jz + cleps job scripts for the 6 missing (non-native) couplings.

Layout: results/{region}/{gcm_method_key}/{method}
Each coupling gets one script per baseline:
  - jz   (sbatch, gpu): bcsd, bicubic, cm, corrdiff_qm, serpentflowclim
  - cleps (sbatch): cdft, dotc (SBCK, cpu_homogen); idownscale (SBCK, gpu)

SBCK-dependent methods (cdft, dotc, idownscale) run on cleps where SBCK is
installed; jz compute nodes have no network (the old `pip install sbck` in the
bcsd script failed there — bcsd itself is pure quantile mapping and stays on jz).

Neural baselines (cm, corrdiff_qm, serpentflowclim) reuse the
existing per-region checkpoint when the coupling's variable set matches the
region-native one; otherwise they retrain (per AGENTS/user decision) with the
coupling's own variable set.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent

# region: (era5_dir_rel, native_gcm)
NATIVE = {
    "france":     ("$BM_CODE/data/era5",              "ipsl_cm6a_lr"),
    "australia":  ("$BM_CODE/data/era5/australia",    "cnrm_cm6_1"),
    "us_central": ("$BM_CODE/data/era5/us_central",   "cesm2"),
}

VARS6 = ["tas", "pr", "psl", "sfcWind", "sfcWindmax", "rsds"]  # IPSL/CNRM couplings
VARS5 = ["tas", "pr", "psl", "sfcWind", "rsds"]                # CESM2 couplings (no sfcWindmax)

# (region, gcm_key, scenario, gcm_dir_rel, vars, neural_reuse_ckpt)
COUPLINGS = [
    ("france",      "cesm2",      "ssp245",
     "cmip6/_factorial/cmip6_cesm2_r4/france/cesm2", VARS5, False),
    ("france",      "cnrm_cm6_1", "ssp370",
     "cmip6/_factorial/cmip6_factorial/france/cnrm_cm6_1", VARS6, True),
    ("australia",   "ipsl_cm6a_lr", "ssp370",
     "cmip6/_factorial/cmip6_factorial/australia/ipsl_cm6a_lr", VARS6, True),
    ("australia",   "cesm2",      "ssp245",
     "cmip6/_factorial/cmip6_cesm2_r4/australia/cesm2", VARS5, False),
    ("us_central",  "ipsl_cm6a_lr", "ssp370",
     "cmip6/_factorial/cmip6_factorial/us_central/ipsl_cm6a_lr", VARS6, False),
    ("us_central",  "cnrm_cm6_1", "ssp370",
     "cmip6/_factorial/cmip6_factorial/us_central/cnrm_cm6_1", VARS6, False),
]

JZ_HEADER = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=gpu_p13
#SBATCH --account=bqu@v100
#SBATCH --qos=qos_gpu-t3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time}
#SBATCH --output=/gpfswork/rech/bqu/uyv87oi/benchmark_downscaling/logs/fact_{job}_%j.out
#SBATCH --error=/gpfswork/rech/bqu/uyv87oi/benchmark_downscaling/logs/fact_{job}_%j.err

set -e
source /etc/profile
module purge
module load pytorch-gpu/py3/2.7.0
source /gpfswork/rech/bqu/uyv87oi/benchmark_downscaling/jz/paths.sh
cd "$BM_CODE"

REGION={region}
GCM_KEY={gcm_key}
SCEN={scen}
VARS=({vars})
ERA5_DIR={era5_rel}
GCM_DIR="$BM_CODE/data/{gcm_rel}"
OUT_DIR="$BM_RESULTS/$REGION/$GCM_KEY/{method}"
mkdir -p "$OUT_DIR" "$BM_LOGS"

echo "==== factorial {region} x {gcm_key} / {method} | {scen} | ${{VARS[*]}} ===="
"""

CLEPS_HEADER = """#!/bin/bash
#SBATCH --job-name=fact_{job}
#SBATCH --partition=cpu_homogen
#SBATCH --account=inria
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time={time}
#SBATCH --output=/home/jkeisler/benchmark_downscaling/logs/fact_{job}_%j.out
#SBATCH --error=/home/jkeisler/benchmark_downscaling/logs/fact_{job}_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
cd $HOME/benchmark_downscaling
export PYTHONPATH=$HOME/benchmark_downscaling/src/evaluation:$HOME/benchmark_downscaling/src
mkdir -p $HOME/benchmark_downscaling/logs

REGION={region}
GCM_KEY={gcm_key}
SCEN={scen}
VARS=({vars})
ERA5_DIR={era5_rel}
GCM_DIR=data/{gcm_rel}
OUT_DIR=results/$REGION/$GCM_KEY/{method}
mkdir -p "$OUT_DIR"

echo "==== factorial {region} x {gcm_key} / {method} | {scen} | ${{VARS[*]}} ===="
"""

CLEPS_GPU_HEADER = """#!/bin/bash
#SBATCH --job-name=fact_{job}
#SBATCH --partition=gpu
#SBATCH --account=inria
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time={time}
#SBATCH --output=/home/jkeisler/benchmark_downscaling/logs/fact_{job}_%j.out
#SBATCH --error=/home/jkeisler/benchmark_downscaling/logs/fact_{job}_%j.err

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sbck-gpu
cd $HOME/benchmark_downscaling
export PYTHONPATH=$HOME/benchmark_downscaling/src/evaluation:$HOME/benchmark_downscaling/src
mkdir -p $HOME/benchmark_downscaling/logs

REGION={region}
GCM_KEY={gcm_key}
SCEN={scen}
VARS=({vars})
ERA5_DIR={era5_rel}
GCM_DIR=data/{gcm_rel}
OUT_DIR=results/$REGION/$GCM_KEY/{method}
mkdir -p "$OUT_DIR"

echo "==== factorial {region} x {gcm_key} / {method} | {scen} | ${{VARS[*]}} ===="
"""


def cleps_era5_rel(region):
    # cleps: france = data/era5 (flat); australia/us_central = data/era5_orig/<region>
    if region == "france":
        return "data/era5"
    return f"data/era5_orig/{region}"


def jz_era5_rel(region):
    return NATIVE[region][0].replace("$BM_CODE/data/", "data/")


# ---------------- jz methods ----------------

def jz_bcsd(region, gcm_key, scen, vars_, gcm_rel, era5_rel):
    return JZ_HEADER.format(
        job=f"{region}_{gcm_key}_bcsd", region=region, gcm_key=gcm_key, scen=scen,
        vars=" ".join(vars_), era5_rel=NATIVE[region][0], gcm_rel=gcm_rel,
        method="bcsd", time="10:00:00", cpus=10) + f"""
python "$BM_CODE/src/baselines/inference_bcsd.py" \\
    --era5-dir    "$ERA5_DIR" \\
    --gcm-dir     "$GCM_DIR" \\
    --out-dir     "$OUT_DIR" \\
    --calib-start 1980 \\
    --calib-end   1999 \\
    --eval-start  2000 \\
    --eval-end    2100 \\
    --scenario    "$SCEN" \\
    --win-size    20 \\
    --step        10 \\
    --trim        5 \\
    --variables   "${{VARS[@]}}"

python "$BM_CODE/src/evaluation/evaluate.py" \\
    --era5-dir   "$ERA5_DIR" \\
    --gcm-dir    "$GCM_DIR" \\
    --scenario   "$SCEN" \\
    --method     bcsd \\
    --variables  "${{VARS[@]}}" \\
    --pred-dir   "$OUT_DIR" \\
    --output-dir "$(dirname "$OUT_DIR")" \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""


def jz_bicubic(region, gcm_key, scen, vars_, gcm_rel, era5_rel):
    return JZ_HEADER.format(
        job=f"{region}_{gcm_key}_bicubic", region=region, gcm_key=gcm_key, scen=scen,
        vars=" ".join(vars_), era5_rel=NATIVE[region][0], gcm_rel=gcm_rel,
        method="bicubic", time="03:00:00", cpus=10) + f"""
python "$BM_CODE/src/baselines/inference_bicubic.py" \\
    --era5-dir    "$ERA5_DIR" \\
    --gcm-dir     "$GCM_DIR" \\
    --out-dir     "$OUT_DIR" \\
    --scenario    "$SCEN" \\
    --variables   "${{VARS[@]}}" \\
    --eval-start  2000 \\
    --eval-end    2100

python "$BM_CODE/src/evaluation/evaluate.py" \\
    --era5-dir   "$ERA5_DIR" \\
    --gcm-dir    "$GCM_DIR" \\
    --scenario   "$SCEN" \\
    --method     bicubic \\
    --variables  "${{VARS[@]}}" \\
    --pred-dir   "$OUT_DIR" \\
    --output-dir "$(dirname "$OUT_DIR")" \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""


def jz_cm(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse):
    if reuse:
        body = f"""
CKPT_DIR="$BM_CHECKPOINTS/consistency_rsds/{region}"
if [ ! -d "$CKPT_DIR" ]; then echo "NO ckpt dir $CKPT_DIR"; exit 1; fi
"""
        time = "05:00:00"
    else:
        # retrain: shared per (region, nvars) — guard single training
        nv = len(vars_)
        body = f"""
CKPT_DIR="$BM_CHECKPOINTS/consistency_{nv}v_{region}_{gcm_key}"
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/best.pt" ]; then
    echo "  [CM] retraining {nv}-var {region} checkpoint ..."
    python "$BM_CODE/src/baselines/train_consistency.py" \\
        --era5-dir    "$ERA5_DIR" \\
        --output-dir  "$CKPT_DIR" \\
        --train-start 1980 \\
        --train-end   1999 \\
        --score-epochs 200 \\
        --n-epochs     150 \\
        --batch-size   32 \\
        --variables   "${{VARS[@]}}"
else
    echo "  [CM] reuse existing $CKPT_DIR/best.pt"
fi
"""
        time = "20:00:00"
    return JZ_HEADER.format(
        job=f"{region}_{gcm_key}_cm", region=region, gcm_key=gcm_key, scen=scen,
        vars=" ".join(vars_), era5_rel=NATIVE[region][0], gcm_rel=gcm_rel,
        method="cm", time=time, cpus=20) + body + f"""
pip install --user --quiet torchdiffeq diffusers

python "$BM_CODE/src/baselines/inference_cm.py" \\
    --era5-dir       "$ERA5_DIR" \\
    --gcm-dir        "$GCM_DIR" \\
    --checkpoint-dir "$CKPT_DIR" \\
    --out-dir        "$OUT_DIR" \\
    --calib-start    1980 \\
    --eval-start     2000 \\
    --eval-end       2100 \\
    --scenario       "$SCEN" \\
    --stroke-time    1.0 \\
    --variables      "${{VARS[@]}}"

python "$BM_CODE/src/evaluation/evaluate.py" \\
    --era5-dir   "$ERA5_DIR" \\
    --gcm-dir    "$GCM_DIR" \\
    --scenario   "$SCEN" \\
    --method     cm \\
    --variables  "${{VARS[@]}}" \\
    --pred-dir   "$OUT_DIR" \\
    --output-dir "$(dirname "$OUT_DIR")" \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""


def jz_corrdiff(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse):
    if reuse:
        src = "$BM_RESULTS/%(region)s/corrdiff_qm/checkpoints" % {"region": region}
        body = f"""
CK_DIR="$OUT_DIR/checkpoints"
if [ ! -d "$CK_DIR" ]; then
    echo "  [corrdiff] copying region {region} checkpoints ..."
    cp -r {src} "$CK_DIR"
fi
"""
        time = "20:00:00"
    else:
        body = None
        time = "20:00:00"
    head = JZ_HEADER.format(
        job=f"{region}_{gcm_key}_corrdiff", region=region, gcm_key=gcm_key,
        scen=scen, vars=" ".join(vars_), era5_rel=NATIVE[region][0],
        gcm_rel=gcm_rel, method="corrdiff_qm", time=time, cpus=10)
    if body:
        head += body
    head += f"""
pip install --user --quiet diffusers torchdiffeq

echo "  [corrdiff] {'reuse' if reuse else 'train+'} infer ..."
python "$BM_CODE/src/baselines/inference_corrdiff.py" \\
    --era5-dir      "$ERA5_DIR" \\
    --gcm-dir       "$GCM_DIR" \\
    --out-dir       "$OUT_DIR" \\
    --calib-start   1980 \\
    --calib-end     1999 \\
    --eval-start    2000 \\
    --eval-end      2100 \\
    --scenario      "$SCEN" \\
    --variables     "${{VARS[@]}}" \\
{f"    --train \\\n" if not reuse else ""}    --predict

python "$BM_CODE/src/evaluation/evaluate.py" \\
    --era5-dir   "$ERA5_DIR" \\
    --gcm-dir    "$GCM_DIR" \\
    --scenario   "$SCEN" \\
    --method     corrdiff_qm \\
    --variables  "${{VARS[@]}}" \\
    --pred-dir   "$OUT_DIR" \\
    --output-dir "$(dirname "$OUT_DIR")" \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""
    return head


def jz_idownscale(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse):
    # iDownscale uses SBCK.CDFt (via _cdft_field_sliding in inference_cdft), so it
    # must run on cleps where SBCK is installed (jz has no network for pip install).
    return CLEPS_GPU_HEADER.format(
        job=f"{region}_{gcm_key}_idownscale", region=region, gcm_key=gcm_key,
        scen=scen, vars=" ".join(vars_), era5_rel=cleps_era5_rel(region),
        gcm_rel=gcm_rel, method="idownscale", time="20:00:00") + f"""
python src/baselines/inference_idownscale.py \\
    --era5-dir    $ERA5_DIR \\
    --gcm-dir     $GCM_DIR \\
    --out-dir     $OUT_DIR \\
    --calib-start 1980 \\
    --calib-end   1999 \\
    --eval-start  2000 \\
    --eval-end    2100 \\
    --scenario    $SCEN \\
    --epochs      30 \\
    --batch-size  32 \\
    --variables   ${{VARS[@]}}

python src/evaluation/evaluate.py \\
    --era5-dir   $ERA5_DIR \\
    --gcm-dir    $GCM_DIR \\
    --scenario   $SCEN \\
    --method     idownscale \\
    --variables  ${{VARS[@]}} \\
    --pred-dir   $OUT_DIR \\
    --output-dir $(dirname $OUT_DIR) \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""


def jz_serpentflow(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse):
    # Three phases: prep (per coupling), train (reuse region ckpt if var-set matches, else retrain), infer
    head = JZ_HEADER.format(
        job=f"{region}_{gcm_key}_sft", region=region, gcm_key=gcm_key,
        scen=scen, vars=" ".join(vars_), era5_rel=NATIVE[region][0],
        gcm_rel=gcm_rel, method="serpentflowclim", time="20:00:00", cpus=20)
    if reuse:
        train = f"""
REG_CKPT="$BM_RESULTS/{region}/serpentflowclim/checkpoints/fm_serpentflowclim_best.pth"
if [ ! -f "$OUT_DIR/checkpoints/fm_serpentflowclim_best.pth" ]; then
    echo "  [SFT] reuse region {region} checkpoint"
    mkdir -p "$OUT_DIR/checkpoints"
    cp "$REG_CKPT" "$OUT_DIR/checkpoints/fm_serpentflowclim_best.pth"
fi
"""
        time = "20:00:00"
    else:
        train = f"""
python "$BM_CODE/src/baselines/inference_serpentflowclim.py" \\
    --phase train \\
    --region "$REGION" \\
    --era5-dir "$ERA5_DIR" \\
    --pred-dir "$OUT_DIR" \\
    --variables "${{VARS[@]}}"
"""
        time = "20:00:00"
    return head + f"""
python "$BM_CODE/src/baselines/inference_serpentflowclim.py" \\
    --phase prep \\
    --region "$REGION" \\
    --era5-dir "$ERA5_DIR" \\
    --gcm-dir  "$GCM_DIR" \\
    --scenario "$SCEN" \\
    --pred-dir "$OUT_DIR" \\
    --variables "${{VARS[@]}}"
{train}
python "$BM_CODE/src/baselines/inference_serpentflowclim.py" \\
    --phase infer \\
    --region "$REGION" \\
    --era5-dir "$ERA5_DIR" \\
    --gcm-dir  "$GCM_DIR" \\
    --scenario "$SCEN" \\
    --pred-dir "$OUT_DIR" \\
    --variables "${{VARS[@]}}"

python "$BM_CODE/src/evaluation/evaluate.py" \\
    --era5-dir   "$ERA5_DIR" \\
    --gcm-dir    "$GCM_DIR" \\
    --scenario   "$SCEN" \\
    --method     serpentflowclim \\
    --variables  "${{VARS[@]}}" \\
    --pred-dir   "$OUT_DIR" \\
    --output-dir "$(dirname "$OUT_DIR")" \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""


# ---------------- cleps methods ----------------

def cleps_cdft_dotc(region, gcm_key, scen, vars_, gcm_rel, era5_rel, method):
    return CLEPS_HEADER.format(
        job=f"{region}_{gcm_key}_{method}", region=region, gcm_key=gcm_key,
        scen=scen, vars=" ".join(vars_), era5_rel=cleps_era5_rel(region),
        gcm_rel=gcm_rel, method=method, time="10:00:00") + f"""
python src/baselines/inference_{method}.py \\
    --era5-dir    $ERA5_DIR \\
    --gcm-dir     $GCM_DIR \\
    --out-dir     $OUT_DIR \\
    --calib-start 1980 \\
    --calib-end   1999 \\
    --eval-start  2000 \\
    --eval-end    2100 \\
    --scenario    $SCEN \\
    --win-size    20 \\
    --step        10 \\
    --trim        5 \\
    --variables   ${{VARS[@]}}

python src/evaluation/evaluate.py \\
    --era5-dir   $ERA5_DIR \\
    --gcm-dir    $GCM_DIR \\
    --scenario   $SCEN \\
    --method     {method} \\
    --variables  ${{VARS[@]}} \\
    --pred-dir   $OUT_DIR \\
    --output-dir $(dirname $OUT_DIR) \\
    --start      2000 \\
    --end        2020 \\
    --gcm-end    2100
echo Done
"""


def main():
    jz_dir = OUT / "jz/jobs_factorial"
    cl_dir = OUT / "cleps_jobs_factorial"
    jz_dir.mkdir(parents=True, exist_ok=True)
    cl_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for coupl in COUPLINGS:
        region, gcm_key, scen, gcm_rel, vars_, reuse = coupl
        era5_rel = jz_era5_rel(region)
        jobs = {
            "bcsd":         lambda: jz_bcsd(region, gcm_key, scen, vars_, gcm_rel, era5_rel),
            "bicubic":      lambda: jz_bicubic(region, gcm_key, scen, vars_, gcm_rel, era5_rel),
            "cm":           lambda: jz_cm(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse),
            "corrdiff_qm":  lambda: jz_corrdiff(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse),
            "idownscale":   lambda: jz_idownscale(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse),
            "serpentflowclim": lambda: jz_serpentflow(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse),
        }
        for m, fn in jobs.items():
            if m == "idownscale":
                continue  # SBCK-dependent → cleps GPU
            p = jz_dir / f"{region}__{gcm_key}__{m}.sh"
            p.write_text(fn())
            written.append(f"jz:  {p.name}")
        for m in ("cdft", "dotc"):
            p = cl_dir / f"{region}__{gcm_key}__{m}.sh"
            p.write_text(cleps_cdft_dotc(region, gcm_key, scen, vars_, gcm_rel, era5_rel, m))
            written.append(f"cl:  {p.name}")
        p = cl_dir / f"{region}__{gcm_key}__idownscale.sh"
        p.write_text(jz_idownscale(region, gcm_key, scen, vars_, gcm_rel, era5_rel, reuse))
        written.append(f"cl:  {p.name}")

    print("\n".join(written))
    print(f"\nWrote {len(written)} scripts to {jz_dir} and {cl_dir}")


if __name__ == "__main__":
    main()
