#!/usr/bin/env python3
"""Package and (optionally) upload the benchmark dashboard to Hugging Face.

Builds a size-minimised copy of the *app-visible* results (the three benchmark
regions and the published baselines — no ``predictions.pt``, no unpublished /
ablation / stray ``*_eval`` dirs, no pre-rendered comparison figures), tars it,
and uploads:

  * the data archive  -> a Hugging Face **dataset** repo (``results.tar.gz`` +
    a dataset card), and
  * the Space files   -> a Hugging Face **Space** (Streamlit SDK):
    ``deploy/hf_space/*`` plus a small copy of the downstream TAM data.

Usage:
    export HF_TOKEN=hf_...            # needs write access
    python tools/deploy_hf.py --upload \
        --dataset-repo Jkeisler75/benchmark-downscaling-results \
        --space-repo   Jkeisler75/downscaling-benchmark

Without ``--upload`` it only stages everything under ``.deploy_build/`` so you
can inspect sizes / contents first.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from baselines.labels import is_unpublished  # noqa: E402  (dependency-free)

REGIONS = ["france", "australia", "us_central"]
# Published baselines only (no unpublished methods, ablations or *_eval dirs).
METHODS = {
    "bicubic", "eqm", "bcsd", "cdft", "dotc",
    "idownscale", "corrdiff_qm", "cm", "serpentflowclim",
}
GCM_DIRS = {"ipsl_cm6a_lr", "cnrm_cm6_1", "cesm2"}
SKIP_FILES = {"predictions.pt"}

TAM_PATTERNS = [
    "metrics_results*.json", "metrics_regional*.json",
    "cdf_results*.npz", "cdf_regional*.npz", "tam_forecasts*.npz",
]
# Arrays in the TAM .npz files indexed by `names`, and the axis that is.
TAM_NPZ_NAME_AXIS = {"curves": 0, "pred": 1}

DEFAULT_DATASET_REPO = "Jkeisler75/benchmark-downscaling-results"
DEFAULT_SPACE_REPO = "Jkeisler75/downscaling-benchmark"

DATASET_CARD = """\
---
license: mit
pretty_name: Downscaling Benchmark for the Electricity System
---

# Downscaling Benchmark — results

Evaluation artifacts (metrics, plot data, maps) for the downscaling benchmark
dashboard. Two archives: `results.tar.gz` (extract → `france/`, `australia/`,
`us_central/`) and `tam.tar.gz` (downstream net-load lens data). Consumed at
runtime by the companion Streamlit apps; see
https://github.com/JulieKeisler/benchmark_downscaling.
"""


def _copy_filtered(src: Path, dst: Path) -> int:
    """Copy `src` -> `dst`, skipping `SKIP_FILES`. Returns files copied."""
    n = 0
    for p in src.rglob("*"):
        if not p.is_file() or p.name in SKIP_FILES:
            continue
        rel = p.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        n += 1
    return n


def stage_results(results_root: Path, out: Path, regions: list[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for r in regions:
        base = results_root / r
        if not base.is_dir():
            print(f"  WARNING: {base} missing, skipping")
            continue
        # native coupling: methods directly under results/{region}/
        for m in sorted(METHODS):
            if (base / m).is_dir():
                total += _copy_filtered(base / m, out / r / m)
        # factorial couplings: results/{region}/{gcm}/{method}
        for g in sorted(GCM_DIRS):
            for m in sorted(METHODS):
                if (base / g / m).is_dir():
                    total += _copy_filtered(base / g / m, out / r / g / m)
        # results/{region}/comparison/ is not staged: its pre-rendered figures
        # can't be filtered by method, and the dashboard draws its own heatmap.
    print(f"  staged {total} files")


def _stage_tam_json(src: Path, dst: Path) -> int:
    """Copy a TAM metrics JSON without unpublished methods. Returns rows dropped."""
    d = json.loads(src.read_text())
    dropped = [k for k in d if is_unpublished(k)]
    d = {k: v for k, v in d.items() if not is_unpublished(k)}
    for bk in ("by_season", "by_wind"):
        if isinstance(d.get(bk), dict):
            d[bk] = {k: v for k, v in d[bk].items() if not is_unpublished(k)}
    dst.write_text(json.dumps(d, indent=2))
    return len(dropped)


def _stage_tam_npz(src: Path, dst: Path) -> int:
    """Copy a TAM .npz without unpublished methods. Returns columns dropped."""
    with np.load(src, allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    names = arrays["names"]
    keep = [i for i, n in enumerate(names) if not is_unpublished(n)]
    arrays["names"] = names[keep]
    for k, axis in TAM_NPZ_NAME_AXIS.items():
        if k in arrays:
            arrays[k] = np.take(arrays[k], keep, axis=axis)
    np.savez_compressed(dst, **arrays)
    return len(names) - len(keep)


def stage_tam(tam_dir: Path, out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    n = dropped = 0
    for pat in TAM_PATTERNS:
        for p in sorted(tam_dir.glob(pat)):
            if not p.is_file():
                continue
            if p.suffix == ".json":
                dropped += _stage_tam_json(p, out / p.name)
            else:
                dropped += _stage_tam_npz(p, out / p.name)
            n += 1
    print(f"  staged {n} TAM files ({dropped} unpublished-method entries dropped)")


def make_archive(results_dir: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz") as tf:
        for child in sorted(results_dir.iterdir()):
            tf.add(child, arcname=child.name)


def stage_space(build: Path) -> Path:
    src = REPO_ROOT / "deploy" / "hf_space"
    dst = build / "space"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    # bundle the small TAM data next to the app
    shutil.copytree(build / "tam", dst / "tam", dirs_exist_ok=True)
    return dst


def _mb(p: Path) -> str:
    if p.is_file():
        return f"{p.stat().st_size / 1048576:,.1f} MB"
    return f"{sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / 1048576:,.1f} MB"


def upload(dataset_repo: str, space_repo: str, build: Path, token: str | None,
           only: str | None = None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if only in (None, "dataset"):
        if not api.repo_exists(dataset_repo, repo_type="dataset"):
            api.create_repo(dataset_repo, repo_type="dataset")
        card = build / "dataset_README.md"
        card.write_text(DATASET_CARD)
        api.upload_file(path_or_fileobj=build / "results.tar.gz", path_in_repo="results.tar.gz",
                        repo_id=dataset_repo, repo_type="dataset")
        if (build / "tam.tar.gz").exists():
            api.upload_file(path_or_fileobj=build / "tam.tar.gz", path_in_repo="tam.tar.gz",
                            repo_id=dataset_repo, repo_type="dataset")
        api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                        repo_id=dataset_repo, repo_type="dataset")
        print(f"  uploaded dataset -> https://huggingface.co/datasets/{dataset_repo}")

    if only in (None, "space"):
        if not api.repo_exists(space_repo, repo_type="space"):
            try:
                api.create_repo(space_repo, repo_type="space", space_sdk="streamlit")
            except Exception as e:
                print(f"  could not create Space via API ({e}).")
                print(f"  -> Create it in the UI (SDK = Streamlit, name "
                      f"'{space_repo.split('/')[-1]}'), then re-run "
                      f"`--upload --only space --reuse-build`.")
                raise
        api.upload_folder(folder_path=build / "space", repo_id=space_repo, repo_type="space")
        print(f"  uploaded space   -> https://huggingface.co/spaces/{space_repo}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--tam-dir", type=Path,
                    default=Path(os.environ.get("TAM_DIR",
                                                "/home/jkeisler/Documents/Code/TAM_downscaling")))
    ap.add_argument("--build-dir", type=Path, default=REPO_ROOT / ".deploy_build")
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--dataset-repo", default=DEFAULT_DATASET_REPO)
    ap.add_argument("--space-repo", default=DEFAULT_SPACE_REPO)
    ap.add_argument("--upload", action="store_true", help="Create/update the HF repos")
    ap.add_argument("--only", choices=["dataset", "space"],
                    help="Upload only the dataset or only the Space")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF token; defaults to $HF_TOKEN, then the `hf auth login` cache")
    ap.add_argument("--reuse-build", action="store_true",
                    help="Skip re-staging results if .deploy_build/results.tar.gz "
                         "already exists (TAM data is always re-staged)")
    args = ap.parse_args()

    build = args.build_dir
    if not args.tam_dir.is_dir():
        print(f"  ERROR: TAM dir not found: {args.tam_dir}", file=sys.stderr)
        return 1

    if args.reuse_build and (build / "results.tar.gz").exists():
        print(f"Reusing existing results build in {build}")
    else:
        if build.exists():
            shutil.rmtree(build)
        build.mkdir(parents=True)
        print("Staging results data …")
        stage_results(args.results_root, build / "results", args.regions)
        print("Creating results.tar.gz …")
        make_archive(build / "results", build / "results.tar.gz")
    # Small, and the unpublished-method filter must apply to every upload.
    print("Staging TAM data …")
    stage_tam(args.tam_dir, build / "tam")
    make_archive(build / "tam", build / "tam.tar.gz")

    space_dir = stage_space(build)

    print("\nBuild summary")
    print(f"  {build / 'results.tar.gz'}  ({_mb(build / 'results.tar.gz')})")
    print(f"  {space_dir}  ({_mb(space_dir)})")
    print(f"  build dir: {build}")

    if args.upload:
        if not args.token:
            from huggingface_hub import get_token
            args.token = get_token()  # cached by `hf auth login`
        if not args.token:
            print("  ERROR: --upload needs a token. Run `hf auth login`, or set "
                  "HF_TOKEN / pass --token.", file=sys.stderr)
            return 1
        print("\nUploading …")
        upload(args.dataset_repo, args.space_repo, build, args.token, args.only)
        print("\nDone. Set Space variable HF_RESULTS_REPO="
              f"{args.dataset_repo} if you changed it from the default.")
    else:
        print("\nStaged only (no --upload). Inspect, then re-run with --upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
