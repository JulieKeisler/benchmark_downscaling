"""Hugging Face Space entry point for the Downscaling Benchmark dashboard.

The Space itself only ships this bootstrap plus a small copy of the downstream
TAM lens data (``tam/``). At startup it:

1. clones the dashboard code from GitHub (single source of truth), and
2. downloads ``results.tar.gz`` from a Hugging Face dataset repo and extracts it.

Both are overridable with Space variables (see README.md). Everything else runs
unchanged from ``src/dashboard/dashboard.py``.
"""
from __future__ import annotations

import os
import runpy
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path("/tmp/benchmark_downscaling")

REPO_URL = os.environ.get(
    "BENCH_REPO_URL", "https://github.com/JulieKeisler/benchmark_downscaling")
REPO_REF = os.environ.get("BENCH_REPO_REF", "main")
RESULTS_REPO = os.environ.get(
    "HF_RESULTS_REPO", "Jkeisler75/benchmark-downscaling-results")
DATA_DIR = Path(os.environ.get("RESULTS_ROOT", "/data/results"))


def _clone_code() -> None:
    target = WORK / "src" / "dashboard" / "dashboard.py"
    if target.exists():
        return
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(WORK)],
        check=True,
    )


def _fetch_results() -> None:
    marker = DATA_DIR / ".ready"
    if marker.exists():
        return
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        repo_id=RESULTS_REPO, repo_type="dataset", filename="results.tar.gz")
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(DATA_DIR, filter="data")  # py>=3.12
        except TypeError:
            tf.extractall(DATA_DIR)
    marker.write_text("ok")


def main() -> None:
    _clone_code()
    _fetch_results()
    os.environ["RESULTS_ROOT"] = str(DATA_DIR)
    os.environ.setdefault("TAM_DIR", str(HERE / "tam"))
    runpy.run_path(
        str(WORK / "src" / "dashboard" / "dashboard.py"), run_name="__main__")


main()
