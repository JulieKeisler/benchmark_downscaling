"""Streamlit Community Cloud entry point for the Downscaling Benchmark dashboard.

Unlike the Hugging Face Space bootstrap, the code is already checked out here
(Streamlit Cloud runs the app straight from the GitHub repo). This entry point
only:

1. downloads ``results.tar.gz`` and ``tam.tar.gz`` from the (public) HF dataset
   repo and extracts them into a writable cache, and
2. runs ``src/dashboard/dashboard.py`` unchanged.

Dataset repo is overridable via the ``HF_RESULTS_REPO`` secret/env var.
"""
from __future__ import annotations

import os
import runpy
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root
CACHE = Path(os.environ.get("BENCH_CACHE", "/tmp/benchmark_downscaling"))
RESULTS_DIR = CACHE / "results"
TAM_DIR = CACHE / "tam"

RESULTS_REPO = os.environ.get(
    "HF_RESULTS_REPO", "Jkeisler75/benchmark-downscaling-results")


def _fetch(filename: str, dest: Path) -> None:
    marker = dest / ".ready"
    if marker.exists():
        return
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download(
        repo_id=RESULTS_REPO, repo_type="dataset", filename=filename)
    with tarfile.open(archive) as tf:
        try:
            tf.extractall(dest, filter="data")  # py>=3.12
        except TypeError:
            tf.extractall(dest)
    marker.write_text("ok")


def main() -> None:
    _fetch("results.tar.gz", RESULTS_DIR)
    _fetch("tam.tar.gz", TAM_DIR)
    os.environ["RESULTS_ROOT"] = str(RESULTS_DIR)
    os.environ["TAM_DIR"] = str(TAM_DIR)
    runpy.run_path(
        str(ROOT / "src" / "dashboard" / "dashboard.py"), run_name="__main__")


main()
