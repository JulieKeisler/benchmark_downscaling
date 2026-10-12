#!/usr/bin/env python3
"""Upload benchmark data to Zenodo via the REST API.

Bundles each directory into a tar.gz (Zenodo limits records to 100 files).

Uploads:
  - data/era5.tar.gz          (ERA5 daily reanalysis, France)
  - data/cmip6/ipsl_cm6a_lr.tar.gz  (IPSL-CM6A-LR CMIP6 GCM)

Usage:
  export ZENODO_TOKEN=<your Zenodo access token>
  python scripts/upload_to_zenodo.py [--sandbox] [--existing ID]

  --sandbox       Use the Zenodo sandbox (sandbox.zenodo.org) for testing.
  --existing ID   Add files to an existing deposition instead of creating one.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import requests

BASE_URL = "https://zenodo.org/api"

UPLOAD_DIRS: list[tuple[str, str]] = [
    ("data/era5", "era5"),
    ("data/cmip6/ipsl_cm6a_lr", "cmip6/ipsl_cm6a_lr"),
]

REPO_ROOT = Path(__file__).resolve().parent.parent

TITLE = "Benchmark data for climate downscaling: ERA5 and IPSL-CM6A-LR (France)"
DESCRIPTION = (
    "Reference ERA5 daily reanalysis (1960-2020) and IPSL-CM6A-LR CMIP6 "
    "GCM output (historical + ssp370, 1960-2100) for the France domain "
    "[55N, 6W, 41S, 10E].\n\n"
    "Variables: tas, pr, sfcWind, sfcWindmax, rsds, psl.\n\n"
    "Source scripts available at "
    "https://github.com/JulieKeisler/benchmark_downscaling"
)

LICENSE = "cc-by-4.0"


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def create_deposition(token: str) -> dict:
    resp = requests.post(
        f"{BASE_URL}/deposit/depositions",
        headers={**_auth(token), "Content-Type": "application/json"},
        json={
            "metadata": {
                "title": TITLE,
                "description": DESCRIPTION,
                "license": LICENSE,
                "access_right": "open",
                "upload_type": "dataset",
                "creators": [{"name": "Keisler, Julien"}],
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    dep = resp.json()
    print(f"Created deposition {dep['id']}: {dep['links']['html']}")
    return dep


def get_deposition(token: str, dep_id: int) -> dict:
    resp = requests.get(
        f"{BASE_URL}/deposit/depositions/{dep_id}",
        headers=_auth(token),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def upload_file(token: str, bucket_url: str, filepath: Path, key: str) -> None:
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  {key} ({size_mb:.1f} MB) ...", end=" ", flush=True)
    with open(filepath, "rb") as fh:
        resp = requests.put(
            f"{bucket_url}/{key}",
            headers=_auth(token),
            data=fh,
            timeout=3600,
        )
    resp.raise_for_status()
    print("done")


def publish(token: str, dep_id: int) -> str:
    resp = requests.post(
        f"{BASE_URL}/deposit/depositions/{dep_id}/actions/publish",
        headers=_auth(token),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    doi = data.get("doi", "")
    url = data.get("links", {}).get("html", "")
    print(f"\nPublished! DOI: {doi}")
    print(f"URL: {url}")
    return doi


def make_tarball(local_dir: Path, arcname: str) -> Path:
    """Create a tar.gz of local_dir, returning the path to the temp file."""
    tarball = Path(tempfile.mktemp(suffix=".tar.gz"))
    with tarfile.open(tarball, "w:gz") as tf:
        for p in sorted(local_dir.iterdir()):
            if p.is_file():
                tf.add(p, arcname=f"{arcname}/{p.name}")
    size_mb = tarball.stat().st_size / (1024 * 1024)
    print(f"  created {tarball.name} ({size_mb:.1f} MB, {len(list(local_dir.iterdir()))} files)")
    return tarball


def collect_archives() -> list[tuple[Path, str]]:
    """Bundle each upload directory into a tar.gz. Returns (tarball_path, upload_name)."""
    archives: list[tuple[Path, str]] = []
    for local_dir_rel, arcname in UPLOAD_DIRS:
        local_dir = REPO_ROOT / local_dir_rel
        if not local_dir.is_dir():
            print(f"WARNING: {local_dir} not found, skipping", file=sys.stderr)
            continue
        tarball = make_tarball(local_dir, arcname)
        archives.append((tarball, f"{arcname.replace('/', '_')}.tar.gz"))
    return archives


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload benchmark data to Zenodo")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org")
    parser.add_argument("--existing", type=int, metavar="ID", help="Add to existing deposition")
    parser.add_argument("--no-publish", action="store_true", help="Upload but don't publish")
    args = parser.parse_args()

    global BASE_URL
    if args.sandbox:
        BASE_URL = "https://sandbox.zenodo.org/api"

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        print("ERROR: set ZENODO_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    archives = collect_archives()
    total_size = sum(p.stat().st_size for p, _ in archives)
    print(f"{len(archives)} archives to upload ({total_size / 1e9:.2f} GB)")

    if args.existing:
        dep = get_deposition(token, args.existing)
        dep_id = args.existing
        print(f"Using existing deposition {dep_id}")
    else:
        dep = create_deposition(token)
        dep_id = dep["id"]

    bucket_url = dep["links"]["bucket"]

    for i, (filepath, key) in enumerate(archives, 1):
        print(f"[{i}/{len(archives)}]")
        upload_file(token, bucket_url, filepath, key)

    if args.no_publish:
        print(f"\nUploaded all files. Deposition {dep_id} is still draft.")
        print(f"Review at: {dep['links']['html']}")
    else:
        publish(token, dep_id)

    for p, _ in archives:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
