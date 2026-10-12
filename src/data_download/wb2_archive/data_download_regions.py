"""Load `src/configs/data/regions.yaml` for the download scripts.

Resolves the repo root relative to this file, so it works when the script is
run directly (e.g. on Jean-Zay from any working directory).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_CONFIG = _HERE.parents[2] / "src" / "configs" / "data" / "regions.yaml"

if not _CONFIG.exists():
    sys.exit(f"regions config not found: {_CONFIG}")

with open(_CONFIG) as _f:
    REGIONS: dict = yaml.safe_load(_f)["regions"]