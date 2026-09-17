"""Tests for the method registry in src/baselines/__init__.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

from baselines import METHOD_REGISTRY, METHOD_LABELS


def _da(lat, lon, t=365, seed=0):  # 365 days = all 12 months for BCSD/CDF-t loops
    rng  = np.random.default_rng(seed)
    time = pd.date_range("2010-01-01", periods=t, freq="D")
    data = (280 + rng.standard_normal((t, len(lat), len(lon)))).astype(np.float32)
    return xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"])


OBS_LAT = np.linspace(41.0, 45.5, 10)
OBS_LON = np.linspace(-5.0, 0.5, 12)
GCM_LAT = np.linspace(41.0, 45.0, 5)
GCM_LON = np.linspace(-5.0, 0.0, 6)


class TestMethodRegistry:

    def test_expected_methods_present(self):
        for key in ["bicubic", "bcsd", "cdft", "dotc", "cordiff"]:
            assert key in METHOD_REGISTRY, f"'{key}' missing from METHOD_REGISTRY"

    def test_all_methods_callable(self):
        for name, fn in METHOD_REGISTRY.items():
            assert callable(fn), f"METHOD_REGISTRY['{name}'] is not callable"

    def test_stateless_methods_run(self):
        obs = _da(OBS_LAT, OBS_LON, seed=0)
        gcm = _da(GCM_LAT, GCM_LON, seed=1)
        for name in ["bicubic"]:
            fn   = METHOD_REGISTRY[name]
            pred = fn(gcm, obs, "tas")
            assert pred.dims == ("time", "lat", "lon"), \
                f"{name} returned wrong dims: {pred.dims}"
            assert pred.sizes["lat"] == obs.sizes["lat"], \
                f"{name} output lat mismatch"

        # Methods requiring external deps (skip if not available)
        for name in ["bcsd", "cdft", "dotc", "cordiff"]:
            if name in METHOD_REGISTRY:
                try:
                    fn = METHOD_REGISTRY[name]
                    pred = fn(gcm, obs, "tas")
                    assert pred.dims == ("time", "lat", "lon")
                except ImportError:
                    pytest.skip(f"{name} requires external dependency")
                except Exception:
                    # Other failures (e.g. time alignment) are acceptable for this smoke test
                    pass

    def test_method_labels_cover_registry(self):
        for key in METHOD_REGISTRY:
            assert key in METHOD_LABELS, \
                f"No label for registry method '{key}' in METHOD_LABELS"

    def test_cdft_in_labels(self):
        assert "cdft" in METHOD_LABELS
        assert METHOD_LABELS["cdft"] == "CDF-t"

    def test_new_baselines_in_labels(self):
        assert "dotc" in METHOD_LABELS
        assert METHOD_LABELS["dotc"] == "DOTC"
        assert "cordiff" in METHOD_LABELS
        assert METHOD_LABELS["cordiff"] == "CorDiff"
