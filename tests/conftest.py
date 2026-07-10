"""Shared pytest fixtures — synthetic ERA5-like and GCM-like DataArrays.

All fixtures use small grids (no real files needed) so tests run fast and
offline. Physical values are chosen to be realistic enough to exercise
unit-conversion and clipping logic.

Grid layout mirrors the real benchmark:
    obs (ERA5-like):  lat 10 pts (0.5° step), lon 12 pts (0.5° step)
    gcm (GCM-like):   lat  5 pts (1.0° step), lon  6 pts (1.0° step)
    time:             120 days covering 2 full years (2010-2011)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# ── Grid definitions ──────────────────────────────────────────────────────────

OBS_LAT = np.linspace(41.0, 45.5, 10)   # south → north, 0.5°
OBS_LON = np.linspace(-5.0, 0.5, 12)    # west → east,  0.5°
GCM_LAT = np.linspace(41.0, 45.0, 5)    # coarser 1.0°
GCM_LON = np.linspace(-5.0, 0.0, 6)
TIME_120 = pd.date_range("2010-01-01", periods=365, freq="D")  # full year covers all 12 months
TIME_60  = pd.date_range("2010-01-01", periods=365, freq="D")


def _make_da(lat, lon, time, base=280.0, noise=5.0, seed=0):
    rng = np.random.default_rng(seed)
    data = base + noise * rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.DataArray(data.astype(np.float32),
                        coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"])


@pytest.fixture
def obs_tas():
    """ERA5-like near-surface temperature [K], fine grid, 120 days."""
    return _make_da(OBS_LAT, OBS_LON, TIME_120, base=280.0, noise=5.0, seed=1)


@pytest.fixture
def gcm_tas():
    """GCM-like near-surface temperature [K], coarse grid, 120 days."""
    return _make_da(GCM_LAT, GCM_LON, TIME_120, base=281.0, noise=4.0, seed=2)


@pytest.fixture
def obs_pr():
    """ERA5-like precipitation [mm/day], fine grid, 120 days, non-negative."""
    rng = np.random.default_rng(3)
    data = np.abs(rng.standard_normal((120, 10, 12))).astype(np.float32) * 3.0
    return xr.DataArray(data, coords={"time": TIME_120, "lat": OBS_LAT, "lon": OBS_LON},
                        dims=["time", "lat", "lon"])


@pytest.fixture
def gcm_pr():
    """GCM-like precipitation [mm/day], coarse grid, 120 days, non-negative."""
    rng = np.random.default_rng(4)
    data = np.abs(rng.standard_normal((120, 5, 6))).astype(np.float32) * 4.0
    return xr.DataArray(data, coords={"time": TIME_120, "lat": GCM_LAT, "lon": GCM_LON},
                        dims=["time", "lat", "lon"])


@pytest.fixture
def obs_fine(obs_tas):
    """Alias: ERA5-like fine-grid DataArray (temperature)."""
    return obs_tas


@pytest.fixture
def gcm_coarse(gcm_tas):
    """Alias: GCM-like coarse-grid DataArray (temperature)."""
    return gcm_tas
