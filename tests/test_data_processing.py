"""Tests for data_processing.py utilities (no real files needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

from data_processing import bicubic_regrid, align_time, save_predictions, load_predictions


@pytest.fixture
def fine_da():
    lat  = np.linspace(41.0, 45.5, 10)
    lon  = np.linspace(-5.0, 0.5, 12)
    time = pd.date_range("2010-01-01", periods=60, freq="D")
    data = np.random.default_rng(0).standard_normal((60, 10, 12)).astype(np.float32) + 280
    return xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"])


@pytest.fixture
def coarse_da():
    lat  = np.linspace(41.0, 45.0, 5)
    lon  = np.linspace(-5.0, 0.0, 6)
    time = pd.date_range("2010-01-01", periods=60, freq="D")
    data = np.random.default_rng(1).standard_normal((60, 5, 6)).astype(np.float32) + 281
    return xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"])


# ── bicubic_regrid ────────────────────────────────────────────────────────────

class TestBicubicRegrid:

    def test_output_shape(self, coarse_da, fine_da):
        out = bicubic_regrid(coarse_da, fine_da.lat.values, fine_da.lon.values)
        assert out.sizes["lat"] == fine_da.sizes["lat"]
        assert out.sizes["lon"] == fine_da.sizes["lon"]
        assert out.sizes["time"] == coarse_da.sizes["time"]

    def test_output_coords(self, coarse_da, fine_da):
        out = bicubic_regrid(coarse_da, fine_da.lat.values, fine_da.lon.values)
        np.testing.assert_allclose(out.lat.values, fine_da.lat.values, atol=1e-6)
        np.testing.assert_allclose(out.lon.values, fine_da.lon.values, atol=1e-6)

    def test_no_nan_in_domain(self, coarse_da, fine_da):
        out = bicubic_regrid(coarse_da, fine_da.lat.values, fine_da.lon.values)
        assert not np.any(np.isnan(out.values)), "bicubic_regrid produced NaN values"

    def test_identity_same_grid(self, fine_da):
        # Regridding to the same grid should return approximately the same values
        out = bicubic_regrid(fine_da, fine_da.lat.values, fine_da.lon.values)
        np.testing.assert_allclose(out.values, fine_da.values, atol=1e-3)


# ── align_time ────────────────────────────────────────────────────────────────

class TestAlignTime:

    def test_identical_time_axes(self, fine_da):
        aligned = align_time({"a": fine_da, "b": fine_da.copy()})
        assert aligned["a"].sizes["time"] == 60
        assert aligned["b"].sizes["time"] == 60

    def test_partial_overlap(self, fine_da):
        da_a = fine_da                          # 2010-01-01 to 2010-03-01
        da_b = fine_da.isel(time=slice(10, 40)) # 2010-01-11 to 2010-02-09
        aligned = align_time({"a": da_a, "b": da_b})
        assert aligned["a"].sizes["time"] == aligned["b"].sizes["time"] == 30

    def test_no_overlap_raises(self):
        lat = np.linspace(41.0, 45.5, 10)
        lon = np.linspace(-5.0, 0.5, 12)
        da_a = xr.DataArray(np.zeros((5, 10, 12)),
                            coords={"time": pd.date_range("2010-01-01", periods=5, freq="D"),
                                    "lat": lat, "lon": lon},
                            dims=["time", "lat", "lon"])
        da_b = xr.DataArray(np.zeros((5, 10, 12)),
                            coords={"time": pd.date_range("2020-01-01", periods=5, freq="D"),
                                    "lat": lat, "lon": lon},
                            dims=["time", "lat", "lon"])
        with pytest.raises(ValueError, match="no common dates"):
            align_time({"a": da_a, "b": da_b})


# ── save_predictions / load_predictions ──────────────────────────────────────

class TestSavePredictions:

    def test_round_trip(self, fine_da, tmp_path):
        import torch
        variables = ["tas", "pr"]
        T, H, W = fine_da.sizes["time"], fine_da.sizes["lat"], fine_da.sizes["lon"]
        tensor = torch.randn(T, len(variables), H, W, dtype=torch.float32)

        save_predictions(tensor, variables, tmp_path, fine_da)

        # Check expected files exist
        assert (tmp_path / "predictions.pt").exists()
        assert (tmp_path / "variables.npy").exists()
        assert (tmp_path / "eval_dates.npy").exists()
        assert (tmp_path / "lat.npy").exists()
        assert (tmp_path / "lon.npy").exists()

    def test_load_round_trip(self, fine_da, tmp_path):
        import torch
        variables = ["tas", "pr"]
        T, H, W = fine_da.sizes["time"], fine_da.sizes["lat"], fine_da.sizes["lon"]
        tensor = torch.randn(T, len(variables), H, W, dtype=torch.float32)

        save_predictions(tensor, variables, tmp_path, fine_da)
        result = load_predictions(tmp_path, variables)

        assert set(result.keys()) == set(variables)
        for var in variables:
            da = result[var]
            assert da.dims == ("time", "lat", "lon")
            assert da.sizes["time"] == T
            assert da.sizes["lat"]  == H
            assert da.sizes["lon"]  == W

    def test_load_values_preserved(self, fine_da, tmp_path):
        import torch
        variables = ["tas"]
        T, H, W = fine_da.sizes["time"], fine_da.sizes["lat"], fine_da.sizes["lon"]
        tensor = torch.ones(T, 1, H, W, dtype=torch.float32) * 273.15

        save_predictions(tensor, variables, tmp_path, fine_da)
        result = load_predictions(tmp_path, variables)

        np.testing.assert_allclose(result["tas"].values, 273.15, atol=1e-4)
