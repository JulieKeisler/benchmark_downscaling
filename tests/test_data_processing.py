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

    def test_restamps_mixed_dtypes_and_timeofday(self):
        """cftime (e.g. 12:00) and np.datetime64 (e.g. 00:00) inputs must share
        one identical date-only np.datetime64 axis so xr.corr inner-joins work."""
        cftime = pytest.importorskip("cftime")
        lat = np.linspace(41.0, 45.5, 10)
        lon = np.linspace(-5.0, 0.5, 12)
        n = 30
        a_time = np.array(["2010-01-" + f"{d:02d}" for d in range(1, n + 1)],
                          dtype="datetime64[D]")
        b_time = [str(t)[:10] for t in
                  [cftime.DatetimeGregorian(2010, 1, d, 12) for d in range(1, n + 1)]]
        da_a = xr.DataArray(np.random.default_rng(0).standard_normal((n, 10, 12)),
                            coords={"time": a_time, "lat": lat, "lon": lon},
                            dims=["time", "lat", "lon"])
        da_b = xr.DataArray(np.random.default_rng(0).standard_normal((n, 10, 12)),
                            coords={"time": np.array(b_time, dtype="datetime64[ns]"),
                                    "lat": lat, "lon": lon},
                            dims=["time", "lat", "lon"])
        aligned = align_time({"a": da_a, "b": da_b})
        # Both outputs must share one identical date-only np.datetime64 axis
        assert aligned["a"].time.dtype == aligned["b"].time.dtype
        assert str(aligned["a"].time.values[0])[:10] == "2010-01-01"
        np.testing.assert_array_equal(aligned["a"].time.values, aligned["b"].time.values)
        out = xr.corr(aligned["a"], aligned["b"], dim="time")
        assert out.shape == (10, 12)
        assert not np.isnan(out.values).all(), "aligned corr should not be all-NaN"


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
