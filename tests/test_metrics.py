"""Tests for all metric functions in src/metrics/."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import (
    bias, clim_rmse, spatial_mean_abs_bias, std_error, ks_stat,
    seasonal_mae, extreme_freq_bias, extreme_intensity_bias,
    temporal_corr_map, mean_temporal_corr, mean_temporal_corr_annual,
    lag1_autocorr, lag1_autocorr_mae, annual_anomaly_bias,
    intervar_corr_mae, spatial_spectrum_metric,
)


# ── Fixtures (inline — don't depend on conftest for isolation) ────────────────

TIME = xr.DataArray(np.arange(120), dims=["time"])


def _da(data, lat_n=10, lon_n=12, t=120):
    import pandas as pd
    lat = np.linspace(41.0, 45.5, lat_n)
    lon = np.linspace(-5.0, 0.5, lon_n)
    time = pd.date_range("2010-01-01", periods=t, freq="D")
    return xr.DataArray(data.astype(np.float32),
                        coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"])


def _const(val, **kw):
    return _da(np.full((120, 10, 12), val), **kw)


@pytest.fixture
def perfect():
    """obs == pred — all error metrics should be zero."""
    rng = np.random.default_rng(0)
    data = (280 + 5 * rng.standard_normal((120, 10, 12))).astype(np.float32)
    da = _da(data)
    return da, da.copy()  # (obs, pred) identical


@pytest.fixture
def biased():
    """pred = obs + 2 — constant warm bias."""
    rng = np.random.default_rng(1)
    data = (280 + 5 * rng.standard_normal((120, 10, 12))).astype(np.float32)
    obs  = _da(data)
    pred = _da(data + 2.0)
    return obs, pred


# ── scalar_metrics_era5 ───────────────────────────────────────────────────────

class TestScalarMetricsERA5:

    def test_bias_perfect(self, perfect):
        obs, pred = perfect
        b = bias(obs, pred)
        np.testing.assert_allclose(b.values, 0.0, atol=1e-5)

    def test_bias_warm(self, biased):
        obs, pred = biased
        b = bias(obs, pred)
        np.testing.assert_allclose(float(b.mean()), 2.0, atol=1e-3)

    def test_bias_returns_spatial_map(self, biased):
        obs, pred = biased
        b = bias(obs, pred)
        assert set(b.dims) == {"lat", "lon"}

    def test_clim_rmse_perfect(self, perfect):
        obs, pred = perfect
        assert clim_rmse(obs, pred) == pytest.approx(0.0, abs=1e-5)

    def test_clim_rmse_biased(self, biased):
        obs, pred = biased
        assert clim_rmse(obs, pred) == pytest.approx(2.0, abs=1e-4)

    def test_spatial_mean_abs_bias_perfect(self, perfect):
        obs, pred = perfect
        assert spatial_mean_abs_bias(obs, pred) == pytest.approx(0.0, abs=1e-5)

    def test_std_error_perfect(self, perfect):
        obs, pred = perfect
        assert std_error(obs, pred) == pytest.approx(0.0, abs=1e-5)

    def test_ks_stat_perfect(self, perfect):
        obs, pred = perfect
        assert ks_stat(obs, pred) == pytest.approx(0.0, abs=1e-5)

    def test_ks_stat_biased(self, biased):
        obs, pred = biased
        # Shifted distribution should have non-zero KS
        assert ks_stat(obs, pred) > 0.1

    def test_ks_stat_upper_tail(self, biased):
        obs, pred = biased
        stat = ks_stat(obs, pred, quantile=0.95)
        assert isinstance(stat, float)

    def test_seasonal_mae_perfect(self, perfect):
        obs, pred = perfect
        assert seasonal_mae(obs, pred) == pytest.approx(0.0, abs=1e-5)

    def test_seasonal_mae_biased(self, biased):
        obs, pred = biased
        assert seasonal_mae(obs, pred) == pytest.approx(2.0, abs=1e-4)

    def test_extreme_freq_bias_perfect(self, perfect):
        obs, pred = perfect
        val = extreme_freq_bias(obs, pred)
        assert abs(val) < 1e-3

    def test_extreme_intensity_bias_perfect(self, perfect):
        obs, pred = perfect
        val = extreme_intensity_bias(obs, pred)
        assert abs(val) < 0.5  # identical obs/pred → zero bias (tolerant for NaN edge)

    def test_extreme_freq_bias_warm_direction(self, biased):
        obs, pred = biased
        # Warm pred → more values exceed obs threshold → |freq bias| > 0
        val = extreme_freq_bias(obs, pred)
        assert val > 0.0


# ── scalar_metrics_gcm ────────────────────────────────────────────────────────

class TestScalarMetricsGCM:

    def _gcm_on_era5(self):
        """GCM already on ERA5 grid (precondition for gcm metrics)."""
        rng = np.random.default_rng(10)
        import pandas as pd
        lat = np.linspace(41.0, 45.5, 10)
        lon = np.linspace(-5.0, 0.5, 12)
        time = pd.date_range("2010-01-01", periods=120, freq="D")
        data_gcm  = (281 + 4 * rng.standard_normal((120, 10, 12))).astype(np.float32)
        data_pred = data_gcm + rng.standard_normal((120, 10, 12)).astype(np.float32) * 0.5
        gcm  = xr.DataArray(data_gcm,  coords={"time": time, "lat": lat, "lon": lon}, dims=["time","lat","lon"])
        pred = xr.DataArray(data_pred, coords={"time": time, "lat": lat, "lon": lon}, dims=["time","lat","lon"])
        return gcm, pred

    def test_temporal_corr_map_perfect(self):
        gcm, _ = self._gcm_on_era5()
        r = temporal_corr_map(gcm, gcm)
        np.testing.assert_allclose(r.values, 1.0, atol=1e-4)

    def test_temporal_corr_map_shape(self):
        gcm, pred = self._gcm_on_era5()
        r = temporal_corr_map(gcm, pred)
        assert r.dims == ("lat", "lon")

    def test_mean_temporal_corr_range(self):
        gcm, pred = self._gcm_on_era5()
        val = mean_temporal_corr(gcm, pred)
        assert -1.0 <= val <= 1.0

    def test_mean_temporal_corr_annual_range(self):
        # Needs ≥ 2 years for annual correlation to be non-NaN
        import pandas as pd
        rng = np.random.default_rng(11)
        lat = np.linspace(41.0, 45.5, 10)
        lon = np.linspace(-5.0, 0.5, 12)
        time = pd.date_range("2008-01-01", periods=730, freq="D")  # 2 years
        data_gcm  = (281 + 4 * rng.standard_normal((730, 10, 12))).astype(np.float32)
        data_pred = data_gcm + rng.standard_normal((730, 10, 12)).astype(np.float32) * 0.5
        gcm2  = xr.DataArray(data_gcm,  coords={"time": time, "lat": lat, "lon": lon}, dims=["time","lat","lon"])
        pred2 = xr.DataArray(data_pred, coords={"time": time, "lat": lat, "lon": lon}, dims=["time","lat","lon"])
        val = mean_temporal_corr_annual(gcm2, pred2)
        assert -1.0 <= val <= 1.0

    def test_lag1_autocorr_perfect(self):
        gcm, _ = self._gcm_on_era5()
        r = lag1_autocorr(gcm)
        assert r.dims == ("lat", "lon")
        # Autocorrelation of random process should be near zero
        assert float(r.mean()) < 0.5

    def test_lag1_autocorr_mae_perfect(self):
        gcm, _ = self._gcm_on_era5()
        val = lag1_autocorr_mae(gcm, gcm)
        assert val == pytest.approx(0.0, abs=1e-5)

    def test_annual_anomaly_bias_perfect(self):
        gcm, _ = self._gcm_on_era5()
        val = annual_anomaly_bias(gcm, gcm)
        assert val == pytest.approx(0.0, abs=1e-5)


# ── inter_variables_metrics ───────────────────────────────────────────────────

class TestInterVariablesMetrics:

    def _two_vars(self):
        import pandas as pd
        rng = np.random.default_rng(20)
        lat = np.linspace(41.0, 45.5, 10)
        lon = np.linspace(-5.0, 0.5, 12)
        time = pd.date_range("2010-01-01", periods=120, freq="D")
        base = rng.standard_normal((120, 10, 12)).astype(np.float32)
        da1_obs  = xr.DataArray(base,       coords={"time":time,"lat":lat,"lon":lon}, dims=["time","lat","lon"])
        da2_obs  = xr.DataArray(base * 0.9, coords={"time":time,"lat":lat,"lon":lon}, dims=["time","lat","lon"])
        da1_pred = xr.DataArray(base + 0.1, coords={"time":time,"lat":lat,"lon":lon}, dims=["time","lat","lon"])
        da2_pred = xr.DataArray(base * 0.9, coords={"time":time,"lat":lat,"lon":lon}, dims=["time","lat","lon"])
        return da1_obs, da2_obs, da1_pred, da2_pred

    def test_perfect_intervar_corr(self):
        o1, o2, _, _ = self._two_vars()
        val = intervar_corr_mae(o1, o2, o1, o2)
        assert val == pytest.approx(0.0, abs=1e-4)

    def test_intervar_corr_nonneg(self):
        o1, o2, p1, p2 = self._two_vars()
        val = intervar_corr_mae(o1, o2, p1, p2)
        assert val >= 0.0


# ── spatial_spectrum_metrics ──────────────────────────────────────────────────

class TestSpatialSpectrumMetrics:

    # spatial_spectrum_metric returns (score: float, plot_data: dict)
    def _field(self, noise=1.0, seed=30):
        import pandas as pd
        rng = np.random.default_rng(seed)
        # Use a larger grid so wavenumber bins have enough coverage
        lat = np.linspace(41.0, 55.0, 29)   # 0.5° × 29 pts
        lon = np.linspace(-6.0, 10.0, 33)   # 0.5° × 33 pts
        time = pd.date_range("2010-01-01", periods=30, freq="D")
        data = (280 + noise * rng.standard_normal((30, 29, 33))).astype(np.float32)
        return xr.DataArray(data, coords={"time":time,"lat":lat,"lon":lon}, dims=["time","lat","lon"])

    def test_perfect_spectrum(self):
        obs = self._field()
        score, _ = spatial_spectrum_metric(obs, obs)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_smoother_pred_nonzero(self):
        obs  = self._field(noise=3.0)
        pred = self._field(noise=0.5, seed=31)
        score, _ = spatial_spectrum_metric(obs, pred)
        assert score > 0.0

    def test_returns_float(self):
        obs  = self._field()
        pred = self._field(seed=32)
        score, plot_data = spatial_spectrum_metric(obs, pred)
        assert isinstance(score, float)
        assert "k_centers_km" in plot_data
