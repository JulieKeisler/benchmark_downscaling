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
    daily_climatology, harmonic_climatology, seasonal_corr, daily_anomalies,
    lag1_autocorr_anom_mae,
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

class TestSeasonalCorr:
    """seasonal_corr: shape of the harmonic 365-day climatology, offset and amplitude ignored."""

    @staticmethod
    def _cycle(phase_days=0.0, offset=0.0, amp=10.0, noise=1.0, seed=0):
        import pandas as pd
        time = pd.date_range("2000-01-01", "2003-12-31", freq="D")   # 2000 is a leap year
        doy = time.dayofyear.values
        rng = np.random.default_rng(seed)
        base = 280 + offset + amp * np.sin(2 * np.pi * (doy - phase_days) / 365.0)
        data = base[:, None, None] + noise * rng.standard_normal((len(time), 3, 4))
        return xr.DataArray(data, coords={"time": time, "lat": [1.0, 2.0, 3.0], "lon": [1.0, 2.0, 3.0, 4.0]},
                            dims=["time", "lat", "lon"])

    def test_climatology_has_365_days_without_feb29(self):
        assert daily_climatology(self._cycle()).sizes["doy"] == 365

    def test_smoothing_keeps_a_constant(self):
        da = self._cycle(amp=0.0) * 0 + 5.0
        assert np.allclose(daily_climatology(da).values, 5.0)

    def test_harmonic_fit_recovers_a_sinusoid(self):
        clim = harmonic_climatology(self._cycle(noise=0.0))
        d = np.arange(1, 366)
        # the series' day of year runs 1..365 (Feb 29 left out), a sinusoid on 365 days is in the basis
        expected = 280 + 10 * np.sin(2 * np.pi * d / 365.0)
        assert np.allclose(clim.isel(lat=0, lon=0).values, expected, atol=0.05)

    def test_perfect(self):
        obs = self._cycle()
        assert seasonal_corr(obs, obs.copy()) == pytest.approx(1.0)

    def test_offset_and_amplitude_ignored(self):
        obs = self._cycle(seed=0)
        assert seasonal_corr(obs, obs * 1.5 + 3.0) == pytest.approx(1.0)

    def test_half_year_shift_anticorrelates(self):
        obs, pred = self._cycle(seed=0), self._cycle(phase_days=182.5, seed=1)
        assert seasonal_corr(obs, pred) < -0.9


class TestPersistence:
    """lag1_autocorr_anom_mae: persistence of daily anomalies, the seasonal cycle removed."""

    @staticmethod
    def _ar1(phi, amp=0.0, seed=0):
        import pandas as pd
        time = pd.date_range("2000-01-01", "2009-12-31", freq="D")
        rng = np.random.default_rng(seed)
        e = rng.standard_normal((len(time), 2, 2))
        x = np.zeros_like(e)
        for t in range(1, len(time)):
            x[t] = phi * x[t - 1] + e[t]
        cycle = amp * np.sin(2 * np.pi * time.dayofyear.values / 365.0)
        return xr.DataArray(x + cycle[:, None, None], dims=["time", "lat", "lon"],
                            coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0]})

    def test_perfect(self):
        a = self._ar1(0.5)
        assert lag1_autocorr_anom_mae(a, a.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_seasonal_cycle_does_not_count(self):
        # same anomalies, one with a large seasonal cycle: the raw lag-1 differs a lot, this score does not
        a, b = self._ar1(0.5), self._ar1(0.5, amp=20.0)
        assert lag1_autocorr_anom_mae(a, b) < 0.02
        assert float(np.abs(lag1_autocorr(a) - lag1_autocorr(b)).mean()) > 0.3

    def test_detects_a_persistence_error(self):
        err = lag1_autocorr_anom_mae(self._ar1(0.3, seed=0), self._ar1(0.7, seed=1))
        assert 0.3 < err < 0.5

    def test_anomalies_have_no_cycle(self):
        an = daily_anomalies(self._ar1(0.0, amp=20.0))
        assert abs(float(daily_climatology(an, window=31).std())) < 0.3


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


class TestFineSpectrum:
    """fine_spectrum_metric: 0 for identical fields, negative bias when smoothed."""

    def _field(self, smooth: bool = False):
        import numpy as np
        import xarray as xr
        from scipy.ndimage import uniform_filter
        rng = np.random.default_rng(0)
        x = rng.standard_normal((30, 57, 65)).astype(np.float32)
        if smooth:
            x = uniform_filter(x, size=(1, 7, 7))
        return xr.DataArray(x, dims=("time", "lat", "lon"),
                            coords={"time": np.arange(30),
                                    "lat": np.arange(55.0, 40.75, -0.25),
                                    "lon": np.arange(-6.0, 10.25, 0.25)})

    def test_identical_is_zero(self):
        from metrics import fine_spectrum_metric
        f = self._field()
        score, bias, data = fine_spectrum_metric(f, f)
        assert score == 0.0 and bias == 0.0
        assert min(data["wavelength_km"]) < 60 and max(data["wavelength_km"]) > 500

    def test_smoothed_is_negative(self):
        from metrics import fine_spectrum_metric
        score, bias, _ = fine_spectrum_metric(self._field(), self._field(smooth=True))
        assert bias < -0.5 and score == -bias


class TestAccDaily:
    """acc_daily_map: correlation of daily anomalies with the GCM, cycle and trend left out."""

    @staticmethod
    def _series(years=(2000, 2039), trend_per_year=0.0, noise_seed=0, amp=10.0):
        import pandas as pd
        time = pd.date_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D")
        doy = time.dayofyear.values
        t_years = (time.year.values - years[0]) + doy / 365.0
        base = 280 + amp * np.sin(2 * np.pi * doy / 365.0) + trend_per_year * t_years
        rng = np.random.default_rng(noise_seed)
        data = base[:, None, None] + rng.standard_normal((len(time), 2, 2))
        return xr.DataArray(data, coords={"time": time, "lat": [1.0, 2.0], "lon": [1.0, 2.0]},
                            dims=["time", "lat", "lon"])

    def test_identical_is_one(self):
        from metrics import acc_daily_map
        g = self._series()
        assert np.allclose(acc_daily_map(g, g.copy()).values, 1.0)

    def test_shared_cycle_not_counted(self):
        from metrics import mean_acc_daily, mean_temporal_corr
        g, p = self._series(noise_seed=0), self._series(noise_seed=1)
        assert mean_temporal_corr(g, p) > 0.9          # the raw daily correlation is the cycle
        assert abs(mean_acc_daily(g, p)) < 0.05

    def test_shared_trend_mostly_removed_by_blocks(self):
        # A shared ramp of total size T adds T^2/12 of common variance; blocks cut T per block.
        from metrics import mean_acc_daily
        g = self._series(trend_per_year=0.2, noise_seed=0)
        p = self._series(trend_per_year=0.2, noise_seed=1)
        full = mean_acc_daily(g, p, block_years=None)        # 8 K ramp vs unit noise: ~0.84
        assert full == pytest.approx(64 / 12 / (64 / 12 + 1), abs=0.05)
        assert mean_acc_daily(g, p, block_years=20) == pytest.approx(16 / 12 / (16 / 12 + 1), abs=0.05)
        assert abs(mean_acc_daily(g, p, block_years=5)) < 0.15

    def test_short_last_block_joins_previous(self):
        from metrics import block_daily_anomalies, daily_anomalies
        g = self._series(years=(2000, 2020), noise_seed=3)    # 2020 alone would be a 1-year block
        assert np.allclose(block_daily_anomalies(g, 20).values, daily_anomalies(g).values)


class TestExtremeMaps:
    """extreme_maps: per-cell tails and the map extreme_freq_bias averages."""

    def test_shift_moves_quantiles(self, biased):
        from metrics import extreme_maps
        obs, pred = biased
        m = extreme_maps(obs.astype("float64"), (obs + 1.0).astype("float64"))
        assert np.allclose(m["pred_q99"] - m["obs_q99"], 1.0)
        assert np.allclose(m["pred_q01"] - m["obs_q01"], 1.0)

    def test_no_low_tail_when_asked(self, perfect):
        from metrics import extreme_maps
        obs, pred = perfect
        assert "obs_q01" not in extreme_maps(obs, pred, q_low=None)

    def test_frequency_map_is_what_extreme_freq_averages(self, biased):
        from metrics import extreme_maps
        obs, pred = biased
        m = extreme_maps(obs, pred)
        assert np.mean(np.abs(m["pred_freq95"] - m["obs_freq95"])) == pytest.approx(
            extreme_freq_bias(obs, pred))


class TestSAL:
    """Object-based SAL quantities on hand-built rain fields."""

    LAT = np.arange(50.0, 44.75, -0.25)   # 21 cells, north to south like ERA5
    LON = np.arange(0.0, 6.25, 0.25)      # 25 cells

    def _series(self, fields):
        import pandas as pd
        time = pd.date_range("2000-01-01", periods=len(fields), freq="D")
        return xr.DataArray(np.stack(fields).astype("float64"),
                            coords={"time": time, "lat": self.LAT, "lon": self.LON},
                            dims=["time", "lat", "lon"])

    def _blank(self):
        return np.zeros((self.LAT.size, self.LON.size))

    def test_objects_split_and_diagonal_join(self):
        from metrics import find_objects
        f = self._blank()
        f[2:4, 2:4] = 10.0
        f[10:12, 15:17] = 10.0
        assert find_objects(f)[1] == 2
        f[4, 4] = 10.0                         # touches the first block diagonally only
        assert find_objects(f)[1] == 2
        g = self._blank()
        g[5, 5] = g[6, 6] = 10.0
        assert find_objects(g)[1] == 1

    def test_flat_and_peaked_objects(self):
        from metrics import sal_daily
        flat = self._blank()
        flat[5:8, 5:8] = 4.0                   # 9 cells, flat: V = 9, flatness 1
        peaked = self._blank()
        peaked[5:8, 5:8] = 1.0
        peaked[6, 6] = 10.0                    # V = (8 + 10) / 10 = 1.8, flatness 0.2
        q = sal_daily(self._series([flat, peaked]))
        assert q["V"][0] == pytest.approx(9.0) and q["V"][1] == pytest.approx(1.8)
        assert q["obj_flat"] == pytest.approx([1.0, 0.2])
        assert np.nanmax(q["cell_flat"][1]) == pytest.approx(0.2)
        assert q["n_obj"].tolist() == [1, 1]

    def test_identical_series_score_zero(self):
        from metrics import grid_geometry, sal_cell_maps, sal_daily, sal_distribution_scores
        rng = np.random.default_rng(0)
        fields = [np.where(rng.random(self._blank().shape) > 0.7, rng.gamma(2, 3, self._blank().shape), 0.0)
                  for _ in range(40)]
        q = sal_daily(self._series(fields))
        d = grid_geometry(self.LAT, self.LON)[3]
        s = sal_distribution_scores(q, q, d)
        assert s["sal_S"] == 0 and s["sal_A"] == 0 and s["sal_L"] == 0 and s["sal_ks_area"] == 0
        _, cell = sal_cell_maps(q, q)
        assert cell["sal_cell_freq"] == 0 and cell["sal_cell_ks_area"] == 0

    def test_spread_rain_has_positive_structure(self):
        from metrics import grid_geometry, sal_daily, sal_distribution_scores
        small, wide = [], []
        for k in range(10):
            a, b = self._blank(), self._blank()
            a[8:10, 8 + k % 3:10 + k % 3] = 9.0      # 4 cells x 9 mm
            b[6:12, 6 + k % 3:12 + k % 3] = 1.0      # 36 cells x 1 mm: same total, wider and flatter
            small.append(a)
            wide.append(b)
        qo, qp = sal_daily(self._series(small)), sal_daily(self._series(wide))
        s = sal_distribution_scores(qo, qp, grid_geometry(self.LAT, self.LON)[3])
        assert s["sal_S"] > 1.0
        assert abs(s["sal_A"]) < 1e-12
        assert s["sal_L"] < 0.02

    def test_shifted_rain_has_location_error_only(self):
        from metrics import grid_geometry, sal_daily, sal_distribution_scores
        obs, pred = [], []
        for k in range(10):
            a = self._blank()
            a[8:11, 4:7] = 5.0 + k
            obs.append(a)
            pred.append(np.roll(a, 12, axis=1))       # 12 cells (3 degrees) east, away from the edge
        qo, qp = sal_daily(self._series(obs)), sal_daily(self._series(pred))
        x, y, _, d = grid_geometry(self.LAT, self.LON)
        s = sal_distribution_scores(qo, qp, d)
        assert abs(s["sal_S"]) < 0.02 and abs(s["sal_A"]) < 1e-12
        assert s["sal_L1"] == pytest.approx(abs(x[9, 17] - x[9, 5]) / d, rel=0.02)


def test_sal_ks_equals_scipy_statistic():
    from scipy.stats import ks_2samp
    from metrics.object_metrics import _ks
    rng = np.random.default_rng(0)
    a, b = rng.gamma(2, 3, 500), np.round(rng.gamma(2.2, 3, 700))   # ties included
    b[:20] = np.nan
    assert _ks(a, b) == pytest.approx(ks_2samp(a, b[np.isfinite(b)]).statistic, abs=1e-12)
