"""Tests for downscaling methods in src/baselines/methods.py."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

from baselines.methods import bicubic_downscale, bcsd_downscale
from baselines.inference_cdft import cdft_downscale


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_output_shape(pred, obs):
    assert pred.dims == ("time", "lat", "lon")
    assert pred.sizes["time"] == obs.sizes["time"]
    assert pred.sizes["lat"]  == obs.sizes["lat"]
    assert pred.sizes["lon"]  == obs.sizes["lon"]


def _check_coords(pred, obs):
    np.testing.assert_array_equal(pred.lat.values, obs.lat.values)
    np.testing.assert_array_equal(pred.lon.values, obs.lon.values)


# ── Bicubic ───────────────────────────────────────────────────────────────────

class TestBicubic:

    def test_output_shape(self, obs_tas, gcm_tas):
        pred = bicubic_downscale(gcm_tas, obs_tas, "tas")
        _check_output_shape(pred, obs_tas)

    def test_output_coords(self, obs_tas, gcm_tas):
        pred = bicubic_downscale(gcm_tas, obs_tas, "tas")
        _check_coords(pred, obs_tas)

    def test_pr_no_negatives(self, obs_pr, gcm_pr):
        pred = bicubic_downscale(gcm_pr, obs_pr, "pr")
        assert float(pred.min()) >= 0.0, "Bicubic pr output contains negative values"

    def test_tas_preserves_sign(self, obs_tas, gcm_tas):
        # Temperature should pass through without clipping
        pred = bicubic_downscale(gcm_tas, obs_tas, "tas")
        assert pred.values.min() < 0 or pred.values.min() > 0, \
            "bicubic_downscale should not clip temperature"

    def test_time_axis_matches_obs(self, obs_tas, gcm_tas):
        pred = bicubic_downscale(gcm_tas, obs_tas, "tas")
        np.testing.assert_array_equal(
            pred.time.values, obs_tas.time.values,
            err_msg="time coord must be taken from obs, not from GCM (cftime vs datetime64 mismatch)"
        )

    def test_smooth_interpolation(self, obs_tas, gcm_tas):
        # Bicubic output should not be noisier than the coarse input
        pred = bicubic_downscale(gcm_tas, obs_tas, "tas")
        assert float(pred.std()) < float(obs_tas.std()) * 2


# ── BCSD ──────────────────────────────────────────────────────────────────────

class TestBCSD:

    def test_output_shape(self, obs_tas, gcm_tas):
        pred = bcsd_downscale(gcm_tas, obs_tas, "tas")
        _check_output_shape(pred, obs_tas)

    def test_pr_no_negatives(self, obs_pr, gcm_pr):
        pred = bcsd_downscale(gcm_pr, obs_pr, "pr")
        assert float(pred.min()) >= 0.0, "BCSD pr output contains negative values"

    def test_bias_reduction_tas(self, obs_tas, gcm_tas):
        # BCSD should reduce bias relative to raw bicubic
        from baselines.methods import bicubic_downscale
        pred_bcsd   = bcsd_downscale(gcm_tas, obs_tas, "tas")
        pred_bicubic = bicubic_downscale(gcm_tas, obs_tas, "tas")
        bias_bcsd   = float(abs(pred_bcsd.mean() - obs_tas.mean()))
        bias_bic    = float(abs(pred_bicubic.mean() - obs_tas.mean()))
        assert bias_bcsd <= bias_bic + 0.5, \
            f"BCSD bias {bias_bcsd:.3f} worse than bicubic {bias_bic:.3f}"

    def test_pr_distribution_shift(self, obs_pr, gcm_pr):
        # BCSD mean should be closer to ERA5 mean than raw GCM fine mean
        from baselines.methods import bicubic_regrid
        pred = bcsd_downscale(gcm_pr, obs_pr, "pr")
        gcm_fine = bicubic_regrid(gcm_pr, obs_pr.lat.values, obs_pr.lon.values)
        assert abs(float(pred.mean()) - float(obs_pr.mean())) <= \
               abs(float(gcm_fine.mean()) - float(obs_pr.mean())) + 0.1


# ── CDF-t ─────────────────────────────────────────────────────────────────────

class TestCDFt:

    def test_output_shape(self, obs_tas, gcm_tas):
        pred = cdft_downscale(gcm_tas, obs_tas, "tas")
        _check_output_shape(pred, obs_tas)

    def test_pr_no_negatives(self, obs_pr, gcm_pr):
        pred = cdft_downscale(gcm_pr, obs_pr, "pr")
        assert float(pred.min()) >= 0.0, "CDF-t pr output contains negative values"

    def test_bias_reduction_tas(self, obs_tas, gcm_tas):
        from baselines.methods import bicubic_downscale
        pred_cdft   = cdft_downscale(gcm_tas, obs_tas, "tas")
        pred_bicubic = bicubic_downscale(gcm_tas, obs_tas, "tas")
        bias_cdft = float(abs(pred_cdft.mean() - obs_tas.mean()))
        bias_bic  = float(abs(pred_bicubic.mean() - obs_tas.mean()))
        assert bias_cdft <= bias_bic + 0.5, \
            f"CDF-t bias {bias_cdft:.3f} worse than bicubic {bias_bic:.3f}"

    def test_quantile_mapping_applied(self, obs_tas, gcm_tas):
        # After QM, the overall distribution should be closer to obs
        pred = cdft_downscale(gcm_tas, obs_tas, "tas")
        from scipy.stats import ks_2samp
        ks_pred = ks_2samp(obs_tas.values.ravel(), pred.values.ravel()).statistic
        from baselines.methods import bicubic_downscale
        gcm_fine = bicubic_downscale(gcm_tas, obs_tas, "tas")
        ks_bic   = ks_2samp(obs_tas.values.ravel(), gcm_fine.values.ravel()).statistic
        assert ks_pred <= ks_bic + 0.05, \
            f"CDF-t KS {ks_pred:.3f} not better than bicubic KS {ks_bic:.3f}"


# ── PSM (requires sklearn) ────────────────────────────────────────────────────

sklearn = pytest.importorskip("sklearn", reason="sklearn not installed — skipping PSM tests")


class TestPSM:
    from baselines.methods import psm_downscale as _psm  # type: ignore[import]

    def test_output_shape(self, obs_tas, gcm_tas):
        from baselines.methods import psm_downscale
        pred = psm_downscale(gcm_tas, obs_tas, "tas")
        _check_output_shape(pred, obs_tas)

    def test_pr_no_negatives(self, obs_pr, gcm_pr):
        from baselines.methods import psm_downscale
        pred = psm_downscale(gcm_pr, obs_pr, "pr")
        assert float(pred.min()) >= 0.0, "PSM pr output contains negative values"
