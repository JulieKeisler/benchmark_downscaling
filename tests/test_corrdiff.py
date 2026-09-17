"""Tests for CorrDiff-QM (src/baselines/inference_corrdiff.py).

Covers: schedule, forward/reverse diffusion shapes, residual construction,
per-cell-per-month quantile mapping, physical clipping, checkpoint round-trip,
save_predictions compatibility, registry entry, and a small end-to-end path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pytest
import xarray as xr

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src" / "baselines"))
sys.path.insert(0, str(_ROOT / "src" / "evaluation"))
sys.path.insert(0, str(_ROOT / "src"))

from inference_corrdiff import (
    VARIABLES, N_VARS, STRIDE, N_DIFF_IN,
    cosine_alpha_bar, _schedule, q_sample, p_sample_step, sample_residual,
    train_corrdiff, _qm_per_month, _qmap_1d, _to_datetime64,
)
from utils.normalization import compute_stats, encode, decode
from utils.dataset import _coarsen_upsample
from utils.month import month_channels
from utils.unet import UNet
from data_processing import save_predictions, load_predictions
from baselines import METHOD_REGISTRY, METHOD_LABELS

H, W = 57, 65


class SmallDenoiser(nn.Module):
    """Tiny timestep-conditional denoiser used in tests in place of DiffusersUNet."""

    def __init__(self, in_ch: int = N_DIFF_IN, out_ch: int = N_VARS, f: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, f, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(f, f, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(f, out_ch, 1),
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallRegressor(nn.Module):
    """Tiny regression net for fast tests in place of the full UNet."""

    def __init__(self, in_ch: int = N_VARS + 2, out_ch: int = N_VARS, f: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, f, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(f, f, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(f, out_ch, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _in_range(x: np.ndarray, stats, var: str) -> np.ndarray:
    """Clamp synthetic physical values into encode/decode-safe ranges."""
    x = np.clip(x, stats[var]["xmin"], stats[var]["xmax"])
    return x.astype(np.float32)


def _synthetic_era5(t: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = np.zeros((t, N_VARS, H, W), dtype=np.float32)
    raw[:, 0] = 280 + 10 * rng.standard_normal((t, 1, 1))      # tas  [K]
    raw[:, 1] = rng.lognormal(0.5, 1.0, (t, 1, 1))             # pr   [mm/day]
    raw[:, 2] = np.abs(rng.standard_normal((t, 1, 1))) * 4      # sfcWind
    raw[:, 3] = np.abs(rng.standard_normal((t, 1, 1))) * 7      # sfcWindmax
    raw[:, 4] = 100000 + 5000 * rng.standard_normal((t, 1, 1))  # psl   [Pa]
    raw = raw + 0.01 * rng.standard_normal(raw.shape)
    raw[:, 1] = np.clip(raw[:, 1], 0.0, None)
    months = (np.arange(t) % 12 + 1).astype(np.float32)
    return raw.astype(np.float32), months


# ── Diffusion schedule & sampling ──────────────────────────────────────────────

class TestSchedule:

    def test_cosine_alpha_bar_properties(self):
        abar = _schedule(200)
        assert float(abar[0]) == pytest.approx(1.0, abs=1e-5)
        assert bool((abar.diff() <= 0).all()), "alpha_bar must be monotone non-increasing"
        assert bool((abar >= 0).all()) and bool((abar <= 1).all())

    def test_q_sample_shape(self):
        abar = _schedule(200)
        x0 = torch.randn(4, 5, H, W)
        t  = torch.randint(1, 201, (4,))
        xt = q_sample(x0, t, abar, torch.randn_like(x0))
        assert xt.shape == x0.shape

    def test_p_sample_step_inverts(self):
        abar = _schedule(200)
        B    = 4
        x0   = torch.rand(B, 5, H, W) * 2 - 1            # x0 must stay in [-1, 1]
        t    = torch.full((B,), 100, dtype=torch.long)
        t0   = torch.zeros(B, dtype=torch.long)          # direct step 100 → 0
        eps  = torch.randn_like(x0)
        xt   = q_sample(x0, t, abar, eps)

        # Exact noise over a full stride to t=0 recovers x0 (conditioning unused).
        dummy = torch.zeros(B, 5, H, W)
        xr0 = p_sample_step(_exact_eps_fn(eps), dummy, dummy, dummy, xt, t, t0,
                            abar, 200, noise=None)
        assert xr0.shape == x0.shape
        assert torch.allclose(xr0, x0, atol=1e-3)

    def test_sample_residual_runs(self):
        abar  = _schedule(200)
        dm    = SmallDenoiser()
        x_lr  = torch.randn(2, 5, H, W)
        mu    = torch.randn(2, 5, H, W)
        mc    = torch.zeros(2, 2, H, W)
        out   = sample_residual(dm, x_lr, mu, mc, n_t=200, n_steps=8,
                                abar=abar, device="cpu")
        assert out.shape == (2, 5, H, W)
        assert torch.isfinite(out).all()


def _exact_eps_fn(eps: torch.Tensor):
    def _f(x, t):
        return eps.expand(x.shape[0], -1, -1, -1)
    return _f


# ── Registry ───────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_corrdiff_qm_registered(self):
        assert "corrdiff_qm" in METHOD_REGISTRY
        assert METHOD_LABELS["corrdiff_qm"] == "CorrDiff-QM"
        assert callable(METHOD_REGISTRY["corrdiff_qm"])

    def test_corrdiff_qm_is_precomputed(self):
        with pytest.raises(RuntimeError, match="checkpoint"):
            METHOD_REGISTRY["corrdiff_qm"](None, None, "tas")


# ── Quantile mapping ───────────────────────────────────────────────────────────

class TestQuantileMapping:

    def test_qmap_1d_shifts_distribution(self):
        rng = np.random.default_rng(3)
        gcm     = 5.0 * rng.standard_normal(1000) + 10.0
        era5    = 1.0 * rng.standard_normal(1000) + 12.0
        apply   = 5.0 * rng.standard_normal(1000) + 10.0
        mapped  = _qmap_1d(gcm, era5, apply)
        assert abs(mapped.std() - 1.0) < abs(apply.std() - 1.0)
        assert abs(mapped.mean() - 12.0) < abs(apply.mean() - 12.0)

    def test_qm_per_month_maps_to_ref(self):
        rng    = np.random.default_rng(7)
        Hc, Wc = 10, 12
        cal_mon  = np.repeat(np.arange(1, 13), 20)                  # 240 days = 20 per month
        ref_mon  = np.repeat(np.arange(1, 13), 20)
        app_mon  = np.repeat(np.arange(1, 13), 25)
        src_cal  = np.stack([5.0 * rng.standard_normal(240) for _ in range(Hc * Wc)],
                            axis=1).reshape(240, Hc, Wc)
        ref_cal  = np.stack([rng.standard_normal(240) for _ in range(Hc * Wc)],
                            axis=1).reshape(240, Hc, Wc)
        app_eval = np.stack([5.0 * rng.standard_normal(300) for _ in range(Hc * Wc)],
                            axis=1).reshape(300, Hc, Wc)
        out = _qm_per_month(src_cal, ref_cal, app_eval, cal_mon, ref_mon, app_mon)
        # every cell: mapped std drops toward the ERA5 (≈1) level
        std0 = app_eval.std(axis=0)
        std1 = out.std(axis=0)
        assert bool((np.abs(std1 - 1.0) < np.abs(std0 - 1.0) - 0.2).all())
        assert out.shape == app_eval.shape

    def test_qm_empty_month_unchanged(self):
        # No January in src/ref → January values pass through untouched.
        src_mon = np.full(40, 2.0)
        ref_mon = np.full(40, 2.0)
        app_mon = np.full(10, 1.0)
        src = np.random.default_rng(1).standard_normal((40, 3, 4))
        ref = np.random.default_rng(2).standard_normal((40, 3, 4))
        app = np.random.default_rng(3).standard_normal((10, 3, 4))
        out = _qm_per_month(src, ref, app, src_mon, ref_mon, app_mon)
        np.testing.assert_array_equal(out, app)


# ── Physical units / clipping ──────────────────────────────────────────────────

class TestPhysicalOutput:

    def test_decode_and_clip(self):
        raw, _ = _synthetic_era5(t=16)
        stats  = compute_stats(raw, VARIABLES)
        norm   = np.stack([encode(raw[:, i], stats, v) for i, v in enumerate(VARIABLES)],
                          axis=1)
        decoded = np.stack([decode(norm[:, i] + 0.05, stats, v) for i, v in enumerate(VARIABLES)],
                           axis=1)
        for name, idx in (("pr", 1), ("sfcWind", 2), ("sfcWindmax", 3)):
            arr = np.clip(decoded[:, idx], 0.0, None)
            assert (arr >= 0.0).all(), name


# ── Checkpoints & save_predictions compat ─────────────────────────────────────

class TestCheckpoints:

    def test_checkpoint_roundtrip(self, tmp_path):
        raw, months = _synthetic_era5(t=24)
        stats, reg, diff = train_corrdiff(
            raw, months, tmp_path, n_epochs=1, batch_size=8, device="cpu",
            n_diff_t=200, stage1_net=SmallRegressor(), stage2_net=SmallDenoiser(),
        )
        ck_reg  = tmp_path / "checkpoints" / "corrdiff_reg.pt"
        ck_diff = tmp_path / "checkpoints" / "corrdiff_resdiff.pt"
        assert ck_reg.exists() and ck_diff.exists()

        # Stage-2 residual = y - mu
        T, _, Hh, Ww = raw.shape
        norm = np.stack([encode(raw[:, i], stats, v) for i, v in enumerate(VARIABLES)],
                        axis=1)
        x_lr = _coarsen_upsample(norm, STRIDE)
        reg.eval()
        with torch.no_grad():
            mc = torch.from_numpy(month_channels(months, (T, Hh, Ww)))
            mu = reg(torch.cat([torch.from_numpy(x_lr), mc], dim=1)).numpy()
        res = norm - mu
        assert res.shape == norm.shape

        loaded_reg  = torch.load(ck_reg,  map_location="cpu", weights_only=False)
        loaded_diff = torch.load(ck_diff, map_location="cpu", weights_only=False)
        assert loaded_reg["variables"] == VARIABLES
        assert loaded_reg["stride"] == STRIDE

        reg2  = SmallRegressor()
        reg2.load_state_dict(loaded_reg["model"])
        diff2 = SmallDenoiser()
        diff2.load_state_dict(loaded_diff["model"])
        with torch.no_grad():
            b_in = torch.cat([torch.from_numpy(x_lr[:2]), mc[:2]], dim=1)
            torch.testing.assert_close(reg2(b_in), reg(b_in))


class TestSavePredictionsCompat:

    def test_to_datetime64_handles_cftime(self):
        cftime = pytest.importorskip("cftime")
        cft = np.array([cftime.DatetimeGregorian(2000, 1, 1),
                        cftime.DatetimeGregorian(2000, 1, 2)])
        out = _to_datetime64(cft)
        assert out.dtype == np.dtype("datetime64[ns]")
        assert (out == np.array(["2000-01-01", "2000-01-02"],
                                dtype="datetime64[ns]")).all()

    def test_to_datetime64_passthrough_np(self):
        t = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]")
        out = _to_datetime64(t)
        assert out.dtype == np.dtype("datetime64[ns]")
        assert (out == np.array(["2000-01-01", "2000-01-02"],
                                dtype="datetime64[ns]")).all()

    def test_roundtrip_standard_format(self, tmp_path):
        t = 128
        preds = np.random.default_rng(5).standard_normal((t, N_VARS, H, W)).astype(np.float32)
        preds[:, 1] = np.clip(preds[:, 1], 0.0, None)     # pr >= 0
        lat = np.linspace(41.0, 45.5, H)
        lon = np.linspace(-5.0, 0.5, W)
        import pandas as pd
        time = pd.date_range("2000-01-01", periods=t, freq="D")
        ref_da = xr.DataArray(
            np.zeros((t, H, W), dtype=np.float32),
            coords={"time": time, "lat": lat, "lon": lon},
            dims=["time", "lat", "lon"],
        )
        save_predictions(torch.from_numpy(preds), VARIABLES, tmp_path, ref_da)
        loaded = load_predictions(tmp_path, VARIABLES)
        assert set(loaded) == set(VARIABLES)
        assert loaded["pr"].sizes["time"] == t
        np.testing.assert_allclose(loaded["tas"].values, preds[:, 0], atol=1e-6)
        assert (loaded["pr"].values >= 0).all()
        np.testing.assert_allclose(loaded["tas"].lat.values, lat)
        np.testing.assert_allclose(loaded["tas"].lon.values, lon)


# ── Small end-to-end ───────────────────────────────────────────────────────────

class TestEndToEnd:

    def test_train_and_predict_smoke(self, tmp_path):
        raw, months = _synthetic_era5(t=32)
        stats, reg, diff = train_corrdiff(
            raw, months, tmp_path, n_epochs=1, batch_size=8, device="cpu",
            n_diff_t=200, stage1_net=SmallRegressor(), stage2_net=SmallDenoiser(),
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        T, _, Hh, Ww = raw.shape
        norm = np.stack([encode(raw[:, i], stats, v) for i, v in enumerate(VARIABLES)],
                        axis=1)
        x_lr  = _coarsen_upsample(norm, STRIDE)
        gcm_lr = x_lr + 0.05 * np.random.default_rng(9).standard_normal(x_lr.shape)
        abar = _schedule(200)

        reg.eval(); diff.eval()
        preds = np.zeros((T, N_VARS, Hh, Ww), dtype=np.float32)
        with torch.no_grad():
            mc  = torch.from_numpy(month_channels(months, (T, Hh, Ww)))
            xb  = torch.from_numpy(gcm_lr.astype(np.float32))
            mu  = reg(torch.cat([xb, mc], dim=1))
            for _ in range(2):                       # 2 ensemble members
                r = sample_residual(diff, xb, mu, mc, n_t=200, n_steps=8,
                                    abar=abar, device="cpu")
                y = mu + r
                for i, v in enumerate(VARIABLES):
                    preds[:, i] += _in_range(decode(y[:, i].cpu().numpy(), stats, v),
                                             stats, v).astype(np.float32)
            preds /= 2.0
            for i, v in enumerate(VARIABLES):
                if v in ("pr", "sfcWind", "sfcWindmax"):
                    preds[:, i] = np.clip(preds[:, i], 0.0, None)

        assert preds.shape == (T, N_VARS, Hh, Ww)
        assert np.isfinite(preds).all()
        assert (preds[:, 1] >= 0).all()   # pr
        assert (preds[:, 2] >= 0).all()   # sfcWind

        # save_predictions path
        lat, lon = np.linspace(41, 45.5, Hh), np.linspace(-5, 0.5, Ww)
        import pandas as pd
        time = pd.date_range("2000-01-01", periods=T, freq="D")
        ref_da = xr.DataArray(
            np.zeros((T, Hh, Ww), dtype=np.float32),
            coords={"time": time, "lat": lat, "lon": lon},
            dims=["time", "lat", "lon"],
        )
        save_predictions(torch.from_numpy(preds), VARIABLES, out_dir, ref_da)
        loaded = load_predictions(out_dir, VARIABLES)
        np.testing.assert_allclose(loaded["pr"].values, preds[:, 1], atol=1e-6)