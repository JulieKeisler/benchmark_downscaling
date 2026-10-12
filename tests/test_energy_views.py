"""Tests for src/evaluation/energy_views.py (day-of-year cycles, events)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

from energy_views import (
    build_doy_block, domain_series, doy_cycle, energy_events, location_cycles,
    resolve_sites, site_catalog, spell_stats,
)


# ── doy_cycle ─────────────────────────────────────────────────────────────────

def _calendar_two_years():
    time = pd.date_range("2010-01-01", periods=730, freq="D")
    month, day = time.month.values, time.day.values
    # sentinel: every observation of a calendar day encodes that day
    values = (month * 100 + day).astype(float)
    return values, month, day


class TestDoyCycle:

    def test_365_bins_calendar_order(self):
        values, month, day = _calendar_two_years()
        cyc = doy_cycle(values, month, day)
        assert len(cyc) == 365
        assert cyc[0] == 101          # Jan 1
        assert cyc[58] == 228         # Feb 28 — Feb 29 was dropped
        assert cyc[59] == 301         # Mar 1 directly follows Feb 28
        assert cyc[-1] == 1231        # Dec 31

    def test_leap_day_dropped_from_raw_dayofyear(self):
        values, month, day = _calendar_two_years()
        cyc = doy_cycle(values, month, day)
        assert 229 not in cyc         # Feb 29 never appears as a bin

    def test_leap_and_non_leap_share_axis(self):
        # (month, day) grouping — not raw dayofyear: on a raw-doy axis
        # 2020-03-01 (doy 61) would be binned with 2021-03-02.
        time = pd.date_range("2020-01-01", periods=731, freq="D")   # 2020 + 2021
        values = np.ones(731)
        i = time.get_loc("2020-03-01")
        values[i] = 11.0
        cyc = doy_cycle(values, time.month.values, time.day.values)
        assert len(cyc) == 365
        assert cyc[59] == pytest.approx(6.0)   # Mar 1: mean(11, 1)
        assert cyc[60] == pytest.approx(1.0)   # Mar 2: mean(1, 1)

    def test_incomplete_year_returns_none(self):
        time = pd.date_range("2010-01-01", periods=60, freq="D")
        cyc = doy_cycle(np.zeros(60), time.month.values, time.day.values)
        assert cyc is None

    def test_length_mismatch_raises(self):
        time = pd.date_range("2010-01-01", periods=10, freq="D")
        with pytest.raises(ValueError):
            doy_cycle(np.zeros(5), time.month.values, time.day.values)


# ── spell_stats ───────────────────────────────────────────────────────────────

class TestSpellStats:

    def test_counts_and_mean_duration(self):
        low = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
        assert spell_stats(low, min_days=2) == (2, pytest.approx(2.5))
        assert spell_stats(low, min_days=3) == (1, 3.0)

    def test_isolated_days_are_not_events(self):
        low = np.array([0, 1, 0, 1, 0], dtype=bool)
        assert spell_stats(low, min_days=2) == (0, 0.0)

    def test_all_true_and_empty(self):
        assert spell_stats(np.ones(5, dtype=bool))[0] == 1
        assert spell_stats(np.zeros(5, dtype=bool)) == (0, 0.0)
        assert spell_stats(np.array([], dtype=bool)) == (0, 0.0)


# ── energy_events ─────────────────────────────────────────────────────────────

def _series(n=7670, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "sfcWind": {"era5": rng.uniform(3, 12, n), "gcm": rng.uniform(3, 12, n)},
        "rsds":    {"era5": rng.uniform(50, 300, n), "gcm": rng.uniform(50, 300, n)},
        "tas":     {"era5": rng.uniform(260, 310, n), "gcm": rng.uniform(260, 310, n)},
    }


class TestEnergyEvents:

    def test_thresholds_from_era5_applied_to_all_sources(self):
        s = _series()
        ev = energy_events(s, period=(2000, 2020))
        assert ev is not None
        thr = ev["meta"]["thresholds"]
        for var, ref in s.items():
            expected = np.nanpercentile(ref["era5"], 20)
            assert thr[var] == pytest.approx(expected, rel=1e-4)
        assert ev["meta"]["period"] == [2000, 2020]
        assert ev["meta"]["min_days"] == 2
        for ev_name in ("dunkelflaute", "cold_low_wind"):
            assert set(ev[ev_name]["count"]) == {"era5", "gcm"}
            assert set(ev[ev_name]["mean_duration"]) == {"era5", "gcm"}

    def test_persistent_low_series_gives_more_events(self):
        n = 7670
        s = _series(n)
        # a product dipping below both thresholds for 7 days out of every 21
        low_week = (np.arange(n) % 21) < 7
        s["sfcWind"]["pred"] = np.where(low_week, 1.0, 10.0)
        s["rsds"]["pred"] = np.where(low_week, 10.0, 200.0)
        s["tas"]["pred"] = s["tas"]["era5"]
        ev = energy_events(s)
        assert ev["dunkelflaute"]["count"]["pred"] > ev["dunkelflaute"]["count"]["era5"]
        assert ev["dunkelflaute"]["mean_duration"]["pred"] == pytest.approx(7.0, rel=0.01)

    def test_missing_era5_reference_returns_none(self):
        s = _series()
        del s["tas"]["era5"]
        assert energy_events(s) is None

    def test_missing_pred_for_one_variable_omits_pred_stats(self):
        s = _series()
        s["sfcWind"]["pred"] = s["sfcWind"]["era5"]
        s["tas"]["pred"] = s["tas"]["era5"]
        # rsds has no pred → dunkelflaute has no pred column
        ev = energy_events(s)
        assert "pred" not in ev["dunkelflaute"]["count"]
        assert "pred" in ev["cold_low_wind"]["count"]

    def test_too_short_series_returns_none(self):
        s = _series(n=100)
        assert energy_events(s) is None


# ── sites / location cycles ───────────────────────────────────────────────────

FRANCE_LAT = np.linspace(55.0, 41.0, 57)
FRANCE_LON = np.linspace(-6.0, 10.0, 65)


class TestSites:

    def test_france_catalog_and_grid_mapping(self):
        cat = site_catalog("france")
        assert len(cat) == 12
        sites = resolve_sites("france", FRANCE_LAT, FRANCE_LON)
        assert len(sites) == 12
        for s in sites:
            assert 0 <= s["ilat"] < 57 and 0 <= s["ilon"] < 65

    def test_sites_outside_grid_are_dropped(self):
        assert resolve_sites("australia", FRANCE_LAT, FRANCE_LON) == []
        assert resolve_sites("unknown_region", FRANCE_LAT, FRANCE_LON) == []

    def test_nearest_cell(self):
        sites = {s["name"]: s for s in resolve_sites("france", FRANCE_LAT, FRANCE_LON)}
        occ = sites["occitanie"]          # lat 43.65, lon 2.44
        assert FRANCE_LAT[occ["ilat"]] == pytest.approx(43.65, abs=0.25)
        assert FRANCE_LON[occ["ilon"]] == pytest.approx(2.44, abs=0.25)


def _field(base, time, lat=FRANCE_LAT, lon=FRANCE_LON, seed=0):
    rng = np.random.default_rng(seed)
    data = (base + rng.normal(0, 1, (len(time), len(lat), len(lon)))).astype(np.float32)
    return xr.DataArray(data, coords={"time": time, "lat": lat, "lon": lon},
                        dims=["time", "lat", "lon"])


class TestLocationCycles:

    def test_region_mean_and_site_keys(self):
        time = pd.date_range("2010-01-01", periods=730, freq="D")
        sites = resolve_sites("france", FRANCE_LAT, FRANCE_LON)
        out = location_cycles(_field(280.0, time), sites)
        assert "global" in out and len(out["global"]) == 365
        assert len(out) == 1 + len(sites)

    def test_build_doy_block_drops_site_without_era5(self):
        time = pd.date_range("2010-01-01", periods=730, freq="D")
        sites = resolve_sites("france", FRANCE_LAT, FRANCE_LON)
        block = build_doy_block(_field(280.0, time), _field(282.0, time), None, sites)
        assert set(next(iter(block.values()))) == {"era5", "gcm"}   # no pred
        assert "global" in block and len(block["global"]["era5"]) == 365
        # all three sources present when available
        block = build_doy_block(_field(280.0, time), None, _field(281.0, time), sites)
        assert set(next(iter(block.values()))) == {"era5", "pred"}

    def test_domain_series_shape(self):
        time = pd.date_range("2010-01-01", periods=10, freq="D")
        s = domain_series(_field(280.0, time))
        assert s.shape == (10,)
