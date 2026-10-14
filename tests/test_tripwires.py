"""Self-tests for the contract tripwires (CONTRACT.md §E): each must be able to fire."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "tools"))

from data_processing import bicubic_regrid, check_same_member, load_gcm  # noqa: E402
import check_gcm_inputs  # noqa: E402
import check_results  # noqa: E402

ERA5_LAT = np.arange(48.0, 33.75, -0.25)
ERA5_LON = np.arange(-105.0, -88.75, 0.25)


def _gcm_file(path: Path, var: str, lon: np.ndarray, years: tuple[int, int],
              member: str | None, offset: float = 0.0) -> None:
    time = xr.cftime_range(f"{years[0]}-01-01", f"{years[1]}-12-31", freq="D", calendar="noleap")
    lat = np.arange(30.0, 52.0, 1.25)
    rng = np.random.default_rng(0)
    data = 280.0 + offset + rng.standard_normal((len(time), len(lat), len(lon))).astype(np.float32)
    ds = xr.Dataset({var: (("time", "lat", "lon"), data)},
                    coords={"time": time, "lat": lat, "lon": lon})
    if member:
        ds.attrs["variant_label"] = member
    ds.to_netcdf(path)


def _relabel_on_variable(path: Path, var: str, member: str) -> None:
    ds = xr.load_dataset(path)
    ds.attrs.pop("variant_label", None)
    ds[var].attrs["variant_label"] = member
    ds.to_netcdf(path)


def test_lon_coverage_fires_on_one_column_case():
    src = xr.DataArray(np.zeros((2, 4, 4)), dims=("time", "lat", "lon"),
                       coords={"time": [0, 1], "lat": [50.0, 45.0, 40.0, 35.0],
                               "lon": [255.0, 260.0, 265.0, 270.0]})
    with pytest.raises(ValueError, match="does not cover"):
        bicubic_regrid(src, ERA5_LAT, ERA5_LON)


def test_lon_normalised_at_load(tmp_path):
    lon360 = np.arange(250.0, 275.0, 1.25)
    _gcm_file(tmp_path / "tas_historical_1960-2014.nc", "tas", lon360, (2010, 2014), "r4i1p1f1")
    da = load_gcm(tmp_path, "tas", 2010, 2010, "ssp245")
    assert da.lon.min() < 0 and da.lon.max() < 0
    out = bicubic_regrid(da, ERA5_LAT, ERA5_LON)       # covered once normalised
    assert out.std("lon").mean() > 0                    # not one repeated column


def test_member_splice_fires(tmp_path):
    lon = np.arange(-110.0, -85.0, 1.25)
    _gcm_file(tmp_path / "tas_historical_1960-2014.nc", "tas", lon, (2014, 2014), "r1i1p1f1")
    _gcm_file(tmp_path / "tas_ssp245_2015-2100.nc", "tas", lon, (2015, 2015), "r4i1p1f1")
    with pytest.raises(ValueError, match="mix realisations"):
        load_gcm(tmp_path, "tas", 2014, 2015, "ssp245")
    assert check_same_member([tmp_path / "tas_ssp245_2015-2100.nc"]) == "r4i1p1f1"


def test_join_step_fires_on_a_15_percent_drop():
    rng = np.random.default_rng(1)
    years = np.arange(1990, 2020)
    v = 5.0 + 0.05 * rng.standard_normal(len(years))   # 1 % interannual sd
    v[years >= 2015] *= 0.85                           # CNRM-like wind step
    assert check_gcm_inputs.join_step_sd(dict(zip(years, v))) > 3.0
    v[years >= 2015] /= 0.85
    assert check_gcm_inputs.join_step_sd(dict(zip(years, v))) < 3.0


def test_results_check_fires_on_nan_and_short_period(tmp_path):
    good = {k: 0.1 for k in check_results.VS_OBS}
    gcm = {k: 0.9 for k in check_results.VS_GCM}
    m = {"tas": {"vs_obs": dict(good, spectral_fine=float("nan")), "vs_gcm": gcm},
         "_delta": {"tas": {"2040_2060": 0.1, "2080_2100": 0.2}}}
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    np.save(tmp_path / "eval_dates.npy", np.array(["2000-01-01", "2020-12-31"]))
    np.save(tmp_path / "variables.npy", np.array(["tas"]))
    fails = check_results.check(tmp_path, ["tas"])
    assert any("spectral_fine" in f for f in fails)
    assert any("not 2000-01-01..2100-12-31" in f for f in fails)


def test_psd_white_noise_and_degenerate_crossings():
    from psd_sweep import radial_psd, last_crossing, t_star_at
    rng = np.random.default_rng(0)
    k, p = radial_psd(0.3 * rng.standard_normal((40, 57, 65)))
    assert abs(p.mean() - 0.09) < 0.005                      # P = s^2 for white noise
    ref = 1.0 / k ** 2
    k_star, found = last_crossing(k, ref, 0.5 * ref)         # GCM below everywhere
    assert not found and k_star == k[0]
    k_star, found = last_crossing(k, ref, 2.0 * ref)         # GCM never drops below
    assert found and k_star == k[-1]
    gcm = np.where(k < 0.1, 1.5 * ref, 0.01 * ref)           # clean crossing near 0.1
    k_star, found = last_crossing(k, ref, gcm)
    assert found and 0.05 < k_star < 0.15
    assert t_star_at(k_star, k, ref) > 0


def test_member_splice_fires_on_variable_level_labels(tmp_path):
    lon = np.arange(-110.0, -85.0, 1.25)
    h, f = tmp_path / "tas_historical_1960-2014.nc", tmp_path / "tas_ssp245_2015-2100.nc"
    _gcm_file(h, "tas", lon, (2014, 2014), "r1i1p1f1")
    _gcm_file(f, "tas", lon, (2015, 2015), "r4i1p1f1")
    _relabel_on_variable(h, "tas", "r1i1p1f1")
    _relabel_on_variable(f, "tas", "r4i1p1f1")
    with pytest.raises(ValueError, match="mix realisations"):
        check_same_member([h, f])


def test_psd_sweep_main_writes_json(tmp_path, monkeypatch):
    """End-to-end: the sweep runs on synthetic files and its JSON is valid."""
    import hashlib
    import torch
    import psd_sweep
    lat = np.arange(48.0, 33.75, -0.25); lon = np.arange(-105.0, -88.75, 0.25)
    era5_dir, gcm_dir, ck = tmp_path / "era5", tmp_path / "gcm", tmp_path / "ck"
    for d in (era5_dir, gcm_dir, ck):
        d.mkdir()
    rng = np.random.default_rng(0)
    for y in (1980, 1981):
        t = xr.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        f = 280 + np.cumsum(np.cumsum(rng.standard_normal((len(t), len(lat), len(lon))), 1), 2) * 0.05
        xr.Dataset({"tas": (("time", "lat", "lon"), f.astype(np.float32))},
                   coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(era5_dir / f"tas_{y}.nc")
    glon = np.arange(-106.25, -87.5, 1.25)
    _gcm_file(gcm_dir / "tas_historical_1960-2014.nc", "tas", glon, (1980, 1981), "r1i1p1f1")
    torch.save({"stats": {"tas": {"xmin": 250.0, "xmax": 310.0, "log_transform": False}}}, ck / "best.pt")
    out = tmp_path / "t.json"
    monkeypatch.setattr(sys, "argv", ["psd_sweep", "--era5-dir", str(era5_dir), "--gcm-dir", str(gcm_dir),
                                      "--checkpoint-dir", str(ck), "--variables", "tas",
                                      "--calib-start", "1980", "--calib-end", "1981", "--out-json", str(out)])
    psd_sweep.main()
    d = json.loads(out.read_text())
    assert d["checkpoint_sha256"] == hashlib.sha256((ck / "best.pt").read_bytes()).hexdigest()
    assert isinstance(d["variables"]["tas"]["degenerate"], bool) and d["t_star"] > 0
