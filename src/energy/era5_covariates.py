"""ERA5 weather covariates for TAM net-load modelling (France-parity).

Replicates the exact pipeline of ``TAM_downscaling/net_demand_model_era5.py``:
  - Loads the 5 ERA5 variables (tas, sfcWind, sfcWindmax, pr, psl) at 0.25 deg
    over the region crop, from ``data/era5/<region>/*_<year>.nc`` (france root).
  - Fits StandardScaler then TruncatedSVD(15 PCs, random_state=42) on the
    training window (2012-01-01 .. 2021-12-31 only), then transforms the full
    2012-2022 series (joint PCA over all grid cells x all variables).
  - Computes spatial (latitude,longitude) means of each variable over the full
    window, with the same aliases / unit conversions as the France script:
        era5_temp_c : tas      K -> degC
        sfcWind     : sfcWind  (m/s)
        sfcWindmax  : sfcWindmax (m/s)
        era5_pr_mm  : pr       stored as DAILY TOTAL in meters -> mm/day (x1000).
                             NOTE: the local WB2 crop stores pr as accumulated
                             m/day (France annual sum ~0.95 m), so we use x1000.
                             This intentionally differs from the France TAM
                             script's x86400*1000 (m/s assumption), which would
                             over-inflate by 86400x on this data.
        psl         : psl      (Pa)
  - Outputs one row per calendar date with columns pc_0..pc_14 plus the five
    spatial means, suitable for the daily (tod==30) energy rows.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

_VARS = ["tas", "sfcWind", "sfcWindmax", "pr", "psl"]
_N_PCS = 15
_YEAR0, _YEAR1 = 2012, 2022
_TRAIN_END = pd.Timestamp("2021-12-31")

_ERA5_DIR = {
    "france": None,  # data/era5 root
    "australia": "australia",
    "us_central": "us_central",
}

_MEAN_ALIASES = {
    "tas": "era5_temp_c",
    "sfcWind": "sfcWind",
    "sfcWindmax": "sfcWindmax",
    "pr": "era5_pr_mm",
    "psl": "psl",
}


def era5_dir(region: str, base: Path) -> Path:
    sub = _ERA5_DIR[region]
    return base / "data" / ("era5" if sub is None else f"era5/{sub}")


def build_era5_covariates(
    region: str,
    base: Path = None,
    year0: int = _YEAR0,
    year1: int = _YEAR1,
    n_pcs: int = _N_PCS,
) -> pd.DataFrame:
    base = Path(base) if base else Path(__file__).resolve().parents[2]
    edir = era5_dir(region, base)

    ds_ref = xr.open_dataset(edir / f"{_VARS[0]}_{year0}.nc")
    da_ref = ds_ref[_VARS[0]]
    n_lat, n_lon = da_ref.sizes["latitude"], da_ref.sizes["longitude"]
    n_grid = n_lat * n_lon
    print(f"[era5:{region}] grid {n_lat}x{n_lon}={n_grid} points")

    all_times = []
    for year in range(year0, year1 + 1):
        with xr.open_dataset(edir / f"{_VARS[0]}_{year}.nc") as ds:
            all_times.extend(ds["time"].values)
    time_axis = np.array(all_times)
    n_time = len(time_axis)
    print(f"[era5:{region}] time {pd.Timestamp(time_axis[0]).date()} .. "
          f"{pd.Timestamp(time_axis[-1]).date()} ({n_time} steps)")

    X_parts = []
    for var in _VARS:
        chunks = []
        for year in range(year0, year1 + 1):
            with xr.open_dataset(edir / f"{var}_{year}.nc") as ds:
                chunks.append(ds[var].values.reshape(ds.sizes["time"], -1))
        var_arr = np.concatenate(chunks, axis=0).astype(np.float32)
        X_parts.append(var_arr)
        print(f"  {var}: {var_arr.shape}")
        del chunks
    X_raw = np.hstack(X_parts).astype(np.float32)
    del X_parts
    gc.collect()
    print(f"[era5:{region}] feature matrix {X_raw.shape}")

    dates = pd.to_datetime(time_axis).normalize()
    train_mask = dates <= _TRAIN_END
    scaler = StandardScaler()
    svd = TruncatedSVD(n_components=n_pcs, random_state=42)

    X_train = X_raw[train_mask]
    X_tr_sc = scaler.fit_transform(X_train)
    pc_train = svd.fit_transform(X_tr_sc)
    pc_all = svd.transform(scaler.transform(X_raw))
    evr = svd.explained_variance_ratio_
    print(f"[era5:{region}] PCs cumulative variance: {np.cumsum(evr).round(3).tolist()}")
    del X_train, X_tr_sc
    gc.collect()

    daily = pd.DataFrame()
    daily["date"] = [t.strftime("%Y-%m-%d") for t in dates]
    for i in range(n_pcs):
        daily[f"pc_{i}"] = pc_all[:, i]

    for var in _VARS:
        means = []
        for year in range(year0, year1 + 1):
            with xr.open_dataset(edir / f"{var}_{year}.nc") as ds:
                means.append(ds[var].mean(dim=["latitude", "longitude"]).values)
        daily[_MEAN_ALIASES[var]] = np.concatenate(means)

    daily["era5_temp_c"] -= 273.15
    daily["era5_pr_mm"] *= 1000.0  # stored as daily total m -> mm/day

    del X_raw, pc_all
    gc.collect()
    return daily


if __name__ == "__main__":
    import sys

    region = sys.argv[1] if len(sys.argv) > 1 else "france"
    out = Path(__file__).resolve().parents[2] / "data/era5_covariates"
    out.mkdir(parents=True, exist_ok=True)
    df = build_era5_covariates(region)
    fp = out / f"{region}_era5_covariates.csv"
    df.to_csv(fp, index=False)
    print(f"[era5:{region}] saved {len(df)} daily rows -> {fp}")
