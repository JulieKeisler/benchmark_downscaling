#!/usr/bin/env python3
"""Evaluate a downscaling method against ERA5 and the source GCM.

Metrics are computed along two axes:
  * vs ERA5 (obs)  — bias, RMSE, KS, seasonal cycle, extremes
  * vs GCM (input) — temporal correlation, lag-1 autocorrelation, annual anomaly

Usage:
    python scripts/evaluate.py                          # uses configs/benchmark.yaml
    python scripts/evaluate.py --config my.yaml
    python scripts/evaluate.py --era5-dir /path/era5 --ipsl-dir /path/cmip6/ipsl_cm6a_lr
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from benchmark_downscaling.src.evaluation.metrics.metrics import (
        spatial_mean_abs_bias, std_error, ks_stat, seasonal_mae,
        extreme_freq_bias, extreme_intensity_bias, clim_rmse,
        mean_temporal_corr, mean_temporal_corr_annual,
        lag1_autocorr_mae, annual_anomaly_bias,
        intervar_corr_mae, spatial_spectrum_metric,
        plot_bias_map, plot_cdf, plot_seasonal_cycle,
        plot_temporal_corr, plot_autocorr_comparison,
        plot_annual_anomalies, plot_radar, plot_intervar_corr,
    )
except ImportError:
    from metrics import (
        spatial_mean_abs_bias, std_error, ks_stat, seasonal_mae,
        extreme_freq_bias, extreme_intensity_bias, clim_rmse,
        mean_temporal_corr, mean_temporal_corr_annual,
        lag1_autocorr_mae, annual_anomaly_bias,
        intervar_corr_mae, spatial_spectrum_metric,
        plot_bias_map, plot_cdf, plot_seasonal_cycle,
        plot_temporal_corr, plot_autocorr_comparison,
        plot_annual_anomalies, plot_radar, plot_intervar_corr,
    )

# Unit conversions — target unit is mm/day for precipitation.
#
# ERA5 from crop_era5_jz.py (WB2 source):
#   pr = daily total in m (from sum of 4 six-hourly values).
#   Annual-mean files contain the mean daily total, so × 1000 → mm/day.
# ERA5 from download_era5.py (CDS daily-stats source):
#   pr = daily mean of hourly accumulations in m → × 24 × 1000 → mm/day.
# Auto-detection: if the spatial mean of the first time step exceeds 0.05,
#   the file contains an annual total (m/year) → apply / 365.25 × 1000.
#   Otherwise it is a daily value in m → × 1000.
def _era5_pr_convert(da: xr.DataArray) -> xr.DataArray:
    ref = float(np.nanmean(da.isel(time=0).values))
    if ref > 0.05:
        # annual total in m → mm/day
        return da * 1000 / 365.25
    else:
        # daily value in m (hourly mean × 24 already; WB2 daily sum)
        return da * 1000

ERA5_UNITS: dict = {
    "pr": _era5_pr_convert,
}
CMIP6_UNITS: dict = {
    "pr": lambda da: da * 86400,            # kg m⁻² s⁻¹ → mm/day
}


def load_era5(era5_dir: Path, variable: str, start: int, end: int) -> xr.DataArray:
    files = sorted(era5_dir.glob(f"{variable}_*.nc"))
    files = [f for f in files
             if start <= int(f.stem.split("_")[-1]) <= end]
    if not files:
        raise FileNotFoundError(f"No ERA5 files for {variable} in {era5_dir}")

    parts = []
    for f in files:
        ds = xr.open_dataset(f)
        renames = {}
        if "latitude" in ds.dims:
            renames["latitude"] = "lat"
        if "longitude" in ds.dims:
            renames["longitude"] = "lon"
        if renames:
            ds = ds.rename(renames)
        parts.append(ds[variable])

    da = xr.concat(parts, dim="time")
    if variable in ERA5_UNITS:
        da = ERA5_UNITS[variable](da)
    return da.load()


def load_ipsl(ipsl_dir: Path, variable: str, start: int, end: int) -> xr.DataArray:
    hist_file = ipsl_dir / f"{variable}_historical_1960-2014.nc"
    ssp_file  = ipsl_dir / f"{variable}_ssp585_2015-2100.nc"
    files = []
    if hist_file.exists() and start <= 2014:
        files.append(str(hist_file))
    if ssp_file.exists() and end >= 2015:
        files.append(str(ssp_file))
    if not files:
        raise FileNotFoundError(f"No IPSL files for {variable} in {ipsl_dir}")
    parts = []
    for f in files:
        ds = xr.open_dataset(f, use_cftime=True)
        renames = {}
        if "latitude" in ds.dims:
            renames["latitude"] = "lat"
        if "longitude" in ds.dims:
            renames["longitude"] = "lon"
        if renames:
            ds = ds.rename(renames)
        parts.append(ds[variable])
    da = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    da = da.sel(time=slice(f"{start}-01-01", f"{end}-12-31"))
    if variable in CMIP6_UNITS:
        da = CMIP6_UNITS[variable](da)
    return da.load()


def bicubic_regrid(gcm: xr.DataArray, target_lat: np.ndarray,
                   target_lon: np.ndarray) -> xr.DataArray:
    """Bicubic (cubic spline) spatial interpolation to the ERA5 grid.

    extrapolate=True so target points outside the GCM domain (e.g. the
    western ERA5 cells at -6..0° when IPSL only covers 0..10°) get a
    plausible value rather than NaN.
    """
    # Sequential 1D so xarray uses interp1d (supports extrapolate) rather than
    # RegularGridInterpolator (multi-dim, rejects fill_value="extrapolate").
    result = gcm.interp(lat=target_lat, method="cubic", kwargs={"fill_value": "extrapolate"})
    return result.interp(lon=target_lon, method="cubic", kwargs={"fill_value": "extrapolate"})


def _qmap_1d(src: np.ndarray, ref: np.ndarray, apply: np.ndarray) -> np.ndarray:
    """Empirical quantile mapping: fit CDF from src→ref, apply to apply."""
    q = np.linspace(0, 1, min(len(src), 200))
    return np.interp(apply, np.quantile(src, q), np.quantile(ref, q))


def bcsd_downscale(gcm: xr.DataArray, obs: xr.DataArray, variable: str) -> xr.DataArray:
    """BCSD: monthly quantile mapping at GCM resolution + spatial disaggregation.

    1. Bilinear interpolation of ERA5 to GCM grid (coarsening).
    2. Monthly empirical quantile mapping at each GCM cell (GCM → coarse ERA5).
    3. Bicubic interpolation of bias-corrected field to ERA5 grid.
    4. Add monthly fine-scale ERA5 spatial detail (additive for T/P/wind,
       multiplicative ratio for precipitation).
    """
    # Step 1: coarsen ERA5 to GCM grid
    obs_coarse = obs.interp(lat=gcm.lat, lon=gcm.lon, method="linear")

    # Step 2: monthly quantile mapping at each GCM cell
    bc = gcm.copy(deep=True)
    months_gcm = gcm.time.dt.month.values
    for m in range(1, 13):
        mask = months_gcm == m
        gcm_m = gcm.values[mask]       # [n_t, n_lat, n_lon]
        obs_m = obs_coarse.values[mask]
        mapped = gcm_m.copy()
        for i in range(gcm_m.shape[1]):
            for j in range(gcm_m.shape[2]):
                mapped[:, i, j] = _qmap_1d(gcm_m[:, i, j], obs_m[:, i, j], gcm_m[:, i, j])
        bc.values[mask] = mapped

    # Step 3: bicubic interpolation to ERA5 grid — sequential 1D so interp1d
    # is used (supports extrapolate); simultaneous 2D would trigger
    # RegularGridInterpolator which rejects fill_value="extrapolate".
    bc_fine = bc.interp(lat=obs.lat, method="cubic", kwargs={"fill_value": "extrapolate"})
    bc_fine = bc_fine.interp(lon=obs.lon, method="cubic", kwargs={"fill_value": "extrapolate"})
    bc_fine = bc_fine.assign_coords(time=obs.time)

    # Step 4: spatial disaggregation — monthly ERA5 fine-scale detail
    obs_clim        = obs.groupby("time.month").mean("time")         # [12, lat, lon]
    obs_clim_coarse = obs_coarse.groupby("time.month").mean("time")  # [12, lat_gcm, lon_gcm]
    obs_clim_coarse_fine = obs_clim_coarse.interp(
        lat=obs.lat, method="cubic", kwargs={"fill_value": "extrapolate"}
    ).interp(lon=obs.lon, method="cubic", kwargs={"fill_value": "extrapolate"})  # [12, lat, lon]

    month_idx = obs.time.dt.month.values - 1  # 0-indexed → array index into dim 0
    pred_np = bc_fine.values.copy()

    if variable == "pr":
        # Multiplicative ratio to avoid negative precipitation
        denom = np.where(obs_clim_coarse_fine.values > 1e-6,
                         obs_clim_coarse_fine.values, 1e-6)
        ratio = np.clip(obs_clim.values / denom, 0.1, 10.0)  # [12, lat, lon]
        pred_np *= ratio[month_idx]                            # [n_time, lat, lon]
        pred_np  = np.clip(pred_np, 0.0, None)
    else:
        detail   = obs_clim.values - obs_clim_coarse_fine.values  # [12, lat, lon]
        pred_np += detail[month_idx]                               # [n_time, lat, lon]

    return xr.DataArray(pred_np, coords=bc_fine.coords, dims=bc_fine.dims)


def gnn_downscale(gcm: xr.DataArray, obs: xr.DataArray, variable: str) -> xr.DataArray:
    """GNN-style bipartite interpolation, Anemoi-inspired.

    Architecture: K-nearest-neighbour bipartite graph (GCM nodes → ERA5 nodes),
    with an MLP that maps local neighbourhood features to the fine-grid value.
    Feature vector per ERA5 node: [lat, lon, sin(month), cos(month),
        (gcm_val, Δlat, Δlon, dist) × K].

    Training: coarsen ERA5 to GCM grid, learn ERA5_coarse → ERA5_fine.
    Inference: apply the trained MLP to IPSL GCM values.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    K = 4          # GCM neighbours per ERA5 node
    N_TRAIN = 200  # ERA5 time steps sampled for training

    # ── Bipartite K-NN graph ────────────────────────────────────────────────
    gcm_lat_g, gcm_lon_g = np.meshgrid(gcm.lat.values, gcm.lon.values, indexing="ij")
    gcm_pos  = np.column_stack([gcm_lat_g.ravel(), gcm_lon_g.ravel()])
    era5_lat_g, era5_lon_g = np.meshgrid(obs.lat.values, obs.lon.values, indexing="ij")
    era5_pos = np.column_stack([era5_lat_g.ravel(), era5_lon_g.ravel()])
    n_gcm, n_era5 = len(gcm_pos), len(era5_pos)
    n_lat, n_lon  = obs.lat.size, obs.lon.size

    nn_model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(gcm_pos)
    graph_dists, graph_idx = nn_model.kneighbors(era5_pos)   # [n_era5, K]
    # Relative position: ERA5 node − GCM neighbour  [n_era5, K, 2]
    delta = era5_pos[:, np.newaxis, :] - gcm_pos[graph_idx]

    def _features(vals_flat: np.ndarray, months: np.ndarray) -> np.ndarray:
        """Build [n_t * n_era5, 4 + K*4] feature matrix."""
        n_t = vals_flat.shape[0]
        X = np.empty((n_t * n_era5, 4 + K * 4), dtype=np.float32)
        X[:, 0] = np.tile(era5_pos[:, 0], n_t)
        X[:, 1] = np.tile(era5_pos[:, 1], n_t)
        X[:, 2] = np.repeat(np.sin(2 * np.pi * months / 12), n_era5)
        X[:, 3] = np.repeat(np.cos(2 * np.pi * months / 12), n_era5)
        gcm_nbr = vals_flat[:, graph_idx]   # [n_t, n_era5, K]
        for k in range(K):
            col = 4 + k * 4
            X[:, col]     = gcm_nbr[:, :, k].ravel()
            X[:, col + 1] = np.tile(delta[:, k, 0], n_t)
            X[:, col + 2] = np.tile(delta[:, k, 1], n_t)
            X[:, col + 3] = np.tile(graph_dists[:, k], n_t)
        return X

    # ── Training: ERA5_coarse → ERA5_fine ───────────────────────────────────
    obs_coarse    = obs.interp(lat=gcm.lat, lon=gcm.lon, method="linear")
    obs_coarse_np = obs_coarse.values.reshape(obs.sizes["time"], n_gcm).astype(np.float32)
    obs_fine_np   = obs.values.reshape(obs.sizes["time"], n_era5).astype(np.float32)
    months_obs    = obs.time.dt.month.values

    rng = np.random.default_rng(42)
    train_t = rng.choice(obs.sizes["time"], size=min(N_TRAIN, obs.sizes["time"]), replace=False)

    print(f"    building features ({len(train_t)} time steps × {n_era5} nodes) ...", flush=True)
    X_tr = _features(obs_coarse_np[train_t], months_obs[train_t])
    y_tr = obs_fine_np[train_t].ravel()

    sx = StandardScaler().fit(X_tr)
    sy = StandardScaler().fit(y_tr.reshape(-1, 1))

    print(f"    training MLP ({len(X_tr):,} samples) ...", flush=True)
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=50,
        batch_size=256,
        learning_rate_init=1e-3,
        random_state=42,
        n_iter_no_change=10,
        verbose=False,
    )
    mlp.fit(sx.transform(X_tr), sy.transform(y_tr.reshape(-1, 1)).ravel())

    # ── Inference on GCM (batched to limit memory) ──────────────────────────
    gcm_np     = gcm.values.reshape(gcm.sizes["time"], n_gcm).astype(np.float32)
    gcm_months = gcm.time.dt.month.values

    print(f"    applying GNN to {gcm.sizes['time']} time steps ...", flush=True)
    parts = []
    for start in range(0, gcm.sizes["time"], 200):
        sl = slice(start, start + 200)
        X_b = _features(gcm_np[sl], gcm_months[sl])
        y_b = sy.inverse_transform(
            mlp.predict(sx.transform(X_b)).reshape(-1, 1)
        ).ravel()
        parts.append(y_b.reshape(-1, n_lat, n_lon))

    pred_np = np.concatenate(parts, axis=0)
    return xr.DataArray(
        pred_np,
        coords={"time": obs.time, "lat": obs.lat, "lon": obs.lon},
        dims=["time", "lat", "lon"],
    )


def serpentflow_load_predictions(
    sf_pt_dir: Path,
    obs_vars: dict[str, xr.DataArray],
    variables: list[str],
) -> dict[str, xr.DataArray]:
    """Load pre-computed SerpentFlow output tensor and return per-variable DataArrays.

    Expects in sf_pt_dir:
      sf_all5_ipsl.pt   — output from inference_serpentflow.py [T, C, H, W]
      sf_norm_stats.json — ERA5 train mean/std per variable (for denormalisation)
      sf_eval_dates.npy  — ISO date strings (eval time axis)
      sf_lat.npy / sf_lon.npy — ERA5 lat/lon arrays

    The output tensor is in the ERA5-normalised space (channel-wise zero-mean,
    unit-variance); sf_norm_stats.json carries the inverse transform.
    """
    import torch, json as _json

    pt_file    = sf_pt_dir / "sf_all5_ipsl.pt"
    stats_file = sf_pt_dir / "sf_norm_stats.json"
    dates_file = sf_pt_dir / "sf_eval_dates.npy"
    lat_file   = sf_pt_dir / "sf_lat.npy"
    lon_file   = sf_pt_dir / "sf_lon.npy"

    for p in [pt_file, stats_file, dates_file]:
        if not p.exists():
            raise FileNotFoundError(f"SerpentFlow output file not found: {p}")

    tensor = torch.load(pt_file, map_location="cpu")   # [T, C, H, W]
    with open(stats_file) as f:
        norm_stats = _json.load(f)
    dates = np.load(dates_file, allow_pickle=True)
    lat   = np.load(lat_file)
    lon   = np.load(lon_file)

    # Use ERA5 time axis: match saved dates against obs time
    ref_obs = next(iter(obs_vars.values()))
    obs_dates = ref_obs.time.dt.strftime("%Y-%m-%d").values
    common = sorted(set(dates) & set(obs_dates))
    if not common:
        raise ValueError("SerpentFlow eval dates do not overlap with ERA5 evaluation period.")
    common_set = set(common)
    obs_idx = [i for i, d in enumerate(obs_dates) if d in common_set]
    sf_idx  = [i for i, d in enumerate(dates)     if d in common_set]
    time_coord = ref_obs.time.isel(time=obs_idx)

    VARIABLES_SF = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
    PHYS_MIN = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0}

    results = {}
    for i, var in enumerate(VARIABLES_SF):
        if var not in variables:
            continue
        s = norm_stats[var]
        # Denormalise: x_physical = x_norm * std + mean
        ch = tensor[sf_idx, i].numpy().astype(np.float64)
        ch = ch * s["std"] + s["mean"]
        if var in PHYS_MIN:
            ch = np.clip(ch, PHYS_MIN[var], None)
        results[var] = xr.DataArray(
            ch.astype(np.float32),
            coords={"time": time_coord, "lat": lat, "lon": lon},
            dims=["time", "lat", "lon"],
        )
        print(f"  [serpentflow] {var}: shape={results[var].shape}, "
              f"mean={float(results[var].mean()):.4g}", flush=True)
    return results


def consistency_downscale_all(
    gcm_vars: dict[str, xr.DataArray],
    obs_vars: dict[str, xr.DataArray],
    checkpoint_dir: Path,
    stroke_time: float = 5.0,
    n_members: int = 1,
) -> dict[str, xr.DataArray]:
    """Multivariate consistency-model downscaling (stroke guidance).

    Based on: "Fast, scale-adaptive and uncertainty-aware downscaling of Earth
    system model fields with generative machine learning" (Nature 2025).
    https://github.com/p-hss/consistency-climate-downscaling

    All variables are processed jointly through a single 5-channel UNet, so
    the model can preserve inter-variable correlations.  The canonical variable
    order (["tas","pr","sfcWind","sfcWindmax","psl"]) must match training.

    Args:
        gcm_vars: dict variable → GCM DataArray (on GCM grid).
        obs_vars: dict variable → ERA5 DataArray (on ERA5 grid, same time axis).
        checkpoint_dir: directory containing best.pt and stats.json.
        stroke_time: noise level t* for stroke guidance.
        n_members: stochastic draws to average (increases quality, costs time).

    Returns dict variable → downscaled DataArray on ERA5 grid / time axis.
    """
    import torch
    import torch.nn.functional as TF
    from diffusers import UNet2DModel

    VARIABLES   = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
    N_VARS      = len(VARIABLES)
    DATA_STD    = 0.5
    T_MIN       = 0.002
    PAD         = (0, 15, 0, 7)
    H_OBS, W_OBS = 57, 65
    LOG_EPS     = 1e-6
    LOG_VARS    = {"pr"}
    CHANNELS    = (128, 128, 256, 256)
    DOWN_BLOCKS = ("DownBlock2D",) * 4
    UP_BLOCKS   = ("UpBlock2D",)   * 4

    ckpt_path = Path(checkpoint_dir) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No consistency checkpoint at {ckpt_path}. "
            "Run scripts/train_consistency.py first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [consistency] loading checkpoint ({device}) ...", flush=True)

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt["stats"]          # dict: variable → {xmin, xmax, log_transform}

    net = UNet2DModel(
        in_channels=N_VARS, out_channels=N_VARS,
        block_out_channels=CHANNELS,
        down_block_types=DOWN_BLOCKS,
        up_block_types=UP_BLOCKS,
    ).to(device)
    net.load_state_dict(ckpt["ema"])
    net.eval()

    # ── Per-variable encode / decode ──────────────────────────────────────────
    def encode(x_np, var):
        if var in LOG_VARS:
            x_np = np.log(x_np + LOG_EPS) - np.log(LOG_EPS)
        s = stats[var]
        return (2.0 * (x_np - s["xmin"]) / (s["xmax"] - s["xmin"]) - 1.0
                ).astype(np.float32)

    def decode(x_np, var):
        s = stats[var]
        x_np = (x_np + 1.0) / 2.0 * (s["xmax"] - s["xmin"]) + s["xmin"]
        if var in LOG_VARS:
            x_np = np.exp(x_np + np.log(LOG_EPS)) - LOG_EPS
        return x_np

    # ── Upsample all GCM variables to ERA5 grid, normalise, stack ─────────────
    # Use the first available obs variable to get the target lat/lon/time
    ref_obs = next(iter(obs_vars.values()))
    T = ref_obs.sizes["time"]

    gcm_norm_list = []
    for var in VARIABLES:
        gcm_up = bicubic_regrid(gcm_vars[var], ref_obs.lat.values, ref_obs.lon.values)
        gcm_up = gcm_up.assign_coords(time=ref_obs.time)
        gcm_arr = gcm_up.values
        if var in LOG_VARS:
            # Bicubic interpolation can produce tiny negatives near zero;
            # clip before log transform to avoid NaN.
            gcm_arr = np.clip(gcm_arr, 0.0, None)
        gcm_norm_list.append(encode(gcm_arr, var))   # [T, H, W]

    gcm_stack = np.stack(gcm_norm_list, axis=1)  # [T, N_VARS, H, W]

    # ── Consistency function ───────────────────────────────────────────────────
    def F_cm(x_t: torch.Tensor, t_val: float) -> torch.Tensor:
        t    = torch.full((x_t.shape[0],), t_val, device=device)
        t4   = t[:, None, None, None]
        skip     = DATA_STD**2 / ((t4 - T_MIN)**2 + DATA_STD**2)
        out_coef = DATA_STD * t4 / (t4**2 + DATA_STD**2)**0.5
        with torch.no_grad():
            pred = net(x_t, t).sample
        return skip * x_t + out_coef * pred

    # ── Batched stroke-guidance inference ─────────────────────────────────────
    print(f"  [consistency] stroke guidance t*={stroke_time}, "
          f"members={n_members}, {T} timesteps ...", flush=True)

    batch_size = 16
    out_chunks = []   # list of [B, N_VARS, H, W] numpy arrays

    for start in range(0, T, batch_size):
        sl   = slice(start, start + batch_size)
        x_np = gcm_stack[sl]                                      # [B, N_VARS, H, W]
        x    = torch.from_numpy(x_np).to(device)                  # [B, N_VARS, H, W]
        x    = TF.pad(x, PAD)                                     # [B, N_VARS, 64, 80]

        member_outs = []
        for _ in range(n_members):
            noise  = torch.randn_like(x)
            x_init = x + noise * stroke_time
            x_out  = F_cm(x_init, stroke_time)                    # [B, N_VARS, 64, 80]
            x_out  = x_out[:, :, :H_OBS, :W_OBS]                 # [B, N_VARS, H, W]
            member_outs.append(x_out.cpu().numpy())               # [B, N_VARS, H, W]

        out_chunks.append(np.mean(member_outs, axis=0))           # [B, N_VARS, H, W]

    out_all = np.concatenate(out_chunks, axis=0)   # [T, N_VARS, H, W]

    # ── Decode each variable and return ───────────────────────────────────────
    # Physical lower bounds after decoding.
    PHYS_MIN = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0}
    # Extreme-value caps to guard against out-of-distribution model outputs.
    # 500 mm/day is a very generous physical cap for daily precipitation.
    PHYS_MAX = {"pr": 500.0}

    results = {}
    for i, var in enumerate(VARIABLES):
        pred_np = decode(out_all[:, i], var)
        if var in PHYS_MIN:
            pred_np = np.clip(pred_np, PHYS_MIN[var], None)
        if var in PHYS_MAX:
            pred_np = np.clip(pred_np, None, PHYS_MAX[var])
        results[var] = xr.DataArray(
            pred_np,
            coords={"time": ref_obs.time, "lat": ref_obs.lat, "lon": ref_obs.lon},
            dims=["time", "lat", "lon"],
        )

    return results


def compute_all_metrics(obs: xr.DataArray, gcm_on_era5: xr.DataArray,
                        pred: xr.DataArray, n_timesteps: int) -> tuple[dict, dict, dict]:
    """Return (metrics_vs_obs, metrics_vs_gcm, plot_series) as dicts.

    plot_series contains time-series data for plotly line charts.
    Metrics requiring daily data are omitted when n_timesteps < 200.
    """
    daily = n_timesteps >= 200

    vs_obs: dict = {
        "bias_mae":  spatial_mean_abs_bias(obs, pred),
        "std_error": std_error(obs, pred),
        "ks_full":   ks_stat(obs, pred),
    }
    if daily:
        vs_obs["ks_q95"]            = ks_stat(obs, pred, quantile=0.95)
        vs_obs["ks_q05"]            = ks_stat(obs, pred, quantile=0.05)
        vs_obs["seasonal_mae"]      = seasonal_mae(obs, pred)
        vs_obs["extreme_freq"]      = extreme_freq_bias(obs, pred, q=0.95)
        vs_obs["extreme_intensity"] = extreme_intensity_bias(obs, pred, q=0.95)

    # Spatial spectrum metric
    spec_score, spec_data = spatial_spectrum_metric(obs, pred)
    vs_obs["spectral_mae"] = spec_score

    vs_gcm: dict = {
        "temporal_corr":        mean_temporal_corr(gcm_on_era5, pred),
        "temporal_corr_annual": mean_temporal_corr_annual(gcm_on_era5, pred),
        "annual_anomaly_bias":  annual_anomaly_bias(gcm_on_era5, pred),
    }
    if daily:
        vs_gcm["lag1_autocorr_mae"] = lag1_autocorr_mae(gcm_on_era5, pred)

    # Time-series data for plotly line charts
    plot_series: dict = {}
    if daily:
        obs_cyc  = obs.groupby("time.month").mean("time").mean(["lat", "lon"])
        pred_cyc = pred.groupby("time.month").mean("time").mean(["lat", "lon"])
        plot_series["seasonal"] = {
            "months": list(range(1, 13)),
            "era5":   [float(v) for v in obs_cyc.values],
            "pred":   [float(v) for v in pred_cyc.values],
        }

    gcm_ann  = gcm_on_era5.mean(["lat", "lon"]).groupby("time.year").mean("time")
    pred_ann = pred.mean(["lat", "lon"]).groupby("time.year").mean("time")
    plot_series["annual"] = {
        "years": [int(y) for y in gcm_ann.year.values],
        "gcm":   [float(v) for v in gcm_ann.values],
        "years_pred": [int(y) for y in pred_ann.year.values],
        "pred":  [float(v) for v in pred_ann.values],
    }

    # CDF (downsampled to 500 pts)
    o_all = obs.values.ravel();  o_all = o_all[np.isfinite(o_all)]; o_all.sort()
    p_all = pred.values.ravel(); p_all = p_all[np.isfinite(p_all)]; p_all.sort()
    idx_o = np.linspace(0, len(o_all) - 1, 500).astype(int)
    idx_p = np.linspace(0, len(p_all) - 1, 500).astype(int)
    plot_series["cdf"] = {
        "era5_x": [float(v) for v in o_all[idx_o]],
        "pred_x": [float(v) for v in p_all[idx_p]],
    }

    plot_series["spectra"] = spec_data

    return vs_obs, vs_gcm, plot_series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--era5-dir", default=None)
    parser.add_argument("--ipsl-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end",   type=int, default=None)
    parser.add_argument("--method",
                        choices=["bicubic", "bcsd", "gnn", "psm", "consistency", "cm",
                                 "serpentflow", "sf"],
                        default="bicubic",
                        help="Downscaling method to evaluate. "
                             "psm=alias for gnn (Perfect Model Setting); "
                             "cm=alias for consistency (Consistency Model); "
                             "sf=alias for serpentflow.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory containing best.pt from train_consistency.py "
                             "(required for --method consistency)")
    parser.add_argument("--sf-pt-dir", default=None,
                        help="Directory with SerpentFlow output: sf_all5_ipsl.pt, "
                             "sf_norm_stats.json, sf_eval_dates.npy, sf_lat.npy, sf_lon.npy "
                             "(required for --method serpentflow)")
    parser.add_argument("--stroke-time", type=float, default=0.468,
                        help="Noise level t* for consistency stroke guidance. "
                             "Default 0.468 from paper (PSD-calibrated: SNR≈1, "
                             "skip_coef≈0.53 — GCM structure preserved while "
                             "ERA5 fine-scale detail is added).")
    parser.add_argument("--n-members", type=int, default=1,
                        help="Stochastic draws to average at inference (default: 1)")
    args = parser.parse_args(argv)

    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    era5_dir = Path(args.era5_dir or cfg["era5"]["dir"])
    ipsl_dir = Path(args.ipsl_dir or cfg["models"]["ipsl_cm6a_lr"]["dir"])
    variables = args.variables or cfg["variables"]
    start = args.start or cfg["period"]["start"]
    end   = args.end   or cfg["period"]["end"]
    # psm/cm/sf are display-name aliases
    effective_method = {
        "psm": "gnn", "cm": "consistency", "sf": "serpentflow"
    }.get(args.method, args.method)
    method_label = {
        "bicubic":     "Bicubic",
        "bcsd":        "BCSD",
        "gnn":         "PSM",
        "psm":         "PSM",
        "consistency": "CM",
        "cm":          "CM",
        "serpentflow": "SF",
        "sf":          "SF",
    }[args.method]

    # Output dir uses the canonical name (psm→psm, cm→cm)
    out_dir = Path(args.output_dir or cfg.get("output_dir", "results")) / args.method

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    all_obs:     dict[str, xr.DataArray] = {}
    all_preds:   dict[str, xr.DataArray] = {}
    all_plot_series: dict[str, dict] = {}

    # ── SerpentFlow: load pre-computed output tensor ────────────────────────────────
    serpentflow_preds: dict[str, xr.DataArray] = {}
    if effective_method == "serpentflow":
        if not args.sf_pt_dir:
            print("ERROR: --sf-pt-dir is required for --method serpentflow")
            return 1
        print("\nLoading SerpentFlow pre-computed predictions ...")
        obs_pre: dict[str, xr.DataArray] = {}
        for variable in variables:
            try:
                obs_pre[variable] = load_era5(era5_dir, variable, start, end)
            except FileNotFoundError as e:
                print(f"  SKIP {variable}: {e}")
        try:
            serpentflow_preds = serpentflow_load_predictions(
                Path(args.sf_pt_dir), obs_pre, variables
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR loading SerpentFlow output: {e}")
            return 1

    # ── Consistency / CM: load + align ALL variables first, then run joint inference ─
    consistency_preds: dict[str, xr.DataArray] = {}
    if effective_method == "consistency":
        if not args.checkpoint_dir:
            print("ERROR: --checkpoint-dir is required for --method consistency")
            return 1
        print("\nPre-loading all variables for joint consistency inference ...")
        gcm_pre: dict[str, xr.DataArray] = {}
        obs_pre: dict[str, xr.DataArray] = {}
        for variable in variables:
            try:
                obs_v = load_era5(era5_dir, variable, start, end)
                gcm_v = load_ipsl(ipsl_dir, variable, start, end)
            except FileNotFoundError as e:
                print(f"  SKIP {variable}: {e}")
                continue
            obs_dates = obs_v.time.dt.strftime("%Y-%m-%d").values
            gcm_dates = gcm_v.time.dt.strftime("%Y-%m-%d").values
            common = sorted(set(obs_dates) & set(gcm_dates))
            if not common:
                continue
            common_set = set(common)
            obs_pre[variable] = obs_v.isel(
                time=[i for i, d in enumerate(obs_dates) if d in common_set])
            gcm_pre[variable] = gcm_v.isel(
                time=[i for i, d in enumerate(gcm_dates) if d in common_set])
            print(f"  {variable}: {len(common)} timesteps")
        consistency_preds = consistency_downscale_all(
            gcm_pre, obs_pre,
            checkpoint_dir=Path(args.checkpoint_dir),
            stroke_time=args.stroke_time,
            n_members=args.n_members,
        )

    for variable in variables:
        print(f"\n{'='*60}")
        print(f"  {variable}  ({start}–{end})")
        print(f"{'='*60}")

        try:
            obs = load_era5(era5_dir, variable, start, end)
        except FileNotFoundError as e:
            print(f"  SKIP ERA5: {e}")
            continue

        try:
            gcm = load_ipsl(ipsl_dir, variable, start, end)
        except FileNotFoundError as e:
            print(f"  SKIP IPSL: {e}")
            continue

        print(f"  ERA5 grid  : {obs.lat.size}×{obs.lon.size}, "
              f"{obs.time.size} timesteps")
        print(f"  IPSL grid  : {gcm.lat.size}×{gcm.lon.size}, "
              f"{gcm.time.size} timesteps")

        # Align time axes: use the intersection of date strings
        obs_dates = obs.time.dt.strftime("%Y-%m-%d").values
        gcm_dates = gcm.time.dt.strftime("%Y-%m-%d").values
        common = sorted(set(obs_dates) & set(gcm_dates))
        if not common:
            print("  SKIP: no overlapping time steps")
            continue
        common_set = set(common)
        obs = obs.isel(time=[i for i, d in enumerate(obs_dates) if d in common_set])
        gcm = gcm.isel(time=[i for i, d in enumerate(gcm_dates) if d in common_set])
        print(f"  Overlap    : {len(common)} timesteps")

        # Downscale: GCM → ERA5 grid
        gcm_on_era5 = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
        gcm_on_era5 = gcm_on_era5.assign_coords(time=obs.time)

        print(f"  {method_label} downscaling ...")
        if effective_method == "bicubic":
            pred = gcm_on_era5
        elif effective_method == "bcsd":
            pred = bcsd_downscale(gcm, obs, variable)
        elif effective_method == "gnn":
            pred = gnn_downscale(gcm, obs, variable)
        elif effective_method == "serpentflow":
            if variable not in serpentflow_preds:
                print(f"  SKIP: no SerpentFlow prediction for {variable}")
                continue
            pred = serpentflow_preds[variable]
            # Re-align to obs time axis
            pred_dates = pred.time.dt.strftime("%Y-%m-%d").values
            obs_dates2  = obs.time.dt.strftime("%Y-%m-%d").values
            common2 = sorted(set(pred_dates) & set(obs_dates2))
            if not common2:
                print(f"  SKIP: SerpentFlow and ERA5 times do not overlap for {variable}")
                continue
            common2_set = set(common2)
            obs = obs.isel(time=[i for i, d in enumerate(obs_dates2) if d in common2_set])
            gcm_on_era5 = gcm_on_era5.isel(
                time=[i for i, d in enumerate(obs_dates2) if d in common2_set])
            pred = pred.isel(time=[i for i, d in enumerate(pred_dates) if d in common2_set])
        else:  # consistency / cm — prediction already computed jointly above
            if variable not in consistency_preds:
                print(f"  SKIP: no consistency prediction for {variable}")
                continue
            pred = consistency_preds[variable]

        all_obs[variable]   = obs
        all_preds[variable] = pred

        # Compute metrics
        n_t = obs.sizes["time"]
        if n_t < 200:
            print(f"  NOTE: ERA5 has {n_t} timesteps (annual means). "
                  "Daily metrics (KS tails, seasonal, extremes, autocorr) are skipped.")
        print("  Computing metrics ...")
        vs_obs, vs_gcm, plot_series = compute_all_metrics(obs, gcm_on_era5, pred, n_t)
        all_metrics[variable]     = {"vs_obs": vs_obs, "vs_gcm": vs_gcm}
        all_plot_series[variable] = plot_series

        # Print summary
        obs_summary = "  ".join(f"{k}={v:.4f}" for k, v in vs_obs.items())
        gcm_summary = "  ".join(f"{k}={v:.4f}" for k, v in vs_gcm.items())
        print(f"  vs ERA5 → {obs_summary}")
        print(f"  vs GCM  → {gcm_summary}")

        # Generate plots
        print("  Plotting ...")
        var_plot_dir = plot_dir / variable
        var_plot_dir.mkdir(exist_ok=True)
        daily = n_t >= 200

        # Obs-based scale limits so all methods share identical axes
        obs_mean_map  = obs.mean("time").values
        obs_std_map   = obs.std("time").values
        # bias_vmax: scale of SPATIAL variability in the ERA5 climatology
        # (remove domain mean first so absolute temperatures don't blow the scale)
        bias_vmax = float(np.nanpercentile(
            np.abs(obs_mean_map - np.nanmean(obs_mean_map)), 95))
        bias_vmax = max(bias_vmax, 1e-6)   # guard against flat fields
        # std_vmax: median temporal variability — a natural scale for std errors
        std_vmax  = float(np.nanmedian(obs_std_map))
        cdf_xlim  = (
            float(np.nanpercentile(obs.values, 0.5)),
            float(np.nanpercentile(obs.values, 99.5)),
        )
        obs_cycle = obs.groupby("time.month").mean("time").mean(["lat", "lon"])
        cyc_range = float(obs_cycle.max() - obs_cycle.min())
        seasonal_ylim = (
            float(obs_cycle.min()) - 0.15 * cyc_range,
            float(obs_cycle.max()) + 0.15 * cyc_range,
        )

        plot_bias_map(obs, pred, variable, method_label, var_plot_dir,
                      bias_vmax=bias_vmax, std_vmax=std_vmax)
        plot_cdf(obs, pred, variable, method_label, var_plot_dir, xlim=cdf_xlim)
        if daily:
            plot_seasonal_cycle(obs, pred, variable, method_label, var_plot_dir,
                                ylim=seasonal_ylim)
        plot_temporal_corr(gcm_on_era5, pred, variable, method_label, var_plot_dir)
        if daily:
            plot_autocorr_comparison(gcm_on_era5, pred, variable, method_label, var_plot_dir)
        plot_annual_anomalies(gcm_on_era5, pred, variable, method_label, var_plot_dir)

    if not all_metrics:
        print("No variables could be evaluated.")
        return 1

    # Inter-variable temporal correlations
    VAR_PAIRS = [
        ("tas", "pr"), ("tas", "sfcWind"), ("tas", "sfcWindmax"),
        ("pr", "sfcWind"), ("pr", "psl"), ("sfcWind", "sfcWindmax"),
    ]
    intervar: dict[str, float] = {}
    print(f"\n{'='*60}")
    print("  Inter-variable correlations")
    print(f"{'='*60}")
    for v1, v2 in VAR_PAIRS:
        if v1 not in all_obs or v2 not in all_obs:
            continue
        try:
            score = intervar_corr_mae(all_obs[v1], all_obs[v2],
                                      all_preds[v1], all_preds[v2])
            intervar[f"{v1}-{v2}"] = score
            print(f"  {v1}-{v2}: {score:.4f}")
            plot_intervar_corr(all_obs[v1], all_obs[v2],
                               all_preds[v1], all_preds[v2],
                               v1, v2, method_label, plot_dir)
        except Exception as exc:
            print(f"  {v1}-{v2}: ERROR {exc}")
    if intervar:
        all_metrics["_intervar"] = intervar

    # Aggregate metrics across variables and generate radar
    var_metrics = {v: m for v, m in all_metrics.items() if not v.startswith("_")}
    agg_obs = {k: float(np.nanmean([var_metrics[v]["vs_obs"][k] for v in var_metrics]))
               for k in next(iter(var_metrics.values()))["vs_obs"]}
    agg_gcm = {k: float(np.nanmean([var_metrics[v]["vs_gcm"][k] for v in var_metrics]))
               for k in next(iter(var_metrics.values()))["vs_gcm"]}
    plot_radar(agg_obs, agg_gcm, method_label, plot_dir)

    # Save metrics JSON
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save time-series data for interactive plotly charts in the dashboard
    # Structure: {variable: {seasonal:{months,era5,pred}, annual:{years,gcm,pred,...},
    #              cdf:{era5_x,pred_x}, spectra:{k_centers_km,era5,pred,...}}}
    plot_data_file = out_dir / "plot_data.json"
    with open(plot_data_file, "w") as f:
        json.dump(all_plot_series, f, indent=2)

    print(f"\nMetrics saved   → {metrics_file}")
    print(f"Plot data saved → {plot_data_file}")
    print(f"Plots saved     → {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
