#!/usr/bin/env python3
"""Evaluate a downscaling method against ERA5 and the source GCM.

Metrics are computed along two axes:
  * vs ERA5 (obs)  — bias, RMSE, KS, seasonal cycle, extremes
  * vs GCM (input) — temporal correlation, lag-1 autocorrelation, annual anomaly

Usage:
    python evaluation/evaluate.py                      # uses configs/benchmark.yaml
    python evaluation/evaluate.py --method bcsd        # run inline
    python evaluation/evaluate.py --method cm \\
        --pred-dir /scratch/.../results/consistency    # load pre-computed predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

# ROOT = src/ locally, downscaling_benchmark/ on Cleps — metrics/ and plots/ live there.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for data_processing

from data_processing import (
    load_era5, load_gcm, bicubic_regrid, align_time, load_predictions,
)

from baselines import METHOD_REGISTRY  # ROOT (=src/) already on sys.path

from metrics import (
    spatial_mean_abs_bias, std_error, ks_stat, seasonal_mae,
    extreme_freq_bias, extreme_intensity_bias, clim_rmse,
    mean_temporal_corr, mean_temporal_corr_annual,
    lag1_autocorr_mae, annual_anomaly_bias,
    intervar_corr_mae, spatial_spectrum_metric,
    delta_change_score,
)
from plots import (
    plot_bias_map, plot_cdf, plot_seasonal_cycle,
    plot_temporal_corr, plot_autocorr_comparison,
    plot_annual_anomalies, plot_radar, plot_intervar_corr, plot_delta_map,
    plot_sample_days,
)

from energy_views import (
    build_doy_block, domain_series, energy_events, resolve_sites,
    EVENT_VARIABLES,
)


def _align_pair(obs: xr.DataArray, gcm: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Subset obs and gcm to their common date intersection."""
    aligned = align_time({"obs": obs, "gcm": gcm})
    return aligned["obs"], aligned["gcm"]


def _infer_region(out_dir: Path, era5_dir: Path | None) -> str | None:
    """Benchmark region for the energy-site views (doy cycles, events).

    Explicit --region wins; otherwise match the output path against
    results/{region}/... and fall back to comparing --era5-dir with
    src/configs/data/regions.yaml. Returns None when unsure (the doy/events
    artefacts are then skipped — everything else still runs).
    """
    regions_file = ROOT / "src" / "configs" / "data" / "regions.yaml"
    try:
        regions = yaml.safe_load(regions_file.read_text()).get("regions", {})
    except Exception:
        return None
    try:
        rel = out_dir.resolve().relative_to((ROOT / "results").resolve())
        if rel.parts and rel.parts[0] in regions:
            return rel.parts[0]
    except ValueError:
        pass
    if era5_dir is not None:
        for name, cfg in regions.items():
            try:
                if (ROOT / cfg["era5_dir"]).resolve() == era5_dir.resolve():
                    return name
            except Exception:
                continue
    return None




def _consistency_downscale_all_legacy(
    gcm_vars: dict[str, xr.DataArray],
    obs_vars: dict[str, xr.DataArray],
    checkpoint_dir: Path,
    stroke_time: float = 5.0,
    n_members: int = 1,
) -> dict[str, xr.DataArray]:
    """Legacy: inline CM inference using old diffusers UNet2DModel checkpoint.
    The new train_consistency.py saves UNetAttention checkpoints; run
    inference_cm.py separately and load via --pred-dir. Kept for backward compat.

    Multivariate consistency-model downscaling (stroke guidance).

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

    VARIABLES    = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
    N_VARS       = len(VARIABLES)
    N_MONTH_CH   = 2
    DATA_STD     = 0.5
    T_MIN        = 0.002
    PAD          = (0, 15, 0, 7)
    H_OBS, W_OBS = 57, 65
    LOG_EPS      = 1e-6
    LOG_VARS     = {"pr"}
    CHANNELS     = (128, 128, 256, 256)
    DOWN_BLOCKS  = ("DownBlock2D",) * 4
    UP_BLOCKS    = ("UpBlock2D",)   * 4

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
    # Support both old checkpoints (in_channels=5) and new (in_channels=7 with month)
    n_in = ckpt.get("in_channels", N_VARS)

    net = UNet2DModel(
        in_channels=n_in, out_channels=N_VARS,
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

    # ── Month covariate (if checkpoint was trained with month channels) ────────
    use_month = (n_in == N_VARS + N_MONTH_CH)
    if use_month:
        import pandas as _pd
        months = np.array([_pd.Timestamp(str(d)).month
                           for d in ref_obs.time.values], dtype=np.float32)
        sin_m = np.sin(2 * np.pi * months / 12)[:, None, None]
        cos_m = np.cos(2 * np.pi * months / 12)[:, None, None]
        sin_ch = np.broadcast_to(sin_m, (T, H_OBS, W_OBS)).copy()
        cos_ch = np.broadcast_to(cos_m, (T, H_OBS, W_OBS)).copy()
        month_stack = np.stack([sin_ch, cos_ch], axis=1)  # [T, 2, H, W]
        gcm_full_stack = np.concatenate([gcm_stack, month_stack], axis=1)  # [T, N_IN, H, W]
    else:
        gcm_full_stack = gcm_stack  # [T, N_VARS, H, W] — old checkpoint

    # ── Consistency function ───────────────────────────────────────────────────
    def F_cm(x_t: torch.Tensor, t_val: float) -> torch.Tensor:
        t    = torch.full((x_t.shape[0],), t_val, device=device)
        t4   = t[:, None, None, None]
        skip     = DATA_STD**2 / ((t4 - T_MIN)**2 + DATA_STD**2)
        out_coef = DATA_STD * t4 / (t4**2 + DATA_STD**2)**0.5
        with torch.no_grad():
            pred = net(x_t, t).sample                    # [B, N_VARS, H, W]
        return skip * x_t[:, :N_VARS] + out_coef * pred

    # ── Batched stroke-guidance inference ─────────────────────────────────────
    print(f"  [consistency] stroke guidance t*={stroke_time}, "
          f"members={n_members}, {T} timesteps ...", flush=True)

    batch_size = 16
    out_chunks = []   # list of [B, N_VARS, H, W] numpy arrays

    for start in range(0, T, batch_size):
        sl   = slice(start, start + batch_size)
        x_np = gcm_full_stack[sl]                                 # [B, n_in, H, W]
        x    = torch.from_numpy(x_np).to(device)
        x    = TF.pad(x, PAD)                                     # [B, n_in, 64, 80]

        member_outs = []
        for _ in range(n_members):
            # Noise only on climate channels; month channels stay clean
            B_cur = x.shape[0]
            noise  = torch.randn(B_cur, N_VARS, x.shape[2], x.shape[3], device=device)
            x_init = x.clone()
            x_init[:, :N_VARS] += noise * stroke_time
            x_out  = F_cm(x_init, stroke_time)                    # [B, N_VARS, 64, 80]
            x_out  = x_out[:, :, :H_OBS, :W_OBS]                 # [B, N_VARS, H, W]
            member_outs.append(x_out.cpu().numpy())               # [B, N_VARS, H, W]

        out_chunks.append(np.mean(member_outs, axis=0))           # [B, N_VARS, H, W]

    out_all = np.concatenate(out_chunks, axis=0)   # [T, N_VARS, H, W]

    # ── Decode each variable and return ───────────────────────────────────────
    # Physical lower bounds after decoding.
    PHYS_MIN = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0, "rsds": 0.0}
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


def compute_all_metrics(
    obs: xr.DataArray,
    pred_obs: xr.DataArray,
    gcm_on_era5: xr.DataArray,
    pred_gcm: xr.DataArray,
    n_timesteps: int,
) -> tuple[dict, dict, dict]:
    """Return (metrics_vs_obs, metrics_vs_gcm, plot_series) as dicts.

    obs / pred_obs  — ERA5 evaluation period (2000-2020): used for all vs-ERA5 metrics.
    gcm_on_era5 / pred_gcm — full GCM period (2000-2100): used for all vs-GCM metrics
                             so that temporal correlation, autocorr, and interannual
                             variability are computed over the full climate scenario.

    Metrics requiring daily data are omitted when n_timesteps < 200.
    """
    daily = n_timesteps >= 200

    vs_obs: dict = {
        "bias_mae":  spatial_mean_abs_bias(obs, pred_obs),
        "std_error": std_error(obs, pred_obs),
        "ks_full":   ks_stat(obs, pred_obs),
    }
    if daily:
        vs_obs["ks_q95"]            = ks_stat(obs, pred_obs, quantile=0.95)
        vs_obs["seasonal_mae"]      = seasonal_mae(obs, pred_obs)
        vs_obs["extreme_freq"]      = extreme_freq_bias(obs, pred_obs, q=0.95)
        vs_obs["extreme_intensity"] = extreme_intensity_bias(obs, pred_obs, q=0.95)

    # Spatial spectrum metric (ERA5 period)
    spec_score, spec_data = spatial_spectrum_metric(obs, pred_obs)
    vs_obs["spectral_mae"] = spec_score

    # All GCM metrics on the full period (pred_gcm / gcm_on_era5 — 2000-2100)
    vs_gcm: dict = {
        "temporal_corr":        mean_temporal_corr(gcm_on_era5, pred_gcm),
        "temporal_corr_annual": mean_temporal_corr_annual(gcm_on_era5, pred_gcm),
        "annual_anomaly_bias":  annual_anomaly_bias(gcm_on_era5, pred_gcm),
    }
    if daily:
        vs_gcm["lag1_autocorr_mae"] = lag1_autocorr_mae(gcm_on_era5, pred_gcm)

    # Time-series data for plotly line charts
    plot_series: dict = {}
    if daily:
        obs_cyc  = obs.groupby("time.month").mean("time").mean(["lat", "lon"])
        pred_cyc = pred_obs.groupby("time.month").mean("time").mean(["lat", "lon"])
        plot_series["seasonal"] = {
            "months": list(range(1, 13)),
            "era5":   [float(v) for v in obs_cyc.values],
            "pred":   [float(v) for v in pred_cyc.values],
        }

    # Annual timeseries over the full GCM period
    gcm_ann  = gcm_on_era5.mean(["lat", "lon"]).groupby("time.year").mean("time")
    pred_ann = pred_gcm.mean(["lat", "lon"]).groupby("time.year").mean("time")
    obs_ann  = obs.mean(["lat", "lon"]).groupby("time.year").mean("time")
    plot_series["annual"] = {
        "years":      [int(y)   for y in gcm_ann.year.values],
        "gcm":        [float(v) for v in gcm_ann.values],
        "years_pred": [int(y)   for y in pred_ann.year.values],
        "pred":       [float(v) for v in pred_ann.values],
        "years_era5": [int(y)   for y in obs_ann.year.values],
        "era5":       [float(v) for v in obs_ann.values],
    }

    # CDF (downsampled to 500 pts, ERA5 period)
    o_all = obs.values.ravel();      o_all = o_all[np.isfinite(o_all)]; o_all.sort()
    p_all = pred_obs.values.ravel(); p_all = p_all[np.isfinite(p_all)]; p_all.sort()
    if len(o_all) > 0 and len(p_all) > 0:
        idx_o = np.linspace(0, len(o_all) - 1, 500).astype(int)
        idx_p = np.linspace(0, len(p_all) - 1, 500).astype(int)
        plot_series["cdf"] = {
            "era5_x": [float(v) for v in o_all[idx_o]],
            "pred_x": [float(v) for v in p_all[idx_p]],
        }
    else:
        plot_series["cdf"] = {"era5_x": [], "pred_x": []}

    plot_series["spectra"] = spec_data

    return vs_obs, vs_gcm, plot_series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--era5-dir", default=None)
    parser.add_argument("--gcm-dir", default=None,
                        help="GCM directory (historical + scenario .nc files). "
                             "Defaults to configs/benchmark.yaml ipsl_cm6a_lr path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variables", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end",   type=int, default=None,
                        help="End of ERA5 evaluation period (default from config).")
    parser.add_argument("--gcm-end", type=int, default=2100,
                        help="End of full GCM period for delta metrics (default 2100). "
                             "Predictions loaded via --pred-dir must cover start–gcm-end.")
    parser.add_argument("--scenario", default=None,
                        help="GCM future experiment (default: config `scenario`, else ssp370). "
                             "Must match the scenario the predictions were made on.")
    parser.add_argument("--method",
                        default="bicubic",
                        help="Method name — any string. Built-in inline methods: "
                             "bicubic, cdft. Pre-compute methods (require --pred-dir): "
                             "bcsd, idownscale, consistency/cm. "
                             "Custom names (hackathon submissions) are accepted when "
                             "--pred-dir is provided. cm=alias for consistency.")
    parser.add_argument("--pred-dir", default=None,
                        help="Directory with pre-computed predictions.pt (standard format). "
                             "Required for neural methods (idownscale, consistency). "
                             "Use output of inference_idownscale/inference_cm.py.")
    parser.add_argument("--region", default=None,
                        help="Benchmark region (france, australia, us_central). Used for the "
                             "day-of-year cycles and energy-event stats written to plot_data.json; "
                             "inferred from --output-dir / --era5-dir when omitted.")
    args = parser.parse_args(argv)

    cfg_path = ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    era5_dir = Path(args.era5_dir or cfg["era5"]["dir"])
    gcm_dir  = Path(args.gcm_dir  or cfg["models"]["ipsl_cm6a_lr"]["dir"])
    variables = args.variables or cfg["variables"]
    start   = args.start   or cfg["period"]["start"]
    end     = args.end     or cfg["period"]["end"]
    gcm_end = args.gcm_end  # full GCM period end for vs-GCM metrics
    scenario = args.scenario or cfg.get("scenario", "ssp370")
    # cm is a display-name alias
    effective_method = {
        "cm": "consistency",
    }.get(args.method, args.method)
    dic_labels = {
        "bicubic":     "Bicubic",
        "bcsd":        "BCSD",
        "idownscale":  "iDownscale",
        "consistency": "CM",
        "cm":          "CM",
    }
    if args.method in dic_labels:
        method_label = dic_labels[args.method]
    else:
        method_label = args.method

    # Output dir uses the canonical name
    out_dir = Path(args.output_dir or cfg.get("output_dir", "results")) / args.method

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Region drives the energy-site day-of-year cycles and the event spells
    # (both stored in plot_data.json, read by the dashboard).
    region = args.region or _infer_region(out_dir, era5_dir)
    if region is None:
        print("\nNOTE: benchmark region unknown — day-of-year cycles and "
              "energy events skipped (pass --region to enable).")

    all_metrics: dict[str, dict] = {}
    all_obs:     dict[str, xr.DataArray] = {}
    all_preds:   dict[str, xr.DataArray] = {}
    all_preds_full: dict[str, xr.DataArray] = {}   # full GCM-period predictions for delta
    all_plot_series: dict[str, dict] = {}
    all_doy:    dict[str, dict] = {}   # variable → {site: {era5, gcm, pred}}
    all_domain: dict[str, dict] = {}   # variable → {era5, gcm, pred} region-mean series
    sites: list | None = None          # resolved once per region

    # ── Pre-load any pre-computed predictions ─────────────────────────────────
    precomputed_preds: dict[str, xr.DataArray] = {}

    if args.pred_dir:
        # Standard format (predictions.pt) — produced by inference_*.py scripts
        print(f"\nLoading pre-computed predictions from {args.pred_dir} ...")
        try:
            precomputed_preds = load_predictions(Path(args.pred_dir), variables)
            print(f"  Loaded: {list(precomputed_preds.keys())}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1

    elif effective_method in ("consistency", "idownscale", "bcsd"):
        # Pre-compute methods: run inference script first, then evaluate via --pred-dir
        if not args.pred_dir:
            script = {
                "bcsd": "inference_bcsd.py",
                "idownscale": "inference_idownscale.py",
                "consistency": "train_consistency.py + inference_cm.py",
            }.get(effective_method, f"inference_{effective_method}.py")
            print(f"ERROR: --pred-dir is required for --method {args.method}. "
                  f"Run {script} first.")
            return 1

    for variable in variables:
        print(f"\n{'='*60}")
        print(f"  {variable}  (ERA5: {start}–{end}  |  GCM: {start}–{gcm_end})")
        print(f"{'='*60}")

        try:
            obs = load_era5(era5_dir, variable, start, end)
        except FileNotFoundError as e:
            print(f"  SKIP ERA5: {e}")
            continue

        # Load GCM for the full period (2000-2100) — used for all vs-GCM metrics
        try:
            gcm_full = load_gcm(gcm_dir, variable, start, gcm_end, scenario)
        except FileNotFoundError as e:
            print(f"  SKIP GCM: {e}")
            continue

        print(f"  ERA5 grid : {obs.lat.size}×{obs.lon.size}, {obs.time.size} timesteps")
        print(f"  GCM grid  : {gcm_full.lat.size}×{gcm_full.lon.size}, "
              f"{gcm_full.time.size} timesteps")

        # Bicubic-regrid full GCM to ERA5 grid (keep GCM's own time axis)
        gcm_on_era5_full = bicubic_regrid(gcm_full, obs.lat.values, obs.lon.values)

        # ── Resolve predictions ───────────────────────────────────────────────
        print(f"  {method_label} downscaling ...")
        if precomputed_preds:
            if variable not in precomputed_preds:
                print(f"  SKIP: no pre-computed prediction for {variable}")
                continue
            pred_full = precomputed_preds[variable]   # full period (2000-2100)
            all_preds_full[variable] = pred_full

            # pred_obs: align predictions to ERA5 period for vs-ERA5 metrics
            obs, pred_obs = _align_pair(obs, pred_full)

            # pred_gcm / gcm_metrics: align pred_full with full-period GCM
            pred_gcm, gcm_metrics = _align_pair(pred_full, gcm_on_era5_full)

        elif effective_method in METHOD_REGISTRY:
            gcm_era5_period, _ = _align_pair(
                gcm_full.sel(time=gcm_full.time.dt.year.isin(range(start, end + 1))),
                obs,
            )
            obs, gcm_era5_period = _align_pair(obs, gcm_era5_period)
            gcm_metrics_era5 = bicubic_regrid(gcm_era5_period, obs.lat.values, obs.lon.values)
            gcm_metrics_era5 = gcm_metrics_era5.assign_coords(time=obs.time)
            pred_obs = METHOD_REGISTRY[effective_method](gcm_era5_period, obs, variable)
            if effective_method == "bicubic":
                # Bicubic full-period prediction = bicubic-regridded full GCM (free, already computed)
                pred_gcm = gcm_on_era5_full
                gcm_metrics = gcm_on_era5_full
                all_preds_full[variable] = gcm_on_era5_full
            else:
                pred_gcm = pred_obs
                gcm_metrics = gcm_metrics_era5
        else:
            print(f"  SKIP: unknown method {effective_method}")
            continue

        # GCM on the ERA5 grid with the ERA5 time axis (for the sample-day plot)
        gcm_on_era5 = None
        try:
            gcm_era5_period = gcm_full.sel(
                time=gcm_full.time.dt.year.isin(range(start, end + 1)))
            _, gcm_era5_period = _align_pair(obs, gcm_era5_period)
            gcm_on_era5 = bicubic_regrid(gcm_era5_period,
                                         obs.lat.values, obs.lon.values)
            gcm_on_era5 = gcm_on_era5.assign_coords(time=obs.time)
        except Exception as exc:
            print(f"  NOTE: sample-days GCM unavailable: {exc}")

        all_obs[variable]   = obs
        all_preds[variable] = pred_obs

        # Day-of-year cycles + region-mean daily series (events) — dashboard
        if region is not None:
            try:
                if sites is None:
                    sites = resolve_sites(region, obs.lat.values, obs.lon.values)
                block = build_doy_block(obs, gcm_on_era5, pred_obs, sites)
                if block:
                    all_doy[variable] = block
                all_domain[variable] = {
                    "era5": domain_series(obs),
                    "gcm":  domain_series(gcm_on_era5) if gcm_on_era5 is not None else None,
                    "pred": domain_series(pred_obs),
                }
            except Exception as exc:
                print(f"  NOTE: day-of-year/events unavailable: {exc}")

        # Compute metrics
        n_t = obs.sizes["time"]
        if n_t < 200:
            print(f"  NOTE: ERA5 has {n_t} timesteps (annual means). "
                  "Daily metrics (KS tails, seasonal, extremes, autocorr) are skipped.")
        print(f"  Computing metrics "
              f"(ERA5: {obs.time.size} steps, GCM: {pred_gcm.time.size} steps) ...")
        vs_obs, vs_gcm, plot_series = compute_all_metrics(
            obs, pred_obs, gcm_metrics, pred_gcm, n_t)
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
        bias_vmax = float(np.nanpercentile(
            np.abs(obs_mean_map - np.nanmean(obs_mean_map)), 95))
        bias_vmax = max(bias_vmax, 1e-6)
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

        # ERA5-period plots (bias, CDF, seasonal)
        plot_bias_map(obs, pred_obs, variable, method_label, var_plot_dir,
                      bias_vmax=bias_vmax, std_vmax=std_vmax)
        plot_cdf(obs, pred_obs, variable, method_label, var_plot_dir, xlim=cdf_xlim)
        if daily:
            plot_seasonal_cycle(obs, pred_obs, variable, method_label, var_plot_dir,
                                ylim=seasonal_ylim)
        if gcm_on_era5 is not None:
            plot_sample_days(obs, pred_obs, gcm_on_era5, variable, method_label,
                             var_plot_dir)
        # GCM-period plots (correlation, autocorr, annual anomalies — full 2000-2100)
        plot_temporal_corr(gcm_metrics, pred_gcm, variable, method_label, var_plot_dir)
        if daily:
            plot_autocorr_comparison(gcm_metrics, pred_gcm, variable, method_label, var_plot_dir)
        plot_annual_anomalies(gcm_metrics, pred_gcm, variable, method_label, var_plot_dir,
                              obs=obs)

    if not all_metrics:
        print("No variables could be evaluated.")
        return 1

    # Inter-variable temporal correlations
    VAR_PAIRS = [
        ("tas", "pr"), ("tas", "sfcWind"), ("tas", "sfcWindmax"),
        ("pr", "sfcWind"), ("pr", "psl"), ("sfcWind", "sfcWindmax"),
        # Radiation: solar-driven warming, cloud/rain anticorrelation, clear skies under highs.
        ("tas", "rsds"), ("pr", "rsds"), ("psl", "rsds"),
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

    # ── Delta change score (climate signal preservation) ──────────────────────
    # Uses full prediction period (start–gcm_end) vs GCM reference.
    # hist = start–end (ERA5 overlap), fut1 = 2040-2060, fut2 = 2080-2100.
    if all_preds_full and gcm_end > end:
        print(f"\n{'='*60}")
        print(f"  Delta change scores (hist={start}–{end}, "
              f"fut1=2040–2060, fut2=2080–2100)")
        print(f"{'='*60}")
        delta_all: dict[str, dict] = {}
        _ref_obs = next(iter(all_obs.values()))
        for variable in variables:
            if variable not in all_preds_full:
                continue
            try:
                gcm_full = load_gcm(gcm_dir, variable, start, gcm_end, scenario)
                gcm_full_up = bicubic_regrid(gcm_full, _ref_obs.lat.values, _ref_obs.lon.values)
                pred_full = all_preds_full[variable]

                hist_pred = pred_full.sel(time=pred_full.time.dt.year.isin(range(start, end + 1)))
                hist_gcm  = gcm_full_up.sel(time=gcm_full_up.time.dt.year.isin(range(start, end + 1)))

                delta_var: dict[str, float] = {}
                for label, (fy0, fy1) in [("2040_2060", (2040, 2060)), ("2080_2100", (2080, 2100))]:
                    fut_pred = pred_full.sel(
                        time=pred_full.time.dt.year.isin(range(fy0, fy1 + 1)))
                    fut_gcm  = gcm_full_up.sel(
                        time=gcm_full_up.time.dt.year.isin(range(fy0, fy1 + 1)))
                    if fut_pred.time.size < 30 or fut_gcm.time.size < 30:
                        print(f"  {variable} {label}: too few timesteps, skip")
                        continue
                    # delta_change_score uses .mean("time") → no need for exact time alignment
                    score = delta_change_score(hist_pred, fut_pred, hist_gcm, fut_gcm)
                    delta_var[label] = score
                    print(f"  {variable} Δ {label}: {score:.4f}")
                    pred_delta = fut_pred.mean("time").values - hist_pred.mean("time").values
                    gcm_delta  = fut_gcm.mean("time").values  - hist_gcm.mean("time").values
                    var_plot_dir = plot_dir / variable
                    var_plot_dir.mkdir(exist_ok=True)
                    plot_delta_map(pred_delta, gcm_delta, pred_delta - gcm_delta,
                                   _ref_obs.lat.values, _ref_obs.lon.values,
                                   variable, method_label, label, var_plot_dir)
                delta_all[variable] = delta_var
            except Exception as exc:
                print(f"  {variable} delta ERROR: {exc}")
        if delta_all:
            all_metrics["_delta"] = delta_all

    # ── Day-of-year cycles + energy-event spells (dashboard) ─────────────────
    if all_doy:
        for variable, block in all_doy.items():
            all_plot_series.setdefault(variable, {})["doy"] = block
        n_locs = len(next(iter(all_doy.values())))
        print(f"\nDay-of-year cycles: {len(all_doy)} variable(s) × "
              f"{n_locs} location(s)")
    if all_domain and all(v in all_domain for v in EVENT_VARIABLES):
        events = energy_events(all_domain, period=(start, end))
        if events:
            all_plot_series["_events"] = events
            c = events["dunkelflaute"]["count"]
            print(f"Energy events (dunkelflaute, ≥{events['meta']['min_days']}-day spells): "
                  f"era5/gcm/pred = "
                  f"{c.get('era5')}/{c.get('gcm', '-')}/{c.get('pred', '-')}")

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
    #              cdf:{era5_x,pred_x}, spectra:{k_centers_km,era5,pred,...},
    #              doy:{site:{era5,gcm,pred}}}, _events:{...}}
    plot_data_file = out_dir / "plot_data.json"
    with open(plot_data_file, "w") as f:
        json.dump(all_plot_series, f, separators=(",", ":"))

    print(f"\nMetrics saved   → {metrics_file}")
    print(f"Plot data saved → {plot_data_file}")
    print(f"Plots saved     → {plot_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
