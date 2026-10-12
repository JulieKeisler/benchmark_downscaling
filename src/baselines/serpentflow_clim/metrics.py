"""Evaluation metrics (paper Appendix A).

This is the single, deduplicated version of the metric functions. The
notebooks defined most of these twice per file (a first "plain" version,
then a near-identical version ~800 lines later adding a `mask=` land/sea
breakdown); only the later definition was actually live at call time. Here
there is only one definition, matching that live (mask-capable) behaviour.
"""

import numpy as np
import xarray as xr
from scipy import stats


def flatten_spacetime(da: xr.DataArray) -> np.ndarray:
    return da.values.reshape(-1)


def _resolve_land(mask, reference) -> xr.DataArray:
    """Binarize a [0,1] land-sea mask (>0.5 = land), aligned to `reference`'s grid."""
    lat, lon = reference["latitude"], reference["longitude"]
    if isinstance(mask, np.ndarray):
        mask = xr.DataArray(mask, dims=["latitude", "longitude"], coords={"latitude": lat, "longitude": lon})
    else:
        mask = mask.reindex(latitude=lat, longitude=lon)
    return mask > 0.5


def _full_or_split(mask, reference, full_value, region_fn):
    """Shared shape for metrics with a land/sea breakdown: `full_value` when
    no mask is given, else `{"full", "land", "sea"}` with `region_fn` applied
    to the (boolean) land mask and its complement.
    """
    if mask is None:
        return full_value
    land = _resolve_land(mask, reference)
    return {"full": full_value, "land": region_fn(land), "sea": region_fn(~land)}


# ------------------------------------------------------- ERA5-relative -----

def mean_std_differences(pred: xr.DataArray, obs: xr.DataArray, mask=None):
    """(Delta mu, Delta sigma) -- Eqs. mu-diff / sigma-diff."""
    mean_pred, mean_obs = pred.mean(dim="time"), obs.mean(dim="time")
    std_pred, std_obs = pred.std(dim="time", ddof=1), obs.std(dim="time", ddof=1)
    abs_mean, abs_std = np.abs(mean_pred - mean_obs), np.abs(std_pred - std_obs)

    delta_mean = float(abs_mean.mean(dim=["latitude", "longitude"]).values)
    delta_std = float(abs_std.mean(dim=["latitude", "longitude"]).values)

    def sp_mean(field):
        return lambda region: float(field.where(region).mean(dim=["latitude", "longitude"]).values)

    return (_full_or_split(mask, mean_pred, delta_mean, sp_mean(abs_mean)),
            _full_or_split(mask, mean_pred, delta_std, sp_mean(abs_std)))


def ks_statistic(pred: xr.DataArray, obs: xr.DataArray, mask=None):
    """Two-sample KS statistic sup|F_pred - F_obs| over pooled space-time (Eq. ks)."""
    full = float(stats.ks_2samp(flatten_spacetime(pred), flatten_spacetime(obs))[0])

    def ks_region(region):
        fp = pred.where(region).values.reshape(-1)
        fo = obs.where(region).values.reshape(-1)
        return float(stats.ks_2samp(fp[~np.isnan(fp)], fo[~np.isnan(fo)])[0])

    return _full_or_split(mask, pred, full, ks_region)


def spatial_spearman_structure(pred: xr.DataArray, obs: xr.DataArray, mask=None):
    """Mean absolute difference between spatial (cell-to-cell) Spearman correlation matrices.

    Diagnostic only -- not one of the metrics reported in the paper's tables.
    """
    def rank_spatial(da):
        flat = da.values.reshape(da.sizes["time"], -1)
        return da.copy(data=stats.rankdata(flat, axis=1).reshape(da.shape))

    pred_anom = pred - pred.mean(dim=["latitude", "longitude"])
    obs_anom = obs - obs.mean(dim=["latitude", "longitude"])
    pred_rank, obs_rank = rank_spatial(pred_anom), rank_spatial(obs_anom)
    n_points = pred_rank.sizes["latitude"] * pred_rank.sizes["longitude"]
    R_pred = np.corrcoef(pred_rank.values.reshape(pred_rank.sizes["time"], n_points).T)
    R_obs = np.corrcoef(obs_rank.values.reshape(obs_rank.sizes["time"], n_points).T)
    full = float(np.mean(np.abs(R_pred - R_obs)))
    if mask is None:
        return full

    land = _resolve_land(mask, pred.isel(time=0))
    p2_all = pred.transpose("time", "latitude", "longitude").values.reshape(pred.sizes["time"], -1)
    o2_all = obs.transpose("time", "latitude", "longitude").values.reshape(obs.sizes["time"], -1)

    def spearman_region(region_bool):
        idx = np.where(region_bool.transpose("latitude", "longitude").values.reshape(-1))[0]
        p2 = p2_all[:, idx] - p2_all[:, idx].mean(axis=1, keepdims=True)
        o2 = o2_all[:, idx] - o2_all[:, idx].mean(axis=1, keepdims=True)
        Rp = np.corrcoef(stats.rankdata(p2, axis=1).T)
        Ro = np.corrcoef(stats.rankdata(o2, axis=1).T)
        return float(np.mean(np.abs(Rp - Ro)))

    return {"full": full, "land": spearman_region(land), "sea": spearman_region(~land)}


def spatial_power_spectrum_metric(pred: xr.DataArray, obs: xr.DataArray, kmax_fraction: float = 0.3, mask=None):
    """Relative RMS deviation of the large-scale isotropic power spectrum (Eq. ssm).

    Eq. (ssm) computes the spectrum per time step and averages the resulting
    power over time -- averaging the *field* over time first (as an earlier
    version of this function did) cancels the day-to-day stochastic
    high-frequency noise a generative model injects before the FFT ever
    sees it, understating its fine-scale content relative to a
    deterministic method whose field has little such noise to cancel.
    """
    def mean_power_spectrum(data3d):
        """data3d: (T, H, W) -> radially-binned power spectrum averaged over T."""
        fft = np.fft.fftshift(np.fft.fft2(data3d, axes=(-2, -1)), axes=(-2, -1))
        power = (np.abs(fft) ** 2).mean(axis=0)
        ny, nx = power.shape
        y, x = np.ogrid[-ny // 2:ny // 2, -nx // 2:nx // 2]
        r = np.sqrt(x ** 2 + y ** 2).astype(int)
        max_r = min(ny, nx) // 2
        bin_power = np.zeros(max_r)
        for i in range(max_r):
            m = r == i
            bin_power[i] = power[m].sum() / max(m.sum(), 1)
        return bin_power

    def metric_from_fields(pred3d, obs3d):
        P_pred, P_obs = mean_power_spectrum(pred3d), mean_power_spectrum(obs3d)
        kmax = max(int(len(P_obs) * kmax_fraction), 1)
        num = np.sqrt(np.mean((P_pred[:kmax] - P_obs[:kmax]) ** 2))
        den = np.mean(P_obs[:kmax])
        return float(num / den) if den > 0 else 0.0

    full = metric_from_fields(pred.values, obs.values)
    if mask is None:
        return full

    land = _resolve_land(mask, pred.isel(time=0))

    def regional_field(da, region):
        # Fill outside `region` with that time step's own region-mean (not a
        # single time-mean scalar) to avoid a sharp edge artifact in the FFT
        # while still varying correctly from one time step to the next.
        fill = da.where(region).mean(dim=["latitude", "longitude"])
        return da.where(region, fill).values

    return {"full": full,
            "land": metric_from_fields(regional_field(pred, land), regional_field(obs, land)),
            "sea": metric_from_fields(regional_field(pred, ~land), regional_field(obs, ~land))}


def inter_variable_correlation_consistency(pred_dict, obs_dict, mask=None):
    """Delta rho: mean |corr_pred - corr_obs| averaged over all variable pairs (Eq. ivcc-avg)."""
    variables = list(pred_dict.keys())
    if len(variables) < 2:
        raise ValueError("Need at least 2 variables")

    land = None if mask is None else _resolve_land(mask, pred_dict[variables[0]].isel(time=0))
    deltas = {"full": []} if mask is not None else []
    per_pair = {}

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v1, v2 = variables[i], variables[j]
            corr_pred = xr.corr(pred_dict[v1], pred_dict[v2], dim="time")
            corr_obs = xr.corr(obs_dict[v1], obs_dict[v2], dim="time")
            diff = np.abs(corr_pred - corr_obs)
            per_pair[f"{v1}-{v2}"] = diff
            if mask is None:
                deltas.append(float(diff.mean(dim=["latitude", "longitude"]).values))
            else:
                deltas["full"].append(float(diff.mean(dim=["latitude", "longitude"]).values))
                deltas.setdefault("land", []).append(float(diff.where(land).mean(dim=["latitude", "longitude"]).values))
                deltas.setdefault("sea", []).append(float(diff.where(~land).mean(dim=["latitude", "longitude"]).values))

    if mask is None:
        return float(np.mean(deltas)), per_pair
    return {k: float(np.mean(v)) for k, v in deltas.items()}, per_pair


def _upper_tail_chi(x: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    """Extremal dependence coefficient chi_q = P(F_Y>q | F_X>q), per grid point.

    x, y: (T, H, W) arrays sharing a time axis. Under independence chi_q ->
    1-q as q -> 1; under perfect positive tail dependence chi_q -> 1. This is
    the standard upper-tail dependence coefficient (Coles et al. 1999),
    computed empirically per pixel rather than via a fitted extreme-value
    model, since it only needs to be internally consistent between the
    model and ERA5, not an absolute estimate of the true tail index.
    """
    thresh_x = np.quantile(x, q, axis=0, keepdims=True)
    thresh_y = np.quantile(y, q, axis=0, keepdims=True)
    exceed_x = x > thresh_x
    exceed_y = y > thresh_y
    n_x = exceed_x.sum(axis=0)
    both = (exceed_x & exceed_y).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(n_x > 0, both / np.maximum(n_x, 1), np.nan)
    return chi


def tail_dependence_consistency(pred_dict, obs_dict, q: float = 0.9, mask=None):
    """Delta chi_q: mean |chi_pred - chi_obs| averaged over all variable pairs,
    the upper-tail-exceedance analogue of `inter_variable_correlation_consistency`
    (Eq. ivcc-avg). Pearson correlation summarizes the whole joint
    distribution and can look unbiased while the *co-occurrence rate of
    extremes* -- what compound-risk applications such as simultaneous high
    demand and low wind availability actually depend on -- is still wrong.
    `chi_q` measures exactly that co-occurrence rate for the upper
    ($q$-quantile-and-above) tail of each variable, independent of sign
    conventions since it only asks how often both variables are
    simultaneously in their own upper tail.
    """
    variables = list(pred_dict.keys())
    if len(variables) < 2:
        raise ValueError("Need at least 2 variables")

    land = None if mask is None else _resolve_land(mask, pred_dict[variables[0]].isel(time=0))
    deltas = {"full": []} if mask is not None else []
    per_pair = {}

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            v1, v2 = variables[i], variables[j]
            chi_pred = _upper_tail_chi(pred_dict[v1].values, pred_dict[v2].values, q)
            chi_obs = _upper_tail_chi(obs_dict[v1].values, obs_dict[v2].values, q)
            diff = np.abs(chi_pred - chi_obs)
            template = pred_dict[v1].isel(time=0).drop_vars("time", errors="ignore")
            diff_da = xr.DataArray(diff, coords=template.coords, dims=template.dims)
            per_pair[f"{v1}-{v2}"] = diff_da
            if mask is None:
                deltas.append(float(np.nanmean(diff)))
            else:
                deltas["full"].append(float(np.nanmean(diff)))
                deltas.setdefault("land", []).append(float(diff_da.where(land).mean(skipna=True).values))
                deltas.setdefault("sea", []).append(float(diff_da.where(~land).mean(skipna=True).values))

    if mask is None:
        return float(np.mean(deltas)), per_pair
    return {k: float(np.mean(v)) for k, v in deltas.items()}, per_pair


# -------------------------------------------------------- GCM-relative -----

def temporal_correlation_with_gcm(pred: xr.DataArray, gcm: xr.DataArray, mask=None):
    """Average temporal Pearson correlation with the raw GCM (Eq. rho-gcm)."""
    corr = xr.corr(pred, gcm, dim="time")
    full = float(corr.mean(dim=["latitude", "longitude"]).values)

    def sp_mean(region):
        return float(corr.where(region).mean(dim=["latitude", "longitude"]).values)

    return _full_or_split(mask, corr, full, sp_mean)


def global_annual_anomalies(pred: xr.DataArray, gcm: xr.DataArray, time_dim: str = "time", mask=None):
    """Mean absolute difference of spatially-averaged annual anomalies (Eq. annual-dev)."""
    pred_annual = pred.groupby(time_dim + ".year").mean(dim=time_dim)
    gcm_annual = gcm.groupby(time_dim + ".year").mean(dim=time_dim)

    def gaa(region):
        pa = pred_annual if region is None else pred_annual.where(region)
        ga = gcm_annual if region is None else gcm_annual.where(region)
        pred_global = pa.mean(dim=["latitude", "longitude"])
        gcm_global = ga.mean(dim=["latitude", "longitude"])
        return float(np.mean(np.abs((pred_global - pred_global.mean()) - (gcm_global - gcm_global.mean()))).values)

    return _full_or_split(mask, pred_annual, gaa(None), gaa)


def relative_climate_change_signal(pred: xr.DataArray, gcm: xr.DataArray, seasonal: bool = False, mask=None,
                                    hist_slice=slice("2000", "2020"),
                                    fut_slices=None):
    """Relative climate-change-signal discrepancy vs. the GCM (Eqs. relchange-*).

    Returns one value PER future window in `fut_slices` (default
    2040-2060 and 2080-2100) -- the paper's Delta_full/Delta_season are each
    a single number per method/variable, and nothing in the original
    notebooks reduced these two windows to one; which window (or an
    average) corresponds to the reported number is not stated in the paper
    and should be pinned down explicitly by the caller, not assumed here.
    """
    fut_slices = fut_slices or {"2040-2060": slice("2040", "2060"), "2080-2100": slice("2080", "2099")}

    def compute_relative_change(da, fut_slice):
        hist = da.sel({da.dims[0]: hist_slice}).mean(dim=da.dims[0])
        fut = da.sel({da.dims[0]: fut_slice}).mean(dim=da.dims[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            return (fut - hist) / hist * 100

    def region_mean(field, region):
        return (field if region is None else field.where(region)).mean().item()

    time_dim = pred.dims[0]
    seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}

    def metric_for(fut_slice, region):
        if not seasonal:
            dp = compute_relative_change(pred, fut_slice)
            dg = compute_relative_change(gcm, fut_slice)
            return float(np.abs(region_mean(dp, region) - region_mean(dg, region)))
        diffs = []
        for months in seasons.values():
            hp = pred.sel({time_dim: hist_slice}).where(pred[time_dim].dt.month.isin(months), drop=True).mean(dim=time_dim)
            fp = pred.sel({time_dim: fut_slice}).where(pred[time_dim].dt.month.isin(months), drop=True).mean(dim=time_dim)
            hg = gcm.sel({time_dim: hist_slice}).where(gcm[time_dim].dt.month.isin(months), drop=True).mean(dim=time_dim)
            fg = gcm.sel({time_dim: fut_slice}).where(gcm[time_dim].dt.month.isin(months), drop=True).mean(dim=time_dim)
            dps, dgs = (fp - hp) / hp * 100, (fg - hg) / hg * 100
            diffs.append(np.abs(region_mean(dps, region) - region_mean(dgs, region)))
        return float(np.mean(diffs))

    land = None if mask is None else _resolve_land(mask, pred.isel({time_dim: 0}))
    results = {}
    for period_name, fut_slice in fut_slices.items():
        full = metric_for(fut_slice, None)
        results[period_name] = full if mask is None else {
            "full": full, "land": metric_for(fut_slice, land), "sea": metric_for(fut_slice, ~land)
        }
    return results


# ------------------------------------------------------- precipitation -----

def quantile_range_ks(pred, obs, q_low: float = 0.9, q_high: float = 1.0, threshold=None):
    """KS statistic restricted to an OBS-quantile-defined value range.

    Returns (ks, n_ratio); n_ratio = n_pred_in_range / n_obs_in_range, which
    the plain KS statistic doesn't reveal (a model can under-produce events
    in a value range while still matching its shape).
    """
    pred, obs = flatten_spacetime(pred), flatten_spacetime(obs)
    if threshold is not None:
        pred, obs = pred[pred > threshold], obs[obs > threshold]

    lo = np.quantile(obs, q_low)
    hi = np.quantile(obs, q_high) if q_high < 1.0 else np.inf
    p = pred[(pred >= lo) & (pred <= hi)]
    o = obs[(obs >= lo) & (obs <= hi)]
    if p.size < 50 or o.size < 50:
        return np.nan, np.nan
    ks, _ = stats.ks_2samp(p, o)
    return float(ks), float(p.size / o.size)


def wasserstein_distance_pr(pred, obs, threshold: float = 0.1, transform: str = "log1p", normalize: bool = True):
    """Wasserstein-1 distance between precipitation distributions (Eq. wasserstein)."""
    p, o = flatten_spacetime(pred), flatten_spacetime(obs)
    if threshold is not None:
        p, o = p[p > threshold], o[o > threshold]
    if p.size < 50 or o.size < 50:
        return np.nan

    if transform == "log1p":
        p, o = np.log1p(np.maximum(p, 0.0)), np.log1p(np.maximum(o, 0.0))
    elif transform == "cbrt":
        p, o = np.cbrt(np.maximum(p, 0.0)), np.cbrt(np.maximum(o, 0.0))
    elif transform is not None:
        raise ValueError(f"unknown transform: {transform}")

    w = stats.wasserstein_distance(p, o)
    if normalize:
        denom = np.mean(o)
        w = w / denom if denom > 0 else np.nan
    return float(w)


def compute_pr_metrics(pred, obs, wet_threshold: float = 1.0) -> dict:
    """{ks, n_ratio, w1_log1p, w1_raw} for one prediction against ERA5 precipitation.

    Returns a dict, not a positional tuple: a plain 4-tuple is exactly what
    let the original notebooks' precip table drop a column silently (see
    reporting.print_pr_ensemble_table) -- keying the result by name makes
    that class of bug impossible to reintroduce downstream.
    """
    ks, nr = quantile_range_ks(pred, obs, q_low=0.9, threshold=wet_threshold)
    w_log = wasserstein_distance_pr(pred, obs, threshold=wet_threshold, transform="log1p", normalize=True)
    w_raw = wasserstein_distance_pr(pred, obs, threshold=wet_threshold, transform=None, normalize=False)
    return {"ks": ks, "n_ratio": nr, "w1_log1p": w_log, "w1_raw": w_raw}
