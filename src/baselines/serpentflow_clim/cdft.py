"""CDF-t bias correction (Michelangeli et al. 2009), three variants.

CDFt-1 (`quantile_transfer`) is a simple empirical-quantile transfer.
CDFt-2 (`cdft_python`) is a Python port of the reference R implementation
(CDFt_JG_new.R), matching Eqs. (qm)/(cdft-Ftest)/(cdft-correct) in the paper.
CDFt-3 (`cdft_with_dry_days`) wraps CDFt-2 with variable-aware handling
(dry-day frequency adaptation / Singularity Stochastic Removal for
precipitation). CDFt-3 is the variant actually used in the pipeline
(`run_cdft_20y(..., required=[3])`); 1 and 2 are kept for comparison.
"""

from typing import Iterable

import numpy as np
import scipy.stats as st
import xarray as xr
from tqdm import tqdm


# ---------------------------------------------------------------- CDFt-1 ---

def quantile_transfer(x_train, y_train, x_test):
    F_xtest = st.rankdata(x_test, method="average") / len(x_test)
    y_sorted = np.sort(y_train)
    mapped_to_y = np.quantile(y_sorted, F_xtest)

    x_train_sorted = np.sort(x_train)
    x_test_sorted = np.sort(x_test)
    F_xtrain_of_mapped = np.searchsorted(x_train_sorted, mapped_to_y, side="right") / len(x_train)
    F_xtrain_of_mapped = np.clip(F_xtrain_of_mapped, 0, 1)
    return np.quantile(x_test_sorted, F_xtrain_of_mapped)


# ---------------------------------------------------------------- CDFt-2 ---

def ecdf_t(x, etype: int = 4):
    """Step-function ECDF matching R's ecdfT (approxfun, method='constant')."""
    xs = np.sort(x.ravel())
    n = len(xs)
    vals, inv = np.unique(xs, return_inverse=True)
    cumcounts = np.cumsum(np.bincount(inv))
    if etype == 4:
        y = cumcounts / n
    elif etype == 5:
        y = (cumcounts - 0.5) / n
    elif etype == 6:
        y = cumcounts / (n + 1)
    elif etype == 7:
        y = (cumcounts - 1) / (n - 1)
    else:
        raise ValueError(f"unsupported etype: {etype}")

    def F(v):
        v = np.atleast_1d(np.asarray(v, dtype=float))
        idx = np.clip(np.searchsorted(vals, v, side="right") - 1, 0, len(y) - 1)
        result = y[idx].copy()
        result[v < vals[0]] = 0.0
        result[v > vals[-1]] = 1.0
        return result

    return F


def cdft_python(DataGp, ObsRp, DataGf, npas: int = 100, dev=None, scale: int = 1, pui: float = 1, etype: int = 4, _max_dev: float = 64.0):
    """Pure-Python port of CDFt_JG_new.R.

    If the future GCM extreme lies outside the ``dev`` extrapolation range
    (reference CDFt error: "dev must be higher"), retry the cell with a larger
    ``dev`` (doubling) up to ``_max_dev`` instead of failing the whole field.
    """
    add = pui >= 1
    if dev is None:
        dev = 2 if add else 1

    dev = min(dev, _max_dev)
    mO, mGp = np.mean(ObsRp), np.mean(DataGp)
    if scale == 1:
        DataGp2 = DataGp + (mO - mGp)
        DataGf2 = DataGf + (mO - mGp)
    elif scale == 2:
        DataGp2 = DataGp * mO / mGp
        DataGf2 = DataGf * mO / mGp
    else:
        DataGp2, DataGf2 = DataGp.copy(), DataGf.copy()

    FRp = ecdf_t(ObsRp, etype)
    FGp = ecdf_t(DataGp2, etype)
    FGf = ecdf_t(DataGf2, etype)

    m = min(np.min(ObsRp), np.min(DataGp), np.min(DataGf))
    M = max(np.max(ObsRp), np.max(DataGp), np.max(DataGf))
    if add:
        a = abs(np.mean(DataGf) - np.mean(DataGp))
        m -= dev * a
        M += dev * a
    else:
        a = abs(np.mean(DataGf) / np.mean(DataGp))
        m = m * a / dev
        M = M * dev * a

    x = np.linspace(m ** pui, M ** pui, npas) ** (1.0 / pui)
    n = len(x)

    gf2s, gp2s, obs_s = np.sort(DataGf2), np.sort(DataGp2), np.sort(ObsRp)
    FRP = FRp(x)
    FGF = np.interp(x, gf2s, FGf(gf2s))

    FGPm1_FGF = np.quantile(DataGp2, FGF)
    FRF = np.interp(FGPm1_FGF, obs_s, FRp(obs_s))

    if np.min(ObsRp) < np.min(DataGf2):
        thresh = np.quantile(ObsRp, FRF[0])
        i = 0
        while i < n and x[i] <= thresh:
            i += 1
        j = 0
        while j < n and x[j] < np.min(DataGf2):
            j += 1
        k = i
        while j >= 0 and k >= 0:
            FRF[j] = FRP[k]
            j -= 1
            k -= 1
        if j >= 0:
            FRF[: j + 1] = 0.0

    if FRF[-1] < 1.0:
        QQ = np.quantile(ObsRp, FRF[-1])
        i = n - 1
        while i >= 0 and x[i] >= QQ:
            i -= 1
        i += 1
        j = n - 2
        while j >= 0 and FRF[j] == FRF[-1]:
            j -= 1
        if j < 0:
            if dev >= _max_dev:
                # Degenerate cell: the whole corrected CDF collapsed to a single
                # value (e.g. an almost-constant future time series), so doubling
                # ``dev`` can never widen the extrapolation range. Fall back to
                # the observed ECDF evaluated on the extended grid, which is a
                # valid monotone CDF, and let the clamp below finish the tail.
                FRF = np.maximum.accumulate(np.asarray(FRp(x), dtype=float))
            else:
                return cdft_python(DataGp, ObsRp, DataGf, npas=npas, dev=dev * 2,
                                   scale=scale, pui=pui, etype=etype, _max_dev=_max_dev)
        if j < 0:
            j = n - 2
        dif = min(n - 1 - j, n - 1 - i)
        FRF[j: j + dif + 1] = FRP[i: i + dif + 1]
        k = j + dif
        if k < n - 1:
            FRF[k:] = 1.0

    qntl = FGf(DataGf2)
    return np.interp(qntl, FRF, x, left=x[0], right=x[-1])


# ---------------------------------------------------------------- CDFt-3 ---

def _ssr_perturb(x, threshold, rng):
    """Singularity Stochastic Removal: replace x < threshold with U(0, threshold)."""
    x = x.copy()
    mask = x < threshold
    if mask.any():
        x[mask] = rng.uniform(0.0, threshold, size=int(mask.sum()))
    return x


def cdft_with_dry_days(DataGp, ObsRp, DataGf, wet_threshold: float = 1.0, clip_min=None,
                        use_ssr: bool = False, ssr_rezero: bool = False, ssr_seed: int = 0,
                        **cdft_kwargs):
    """CDF-t with dry-day handling, for precipitation."""
    DataGp = np.asarray(DataGp, dtype=float).ravel()
    ObsRp = np.asarray(ObsRp, dtype=float).ravel()
    DataGf = np.asarray(DataGf, dtype=float).ravel()

    if use_ssr:
        rng = np.random.default_rng(ssr_seed)
        Gp_p = _ssr_perturb(DataGp, wet_threshold, rng)
        Obs_p = _ssr_perturb(ObsRp, wet_threshold, rng)
        Gf_p = _ssr_perturb(DataGf, wet_threshold, rng)
        corrected = cdft_python(Gp_p, Obs_p, Gf_p, **cdft_kwargs)
        if ssr_rezero:
            corrected = np.where(corrected < wet_threshold, 0.0, corrected)
        return np.maximum(corrected, 0.0)

    if clip_min is None:
        clip_min = wet_threshold

    obs_wet_freq = float(np.mean(ObsRp >= wet_threshold))
    if obs_wet_freq == 0.0:
        return np.zeros_like(DataGf)
    gcm_thresh = 0.0 if obs_wet_freq >= 1.0 else float(np.quantile(DataGp, 1.0 - obs_wet_freq))

    gp_wet = DataGp >= gcm_thresh
    gf_wet = DataGf >= gcm_thresh
    obs_wet = ObsRp >= wet_threshold

    if gp_wet.sum() < 2 or gf_wet.sum() < 2 or obs_wet.sum() < 2:
        out = np.zeros_like(DataGf)
        out[gf_wet] = np.maximum(DataGf[gf_wet], clip_min)
        return out

    corrected_wet = cdft_python(DataGp[gp_wet], ObsRp[obs_wet], DataGf[gf_wet], **cdft_kwargs)
    out = np.zeros_like(DataGf)
    out[gf_wet] = np.maximum(corrected_wet, clip_min)
    return out


CDFT3_CONFIG = {
    "sfcWind": dict(scale=0, pui=1, dev=2, npas=300, clip_min=0.0),
    "tas": dict(),  # unbounded variable: no clip_min / dry-day handling needed
    "pr": dict(use_dry_day_correction=True, wet_threshold=0.1, use_ssr=True,
               ssr_rezero=False, ssr_seed=0, scale=0, pui=0.5),
    # Benchmark environment additions (all-6-variable port):
    "sfcWindmax": dict(scale=0, pui=1, dev=2, npas=300, clip_min=0.0),
    # psl (GCM "msl" climatology analog): mean-bias additive correction, no clip.
    "psl": dict(scale=1, pui=1, dev=2, npas=300),
    # rsds: non-negative radiative flux, additive mean correction, physical clip at 0.
    "rsds": dict(scale=1, pui=1, dev=2, npas=300, clip_min=0.0),
}


def get_cdft3_params(var: str) -> dict:
    if var not in CDFT3_CONFIG:
        raise ValueError(f"Unknown variable: {var}")
    return CDFT3_CONFIG[var]


# --------------------------------------------------------- xarray wiring ---

def _groupby_month(train_data_gcm, train_data_era5, test_data, var, apply_fn):
    results = []
    for month in tqdm(range(1, 13), desc="Processing by month"):
        train_sel = train_data_gcm.sel(time=train_data_gcm.time.dt.month == month)[var]
        obs_sel = train_data_era5.sel(time=train_data_era5.time.dt.month == month)[var]
        test_sel = test_data.sel(time=test_data.time.dt.month == month)[var]
        if train_sel.time.size == 0 or obs_sel.time.size == 0 or test_sel.time.size == 0:
            continue

        corrected = xr.apply_ufunc(
            apply_fn,
            train_sel.drop_vars("time"), obs_sel.drop_vars("time"), test_sel.drop_vars("time"),
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[["time"]],
            vectorize=True, dask="parallelized",
            output_dtypes=[test_sel.dtype], exclude_dims=set(("time",)),
        )
        corrected["time"] = test_sel["time"]
        results.append(corrected)

    if not results:
        raise ValueError("No valid month found for correction.")
    return xr.concat(results, dim="time").sortby("time")


def cdft_groupby1(train_data_gcm, train_data_era5, test_data, var, function=quantile_transfer):
    return _groupby_month(train_data_gcm, train_data_era5, test_data, var, function)


def cdft_groupby2(train_data_gcm, train_data_era5, test_data, var):
    return _groupby_month(train_data_gcm, train_data_era5, test_data, var, cdft_python)


def cdft_groupby3(train_data_gcm, train_data_era5, test_data, var,
                   use_dry_day_correction: bool = False, wet_threshold: float = 1.0,
                   clip_min=None, use_ssr: bool = False, ssr_rezero: bool = False,
                   ssr_seed: int = 0, **cdft_kwargs):
    def _apply(gp, obs, gf):
        if use_dry_day_correction:
            return cdft_with_dry_days(gp, obs, gf, wet_threshold=wet_threshold, clip_min=clip_min,
                                       use_ssr=use_ssr, ssr_rezero=ssr_rezero, ssr_seed=ssr_seed,
                                       **cdft_kwargs)
        out = cdft_python(gp, obs, gf, **cdft_kwargs)
        return np.maximum(out, clip_min) if clip_min is not None else out

    return _groupby_month(train_data_gcm, train_data_era5, test_data, var, _apply)


def split_20y(data):
    years = np.unique(data.time.dt.year.values)
    y0, y1 = int(years[0]), int(years[-1])
    while y0 <= y1:
        mask = (data.time.dt.year >= y0) & (data.time.dt.year < y0 + 20)
        chunk = data.sel(time=mask)
        if chunk.time.size > 0:
            yield y0, y0 + 20, chunk
        y0 += 20


def run_cdft_20y(gcm_test_clim, era5_train_clim, gcm_train_clim, var: str = "tas",
                  required: Iterable[int] = (1, 2, 3)):
    """Apply CDF-t on independent rolling 20-year GCM chunks (no stationary-future
    assumption), calibrated once from the fixed ERA5/GCM training period."""
    cdft1_list, cdft2_list, cdft3_list = [], [], []
    cdft3_params = get_cdft3_params(var)

    for y_start, y_end, gcm_chunk in split_20y(gcm_test_clim):
        print(f"Running CDF-t for {y_start}-{y_end}")
        out1 = (cdft_groupby1(gcm_train_clim, era5_train_clim, gcm_chunk, var=var,
                               function=quantile_transfer).transpose("time", "latitude", "longitude")
                if 1 in required else None)
        out2 = (cdft_groupby2(gcm_train_clim, era5_train_clim, gcm_chunk, var=var)
                .transpose("time", "latitude", "longitude") if 2 in required else None)
        out3 = (cdft_groupby3(gcm_train_clim, era5_train_clim, gcm_chunk, var=var, **cdft3_params)
                .transpose("time", "latitude", "longitude") if 3 in required else None)
        cdft1_list.append(out1)
        cdft2_list.append(out2)
        cdft3_list.append(out3)

    out1 = xr.concat(cdft1_list, dim="time").sortby("time") if 1 in required else None
    out2 = xr.concat(cdft2_list, dim="time").sortby("time") if 2 in required else None
    out3 = xr.concat(cdft3_list, dim="time").sortby("time") if 3 in required else None
    return out1, out2, out3
