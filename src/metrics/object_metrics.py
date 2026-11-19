"""Object-based precipitation metrics after SAL (Wernli et al. 2008), for unpaired fields.

SAL compares a forecast with the observation of the same day: its structure (S),
amplitude (A) and location (L) components are differences between the two
fields' rain objects. A downscaled GCM day has no ERA5 counterpart, so here the
SAL quantities are computed for each map on its own, every day, and ERA5 and the
prediction are compared through their distributions over the evaluation period:

1. Domain distributions (``sal_daily`` then ``sal_distribution_scores``): per
   day, the domain-mean rain D, the centre of mass of the rain, the mean
   distance r of the objects from it, and the structure parameter V (the
   volume-weighted scaled volume of the objects: small for small, peaked
   objects, large for wide, flat ones).
2. Per-cell distributions (``sal_cell_maps``): every day, each cell inside an
   object takes that object's area and flatness, so each cell gets a 20-year
   distribution of the size and shape of the rain objects that cover it.

Main functions
--------------
grid_geometry
    Cell centres in km and cell areas.
find_objects
    Rain objects of one map (Wernli threshold).
sal_daily
    SAL quantities of every day of one series, and per-cell object attributes.
sal_distribution_scores
    S, A, L of the period means and KS distances of the daily distributions.
sal_cell_maps
    Per-cell maps of object frequency, size and flatness, ERA5 vs prediction.
"""
from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from scipy import ndimage

KM_PER_DEG = 111.2
# Wernli et al. (2008, 2009): objects are the connected cells above f * R95,
# with R95 the 95th percentile of the map's values above 0.1 mm/day.
THR_FACTOR = 1.0 / 15.0
THR_QUANTILE = 0.95
WET_MM = 0.1
# Diagonal neighbours belong to the same object.
STRUCTURE = np.ones((3, 3), dtype=bool)


def grid_geometry(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Cell centres (x, y) in km, cell areas in km², and the domain's largest extent d in km.

    Sinusoidal projection about the mean longitude: x = 111.2 (lon - lon0) cos(lat),
    y = 111.2 lat. d is the largest distance between two corner cells (SAL's d).
    """
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    coslat = np.cos(np.deg2rad(lat2d))
    x = KM_PER_DEG * (lon2d - lon.mean()) * coslat
    y = KM_PER_DEG * lat2d
    area = (KM_PER_DEG * abs(lat[1] - lat[0])) * (KM_PER_DEG * abs(lon[1] - lon[0])) * coslat
    corners = [(i, j) for i in (0, -1) for j in (0, -1)]
    d = max(np.hypot(x[a] - x[b], y[a] - y[b]) for a in corners for b in corners)
    return x, y, area, float(d)


def find_objects(field: np.ndarray) -> tuple[np.ndarray, int]:
    """Label the rain objects of one map; returns (labels, number of objects).

    Threshold R* = THR_FACTOR * R95, R95 the THR_QUANTILE quantile of the cells
    above WET_MM. R* is raised to WET_MM when lower, so an object needs
    measurable rain (on light-rain days f * R95 falls below 0.1 mm/day).
    """
    wet = field[field > WET_MM]
    if wet.size == 0:
        return np.zeros(field.shape, dtype=np.int32), 0
    thr = max(THR_FACTOR * float(np.quantile(wet, THR_QUANTILE)), WET_MM)
    labels, n = ndimage.label(field >= thr, structure=STRUCTURE)
    return labels, int(n)


def sal_daily(da: xr.DataArray, keep_cells: bool = True) -> dict[str, np.ndarray]:
    """SAL quantities of every day of a precipitation series [time, lat, lon] in mm/day.

    Per day (arrays of length T; NaN for r and V on days without objects):
    ``D`` domain mean (unweighted, as the other metrics), ``xc``/``yc`` centre of
    mass of the whole field (km), ``r`` rain-weighted mean distance of the
    objects' centres of mass from it (km), ``V`` = sum_n R_n V_n / sum_n R_n with
    R_n the rain summed over object n and V_n = R_n / max(object n), ``n_obj``,
    ``total`` (sum of the field, the weight of the centre of mass).
    Per object (pooled over days): ``obj_area`` (km²), ``obj_flat`` = mean / max
    rain in the object (1 = flat), ``obj_day``.
    With ``keep_cells``, ``cell_area`` and ``cell_flat`` [T, lat, lon] (float32)
    give the area and flatness of the object covering each cell, NaN outside objects.
    """
    values = da.transpose("time", "lat", "lon").values
    x, y, area, _ = grid_geometry(da.lat.values, da.lon.values)
    n_t = values.shape[0]
    out = {k: np.full(n_t, np.nan) for k in ("D", "xc", "yc", "r", "V", "total")}
    out["n_obj"] = np.zeros(n_t, dtype=np.int32)
    obj_area, obj_flat, obj_day = [], [], []
    if keep_cells:
        cell_area = np.full(values.shape, np.nan, dtype=np.float32)
        cell_flat = np.full(values.shape, np.nan, dtype=np.float32)
    for t in range(n_t):
        f = np.where(np.isfinite(values[t]), values[t], 0.0)
        f = np.clip(f, 0.0, None)                      # ERA5 has tiny negative values
        out["D"][t] = f.mean()
        tot = f.sum()
        out["total"][t] = tot
        if tot > 0:
            out["xc"][t] = (f * x).sum() / tot
            out["yc"][t] = (f * y).sum() / tot
        labels, n = find_objects(f)
        out["n_obj"][t] = n
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        r_n = np.asarray(ndimage.sum(f, labels, idx))
        r_max = np.asarray(ndimage.maximum(f, labels, idx))
        n_cells = np.asarray(ndimage.sum(np.ones_like(f), labels, idx))
        a_n = np.asarray(ndimage.sum(area, labels, idx))
        x_n = np.asarray(ndimage.sum(f * x, labels, idx)) / r_n
        y_n = np.asarray(ndimage.sum(f * y, labels, idx)) / r_n
        v_n = r_n / r_max
        out["V"][t] = (r_n * v_n).sum() / r_n.sum()
        out["r"][t] = (r_n * np.hypot(x_n - out["xc"][t], y_n - out["yc"][t])).sum() / r_n.sum()
        flat_n = v_n / n_cells
        obj_area.append(a_n)
        obj_flat.append(flat_n)
        obj_day.append(np.full(n, t, dtype=np.int32))
        if keep_cells:
            inside = labels > 0
            cell_area[t][inside] = a_n[labels[inside] - 1]
            cell_flat[t][inside] = flat_n[labels[inside] - 1]
    out["obj_area"] = np.concatenate(obj_area) if obj_area else np.zeros(0)
    out["obj_flat"] = np.concatenate(obj_flat) if obj_flat else np.zeros(0)
    out["obj_day"] = np.concatenate(obj_day) if obj_day else np.zeros(0, dtype=np.int32)
    if keep_cells:
        out["cell_area"] = cell_area
        out["cell_flat"] = cell_flat
    return out


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic, NaN left out (scipy's ks_2samp statistic, without its
    exact p-value, which takes seconds per call below 10,000 values)."""
    a = np.sort(a[np.isfinite(a)])
    b = np.sort(b[np.isfinite(b)])
    if a.size < 2 or b.size < 2:
        return float("nan")
    both = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, both, side="right") / a.size
    cdf_b = np.searchsorted(b, both, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _rel_diff(pred: float, obs: float) -> float:
    """SAL's normalised difference, in [-2, 2]."""
    return float((pred - obs) / (0.5 * (pred + obs)))


def sal_distribution_scores(obs_q: dict, pred_q: dict, d_km: float) -> dict[str, float]:
    """Compare the daily SAL quantities of ERA5 (obs_q) and a prediction (pred_q) over the period.

    S, A and L take Wernli's formulas, applied to the period's mean quantities:
    ``sal_S`` from the mean of V over days with objects (> 0: objects too large
    or flat), ``sal_A`` from the mean of D (> 0: too much rain), ``sal_L`` =
    |centre of mass of the period's total rain| difference / d + 2 |mean r
    difference| / d (0 to 2). ``sal_ks_*`` are KS distances between the daily
    distributions (V, D, r, number of objects) and between the pooled object
    areas. ``sal_days_no_object_*`` are the fractions of days without an object.
    """
    def centre(q):
        w = q["total"]
        ok = np.isfinite(q["xc"]) & (w > 0)
        return (np.average(q["xc"][ok], weights=w[ok]), np.average(q["yc"][ok], weights=w[ok]))

    xo, yo = centre(obs_q)
    xp, yp = centre(pred_q)
    l1 = float(np.hypot(xp - xo, yp - yo)) / d_km
    l2 = 2.0 * abs(np.nanmean(pred_q["r"]) - np.nanmean(obs_q["r"])) / d_km
    return {
        "sal_S": _rel_diff(np.nanmean(pred_q["V"]), np.nanmean(obs_q["V"])),
        "sal_A": _rel_diff(np.nanmean(pred_q["D"]), np.nanmean(obs_q["D"])),
        "sal_L": float(l1 + l2),
        "sal_L1": float(l1),
        "sal_L2": float(l2),
        "sal_ks_V": _ks(obs_q["V"], pred_q["V"]),
        "sal_ks_D": _ks(obs_q["D"], pred_q["D"]),
        "sal_ks_r": _ks(obs_q["r"], pred_q["r"]),
        "sal_ks_nobj": _ks(obs_q["n_obj"].astype(float), pred_q["n_obj"].astype(float)),
        "sal_ks_area": _ks(obs_q["obj_area"], pred_q["obj_area"]),
        "sal_days_no_object_obs": float(np.mean(obs_q["n_obj"] == 0)),
        "sal_days_no_object_pred": float(np.mean(pred_q["n_obj"] == 0)),
    }


def _cell_ks(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """KS distance per cell between two [T, lat, lon] arrays, NaN days left out."""
    n_lat, n_lon = a.shape[1:]
    out = np.full((n_lat, n_lon), np.nan)
    for i in range(n_lat):
        for j in range(n_lon):
            out[i, j] = _ks(a[:, i, j], b[:, i, j])
    return out


def sal_cell_maps(obs_q: dict, pred_q: dict) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Per-cell maps of the rain objects covering each cell, ERA5 vs prediction.

    Maps: ``obs_freq``/``pred_freq`` fraction of days the cell is inside an
    object (where it rains as part of an object), ``obs_area``/``pred_area``
    median area (km²) and ``obs_flat``/``pred_flat`` median flatness of the
    objects covering it, ``ks_area``/``ks_flat`` KS distance between the ERA5
    and prediction distributions at each cell.
    Scores (spatial means): ``sal_cell_freq`` = |freq difference|,
    ``sal_cell_ks_area``, ``sal_cell_ks_flat``.
    """
    maps = {}
    for tag, q in (("obs", obs_q), ("pred", pred_q)):
        maps[f"{tag}_freq"] = np.isfinite(q["cell_area"]).mean(axis=0)
        with warnings.catch_warnings():   # a cell never inside an object has an all-NaN column
            warnings.simplefilter("ignore", RuntimeWarning)
            maps[f"{tag}_area"] = np.nanmedian(q["cell_area"], axis=0)
            maps[f"{tag}_flat"] = np.nanmedian(q["cell_flat"], axis=0)
    maps["ks_area"] = _cell_ks(obs_q["cell_area"], pred_q["cell_area"])
    maps["ks_flat"] = _cell_ks(obs_q["cell_flat"], pred_q["cell_flat"])
    scores = {
        "sal_cell_freq": float(np.mean(np.abs(maps["pred_freq"] - maps["obs_freq"]))),
        "sal_cell_ks_area": float(np.nanmean(maps["ks_area"])),
        "sal_cell_ks_flat": float(np.nanmean(maps["ks_flat"])),
    }
    return maps, scores
