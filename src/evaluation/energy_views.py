"""Day-of-year cycles and energy-event spells for the dashboard.

Two dashboard features are computed here and stored in
`results/{region}/{method}/plot_data.json`:

* **DOY cycles** — `plot_data[var]["doy"] = {site: {"era5", "gcm", "pred"}}`,
  a 365-value daily climatology (Feb 29 removed) of the region-wide mean and
  of individual energy sites. The dashboard's *Line plots → doy* view uses it
  to spot day-to-day (month-boundary) jumps of month-wise methods (BCSD,
  CDF-t, ...) against ERA5 and the raw GCM.

* **Energy events** — `plot_data["_events"]`, counts and mean spell lengths of
  Dunkelflaute days (low wind AND low solar) and cold-and-calm days
  (low temperature AND low wind), for ERA5, the raw GCM and the downscaled
  product. All series use the same ERA5-derived thresholds (region-mean daily
  series, `EVENT_QUANTILE`th percentile) so the three are comparable.

Produced by `evaluate.py` (region inferred from the output path) and
backfilled for existing results by `backfill_energy.py`.
Read by `src/dashboard/dashboard.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

# ── Event definitions ─────────────────────────────────────────────────────────
EVENT_QUANTILE = 20.0      # "low" = below this percentile of the ERA5 series
EVENT_MIN_DAYS = 2         # a spell must last at least this many days
EVENT_VARIABLES = ("sfcWind", "rsds", "tas")

# key → (event, required variables) — dunkelflaute = wind & solar, cold = tas & wind
EVENT_DEFS = {
    "dunkelflaute": ("sfcWind", "rsds"),
    "cold_low_wind": ("tas", "sfcWind"),
}

_SITES_FILE = Path(__file__).resolve().parent.parent / "configs" / "data" / "energy_sites.yaml"


def _r5(x) -> float:
    """Round to 5 significant figures (JSON size vs. precision)."""
    return float(f"{float(x):.5g}")


# ── Site configuration ────────────────────────────────────────────────────────

def site_catalog(region: str) -> dict[str, dict]:
    """name → {label, lat, lon} from configs/data/energy_sites.yaml.

    Returns an empty dict when the region has no configured sites (unknown
    region, missing file) — callers fall back to whatever is in the data.
    """
    try:
        yml = yaml.safe_load(_SITES_FILE.read_text()) or {}
    except FileNotFoundError:
        return {}
    sites = yml.get("energy_sites", {}).get(region) or []
    return {
        s["name"]: {"label": s.get("label", s["name"]),
                    "lat": float(s["lat"]), "lon": float(s["lon"])}
        for s in sites
    }


def resolve_sites(region: str, lat: np.ndarray, lon: np.ndarray) -> list[dict]:
    """Configured sites mapped to the nearest grid cell of this region.

    Returns one entry per in-domain site —
    ``{name, label, lat, lon, ilat, ilon}`` — in configuration order.
    Sites further than one grid step outside the box are dropped (a site list
    may cover more regions than one grid does).
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    if lat.size == 0 or lon.size == 0:
        return []
    step = float(np.min(np.abs(np.diff(lat)))) if lat.size > 1 else 1.0
    step_lon = float(np.min(np.abs(np.diff(lon)))) if lon.size > 1 else 1.0
    pad_lat, pad_lon = max(step, 1e-9), max(step_lon, 1e-9)

    out: list[dict] = []
    for name, s in site_catalog(region).items():
        if not (lat.min() - pad_lat <= s["lat"] <= lat.max() + pad_lat):
            continue
        if not (lon.min() - pad_lon <= s["lon"] <= lon.max() + pad_lon):
            continue
        out.append({
            "name": name, "label": s["label"],
            "lat": s["lat"], "lon": s["lon"],
            "ilat": int(np.argmin(np.abs(lat - s["lat"]))),
            "ilon": int(np.argmin(np.abs(lon - s["lon"]))),
        })
    return out


# ── Day-of-year cycles ────────────────────────────────────────────────────────

def doy_cycle(values: np.ndarray, month: np.ndarray, day: np.ndarray) -> list[float] | None:
    """Mean of a daily series per calendar day → 365 values in calendar order.

    Grouping is by (month, day-of-month), not by raw day-of-year, so leap and
    non-leap years stay on the same calendar axis; Feb 29 (key 229) is dropped.
    Returns None when the series does not cover every calendar day.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return None
    key = np.asarray(month, dtype=int) * 100 + np.asarray(day, dtype=int)
    if key.size != v.size:
        raise ValueError(f"values ({v.size}) and month/day ({key.size}) length mismatch")
    keep = key != 229
    key, v = key[keep], v[keep]
    uniq, inv = np.unique(key, return_inverse=True)   # sorted → calendar order
    if uniq.size != 365:
        return None                                   # incomplete year coverage
    sums = np.zeros(365)
    cnts = np.zeros(365)
    np.add.at(sums, inv, v)
    np.add.at(cnts, inv, 1.0)
    with np.errstate(invalid="ignore"):
        out = sums / cnts
    out = np.where(cnts > 0, out, np.nan)
    if not np.all(np.isfinite(out)):
        return None
    return [_r5(x) for x in out]


def month_day(time) -> tuple[np.ndarray, np.ndarray]:
    """(month, day-of-month) arrays from an xarray time coordinate."""
    return (np.asarray(time.dt.month.values, dtype=int),
            np.asarray(time.dt.day.values, dtype=int))


def location_cycles(da, sites: list[dict]) -> dict[str, list[float]]:
    """DOY cycles of a (time, lat, lon) field for the region mean + sites.

    ``da.mean(["lat", "lon"])`` for the region-wide cycle (key ``global``),
    the nearest grid cell for each site. Sources whose data do not cover the
    full calendar are silently left out.
    """
    month, day = month_day(da["time"])
    out: dict[str, list[float]] = {}
    cyc = doy_cycle(da.mean(["lat", "lon"]).values, month, day)
    if cyc is not None:
        out["global"] = cyc
    for s in sites:
        series = da.isel(lat=s["ilat"], lon=s["ilon"]).values
        cyc = doy_cycle(series, month, day)
        if cyc is not None:
            out[s["name"]] = cyc
    return out


def build_doy_block(obs, gcm, pred, sites: list[dict]) -> dict:
    """``{site: {"era5": ..., "gcm": ..., "pred": ...}}`` for one variable.

    ``gcm``/``pred`` may be None (unavailable source); sites without an ERA5
    reference are dropped.
    """
    block: dict[str, dict] = {}
    for source, da in (("era5", obs), ("gcm", gcm), ("pred", pred)):
        if da is None:
            continue
        for key, vals in location_cycles(da, sites).items():
            block.setdefault(key, {})[source] = vals
    return {k: v for k, v in block.items() if "era5" in v}


def domain_series(da) -> np.ndarray:
    """Region-mean daily series [time] (unrounded — used for event spells)."""
    return np.asarray(da.mean(["lat", "lon"]).values, dtype=np.float64)


# ── Energy events ─────────────────────────────────────────────────────────────

def spell_stats(low: np.ndarray, min_days: int = EVENT_MIN_DAYS) -> tuple[int, float]:
    """Number of spells of True lasting ≥ ``min_days`` and their mean length.

    Isolated days (shorter than ``min_days``) are not events.
    """
    a = np.asarray(low, dtype=bool).ravel()
    if a.size == 0 or not a.any():
        return 0, 0.0
    edges = np.diff(np.concatenate(([False], a, [False])).astype(np.int8))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    lengths = lengths[lengths >= min_days]
    if lengths.size == 0:
        return 0, 0.0
    return int(lengths.size), _r5(float(lengths.mean()))


def energy_events(
    series: dict[str, dict[str, np.ndarray | None]],
    period: tuple[int, int] | None = None,
    quantile: float = EVENT_QUANTILE,
    min_days: int = EVENT_MIN_DAYS,
) -> dict | None:
    """Counts and mean spell lengths of energy-relevant low-field events.

    Args:
        series: ``{variable: {"era5": arr, "gcm": arr, "pred": arr}}`` of
            region-mean daily values on a common time axis (None = source
            unavailable for that variable).
        period: (start, end) years recorded in the output metadata.

    Thresholds are the ``quantile``th percentile of each ERA5 series and are
    applied unchanged to GCM and predictions, so all three are classified with
    the same absolute cutoffs. Returns None when a required series is missing.
    """
    for _, needed in EVENT_DEFS.items():
        for v in needed:
            ref = series.get(v, {}).get("era5")
            if ref is None or np.asarray(ref).size == 0:
                return None

    lengths = [np.asarray(a).size
               for v in EVENT_VARIABLES
               for a in series.get(v, {}).values() if a is not None]
    n = min(lengths) if lengths else 0
    if n < 400:                      # need at least ~a year of daily data
        return None

    def _arr(var: str, src: str) -> np.ndarray | None:
        a = series.get(var, {}).get(src)
        if a is None:
            return None
        return np.asarray(a, dtype=np.float64)[:n]

    sources = ["era5"] + [s for s in ("gcm", "pred")
                          if any(_arr(v, s) is not None for v in EVENT_VARIABLES)]

    thresholds = {v: _r5(float(np.nanpercentile(_arr(v, "era5"), quantile)))
                  for v in EVENT_VARIABLES}

    out: dict = {
        "meta": {
            "period": list(period) if period else None,
            "quantile": quantile,
            "min_days": min_days,
            "n_days": int(n),
            "thresholds": thresholds,       # ERA5 region-mean daily values
            "units": {"sfcWind": "m/s", "rsds": "W/m^2", "tas": "K"},
        },
    }

    for ev_name, (v_lo, v_hi) in EVENT_DEFS.items():
        stats: dict[str, dict] = {"count": {}, "mean_duration": {}}
        for src in sources:
            a_lo, a_hi = _arr(v_lo, src), _arr(v_hi, src)
            if a_lo is None or a_hi is None:
                continue
            low = (a_lo < thresholds[v_lo]) & (a_hi < thresholds[v_hi])
            cnt, dur = spell_stats(low, min_days=min_days)
            stats["count"][src] = cnt
            stats["mean_duration"][src] = dur
        if "era5" not in stats["count"]:
            return None
        out[ev_name] = stats

    return out
