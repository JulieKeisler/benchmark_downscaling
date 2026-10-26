"""Build australia energy dataset from the OpenElectricity (ex-OpenNEM) API.

Region decision: australia = NSW1 + VIC1 + SA1 (combined), the NEM regions whose load
lies inside the weather box [24-38S, 136-152E] (Brisbane, QLD1, lies outside it).
NetDemand = TOTALDEMAND - wind - solar_utility, summed over the three regions
  - demand        = AEMO TOTALDEMAND (operational demand)  /v4/market/network/NEM
  - wind          = utility wind    (scheduled + semi-scheduled)
  - solar_utility = utility PV
  Rooftop PV stays inside demand (it lowers TOTALDEMAND), in every year: its estimate
  only exists from 2018, so adding it back would break the target in 2018. Same
  convention as EIA-930 us_central and RTE France: NetDemand = load - wind - solar.

Conventions match the France 30-min dataset:
  - 5-min API samples resampled to a 30-min grid (mean)
  - date column = 30-min interval START, network wall-clock (Sydney), "%Y-%m-%d %H:%M:%S"
  - calendar covariates: build_calendar(times, "australia")
  - weather covariates:  australia_era5_covariates.csv (15 PCs + 5 spatial means)

Note: Community plan only exposes the last 730 days; full history needs the free
Academic plan on platform.openelectricity.org.au.

Usage:
  python src/energy/build_australia.py --key oe_... --start 2025-09-01 --end 2025-09-07
  python src/energy/build_australia.py --key oe_...                    # 2012 -> 2022
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://api.openelectricity.org.au"
REGIONS = ["NSW1", "VIC1", "SA1"]
FUELTECHS = {"wind": "wind", "solar_utility": "solar_utility"}
START = "2012-01-01"
END = "2022-12-31"
CHUNK_DAYS = 14  # keep query payload/credits modest (Pro 5m bucket limit is per-query)


def _get(url: str, key: str) -> dict:
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)


def _wall(ts: list[str]) -> pd.DatetimeIndex:
    """API timestamps -> naive wall-clock, keeping the clock face as-is.

    The OpenElectricity API labels NEM series with a fixed +10:00 (AEST)
    offset year-round (no DST shifting: even DST months show +10:00 and DST
    boundary days return exactly 24h of points). So the wall-clock face is
    already the canonical naive local time - drop the tz, don't convert.
    Parsed in one call: one string at a time took ~10 s per series.
    """
    return pd.DatetimeIndex(pd.to_datetime(ts)).tz_localize(None)


def _fetch_market_series(metrics, start: str, end: str, key: str) -> dict:
    q = urllib.parse.urlencode({
        "metrics": metrics, "interval": "5m",
        "date_start": start, "date_end": end,
        "primary_grouping": "network_region",
    })
    d = _get(f"{API}/v4/market/network/NEM?{q}", key)
    if not d.get("success") or "data" not in d:
        return {}
    out = {}
    for rd in d["data"]:
        for r in rd["results"]:
            region = r["name"].rsplit("_", 1)[-1]
            if region in REGIONS:
                out[(region, rd["metric"])] = pd.Series(
                    [v for _, v in r["data"]], index=_wall([t for t, _ in r["data"]]))
    return out


def _fetch_power(fueltech: str, start: str, end: str, key: str) -> dict:
    q = urllib.parse.urlencode({
        "metrics": "power", "interval": "5m",
        "date_start": start, "date_end": end,
        "fueltech": fueltech, "primary_grouping": "network_region",
    })
    try:
        d = _get(f"{API}/v4/data/network/NEM?{q}", key)
    except Exception:
        return {}
    if not d.get("success") or "data" not in d or not d.get("data"):
        return {}
    out = {}
    for rd in d["data"]:
        for r in rd["results"]:
            region = r.get("columns", {}).get("region")
            if region in REGIONS:
                out[(region, fueltech)] = pd.Series(
                    [v for _, v in r["data"]], index=_wall([t for t, _ in r["data"]]))
    return out


def _fetch_chunk(start: str, end: str, key: str) -> dict:
    """All series for a [start,end) Sydney-wall-clock window, 5-min, keyed (region, series)."""
    out = _fetch_market_series("demand", start, end, key)
    set_fueltechs = set(out)  # only query fueltechs for regions that returned demand
    regions = {rg for (rg, _) in set_fueltechs}
    y0 = int(start[:4])
    for ft in FUELTECHS.values():
        if ft == "solar_utility" and y0 < 2015:
            continue  # utility solar absent before 2015
        if ft == "solar_rooftop" and y0 < 2018:
            continue  # rooftop estimates absent before 2018
        p = _fetch_power(ft, start, end, key)
        for (rg, sname), sv in p.items():
            if rg in regions:
                out[(rg, sname)] = sv
    return out


def build_australia(key: str, start: str, end: str, verbose: bool = False) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from energy.calendar_covariates import build_calendar, CAL_COLS

    idx = pd.date_range(f"{start} 00:00", f"{end} 23:30", freq="30min")

    series = {(r, s): [] for r in REGIONS for s in ["demand", "wind", "solar_utility"]}
    chunks = list(pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1),
                                freq=f"{CHUNK_DAYS}D"))
    if len(chunks) == 1:  # window shorter than CHUNK_DAYS -> single chunk
        chunks.append(pd.Timestamp(end) + pd.Timedelta(days=1))
    windows = [(cs.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d"))
               for cs, ce in zip(chunks[:-1], chunks[1:])]

    from concurrent.futures import ThreadPoolExecutor
    import threading
    _lock = threading.Lock()
    fetched = [None] * len(windows)

    def _one(wi):
        a, b = windows[wi]
        try:
            c = _fetch_chunk(a, b, key)
        except Exception as e:  # 429 / 400 -> Community 730-day cap or rate limit
            print(f"  chunk {a}: {type(e).__name__} {e}")
            return
        with _lock:
            fetched[wi] = c
            if verbose:
                print(f"  chunk {a}..{b}: {len(c)} series ok")

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_one, range(len(windows))))
    for wi, (a, b) in enumerate(windows):
        c = fetched[wi]
        if not c:
            continue
        for (rg, sname), sv in c.items():
            v = sv.sort_index()
            v = v[~v.index.duplicated(keep="first")]
            v30 = v.resample("30min").mean().reindex(idx[
                (idx >= a) & (idx <= (pd.Timestamp(b) - pd.Timedelta(minutes=30)))
            ])
            series[(rg, sname)].append(v30)

    # per-region frames: a missing demand value stays NaN and is forward-filled (a gap,
    # not a zero load); wind and solar are 0 where absent (utility solar before 2015)
    acc = {}
    for r in REGIONS:
        parts = []
        for s in ["demand", "wind", "solar_utility"]:
            chunk = series[(r, s)]
            if not chunk:
                if s == "demand":
                    raise SystemExit(f"no demand returned for {r}")
                v = pd.Series(0.0, index=idx)
            else:
                v = pd.concat(chunk).groupby(level=0).mean().reindex(idx)
                v = v.ffill() if s == "demand" else v.fillna(0.0)
            parts.append(v.rename(s))
        acc[r] = pd.concat(parts, axis=1)

    tot = sum(acc[r] for r in REGIONS)
    df = pd.DataFrame({
        "NetDemand": tot["demand"] - tot["wind"] - tot["solar_utility"],
        "load_adj": tot["demand"],
        "wind_adj": tot["wind"],
        "solar_adj": tot["solar_utility"],
    })
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    cal = build_calendar(idx, "australia")
    out = cal.merge(df, on="date", how="left")
    base = Path(__file__).resolve().parents[2]
    e5 = pd.read_csv(base / "data/era5_covariates/australia_era5_covariates.csv")
    out = out.merge(e5.rename(columns={"date": "_day"}),
                    left_on=out["date"].str.slice(0, 10), right_on="_day",
                    how="left").drop(columns=["_day"])
    keep = ["date", "NetDemand", "load_adj", "wind_adj", "solar_adj", "rooftop_adj"] + \
        [c for c in CAL_COLS if c != "date"] + [c for c in e5.columns if c != "date"]
    out = out[[c for c in keep if c in out.columns]].sort_values("date").reset_index(drop=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--key", default="")
    p.add_argument("--start", default=START)
    p.add_argument("--end", default=END)
    p.add_argument("--out", default="")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    # key from --key, else $OE_KEY, else ~/.config/oe_key (keeps it off the command line)
    key = a.key or os.environ.get("OE_KEY", "")
    if not key and (Path.home() / ".config/oe_key").exists():
        key = (Path.home() / ".config/oe_key").read_text().strip()
    if not key:
        raise SystemExit("no OpenElectricity key: pass --key, set OE_KEY or write ~/.config/oe_key")
    repo = Path(__file__).resolve().parents[2]
    df = build_australia(key, a.start, a.end, verbose=a.verbose)
    out = Path(a.out) if a.out else repo / "data/australia_energy.csv"
    df.to_csv(out, index=False, sep=";")
    print(f"[australia] saved {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
