"""Build us_central energy dataset from EIA-930 (region decision: MISO+SPP+ERCOT).

Conventions match the France 30-min dataset (TAM_downscaling/data/...):
  - date column = 30-min interval START, "%Y-%m-%d %H:%M:%S"
  - NetDemand = Sum_{BA in {MISO,SWPP,ERCO}} (Demand(Adj) - Wind(Adj) - Solar(Adj))
  - EIA-930 supplies hourly "Local Time at End of Hour"; we shift back 1h to get
    the interval-start label and upsample each hour to two 30-min rows (ffill).
  - Wind/Solar fuel splits (EIA 'Net Generation from Wind/Solar (Adjusted)') only
    exist from 2018-07 onward => dataset starts 2018-07-01 (documented).
  - Rooftop solar is part of EIA-930 'Demand' (only utility-scale Solar reported),
    so NetDemand keeps rooftop in the load term - same choice as EIA-930/Australia.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SIX_MONTHS = {
    2018: ["Jul_Dec"],
    2019: ["Jan_Jun", "Jul_Dec"],
    2020: ["Jan_Jun", "Jul_Dec"],
    2021: ["Jan_Jun", "Jul_Dec"],
    2022: ["Jan_Jun", "Jul_Dec"],
    2023: ["Jan_Jun", "Jul_Dec"],
}
START = "2018-07-01"
END = "2022-12-31"
BAS = ["MISO", "SWPP", "ERCO"]
URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{y}_{half}.csv"


def fetch_balance(year: int, half: str, cache: Path) -> Path:
    path = cache / f"EIA930_BALANCE_{year}_{half}.csv"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    import urllib.request

    url = URL.format(y=year, half=half)
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
        f.write(r.read())
    return path


def load_series(cache: Path) -> pd.DataFrame:
    rows = []
    for year, halves in SIX_MONTHS.items():
        for half in halves:
            fp = fetch_balance(year, half, cache)
            d = pd.read_csv(fp)
            d = d[d["Balancing Authority"].isin(BAS)].copy()
            d["dt_end"] = pd.to_datetime(d["Local Time at End of Hour"])
            d = d.groupby("dt_end")[
                ["Demand (MW) (Adjusted)",
                 "Net Generation (MW) from Wind (Adjusted)",
                 "Net Generation (MW) from Solar (Adjusted)"]
            ].sum().reset_index()  # sum across MISO+SWPP+ERCO
            rows.append(d)
    s = pd.concat(rows, ignore_index=True).sort_values("dt_end").drop_duplicates("dt_end")
    s = s.rename(columns={
        "Demand (MW) (Adjusted)": "load_adj",
        "Net Generation (MW) from Wind (Adjusted)": "wind_adj",
        "Net Generation (MW) from Solar (Adjusted)": "solar_adj",
    })
    return s


def build_us_central(cache: Path, out_csv: Path | None = None) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from energy.calendar_covariates import build_calendar, CAL_COLS
    from energy.era5_covariates import era5_dir

    s = load_series(cache)
    # interval START label = end-of-hour minus 1h
    s["dt"] = s["dt_end"] - pd.Timedelta(hours=1)
    s = s.set_index("dt")[["load_adj", "wind_adj", "solar_adj"]]

    idx = pd.date_range(f"{START} 00:00", f"{END} 23:30", freq="30min")
    df = s.resample("30min").first().reindex(idx).ffill()  # hourly -> 30-min ffill
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["NetDemand"] = df["load_adj"] - df["wind_adj"].fillna(0) - df["solar_adj"].fillna(0)

    cal = build_calendar(idx, "us_central")
    out = cal.merge(df, on="date", how="left")
    base = Path(__file__).resolve().parents[2]
    e5 = pd.read_csv(base / "data/era5_covariates/us_central_era5_covariates.csv")
    out = out.merge(e5.rename(columns={"date": "_day"}),
                    left_on=out["date"].str.slice(0, 10), right_on="_day",
                    how="left").drop(columns=["_day"])
    keep = ["date", "NetDemand", "load_adj", "wind_adj", "solar_adj"] + \
        [c for c in CAL_COLS if c != "date"] + [c for c in e5.columns if c != "date"]
    out = out[[c for c in keep if c in out.columns]].sort_values("date").reset_index(drop=True)
    if out_csv:
        out.to_csv(out_csv, index=False, sep=";")
        print(f"[us_central] saved {len(out)} rows -> {out_csv}")
    return out


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[2]
    cache = repo / "data/eia930"
    cache.mkdir(exist_ok=True)
    build_us_central(cache, repo / "data/us_central_energy.csv")
