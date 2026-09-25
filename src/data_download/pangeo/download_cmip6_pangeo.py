#!/usr/bin/env python3
"""Download daily CMIP6 fields from the Pangeo / Google Cloud CMIP6 Zarr mirror.

Fallback for datasets missing from the ESGF search index: the public bucket
`gs://cmip6` (https://pangeo-data.github.io/pangeo-cmip6-cloud/) carries some
day-table datasets ESGF no longer lists — e.g. CESM2 ssp245 r4i1p1f1 `rsds`.
Anonymous access, and only the regional crop is transferred.

Output is identical in layout to download_cmip6_esgf.py:
    <output_dir>/<model>/<var>_<experiment>_<start>-<end>.nc
cropped to `area` ([N, W, S, E]) with longitudes in [-180, 180].

Usage:
    python src/data_download/pangeo/download_cmip6_pangeo.py \\
        --config src/configs/data/cmip6_uscentral.yaml \\
        --models cesm2 --variables rsds --experiments ssp245 --member r4i1p1f1

Requirements:
    pip install xarray zarr gcsfs netcdf4 pyyaml pandas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import xarray as xr
import yaml

CATALOG = "https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv"
REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE_IDS = {
    "ipsl_cm6a_lr":  "IPSL-CM6A-LR",
    "cnrm_cm6_1":    "CNRM-CM6-1",
    "cesm2":         "CESM2",
    "access_esm1_5": "ACCESS-ESM1-5",
}


def find_store(catalog: pd.DataFrame, source_id: str, experiment: str,
               variable: str, member: str | None) -> pd.Series:
    """Latest day-table store for (model, experiment, variable), preferring `member`."""
    rows = catalog[(catalog.source_id == source_id) & (catalog.experiment_id == experiment)
                   & (catalog.variable_id == variable) & (catalog.table_id == "day")]
    if member and (rows.member_id == member).any():
        rows = rows[rows.member_id == member]
    if rows.empty:
        raise LookupError(f"{source_id} {experiment} day/{variable}: not in the Pangeo catalog")
    return rows.sort_values("version").iloc[-1]


def crop(ds: xr.Dataset, area: list[float], start: int, end: int) -> xr.Dataset:
    """Same crop semantics as download_cmip6_esgf._crop_stage."""
    north, west, south, east = area
    if float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180).sortby("lon")
    return ds.sel(lat=slice(south, north), lon=slice(west, east),
                  time=slice(f"{start}-01-01", f"{end}-12-31"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--experiments", nargs="+", required=True)
    parser.add_argument("--member", default=None,
                        help="preferred member (default: config `members`, else latest any)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    out_root = REPO_ROOT / cfg["output_dir"]
    catalog = pd.read_csv(CATALOG)
    failures = 0
    for model in args.models or cfg["models"]:
        area = cfg.get("areas", {}).get(model, cfg["area"])
        member = args.member or cfg.get("members", {}).get(model)
        for experiment in args.experiments:
            start, end = (1960, 2014) if experiment == "historical" else (2015, 2100)
            for variable in args.variables:
                target = out_root / model / f"{variable}_{experiment}_{start}-{end}.nc"
                if target.exists() and not args.overwrite:
                    print(f"[skip] {target}")
                    continue
                try:
                    row = find_store(catalog, SOURCE_IDS[model], experiment, variable, member)
                except LookupError as exc:
                    print(f"[missing] {exc}")
                    failures += 1
                    continue
                print(f"[fetch] {row.zstore} ({row.member_id})")
                ds = xr.open_zarr(row.zstore, storage_options={"token": "anon"},
                                  decode_times=xr.coders.CFDatetimeCoder(use_cftime=True))
                out = crop(ds[[variable]], area, start, end).load()
                out.attrs = {**ds.attrs, "history":
                             f"{ds.attrs.get('history', '')}\ncropped from {row.zstore} "
                             "by src/data_download/pangeo/download_cmip6_pangeo.py".strip()}
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_suffix(".tmp.nc")
                out.to_netcdf(tmp)
                tmp.rename(target)
                print(f"[done] {target} {dict(out.sizes)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
