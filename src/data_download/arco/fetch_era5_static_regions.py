#!/usr/bin/env python3
"""Fetch ERA5 static conditioning (4 channels) for all benchmark regions from CDS.

Channels follow SerpentFlow-CLIM Data_Preparation.ipynb exactly:
    lsm  : land-sea mask, kept in [0, 1]
    sdor : standard deviation of sub-gridscale orography, log1p -> minmax [-1,1]
    slor : slope of sub-gridscale orography,            minmax       [-1,1]
    isor : standard deviation of filtered sub-gridscale orography, minmax [-1,1]

CDS variable names (reanalysis-era5-single-levels, static fields, no time):
    land_sea_mask
    standard_deviation_of_sub_gridscale_orography
    slope_of_sub_gridscale_orography
    standard_deviation_of_filtered_subgridscale_orography

Writes per region: <output>/<region>/static_fields_v1.pt   (4, H, W) float32
                    <output>/<region>/static_fields.nc
and a merged <output>/static_fields_all.pt keyed by region.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
REGIONS_YAML = REPO_ROOT / "src" / "configs" / "data" / "regions.yaml"

CDS_VARS = {
    "lsm": "land_sea_mask",
    "sdor": "standard_deviation_of_sub_gridscale_orography",
    "slor": "slope_of_sub_gridscale_orography",
    "isor": "standard_deviation_of_filtered_subgridscale_orography",
}


def cdstime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def download_area(area: list[float], out_nc: Path) -> None:
    import cdsapi
    import xarray as xr

    c = cdsapi.Client()
    for var, cds_name in CDS_VARS.items():
        path = out_nc.parent / f"static_{var}.nc"
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [cds_name],
                "year": "2000", "month": "01", "day": "01", "time": "00:00",
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": list(area),  # N/W/S/E
                "grid": [0.25, 0.25],
            },
            str(path),
        )
        print(f"[{cdstime()}] {cds_name} -> {path}", flush=True)


def build_static(area: list[float], out_dir: Path) -> None:
    import torch
    import xarray as xr

    raw = {}
    for var in CDS_VARS:
        ds = xr.open_dataset(out_dir / f"static_{var}.nc")
        da = next(iter(ds.data_vars.values()))
        for d in ("valid_time", "time", "number", "expver"):
            if d in da.dims:
                da = da.isel({d: 0})
        lat = ds.latitude.values
        lon = ds.longitude.values
        raw[var] = np.asarray(da.values, dtype=np.float32)
    print(f"[{cdstime()}] {area}: lat {lat[0]}..{lat[-1]} ({len(lat)}) "
          f"lon {lon[0]}..{lon[-1]} ({len(lon)})", flush=True)

    # normalization identical to Data_Preparation.ipynb
    lsm = raw["lsm"]
    sdor_log = np.log1p(raw["sdor"])
    slor = raw["slor"]
    isor = raw["isor"]
    static = np.stack([
        lsm,
        2 * (sdor_log - sdor_log.min()) / (sdor_log.max() - sdor_log.min() + 1e-8) - 1,
        2 * (slor - slor.min()) / (slor.max() - slor.min() + 1e-8) - 1,
        2 * (isor - isor.min()) / (isor.max() - isor.min() + 1e-8) - 1,
    ]).astype(np.float32)  # (4, H, W)
    torch.save(torch.from_numpy(static), out_dir / "static_fields_v1.pt")
    print(f"[{cdstime()}] wrote {out_dir / 'static_fields_v1.pt'} {static.shape}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "static",
                        help="parent dir; per-region subdirs are created")
    args = parser.parse_args(argv)

    regions = yaml.safe_load(REGIONS_YAML.read_text())["regions"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_static = {}
    for name, cfg in regions.items():
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"[{cdstime()}] == {name} area={cfg['area']}", flush=True)
        download_area(cfg["area"], out / "static.nc")
        build_static(cfg["area"], out)
        import torch
        all_static[name] = torch.load(out / "static_fields_v1.pt", map_location="cpu")
    torch.save(all_static, args.output_dir / "static_fields_all.pt")
    print(f"[{cdstime()}] wrote {args.output_dir / 'static_fields_all.pt'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())