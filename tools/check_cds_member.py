#!/usr/bin/env python3
"""Check that the CDS files without a variant_label are the expected member.

The CDS delivers rsds and sfcWindmax for three pairs without metadata (12 files: two
variables, two periods). Each crop is compared, over every day it holds, with the same
grid points read by OPeNDAP from the published member on ESGF (CEDA); `--control` also
compares one crop with another member, which must differ.

Usage: python tools/check_cds_member.py [--control]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "data" / "cmip6" / "_v2_inputs"
TDS = "https://esgf.ceda.ac.uk/thredds/dodsC/esg_cmip6/CMIP6"
REMOTE = {  # (model, variable, period) -> files of member {m}
    ("IPSL-CM6A-LR", "rsds", "historical"): ["CMIP/IPSL/IPSL-CM6A-LR/historical/{m}/day/rsds/gr/v20180803/rsds_day_IPSL-CM6A-LR_historical_{m}_gr_18500101-20141231.nc"],
    ("IPSL-CM6A-LR", "rsds", "ssp370"): ["ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp370/{m}/day/rsds/gr/v20190119/rsds_day_IPSL-CM6A-LR_ssp370_{m}_gr_20150101-21001231.nc"],
    ("IPSL-CM6A-LR", "sfcWindmax", "historical"): ["CMIP/IPSL/IPSL-CM6A-LR/historical/{m}/day/sfcWindmax/gr/v20180803/sfcWindmax_day_IPSL-CM6A-LR_historical_{m}_gr_18500101-20141231.nc"],
    ("IPSL-CM6A-LR", "sfcWindmax", "ssp370"): ["ScenarioMIP/IPSL/IPSL-CM6A-LR/ssp370/{m}/day/sfcWindmax/gr/v20190119/sfcWindmax_day_IPSL-CM6A-LR_ssp370_{m}_gr_20150101-21001231.nc"],
    ("CNRM-CM6-1", "rsds", "historical"): ["CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/{m}/day/rsds/gr/v20180917/rsds_day_CNRM-CM6-1_historical_{m}_gr_19500101-20141231.nc"],
    ("CNRM-CM6-1", "rsds", "ssp370"): ["ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp370/{m}/day/rsds/gr/v20190219/rsds_day_CNRM-CM6-1_ssp370_{m}_gr_20150101-21001231.nc"],
    ("CNRM-CM6-1", "sfcWindmax", "historical"): ["CMIP/CNRM-CERFACS/CNRM-CM6-1/historical/{m}/day/sfcWindmax/gr/v20180917/sfcWindmax_day_CNRM-CM6-1_historical_{m}_gr_19500101-20141231.nc"],
    ("CNRM-CM6-1", "sfcWindmax", "ssp370"): ["ScenarioMIP/CNRM-CERFACS/CNRM-CM6-1/ssp370/{m}/day/sfcWindmax/gr/v20190219/sfcWindmax_day_CNRM-CM6-1_ssp370_{m}_gr_20150101-21001231.nc"],
}
LOCAL = {"historical": "1960-2014", "ssp370": "2015-2100"}
PAIRS = [("australia/ipsl_cm6a_lr", "IPSL-CM6A-LR", "r1i1p1f1"),
         ("us_central/ipsl_cm6a_lr", "IPSL-CM6A-LR", "r1i1p1f1"),
         ("us_central/cnrm_cm6_1", "CNRM-CM6-1", "r1i1p1f2")]


def identical(sub: str, model: str, var: str, period: str, member: str) -> bool:
    loc = xr.open_dataset(INPUTS / sub / f"{var}_{period}_{LOCAL[period]}.nc", use_cftime=True)[var]
    days = [str(t)[:10] for t in loc.time.values]
    parts, rdays = [], []
    for f in REMOTE[(model, var, period)]:
        rem = xr.open_dataset(f"{TDS}/{f.format(m=member)}", use_cftime=True)[var]
        # contiguous index slices: the CEDA OPeNDAP server refuses large or scattered requests
        ilat = np.searchsorted(rem.lat.values, loc.lat.values)
        ilon = np.searchsorted(rem.lon.values, loc.lon.values % 360)
        rem = rem.isel(lat=slice(ilat.min(), ilat.max() + 1), lon=slice(ilon.min(), ilon.max() + 1))
        assert np.array_equal(rem.lat.values, loc.lat.values)
        assert np.array_equal(rem.lon.values, loc.lon.values % 360)
        rt = [str(t)[:10] for t in rem.time.values]
        keep = [i for i, d in enumerate(rt) if days[0] <= d <= days[-1]]
        if not keep:
            continue
        for i0 in range(keep[0], keep[-1] + 1, 3650):   # 10-year blocks
            i1 = min(i0 + 3650, keep[-1] + 1)
            parts.append(rem.isel(time=slice(i0, i1)).values)
            rdays += rt[i0:i1]
    rem_v = np.concatenate(parts)
    same_dates = rdays == days
    same = same_dates and np.array_equal(loc.values, rem_v)
    diff = np.abs(loc.values - rem_v).max() if same_dates else float("nan")
    print(f"{sub} {var} {period} vs ESGF {member}: {len(days)} days, dates equal {same_dates}, "
          f"max|diff| {diff:.3g} -> {'IDENTICAL' if same else 'DIFFERENT'}", flush=True)
    return same


def main() -> None:
    ok = all([identical(sub, model, var, period, member) for sub, model, member in PAIRS
              for var in ("rsds", "sfcWindmax") for period in ("historical", "ssp370")])
    if "--control" in sys.argv and identical("australia/ipsl_cm6a_lr", "IPSL-CM6A-LR", "rsds",
                                             "ssp370", "r2i1p1f1"):
        sys.exit("control member is identical: the check cannot discriminate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
