#!/usr/bin/env python3
"""Tripwire: check a GCM input dir before any run (contract §E1–§E3).

For every variable: one realisation across the historical and scenario files,
a longitude range that covers the ERA5 domain once normalised to -180..180,
and no step at the 2014/2015 file join — the 2015-2019 minus 2010-2014
domain-mean change must stay within `--max-sd` interannual standard deviations
(1990-2014). Exits non-zero on any failure.

Usage:
    python tools/check_gcm_inputs.py --gcm-dir data/cmip6/australia/cnrm_cm6_1 \
        --era5-dir data/era5/australia --scenario ssp370 --variables tas pr psl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "evaluation"))


def join_step_sd(annual: dict[int, float]) -> float:
    """Step at the 2015 join, in standard errors of a 5-year mean.

    The 1990-2014 linear trend is extrapolated to 2015-2019; the departure of
    the 2015-2019 mean from it is divided by the residual sd / sqrt(5)
    (plus the extrapolation error). Detects steps of ~1.5 interannual sd.
    """
    yrs = np.array(sorted(annual))
    v = np.array([annual[y] for y in yrs])
    base = (yrs >= 1990) & (yrs <= 2014)
    after = (yrs >= 2015) & (yrs <= 2019)
    coef = np.polyfit(yrs[base], v[base], 1)
    resid = v[base] - np.polyval(coef, yrs[base])
    sd = resid.std(ddof=2)
    x0 = yrs[after].mean()
    se_fit = sd * np.sqrt(1 / base.sum() + (x0 - yrs[base].mean()) ** 2
                          / ((yrs[base] - yrs[base].mean()) ** 2).sum())
    se = np.sqrt(sd ** 2 / after.sum() + se_fit ** 2)
    return float(abs(v[after].mean() - np.polyval(coef, x0)) / se)


def check(gcm_dir: Path, era5_dir: Path, scenario: str, variables: list[str],
          max_sd: float = 3.0, allow_unlabelled: tuple[str, ...] = ()) -> list[str]:
    from data_processing import load_era5, load_gcm, check_same_member  # noqa: PLC0415

    failures = []
    members = {}
    ref = load_era5(era5_dir, variables[0], 2000, 2000)
    for var in variables:
        files = [gcm_dir / f"{var}_historical_1960-2014.nc",
                 gcm_dir / f"{var}_{scenario}_2015-2100.nc"]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            failures.append(f"{var}: missing {missing}")
            continue
        try:
            member = check_same_member(files)
            members[var] = member
        except ValueError as e:
            failures.append(f"{var}: {e}")
            continue
        da = load_gcm(gcm_dir, var, 1990, 2019, scenario)
        lon, lat = da.lon.values, da.lat.values
        dlon, dlat = abs(lon[1] - lon[0]), abs(lat[1] - lat[0])
        if (lon.min() > ref.lon.min() + dlon or lon.max() < ref.lon.max() - dlon
                or lat.min() > ref.lat.min() + dlat or lat.max() < ref.lat.max() - dlat):
            failures.append(f"{var}: GCM lon/lat [{lon.min()},{lon.max()}]x[{lat.min()},{lat.max()}] "
                            f"does not cover ERA5 [{ref.lon.min().item()},{ref.lon.max().item()}]x"
                            f"[{ref.lat.min().item()},{ref.lat.max().item()}]")
        annual = da.mean(["lat", "lon"]).groupby("time.year").mean("time")
        z = join_step_sd(dict(zip(annual.year.values.tolist(), annual.values.tolist())))
        status = "FAIL" if z > max_sd else "ok"
        print(f"  {var:10s} member={member}  lon=[{lon.min():.2f},{lon.max():.2f}]  "
              f"2015 step = {z:.2f} sd  {status}", flush=True)
        if z > max_sd:
            failures.append(f"{var}: step at the 2015 file join = {z:.2f} interannual sd")
    labelled = {v: m for v, m in members.items() if m is not None}
    if len(set(labelled.values())) > 1:
        failures.append(f"variables come from different realisations: {labelled}")
    unlabelled = sorted(set(members) - set(labelled) - set(allow_unlabelled))
    if unlabelled:
        failures.append(f"no variant_label for {unlabelled} (allow-list derived products "
                        "with --allow-unlabelled)")
    return failures


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--gcm-dir", required=True)
    p.add_argument("--era5-dir", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--max-sd", type=float, default=3.0)
    p.add_argument("--allow-unlabelled", nargs="*", default=[],
                   help="variables allowed without variant_label (derived products)")
    args = p.parse_args()
    print(f"check_gcm_inputs {args.gcm_dir} ({args.scenario})", flush=True)
    failures = check(Path(args.gcm_dir), Path(args.era5_dir), args.scenario,
                     args.variables, args.max_sd, tuple(args.allow_unlabelled))
    if failures:
        print("TRIPWIRE FAILED:\n  " + "\n  ".join(failures), flush=True)
        sys.exit(1)
    print("all GCM input checks passed", flush=True)


if __name__ == "__main__":
    main()
