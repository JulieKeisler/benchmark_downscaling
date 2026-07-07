import time
import zipfile
from pathlib import Path

import cfgrib
import numpy as np
import xarray as xr
import cdsapi

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("data/raw/era5_yearly")  # one zarr per year: era5_{year}.zarr
TEMP_DIR = Path("data/raw/era5_tmp")
YEARS = list(range(1960, 2021))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "reanalysis-era5-single-levels"

VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
]

# Variables expected in the final daily dataset
EXPECTED_VARS = {"tas", "pr", "rsds", "psl", "ps", "sfcWind", "sfcWindmax"}

AREA = [55, -6, 41, 10]

client = cdsapi.Client()


# =========================
# SKIP LOGIC
# =========================
def year_is_done(year: int) -> bool:
    """True if the year's zarr exists and has all expected variables."""
    path = OUTPUT_DIR / f"era5_{year}.zarr"
    if not path.exists():
        return False
    try:
        ds = xr.open_zarr(path)
        done = EXPECTED_VARS.issubset(set(ds.data_vars))
        ds.close()
        return done
    except Exception:
        return False


# =========================
# DOWNLOAD ONE YEAR
# =========================
def download_year(year: int) -> Path:
    """Download one year from CDS. Returns path to extracted GRIB file(s)."""
    zip_path = TEMP_DIR / f"era5_{year}.zip"
    extract_dir = TEMP_DIR / f"era5_{year}"

    if extract_dir.exists() and any(extract_dir.glob("*.grib")):
        print(f"  [{year}] already downloaded, skipping")
        return extract_dir

    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "grib",
        "download_format": "zip",
        "area": AREA,
    }

    print(f"  [{year}] submitting CDS request...")
    client.retrieve(DATASET, request).download(str(zip_path))

    print(f"  [{year}] extracting zip...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    zip_path.unlink()

    return extract_dir


# =========================
# OPEN GRIB → xarray
# =========================
def open_grib_dir(extract_dir: Path) -> xr.Dataset:
    """Open all GRIB files in directory, return merged Dataset.

    Uses cfgrib.open_datasets (plural) which splits mixed-step-type GRIB
    files into separate compatible datasets before merging — this avoids
    DatasetBuildError when accumulated variables (tp, ssrd) share a file
    with instantaneous ones (u10, v10, t2m, msl, sp).
    """
    grib_files = sorted(extract_dir.glob("*.grib")) + sorted(extract_dir.glob("*.grib2"))
    if not grib_files:
        grib_files = sorted(extract_dir.iterdir())

    all_ds = []
    for f in grib_files:
        try:
            dss = cfgrib.open_datasets(str(f), backend_kwargs={"indexpath": ""})
            for ds in dss:
                # Drop scalar coordinates that vary between datasets and
                # would block the merge (step, surface, number, valid_time).
                scalar = [c for c in ds.coords if ds[c].dims == ()]
                all_ds.append(ds.drop_vars(scalar, errors="ignore"))
        except Exception as e:
            print(f"    Warning opening {f.name}: {e}")

    if not all_ds:
        raise RuntimeError(f"No data opened from {extract_dir}")

    return xr.merge(all_ds, join="inner", compat="override")


# =========================
# AGGREGATE TO DAILY
# =========================
def aggregate_daily(ds: xr.Dataset) -> xr.Dataset:
    """
    GRIB short names:
      u10  = 10m U wind component
      v10  = 10m V wind component
      t2m  = 2m temperature
      msl  = mean sea level pressure
      sp   = surface pressure
      tp   = total precipitation (hourly accumulation)
      ssrd = surface solar radiation downwards (hourly accumulation)
    """
    daily = xr.Dataset()

    if "t2m" in ds:
        daily["tas"] = ds["t2m"].resample(time="1D").mean()

    if "tp" in ds:
        daily["pr"] = ds["tp"].resample(time="1D").sum()

    if "ssrd" in ds:
        daily["rsds"] = ds["ssrd"].resample(time="1D").mean()

    if "msl" in ds:
        daily["psl"] = ds["msl"].resample(time="1D").mean()

    if "sp" in ds:
        daily["ps"] = ds["sp"].resample(time="1D").mean()

    if "u10" in ds and "v10" in ds:
        hourly_ws = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        daily["sfcWind"] = hourly_ws.resample(time="1D").mean()
        daily["sfcWindmax"] = hourly_ws.resample(time="1D").max()

    return daily


# =========================
# WRITE ONE YEAR
# =========================
def write_year(ds: xr.Dataset, year: int) -> None:
    path = OUTPUT_DIR / f"era5_{year}.zarr"
    ds.to_zarr(path, mode="w")


# =========================
# PROCESS ONE YEAR
# =========================
def process_year(year: int) -> None:
    print(f"\n{'='*50}")
    print(f"[{year}] START")
    print(f"{'='*50}")

    t0 = time.time()

    extract_dir = download_year(year)

    print(f"  [{year}] opening GRIB files...")
    ds = open_grib_dir(extract_dir)
    print(f"  [{year}] variables found: {list(ds.data_vars)}")

    print(f"  [{year}] aggregating to daily...")
    daily = aggregate_daily(ds)
    daily = daily.compute()
    missing = EXPECTED_VARS - set(daily.data_vars)
    if missing:
        raise RuntimeError(f"Missing expected variables after aggregation: {missing}")
    print(f"  [{year}] daily shape: {dict(daily.sizes)}")

    print(f"  [{year}] writing to zarr...")
    write_year(daily, year)

    ds.close()
    import shutil
    shutil.rmtree(extract_dir, ignore_errors=True)

    elapsed = time.time() - t0
    print(f"  [{year}] DONE in {elapsed/60:.1f} min")


# =========================
# MAIN
# =========================
def main() -> None:
    print("Starting ERA5 hourly pipeline")
    print(f"  Years: {YEARS[0]}-{YEARS[-1]}")
    print(f"  Variables: {len(VARIABLES)}")
    print(f"  Output: {OUTPUT_DIR}/era5_{{year}}.zarr")
    print()

    skipped = [y for y in YEARS if year_is_done(y)]
    todo = [y for y in YEARS if not year_is_done(y)]
    if skipped:
        print(f"Skipping {len(skipped)} already-complete year(s): "
              f"{skipped[0]}-{skipped[-1]}")
    print(f"{len(todo)} year(s) to process\n")

    for year in todo:
        try:
            process_year(year)
        except Exception as e:
            print(f"\n  [{year}] FAILED: {e}")
            print(f"  [{year}] continuing to next year...\n")
            time.sleep(30)
            continue

    print(f"\n{'='*50}")
    print(f"ALL DONE -> {OUTPUT_DIR}")

    done = [y for y in YEARS if year_is_done(y)]
    print(f"  Complete years: {len(done)}/{len(YEARS)}")
    if done:
        ds = xr.open_zarr(OUTPUT_DIR / f"era5_{done[0]}.zarr")
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Shape (one year): {dict(ds.sizes)}")
        ds.close()


if __name__ == "__main__":
    main()
