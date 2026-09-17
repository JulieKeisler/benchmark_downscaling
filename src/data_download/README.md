# Data download

## Sources

| Dataset | What it provides | Script |
|---------|-----------------|--------|
| ERA5 (CDS daily statistics) | Temperature, precipitation, radiation, sea-level pressure | `cds/download_era5.py` |
| ERA5 (CDS hourly) | u/v wind components → daily wind speed/max | `cds/download_era5.py` (wind pipeline) |
| CMIP6 (CDS projections) | GCM fields, historical + ssp585, 1960–2100 | `cds/download_cmip6.py` |
| CMIP6 (ESGF) | Variables missing from CDS (psl, sfcWindmax on IPSL-CM6A-LR) | `esgf/download_cmip6_esgf.py` |

---

## One-time CDS setup

1. Create a free account at <https://cds.climate.copernicus.eu> and accept the licence for each dataset.
2. Write your API key to `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-access-token>
```

---

## Scripts

### `cds/download_era5.py`

Downloads ERA5 reference data to `data/raw/era5/`. Config: `configs/data/era5.yaml`.

```bash
python src/data_download/cds/download_era5.py
python src/data_download/cds/download_era5.py --variables tas pr   # subset
python src/data_download/cds/download_era5.py --jobs 8             # parallel
```

**Two internal pipelines:**

1. **Daily statistics** (`derived-era5-single-levels-daily-statistics`) — used for
   `tas`, `pr`, `rsds`, `psl`. The CDS cost limit is 400 variable-days per request,
   so the script issues one request per variable per year (365 days, cost 365).
   Output: `<short_name>_<year>.nc`.

2. **Wind** (`reanalysis-era5-single-levels`, hourly) — the daily-statistics dataset
   has no wind-speed variable (only u/v components). The script downloads hourly
   `10m_u/v` for one year per request (u+v ≈ 105 k of the 121 k cost limit), then
   computes `√(u²+v²)` per hour and aggregates locally to daily mean (`sfcWind`)
   and daily max (`sfcWindmax`). Raw hourly chunks are deleted after aggregation
   unless `keep_hourly: true` in the config.

The script pre-flights each pipeline against the `/costing` endpoint before
submitting real requests and aborts with the actual cost/limit numbers if a
request is oversized.

**Known issue (June 2026):** the `*_since_previous_post_processing` parameters
(max/min 2 m temperature, wind gust, max/min precipitation rate) have a +1 h
window shift due to an ECMWF processing bug. The script refuses them
(`KNOWN_ISSUE_VARIABLES`); clear that set once ECMWF announces the fix. None of
the default variables are affected.

### `cds/download_cmip6.py`

Downloads CMIP6 GCM fields to `data/raw/cmip6/<model>/`. Config: `configs/data/cmip6.yaml`.

```bash
python src/data_download/cds/download_cmip6.py
python src/data_download/cds/download_cmip6.py --models cesm2      # single GCM
python src/data_download/cds/download_cmip6.py --variables tas pr  # subset
python src/data_download/cds/download_cmip6.py --jobs 4
```

The requested period is split into `historical` (≤ 2014) and the configured
scenario (≥ 2015, default `ssp585`) by `experiment_spans()`. CDS delivers a zip;
the script extracts the single `.nc` inside. Not every GCM publishes every variable
at daily resolution — the script continues past failures and reports them at the end.

**CDS API v2 (2025):** `"date"` was replaced by `"year"` and `"month"` arrays;
`"format"` was removed (server decides). The script is updated accordingly.

### `esgf/download_cmip6_esgf.py`

Downloads variables not available on CDS — specifically `psl` and `sfcWindmax` for
IPSL-CM6A-LR — directly from ESGF nodes. Ensemble member IDs are in `configs/data/cmip6.yaml`
under `members`.

### `cds/hourly_era5.py`

Helper called by `download_era5.py` for the wind pipeline. Can also be run
standalone to download any hourly ERA5 variable.

---

## Variables

| Short name | Long name | ERA5 source | CMIP6 variable |
|-----------|-----------|-------------|----------------|
| `tas` | Near-surface temperature | `2m_temperature` (daily mean) | `tas` |
| `pr` | Precipitation | `total_precipitation` (daily mean of hourly accum.) | `pr` |
| `sfcWind` | Near-surface wind speed | hourly `10m_u/v` → daily mean of `√(u²+v²)` | `sfcWind` |
| `sfcWindmax` | Daily max wind speed | hourly `10m_u/v` → daily max of `√(u²+v²)` | `sfcWindmax` |
| `psl` | Sea-level pressure | `mean_sea_level_pressure` (daily mean) | `psl` |
| `rsds` | Surface solar radiation | `surface_solar_radiation_downwards` (daily mean of hourly accum.) | `rsds` |

Default domain: `[55.0, -6.0, 41.0, 10.0]` (N, W, S, E) — France.

---

## Expected output format

After running both scripts the tree should look like:

```
data/raw/
  era5/
    tas_1960.nc  tas_1961.nc  ...  tas_2020.nc
    pr_1960.nc   ...
    rsds_1960.nc ...
    psl_1960.nc  ...
    sfcWind_1960.nc     ...     sfcWind_2020.nc
    sfcWindmax_1960.nc  ...  sfcWindmax_2020.nc

  cmip6/
    ipsl_cm6a_lr/
      tas_historical_1960-2014.nc
      tas_ssp585_2015-2100.nc
      pr_historical_1960-2014.nc
      ...
    cnrm_cm6_1/
      ...
    cesm2/
      ...
```

### ERA5 file content

Each `<var>_<year>.nc` contains one variable on the ERA5 0.25° grid cropped to
the configured domain. Dimensions: `(time, latitude, longitude)`.

Unit as delivered by CDS (before `data_processing.py` converts on load):

| Variable | CDS unit | Conversion in `data_processing.py` |
|----------|----------|-------------------------------------|
| `tas` | K | none (already K) |
| `pr` | m (mean of hourly accumulations) | × 1000 × 24 → mm day⁻¹ |
| `rsds` | J m⁻² (mean of hourly accum.) | ÷ 3600 → W m⁻² |
| `psl` | Pa | none |
| `sfcWind`, `sfcWindmax` | m s⁻¹ | none |

### CMIP6 file content

Each file covers the full time span of one experiment for one variable. Dimensions:
`(time, lat, lon)`. Native GCM resolution (typically 1–2° for IPSL, CNRM, CESM2).

Unit as delivered (before `data_processing.py` converts on load):

| Variable | CMIP6 unit | Conversion in `data_processing.py` |
|----------|-----------|--------------------------------------|
| `tas` | K | none |
| `pr` | kg m⁻² s⁻¹ | × 86 400 → mm day⁻¹ |
| `psl` | Pa | none |
| `sfcWind`, `sfcWindmax`, `rsds` | native SI | none |

### GCMs

| GCM | CDS id | Notes |
|-----|--------|-------|
| IPSL-CM6A-LR | `ipsl_cm6a_lr` | `psl`, `sfcWindmax` not on CDS → use ESGF script |
| CNRM-CM6-1 | `cnrm_cm6_1` | uses ensemble member `r1i1p1f2` |
| CESM2 | `cesm2` | ssp585 uses `r4i1p1f1` (no `r1i1p1f1` available) |
