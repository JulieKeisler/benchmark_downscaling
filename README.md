# benchmark_downscaling

A benchmark for evaluating downscaling and bias correction methods for
climate impact studies (in progress — currently: data download). Reference
data is ERA5 (reanalysis, used as pseudo-observations, 1960–2020); model
data comes from CMIP6 GCMs (1960–2100). The default domain is centered over
France, daily resolution.

## Repository layout

```
configs/data/   era5.yaml, cmip6.yaml — area, period, variables, models
scripts/        download_era5.py, download_cmip6.py
data/raw/       downloaded NetCDF files (not committed)
papers/         reference literature
```

## Installation

```bash
pip install -r requirements.txt
```

## Downloading the data

Both scripts use the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu). One-time setup:

1. Create a free account at https://cds.climate.copernicus.eu and accept the
   licence of each dataset on its page ("ERA5 post-processed daily statistics
   on single levels" and "CMIP6 climate projections").
2. Put your API key in `~/.cdsapirc`:

   ```
   url: https://cds.climate.copernicus.eu/api
   key: <your-personal-access-token>
   ```

Then:

```bash
# ERA5 reference data -> data/raw/era5/
python scripts/download_era5.py

# CMIP6 GCM data -> data/raw/cmip6/<model>/<variable>_<experiment>_<start>-<end>.nc
python scripts/download_cmip6.py

# Common options
python scripts/download_era5.py --variables tas pr   # subset of variables
python scripts/download_era5.py --no-wind            # skip the wind pipeline
python scripts/download_cmip6.py --models cesm2      # subset of models
python scripts/download_era5.py --jobs 8             # parallel CDS requests
python scripts/download_era5.py --config my.yaml     # alternative config
```

Both scripts skip outputs that already exist, so they can be interrupted
and re-run at any time.

Each request is sized to sit just under the CDS cost limits (checked
against the `/costing` API; the script also verifies this before
submitting anything, so an oversized selection fails immediately with the
actual numbers instead of one 403 per request):

- Daily statistics: the cost limit is **400 variable-days per request**,
  so one variable × one year (365) is the largest selection that fits →
  `data/raw/era5/<short_name>_<year>.nc`, 61 requests per variable.
- Hourly wind: the cost limit is 121,000 and one year of `u+v` costs
  ~105,120 → one year per request, 61 requests, then local aggregation to
  `sfcWind_<year>.nc` / `sfcWindmax_<year>.nc`.

Requests are submitted in parallel (`--jobs`, default 4) — this is the
main throughput lever, since the request sizes are maxed out. Raising
`--jobs` beyond ~8 rarely helps: the CDS caps how many of *your* requests
it processes concurrently, so the queue — not your connection — is the
bottleneck.

**Wind is derived, not downloaded:** the ERA5 daily-statistics dataset has
no wind-speed variable (only `u`/`v` components, and daily statistics of
components cannot give daily mean or max *speed*). The script therefore
downloads hourly `10m_u/v`, computes `speed = √(u² + v²)` per hour, and
aggregates locally to daily mean (`sfcWind`) and daily max (`sfcWindmax`) —
matching the CMIP6 definitions. The raw hourly chunks (~250 MB/year) are
deleted after aggregation unless `keep_hourly: true` is set in the config.

### What gets downloaded

| Short name   | Description                        | ERA5 source                                       | CMIP6 variable |
| ------------ | ---------------------------------- | ------------------------------------------------- | -------------- |
| `tas`        | Near-surface air temperature       | `2m_temperature` (daily mean)                     | `tas`          |
| `sfcWind`    | Near-surface wind speed            | hourly `10m_u/v` → daily mean of √(u²+v²)         | `sfcWind`      |
| `sfcWindmax` | Daily maximum wind speed           | hourly `10m_u/v` → daily max of √(u²+v²)          | `sfcWindmax`   |
| `pr`         | Precipitation                      | `total_precipitation` (daily mean)                | `pr`           |
| `rsds`       | Surface downwelling solar radiation| `surface_solar_radiation_downwards` (daily mean)  | `rsds`         |
| `psl`        | Sea level pressure                 | `mean_sea_level_pressure` (daily mean)            | `psl`          |

GCMs: **IPSL-CM6A-LR**, **CNRM-CM6-1**, **CESM2**. ERA5 covers 1960–2020;
the CMIP6 download covers 1960–2100, with 1960–2014 from the `historical`
experiment and 2015 onwards from `ssp585`.

Domain: `[55.0, -6.0, 41.0, 10.0]` = North, West, South, East (France).

**Unit caveats:** ERA5 accumulated variables are delivered as daily *means
of hourly accumulations*: multiply `pr` by `24 × 1000` for mm/day, divide
`rsds` by `3600` for W m⁻². CMIP6 `pr` is a flux in kg m⁻² s⁻¹: multiply by
`86400` for mm/day. Not every GCM publishes every variable at daily
resolution on the CDS; the script reports failures at the end and keeps
going (the [ESGF](https://esgf-node.llnl.gov/search/cmip6/) is the fallback
for anything missing).

**ERA5 known issue (June 2026):** the `*_since_previous_post_processing`
parameters of the daily-statistics dataset (max/min 2m temperature, 10m
wind gust, max/min total precipitation rate) are computed over a time
window shifted by +1 hour: these are interval variables whose 00:00 step
describes the previous day, and ECMWF's processing missed the required
one-hour shift. Daily means are barely affected, but extremes near midnight
land on the wrong day — do not use these parameters until the fix announced
in the [ECMWF forum](https://forum.ecmwf.int/t/issue-affecting-some-parameters-from-the-era5-post-processed-daily-statistics-on-single-levels/15057)
is released ([known-issues page](https://confluence.ecmwf.int/display/CKB/ERA5+family+post-processed+daily+statistics+documentation)).
None of the default variables here are affected (`sfcWindmax` is the daily
maximum of instantaneous `10m_wind_speed`, not the wind gust parameter), and
`scripts/download_era5.py` refuses to download the affected parameters
(`KNOWN_ISSUE_VARIABLES`; clear it once the fix lands). If you need daily
max/min temperature, use `2m_temperature` with `daily_maximum` /
`daily_minimum` instead.

### Troubleshooting

- **`Unknown API state 'rejected'` on every request:** your `cdsapi` is
  outdated and cannot talk to the current CDS — `pip install -U cdsapi`
  (you need ≥ 0.7). Also check that `~/.cdsapirc` uses the new format shown
  above (no `/v2` in the URL, no `UID:` prefix in the key).
- **`cost limits exceeded` (403):** the selection in your config is too
  large for one request. The script checks one representative request
  against the CDS `/costing` endpoint before submitting and prints the
  actual cost vs. limit; shrink the selection accordingly (e.g. wind
  `years_per_request` must stay 1).
- A re-run only downloads what is missing, so after fixing an error just
  run the same command again.

### Adding variables, areas, or GCMs

- **New ERA5 variable:** add `short_name: era5_name` under the appropriate
  statistic in the `daily_statistics` section of `configs/data/era5.yaml`
  (era5 names are listed on the CDS dataset page).
- **New CMIP6 variable:** add the CMIP6 short name to
  `configs/data/cmip6.yaml` and, if it is not already there, its CDS long
  name to `CDS_VARIABLES` in `scripts/download_cmip6.py`.
- **New GCM:** add its CDS model id (lowercase, underscores — e.g.
  `mpi_esm1_2_lr`) to the `models` list in `configs/data/cmip6.yaml`.
- **Different area or period:** edit `area` (N, W, S, E in degrees) and
  `years` in both config files. Keep the two consistent so observations and
  models cover the same domain.
