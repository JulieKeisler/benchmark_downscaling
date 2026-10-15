# Data download

## Sources

| Dataset | What it provides | Script |
|---------|-----------------|--------|
| ERA5 (WB2 archive on Jean-Zay) | Daily tas/psl/pr/sfcWind/sfcWindmax, 0.25°, 1979–2022, no account/queue | `wb2_archive/crop_jz_archive.py` (run on Jean-Zay) |
| ERA5 (ARCO-ERA5, Google public GCS) | Daily `rsds` from hourly 0.25° ERA5, 1979–2022, no account/queue | `arco/download_era5_arco.py` |
| ERA5 (CDS) | Daily tas/pr/rsds/psl and wind pipeline (optional, needs API key; slow queue) | `cds/download_era5.py` |
| CMIP6 (ESGF) | Multi-scenario GCM fields (historical + ssp245/ssp370/ssp585) | `esgf/download_cmip6_esgf.py` |
| CMIP6 (Pangeo, Google public GCS) | Day-table Zarr mirror; fills datasets missing from the ESGF index | `pangeo/download_cmip6_pangeo.py` |
| CMIP6 (CDS) | GCM fields historically, limited variable/experiment coverage | `cds/download_cmip6.py` |

**Primary pipeline (used for the benchmark):**

```
ERA5:  WB2 archive on Jean-Zay   ->  crop_jz_archive.py  ->  data/era5/<region>/<var>_<year>.nc
ERA5 rsds: ARCO-ERA5 (GCS)      ->  download_era5_arco.py -> data/era5/<region>/rsds_<year>.nc
GCM:   ESGF (dkrz fast mirror)   ->  download_cmip6_esgf.py -> data/cmip6/<region>/<model>/<var>_<exp>_<years>.nc
```

---

## Configs (`src/configs/data/`)

| File | Purpose |
|------|---------|
| `regions.yaml` | Region definitions: `france`, `australia`, `us_central` — labels, `area` (N,W,S,E), `era5_dir`, `gcm_dir`, `gcm` per region |
| `cmip6.yaml` | France CMIP6 ESGF download config (models, scenarios, ignore_missing) |
| `cmip6_australia.yaml` | Australia config: CNRM-CM6-1, `scenarios: [ssp370, ssp245]` |
| `cmip6_uscentral.yaml` | US central config: CESM2, `scenarios: [ssp370, ssp245]` — **benchmark runs use ssp245** (only complete scenario incl. `rsds`) |
| `era5.yaml` | CDS-only ERA5 config (default France; `regions.yaml` drives the WB2/jz path) |

New regions: add an entry to `regions.yaml`, then `cmip6_<region>.yaml`
(`--area N W S E` also works on the ESGF script without editing yaml).

### Known variable gaps (confirmed on ESGF **and** CDS)

| Model | Variable | Availability |
|-------|----------|--------------|
| CNRM-CM6-1 | `sfcWind` | day table only for `historical` (ssp370/ssp245 not published, any member/table) — **filled** by deriving `√(uas²+vas²)` from daily-mean 10 m wind components (`jz/jobs/derive_sfcwind.py`) uniformly across historical + ssp245 + ssp370 |
| CESM2 | `sfcWindmax` | not published at all (day/6hrLev/3hr/Amon, any member) |
| CESM2 | `rsds` ssp370 | daily only in r1/r2, both **retracted**; r4i1p1f1 has only `Amon` rsds (+ daily `rsus`). Not filled — a `rsus_day / albedo_month` reconstruction validated on 1995–2014 has RMSE 26 W m⁻² (DJF 21 %), too noisy for daily metrics |
| CESM2 | `rsds` ssp245 | absent from the ESGF index but on Pangeo GCS (r4i1p1f1, v20200528) — fetched with `pangeo/download_cmip6_pangeo.py` |

Notes:

- **CNRM-CM6-1 `sfcWind`: derived, not native.** Native CMIP daily `sfcWind` =
  time-mean of the *instantaneous* speed; with only daily `uas`/`vas`
  published for the SSPs, the native definition cannot be reproduced
  (`√(uas²+vas²)` of daily-mean components runs ~12% low — mean 3.64 vs
  4.15 m s⁻¹ for 2000-2020). By benchmark decision (2026-09-18) the derived
  product is applied to **all three experiments** so the definition is
  consistent across 1960-2100 (no seam at 2014/2015). Provenance is kept in
  the NetCDF attributes (`derivation`, `comment`, `history`).
- The US central set has `tas/pr/psl/sfcWind` (no `sfcWindmax`). Configure
  these in `ignore_missing` so the script reports them as expected warnings
  rather than failures.

### CESM2 ensemble members

- historical: `r1i1p1f1` is the default download, but the benchmark uses `r4i1p1f1` in
  both periods (one realisation; `load_gcm` refuses a mixed-member pair)
- ssp370 / ssp245: only `r4i1p1f1` exists — the script auto-detects the member
  (falls back to a member-filter-free search, then keeps only the lowest-numbered member)
  when the configured one has no files. Before this was fixed (2026-09-23) the fallback
  concatenated r4+r10+r11 — `pr_ssp245` had 3× the time steps; it was re-fetched from Pangeo.

---

## Scripts

### `wb2_archive/crop_jz_archive.py`  (run on Jean-Zay)

Crops WeatherBench2 0.25° ERA5 from `/lustre/fsstor/projects/rech/idl/uzm26qx/era5/1440x721/weatherbench2/4xyearly`
to a region and aggregates the 6-hourly files to daily:

```bash
# from $BM_CODE on Jean-Zay (module pytorch-gpu/py3/2.7.0 provides h5py/hdf5plugin)
python src/data_download/wb2_archive/crop_jz_archive.py \
    --region us_central --output-dir $BM_CODE/data/era5/us_central \
    --years 1979 2022 --variables tas psl pr sfcWind sfcWindmax
```

Daily aggregation: `tas`/`psl`/`surface_pressure` → daily mean; `total_precipitation_6hr`
→ daily sum; `sfcWind` → daily mean of `√(u²+v²)`; `sfcWindmax` → daily max.
Lat/lon outputs are north→south (57×65 for france/australia/us_central), matching the
France convention. The script imports `REGIONS` from `data_download_regions.py`
(regions.yaml loader); `sys.path` is set to the script directory automatically.
Writes `<var>_<year>.nc` (CF-time `time`, not cftime). SLURM job scripts:
`jz/jobs/job_crop_era5_<region>.sh` (partition `compil`, account `bqu@v100` — the
`cpu_p1` partition rejects GPU accounts).

### `arco/download_era5_arco.py`  (run anywhere with internet)

Reads Google's ARCO-ERA5 (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`,
anonymous) — the full hourly 0.25° ERA5 single-level archive. Preferred over CDS for `rsds`.

```bash
python src/data_download/arco/download_era5_arco.py                       # all regions in regions.yaml, 1979-2022
python src/data_download/arco/download_era5_arco.py --regions france --years 2000 2000
python src/data_download/arco/download_era5_arco.py --area 55 -6 41 10 --output-dir data/era5
```

- Chunks are one **global** hourly field, so a year costs ~8.8k chunk reads regardless of domain
  size; the script streams each chunk once and crops **all regions from the same read**
  (~2–3 min/year for the three regions on a good connection, ~2 h for 1979–2022).
- Daily `rsds` = mean of the 24 hourly accumulations ending in (D 00:00, D+1 00:00] UTC, / 3600
  → W m⁻² (ERA5 stamps accumulations at the end of the hour). Same definition as CMIP6 day `rsds`.
- Skips existing `<var>_<year>.nc` (use `--overwrite`). Grid/time layout identical to the WB2 crop.
- The WB2 Zarr (`gs://weatherbench2/...`) only has *net* surface shortwave, not downwelling.

### `pangeo/download_cmip6_pangeo.py`

Day-table CMIP6 from the Pangeo Google Cloud mirror (`gs://cmip6`, catalog
`cmip6-zarr-consolidated-stores.csv`). Same config files, crop and output names as the ESGF
script; use it for datasets the ESGF index does not list:

```bash
python src/data_download/pangeo/download_cmip6_pangeo.py --config src/configs/data/cmip6_uscentral.yaml \
    --models cesm2 --variables rsds --experiments ssp245 --member r4i1p1f1
```

### `cds/download_era5.py`

Optional CDS alternative (needs `~/.cdsapirc`); daily-statistics + hourly wind pipelines.
Preferred only for `rsds` (radiation, absent from WB2) or when jz is unreachable.
Output maps to `data/era5/<region>/` the same `<var>_<year>.nc` layout.

### `esgf/download_cmip6_esgf.py`

Main CMIP6 source. Supports multiple scenarios and `ignore_missing`:

```bash
python src/data_download/esgf/download_cmip6_esgf.py \
    --config src/configs/data/cmip6_australia.yaml [--models cnrm_cm6_1] \
    [--variables tas pr] [--jobs 2]
```

Config keys: `scenarios` (list) or legacy `scenario`; `ignore_missing`
(`var:experiment` pairs reported as warnings); `areas` (per-model override of
`area`); `members`. `--area N W S E` overrides on the CLI.

**Mirror handling:** mirrors are ordered dkrz-first (`ESGF_MIRROR_ORDER` env var can
override), with a deterministic `esgf3.dkrz.de` fast-mirror candidate constructed from
the canonical CMIP6 DRS layout when the search index omits it (dkrz ≈ 22 MB/s vs
CEDA ≈ 1.3 MB/s). The `esg1.umr-cnrm.fr` mirror is dead (connect timeout in < 10 s).

**Notes / gotchas:**

- Downstream `load_gcm` (`src/evaluation/data_processing.py`) expects
  `{var}_historical_1960-2014.nc` and `{var}_{scenario}_2015-2100.nc` — keep the
  default `1960-2100` years.
- The historical file may start at 1850 (global spans are clipped to `[1960, 2014]` server-side).
- Non-NetCDF responses (HTML error pages) are detected by magic bytes and reported as errors.
- The dkrz search index intermittently returns empty result sets; the script retries (×4, backoff).

### `cds/download_cmip6.py`

CDS fallback (licensed `projections-cmip6`). Same layout; continue only if ESGF
is unreachable — CDS gaps mirror the table above.

---

## Variables

| Short name | Long name | ERA5 (WB2) source | CMIP6 variable |
|-----------|-----------|-------------------|----------------|
| `tas` | Near-surface temperature | `2m_temperature` (daily mean) | `tas` |
| `pr` | Precipitation | `total_precipitation_6hr` (daily sum, m day⁻¹) | `pr` |
| `sfcWind` | Near-surface wind speed | daily mean of `√(u²+v²)` | `sfcWind` |
| `sfcWindmax` | Daily max wind speed | daily max of `√(u²+v²)` | `sfcWindmax` |
| `psl` | Sea-level pressure | `mean_sea_level_pressure` (daily mean) | `psl` |
| `rsds` | Surface solar radiation | not in WB2 → ARCO-ERA5 `surface_solar_radiation_downwards` (hourly J m⁻² → daily mean W m⁻²) | `rsds` |

Regions (N, W, S, E): France `[55, -6, 41, 10]`, Australia `[-24, 136, -38, 152]`,
US central `[48, -105, 34, -89]` — each 57×65 grid cells at 0.25°.

---

## Expected output format

```
data/era5/<region>/            # e.g. data/era5/australia
  tas_1979.nc  tas_1980.nc  ... tas_2022.nc
  sfcWindmax_1979.nc ...
data/cmip6/<region>/<model>/   # e.g. data/cmip6/us_central/cesm2
  tas_historical_1960-2014.nc
  tas_ssp370_2015-2100.nc
  tas_ssp245_2015-2100.nc
  ...
```

### ERA5 file content

Each `<var>_<year>.nc` holds one variable on the 0.25° grid cropped to the region
(`(time, latitude, longitude)`, latitude north→south). Units as delivered (converted
on load by `data_processing.py`): `tas` K, `pr` m day⁻¹ (× 1000 → mm day⁻¹),
`psl` Pa, `sfcWind`/`sfcWindmax` m s⁻¹, `rsds` W m⁻² (no conversion needed).

### CMIP6 file content

Each file covers one experiment's full span for one variable, native GCM resolution.
`tas` K, `pr` kg m⁻² s⁻¹ (× 86 400 → mm day⁻¹), `psl` Pa, `sfcWind`/`sfcWindmax` m s⁻¹, `rsds` W m⁻².