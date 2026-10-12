# Downscaling Benchmark for the Electricity System

A benchmark for evaluating downscaling and bias-correction methods for the
electricity system. Reference data is ERA5 daily reanalysis (pseudo-observations,
1960–2020); model data comes from CMIP6 GCMs (1960–2100). Default domain:
France, daily resolution.

We downscale the variables that drive the electric system — temperature
(demand), 10-m wind and gusts (wind generation), precipitation (hydrology),
sea-level pressure (weather regimes) — to ~25 km and score each baseline both
against the observations (accuracy) and against the GCM's future signal
(fidelity), plus a downstream net-load forecasting task.

- **Code & dashboard:** <https://github.com/JulieKeisler/benchmark_downscaling>
- **Dashboard:** `streamlit run src/dashboard/dashboard.py`

Compared to existing downscaling/bias-correction benchmark initiatives
(VALUE — Maraun et al. 2015; CORDEX/EURO-CORDEX — Jacob et al. 2014;
ISIMIP — Frieler et al. 2017; WeatherBench — Rasp et al. 2020/2024), this
benchmark is electricity-oriented, transparent (every baseline/metric links to
its implementation), projection-aware (all methods on the same GCM out to 2100),
and fully open-source.

---

## Repository layout

```
src/
  baselines/
    inference_bicubic.py       bicubic regridding baseline (inline)
    inference_bcsd.py          BCSD (quantile mapping + delta scaling)
    inference_cdft.py          CDF-t (quantile mapping, SBCK)
    inference_dotc.py          DOTC (trend-preserving CDF-t)
    inference_idownscale.py    iDownscale (perfect-prognosis U-Net)
    inference_corrdiff.py      CorrDiff-QM (residual corrective diffusion + QM)
    inference_cm.py            ConsistencyModel (restricted flow-matching)
    inference_serpentflowclim.py  SerpentFlow-CLIM (flow-matching gen.) — the
                                 GHG/ERA5 boundary-condition setup lives in
                                 serpentflow_clim/ below
    methods.py                 shared physical post-processing helpers
    __init__.py                METHOD_REGISTRY + METHOD_LABELS
    serpentflow_clim/          shared SF library (cdft, climatology, dataset,
                                 ema, filters, inference, model, time_utils,
                                 training) used by SerpentFlow-CLIM
    utils/                     dataset.py, training.py, inference_utils.py,
                                 unet.py, normalization.py, month.py, ema.py

  configs/
    benchmark.yaml             evaluation config (period, variables, paths)
    methods.yaml               documented hyperparameters per method
    data/
      regions.yaml             benchmark regions (area, ERA5/GCM dirs, GCM)
      energy_sites.yaml        dashboard day-of-year site lists per region
      era5*.yaml, cmip6*.yaml  download configs (area, period, variables)

  data_download/
    README.md                  data sources, expected formats, unit traps
    cds/                       download_era5.py, download_cmip6.py, hourly_era5.py
    esgf/                      download_cmip6_esgf.py (primary CMIP6 mirror)
    pangeo/                    download_cmip6_pangeo.py (GCS fallback)
    arco/                      download_era5_arco.py (GCS ERA5, no CDS),
                                 fetch_era5_static_regions.py
    wb2_archive/               download_wb2.py, download_wb2_gcs.py,
                                 crop_jz_archive.py, data_download_regions.py

  evaluation/
    data_processing.py         load_era5, load_gcm, bicubic_regrid, save_predictions
    evaluate.py                compute all metrics, write metrics.json + plot_data.json
    backfill_energy.py         energy backfill tool (see below)
    backfill_samples.py        regenerate samples_data_*.npz for old results
    energy_views.py            day-of-year / inter-variable event computation
    plot_comparison.py         multi-method comparison figures
    psd_sweep.py               spectral diagnostics sweep (t★ calibration)

  metrics/                     scalar_metrics_{era5,gcm}.py,
                                 inter_variables_metrics.py,
                                 spatial_spectrum_metrics.py
  plots/                       spatial_plots.py, line_plots.py, radar_plots.py, utils.py
  energy/                      downstream net-load forecast dataset builders
                                 (see LOAD_FORECASTING.md)
  dashboard/
    dashboard.py               Streamlit app — browse results interactively

tools/                         cluster/ops helpers: gen_factorial_jobs.py,
                                 backfill_energy_cluster.sh, check_energy_*.py,
                                 flatten_factorial_results.sh, ...
tests/
  conftest.py
  test_data_processing.py
  test_methods.py
  test_metrics.py
  test_registry.py

CONTRIBUTING.md                guide for adding a new method
TUTORIALS.md                   step-by-step walk-throughs
LOAD_FORECASTING.md            downstream net-load forecasting task
requirements.txt               cdsapi, pyyaml, xarray, netcdf4, requests,
                                 diffusers, sbck, streamlit, plotly, …
```

---

## Installation

```bash
pip install -r requirements.txt
```

For neural baselines (iDownscale, CorrDiff-QM, CM) also install:

```bash
pip install torch diffusers torchdiffeq
```

Python 3.10+ recommended.

---

## Data

See [`src/data_download/README.md`](src/data_download/README.md) for data sources,
script descriptions, and expected file formats.

```bash
python src/data_download/cds/download_era5.py    # needs ~/.cdsapirc
python src/data_download/cds/download_cmip6.py
# both accept --config, --variables, --jobs (parallel requests)
# cmip6 also accepts --models
```

---

## Baselines

| Method | Code | Type | Reference |
|--------|------|------|-----------|
| Bicubic | `src/baselines/inference_bicubic.py` | Inline | Interpolates GCM to ERA5 grid via bicubic splines. No learning. |
| BCSD | `src/baselines/inference_bcsd.py` | Pre-computed (–pred-dir) | Wood et al. (2002). QM on monthly means + spatial disaggregation. |
| CDF-t | `src/baselines/inference_cdft.py` | Inline | Michelangeli et al. (2009), via SBCK (`pip install sbck`). Sliding-window CDF-t. |
| DOTC | `src/baselines/inference_dotc.py` | Pre-computed (–pred-dir) | Vrac et al. (2012). Trend-preserving CDF-t: detrend → QM → re-add trend. |
| iDownscale | `src/baselines/inference_idownscale.py` | Pre-computed (–pred-dir) | Soares et al. (2024). CERFACS perfect-prognosis U-Net. |
| CorrDiff-QM | `src/baselines/inference_corrdiff.py` | Pre-computed (–pred-dir) | Mardani et al. (2025). Two-stage residual diffusion (UNet regression + Denoising Diffusion) + quantile mapping, stochastic ensemble. |
| ConsistencyModel | `src/baselines/inference_cm.py` | Pre-computed (–pred-dir) | Song et al. (2023) restricted flow-matching, ERA5-only. |
| SerpentFlow-CLIM | `src/baselines/inference_serpentflowclim.py` | Pre-computed (–pred-dir) | Zheng et al. (2025). Flow-matching generative downscaling conditioned on GHG + ERA5 static fields. |

Pre-computed methods require their inference script to be run first, then evaluated with `--pred-dir`. Each region is evaluated against its benchmark GCM
(native layout: one coupling per region) and in factorial layout (all GCM × region couplings); see `src/configs/data/regions.yaml`.

---

## Running the evaluation

**Step 1 — run inference for a method:**

```bash
# Bicubic (no GPU, no training)
python src/baselines/inference_bicubic.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/bicubic

# BCSD
python src/baselines/inference_bcsd.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/bcsd

# CDF-t (sliding-window, SBCK)
python src/baselines/inference_cdft.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/cdft

# DOTC (trend-preserving CDF-t)
python src/baselines/inference_dotc.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/dotc
```

Pre-computed neural methods (iDownscale, CorrDiff-QM, CM) first train on
ERA5 (`--train`), then infer on the GCM period (`--predict`):

```bash
# CorrDiff-QM (train on ERA5 1980-1999, predict GCM 2000-2100, 10-member ensemble)
python src/baselines/inference_corrdiff.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/corrdiff_qm \
    --train --predict --n-members 10

# iDownscale
python src/baselines/inference_idownscale.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/idownscale \
    --train --predict
```

**Step 2 — evaluate:**

```bash
python src/evaluation/evaluate.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --method bicubic \
    --pred-dir /path/to/results/bicubic \
    --output-dir /path/to/results \
    --start 2000 --end 2020 --gcm-end 2100
```

**Step 3 — dashboard:**

```bash
streamlit run src/dashboard/dashboard.py
# Open http://localhost:8501
# Results are auto-discovered from results/*/metrics.json
```

---

## Energy views & backfill

The dashboard's "Line plots → day-of-year" and "Inter-variable" views read
per-variable `doy` (region mean + energy sites from `src/configs/data/energy_sites.yaml`)
and `_events` blocks written into `plot_data.json` by `evaluate.py`. For results
that predate this feature (or were produced/rsync'd without it), backfill only
those artefacts — no full re-evaluation needed:

```bash
python src/evaluation/backfill_energy.py --region <france|australia|us_central>
```

`backfill_energy.py` resembles an ERA5/GCM-loaded evaluation: it needs `predictions.pt`
(or an inline registry method such as `bicubic`), re-stamps the GCM time axis, clips
physical variables, and writes a native single `.nc` per method for the extremes study.
On HPC, launch it through `tools/backfill_energy_cluster.sh` (and re-run `evaluate.py`
on the cluster for factorial couplings).

## Tools

Cluster/ops helpers live in `tools/`: per-coupling job generation
(`gen_factorial_jobs.py`), energy-backfill launchers and checkers
(`backfill_energy_cluster.sh`, `backfill_jz.tpl.sh`, `backfill_cleps.tpl.sh`,
`check_energy_backfill.py`, `check_energy_factorial.py`), and post-processing
helpers (`flatten_factorial_results.sh`). These are run
directly (`python tools/...`), not installed as a package.

---

## Adding a new method

See `CONTRIBUTING.md` for the full walk-through.
