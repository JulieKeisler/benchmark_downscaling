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
    inference_bicubic.py      bicubic regridding baseline
    inference_bcsd.py         BCSD (quantile mapping + delta scaling)
    inference_cdft.py         CDF-t (quantile mapping)
    inference_dotc.py         DOTC (distribution of trends correction)
    inference_idownscale.py   iDownscale (CDF-t on coarse data + MSE regression UNet, trained on perfect prognosis)
    inference_corrdiff.py     CorrDiff-QM (residual corrective diffusion + QM, two-stage ERA5-only training, stochastic ensemble)
    methods.py                shared physical post-processing helpers
    __init__.py               METHOD_REGISTRY + METHOD_LABELS
    utils/
      dataset.py              ERA5DownscalingDataset (idownscale / consistency modes)
      training.py             unified train loop (idownscale / consistency)
      inference_utils.py      neural_predict (regression / stroke guidance)
      unet.py                 UNet and UNetAttention architectures
      normalization.py        encode / decode, compute_stats, LOG_VARS
      month.py                month_channels (sin/cos encoding)
      ema.py                  EMA (exponential moving average of weights)

  configs/
    benchmark.yaml            evaluation config (period, variables, paths)
    methods.yaml              documented hyperparameters per method
    data/
      era5.yaml               ERA5 download config (area, period, variables)
      cmip6.yaml              CMIP6 download config (models, scenario)

  data_download/
    DATA.md                   data sources, expected formats, unit traps
    cds/
      download_era5.py        ERA5 daily stats + wind → data/raw/era5/
      download_cmip6.py       CMIP6 GCMs via CDS API → data/raw/cmip6/
      hourly_era5.py          hourly ERA5 wind → aggregated sfcWind files
    esgf/
      download_cmip6_esgf.py  ESGF download (psl / sfcWindmax missing from CDS)
    local_data/
    wb2_archive/
      download_wb2.py         [archived] WeatherBench2 download

  evaluation/
    data_processing.py        load_era5, load_gcm, bicubic_regrid, save_predictions
    evaluate.py               compute all metrics, write metrics.json + plot_data.json
    plot_comparison.py        multi-method comparison figures
    psd_sweep.py              spectral diagnostics sweep

  metrics/
    scalar_metrics_era5.py    bias, RMSE, KS stat, seasonal MAE, extremes
    scalar_metrics_gcm.py     temporal correlation, lag-1 autocorrelation, delta scores
    inter_variables_metrics.py  multivariate Spearman correlation
    spatial_spectrum_metrics.py  isotropic power spectral density (spectral_mae)

  plots/
    spatial_plots.py          bias maps, temporal correlation, delta maps
    line_plots.py             CDF, seasonal cycle, annual anomalies, spectra
    radar_plots.py            radar chart across all metrics
    utils.py                  cartopy helpers

  dashboard/
    dashboard.py              Streamlit app — browse results interactively

tests/
  conftest.py
  test_data_processing.py
  test_methods.py
  test_metrics.py
  test_registry.py

CONTRIBUTING.md               guide for adding a new method
TUTORIALS.md                  step-by-step walk-throughs
requirements.txt              cdsapi, pyyaml, xarray, netcdf4, requests, diffusers, sbck
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

See [`src/data_download/DATA.md`](src/data_download/DATA.md) for data sources,
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

Pre-computed methods require their inference script to be run first, then evaluated with `--pred-dir`.

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

## Adding a new method

See `CONTRIBUTING.md` for the full walk-through.
