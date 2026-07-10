# Downscaling benchmark for climate impact studies

A benchmark for evaluating downscaling and bias-correction methods for climate
impact studies. Reference data is ERA5 daily reanalysis (pseudo-observations,
1980–2020); model data comes from CMIP6 GCMs (1960–2100). Default domain:
France, daily resolution.

---

## Repository layout

```
src/
  baselines/
    inference_bicubic.py      bicubic regridding baseline
    inference_bcsd.py         BCSD (quantile mapping + delta scaling)
    inference_cdft.py         CDF-t (quantile mapping)
    inference_idownscale.py   iDownscale (CDF-t on coarse data + MSE regression UNet, trained on perfect prognosis)
    inference_serpentflow.py  SerpentFlow (spatial frequency-base alignment+flow-matching UNet, train + infer)
    inference_cm.py           Consistency model inference
    train_consistency.py      Consistency model training
    methods.py                shared physical post-processing helpers
    __init__.py               METHOD_REGISTRY + METHOD_LABELS
    utils/
      dataset.py              ERA5DownscalingDataset (all three neural modes)
      training.py             unified train loop (idownscale / serpentflow / cm)
      inference_utils.py      neural_predict (regression / ODE / stroke guidance)
      unet.py                 UNet and UNetAttention architectures
      normalization.py        encode / decode, compute_stats, LOG_VARS
      fourier.py              spectral_lowpass (2D Fourier low-pass filter)
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
      convert_serpentflow_wind.py  unit conversion for local SF wind data
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
requirements.txt              cdsapi, pyyaml, xarray, netcdf4, requests
```

---

## Installation

```bash
pip install -r requirements.txt
```

For neural baselines (iDownscale, SerpentFlow, Consistency Model) also install:

```bash
pip install torch torchdiffeq
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

# CDF-t
python src/baselines/inference_cdft.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/cdft

# SerpentFlow (trains then infers, needs GPU)
python src/baselines/inference_serpentflow.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir /path/to/results/serpentflow

# Consistency Model — train first, then infer
python src/baselines/train_consistency.py \
    --era5-dir /path/to/era5 --output-dir /path/to/checkpoints
python src/baselines/inference_cm.py \
    --era5-dir /path/to/era5 --gcm-dir /path/to/cmip6/ipsl_cm6a_lr \
    --checkpoint-dir /path/to/checkpoints --out-dir /path/to/results/cm
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
