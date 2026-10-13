# Net-Load Downstream Evaluation Lens (TAM)

## Overview

In energy-system planning, climate model outputs must be downscaled to drive power-system models. However, because Global Climate Models (GCMs) are free-running simulations not tied to a historical calendar, pointwise temporal metrics (like daily MAE or RMSE against historical reanalysis) are not well-defined.

To overcome this, our benchmark incorporates a **downstream net-load evaluation lens** based on a Time series Additive Model (TAM). Instead of comparing high-dimensional weather distributions directly, every climate product (ERA5 reference, raw GCM, bicubic, BCSD, CDF-t, dOTC, iDownscale, Consistency Model, CorrDiff-QM, SerpentFlow-CLIM) is passed through a **fixed weather-to-net-load model** trained once per region on ERA5 reanalysis and that region's observed net demand (2012–2021).

The lens is maintained in the separate TAM-downscaling repo (`../TAM_downscaling`):

| Script | Coverage | Writes |
|---|---|---|
| `net_lens_region.py` | any region × GCM coupling, Full Grid PCA | `metrics_results_{region}[_{gcm}].json`, `cdf_results_…npz`, `tam_forecasts_results_…npz` |
| `analyze_degradation.py` | france × IPSL-CM6A-LR, Full Grid PCA (legacy) | `metrics_results.json`, `cdf_results.npz`, `tam_forecasts_results.npz` |
| `analyze_degradation_regional.py` | france × IPSL-CM6A-LR, Regional Barycenters | `metrics_regional.json`, `cdf_regional.npz`, `tam_forecasts_regional.npz` |
| `plot_regional_full_performance.py` | france × IPSL-CM6A-LR, Regional + Renewables | `metrics_regional_full.json`, `cdf_regional_full.npz`, `tam_forecasts_regional_full.npz` |

```bash
python net_lens_region.py --region australia                       # native GCM
python net_lens_region.py --region australia --gcm ipsl_cm6a_lr    # factorial coupling
```

`net_lens_region.py` scores every method in its default list that has a
`results/{region}[/{gcm}]/<method>/predictions.pt`; the per-region
`res_dir_alias` picks which result directory stands for each method. The lens
files may therefore hold unpublished methods: the dashboard and
`tools/deploy_hf.py` drop them (`baselines.labels.is_unpublished`).

> **France has two lenses.** For france × IPSL-CM6A-LR the dashboard reads the
> legacy files (`metrics_results.json`, …), not the `metrics_results_france.json`
> that `net_lens_region.py --region france` writes. They come from different TAM
> fits, so their numbers are not comparable — do not mix them.

## Regions and GCM Couplings

| Region | Native GCM (scenario) | Net-demand source | Actuals | Weather variables |
|---|---|---|---|---|
| **france** | IPSL-CM6A-LR (ssp370) | RTE, metropolitan France | 2012– | `tas pr sfcWind sfcWindmax psl rsds` |
| **us_central** | CESM2 (ssp245) | EIA-930, MISO + SWPP + ERCO | 2018-07-01– | `tas pr sfcWind psl rsds` (CESM2 has no `sfcWindmax`) |
| **australia** | CNRM-CM6-1 (ssp370) | AEMO via the OpenElectricity API, NSW1 + QLD1 | 2012– | `tas pr sfcWind sfcWindmax psl rsds` |

ERA5 boxes are those of `src/configs/data/regions.yaml`. Each region can also be
driven by the other two GCMs (`--gcm`, 9 couplings in `GCM_COUPLINGS`; CESM2 is
always ssp245). Factorial couplings read methods from `results/{region}/{gcm}/`
and write files suffixed `_{region}_{gcm}`.

**Status (2026-10-13):** only the three native couplings have been run and
published; no factorial lens output exists yet.

## Net Demand Definition

Daily net demand follows the same convention in each region:
$$\text{NetDemand} = \text{load} - \text{wind} - \text{solar}$$

- **France**: `rte_load_france − rte_wind_france − rte_solar_france`, from the RTE 30-min series.
- **us_central**: EIA-930 `Demand(Adj) − Wind(Adj) − Solar(Adj)` summed over MISO+SWPP+ERCO (utility-scale solar only; rooftop is inside demand). Wind/solar splits only exist from 2018-07, hence the later start. Built by `src/energy/build_us_central.py`.
- **australia**: AEMO `TOTALDEMAND` + rooftop PV − wind − utility PV, summed over NSW1+QLD1. Rooftop PV is added back because `TOTALDEMAND` excludes it, so it stays in the load term as in the other regions. Built by `src/energy/build_australia.py` (the 2012 history needs the OpenElectricity Academic plan).

This compounds the temperature sensitivity of electricity demand with the wind and solar sensitivities of renewable generation, capturing key multi-variable interactions relevant to grid stability and residual demand.

## TAM Model

- **Terms**: long-term trend (linear; a smooth spline for australia), calendar factors from the region's holiday calendar, a time-of-year spline, splines on the 15 weather PCs, on the region-wide mean of each variable, and on 3- and 7-day temperature moving averages. No lagged net demand.
- **Fitting**: the PCA and scaler are fitted on ERA5 2012–2021; TAM is fitted on the ERA5 days of 2012–2021 that have observed net demand (from 2018-07 for us_central).
- **Inference**: the fitted model is applied unchanged over 2000–2022 to ERA5, to the raw GCM (bicubic onto the ERA5 grid) and to every downscaled product.

## Spatial Representations

To isolate representation choices from downscaling performance, three spatial summarizations of the multi-variable weather fields are evaluated:
1. **Full Grid PCA**: Joint PCA over the full multi-variable grid (15 principal components + one region-wide spatial mean per variable). Available for all three regions.
2. **Regional Barycenters**: Fields sampled at regional load barycenters (France: 12 RTE regions). France only.
3. **Regional + Renewables**: Per-variable barycenters (solar-weighted temperature, wind-weighted wind, load-weighted precipitation/pressure). France only.

## Evaluation Metrics

Induced net-load distributions over the 2000–2022 evaluation window are compared against the ERA5-driven TAM predictions using:
- **KS (Kolmogorov-Smirnov Statistic)**: Overall distributional distance.
- **Wasserstein Distance (MW)**: Earth-mover distance measuring absolute distribution shift.
- **KL Divergence**: Relative entropy between induced probability distributions.
- **AE@q01 (MW)**: Absolute error at the 1st lower percentile (low demand / surplus generation).
- **AE@q99 (MW)**: Absolute error at the 99th upper percentile (high demand / peak load stress).
- **by_season / by_wind**: KS within each season (DJF, MAM, JJA, SON) and within each tercile of ERA5 region-mean `sfcWind`.

MW values scale with the size of each power system, so compare methods within a region, not across regions.

### Results

Per-method net-load scores (KS, Wasserstein, KL, and quantile absolute errors)
and the spatial-representation comparisons are computed by the scripts above
and rendered live in the dashboard's **Load Forecasting (TAM)** tab. They are
intentionally not reproduced here, so that this document stays a description of
the lens rather than a snapshot of results.


## Streamlit Dashboard Integration

The Streamlit dashboard includes a dedicated **"⚡ Load Forecasting (TAM)"** tab (region-aware; shown for france, us_central and australia) featuring:
- Interactive spatial representation selection (france only; us_central and australia expose Full Grid PCA).
- Region + GCM selection: a non-native GCM reads the `_{region}_{gcm}` files, which do not exist yet.
- Performance metrics tables with automatic highlighting of the best results per metric.
- Interactive Plotly error comparisons, tail quantile profiles, CDF curves, and seasonal/wind-level breakdown charts.

The tab reads from `TAM_DIR` (default: the local TAM-downscaling checkout). For
the hosted app, `tools/deploy_hf.py` stages every `metrics_*`, `cdf_*` and
`tam_forecasts_*` file from `TAM_DIR`, minus unpublished methods, into
`tam.tar.gz` on the Hugging Face dataset
`Jkeisler75/benchmark-downscaling-results`; the Streamlit Cloud app and the HF
Space download it at startup. Re-deploy after re-running the lens.
