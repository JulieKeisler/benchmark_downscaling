# Net-Load Downstream Evaluation Lens (TAM)

## Overview

In energy-system planning, climate model outputs must be downscaled to drive power-system models. However, because Global Climate Models (GCMs) are free-running simulations not tied to a historical calendar, pointwise temporal metrics (like daily MAE or RMSE against historical reanalysis) are not well-defined. 

To overcome this, our benchmark incorporates a **downstream net-load evaluation lens** based on a Time series Additive Model (TAM). Instead of comparing high-dimensional weather distributions directly, every climate product (ERA5 reference, raw GCM, BCSD, CDF-t, dOTC, iDownscale, Consistency Models, ChimeraFlow) is passed through a **fixed weather-to-net-load model** trained on ERA5 reanalysis (2012–2021).

The lens is implemented as a single region-parameterized script — [`net_lens_region.py`](TAM_downscaling/net_lens_region.py) — driven by a per-region config (ERA5 box, paired GCM, downscaling result directory, energy dataset, variable set). Currently configured for **france** (5 vars incl. `sfcWindmax`; IPSL-CM6A-LR GCM) and **us_central** (4 vars — CESM2 has no `sfcWindmax`; CESM2 GCM, ERA5 box 34–48°N, 105–89°W).

## Net Demand Definition

Daily net demand is defined identically in each region as:
$$\text{NetDemand} = \text{load} - \text{wind} - \text{solar}$$

- **France**: `rte_load_france − rte_wind_france − rte_solar_france`, from the RTE 30-min series.
- **us_central**: EIA-930 `Demand(Adj) − Wind(Adj) − Solar(Adj)` over MISO+SWPP+ERCO (utility-scale solar only; rooftop is inside demand).

This compounds the temperature sensitivity of electricity demand with the wind and solar sensitivities of renewable generation, capturing key multi-variable interactions relevant to grid stability and residual demand.

## Spatial Representations

To isolate representation choices from downscaling performance, three spatial summarizations of the multi-variable weather fields (`tas`, `sfcWind`, [`sfcWindmax`], `pr`, `psl`) are evaluated:
1. **Full Grid PCA**: Joint PCA over the full multi-variable grid (15 principal components + 5 region-wide spatial means). This is the representation currently exported for us_central.
2. **Regional Barycenters**: Fields sampled at regional load barycenters (France: 12 RTE regions).
3. **Regional + Renewables**: Per-variable barycenters (solar-weighted temperature, wind-weighted wind, load-weighted precipitation/pressure).

## Evaluation Metrics

Induced net-load distributions over the 2000–2022 evaluation window are compared against the ERA5 reanalysis reference using:
- **KS (Kolmogorov-Smirnov Statistic)**: Overall distributional distance.
- **Wasserstein Distance (MW)**: Earth-mover distance measuring absolute distribution shift.
- **KL Divergence**: Relative entropy between induced probability distributions.
- **AE@q01 (MW)**: Absolute error at the 1st lower percentile (low demand / surplus generation).
- **AE@q99 (MW)**: Absolute error at the 99th upper percentile (high demand / peak load stress).

### France (Full Grid PCA)

| Method | KS | Wasserstein (MW) | KL | AE@q01 (MW) | AE@q99 (MW) |
|---|---|---|---|---|---|
| IPSL (bicubic) | 0.0464 | 1084.3 | 0.0288 | 676.6 | 1117.9 |
| BCSD | 0.0239 | 411.7 | 0.0293 | 507.2 | 1299.0 |
| CDF-t | 0.0408 | 1160.8 | 0.0457 | 128.2 | 4168.7 |
| iDownscale | 0.0248 | 422.1 | 0.0246 | 157.1 | 1694.6 |
| Consistency | 0.0236 | 396.3 | 0.0266 | 85.9 | 767.6 |
| dOTC | 0.0209 | 366.8 | 0.0240 | 109.4 | 2063.4 |
| ChimeraFlow | 0.0217 | 414.0 | 0.0251 | 159.7 | 254.8 |

Downscaled products cut the KS/Wasserstein error of raw IPSL roughly in half; dOTC is best on the bulk (KS, W), while ChimeraFlow is best on high-demand tails (AE@q99).

### us_central (Full Grid PCA)

| Method | KS | Wasserstein (MW) | KL | AE@q01 (MW) | AE@q99 (MW) |
|---|---|---|---|---|---|
| CESM2 (bicubic) | 0.0475 | 3743.2 | 0.0470 | 2317.5 | 11372.1 |
| iDownscale | 0.0409 | 3599.3 | 0.0354 | 4710.2 | 11444.9 |
| ChimeraFlow | 0.0270 | 2142.2 | 0.0226 | 5205.9 | 6364.8 |

Same qualitative ordering as France: ChimeraFlow is best on KS/Wasserstein/KL and halves the peak-load (AE@q99) error of the raw CESM2 bicubic baseline, but its lower-tail (AE@q01) is worse than the GCM — a bulk-vs-tail trade-off to report. Only methods with locally synced `predictions.pt` (iDownscale, ChimeraFlow) are shown; the remaining us_central methods await prediction files from the cluster. RMSE on the training overlap: France 1778.4 MW, us_central 5532.1 MW (us_central actuals start 2018-07).

## Streamlit Dashboard Integration

The Streamlit dashboard includes a dedicated **"⚡ Load Forecasting (TAM)"** tab (region-aware; shown for france and us_central) featuring:
- Interactive spatial representation selection (us_central currently exposes Full Grid PCA).
- Performance metrics tables with automatic highlighting of the best results per metric.
- Interactive Plotly error comparisons, tail quantile profiles, CDF curves, and seasonal/wind-level breakdown charts.
