# Net-Load Downstream Evaluation Lens (TAM)

## Overview

In energy-system planning, climate model outputs must be downscaled to drive power-system models. However, because Global Climate Models (GCMs) are free-running simulations not tied to a historical calendar, pointwise temporal metrics (like daily MAE or RMSE against historical reanalysis) are not well-defined. 

To overcome this, our benchmark incorporates a **downstream net-load evaluation lens** based on a Time series Additive Model (TAM). Instead of comparing high-dimensional weather distributions directly, every climate product (ERA5 reference, raw IPSL GCM, BCSD, CDF-t, dOTC, iDownscale, and Consistency Models) is passed through a **fixed weather-to-net-load model** trained on ERA5 reanalysis (2012–2021).

## Net Demand Definition

Daily French net demand is defined as:
$$\text{NetDemand} = \text{load} - \text{wind} - \text{solar}$$

This compounds the temperature sensitivity of electricity demand with the wind and solar sensitivities of renewable generation, capturing key multi-variable interactions relevant to grid stability and residual demand.

## Spatial Representations

To isolate representation choices from downscaling performance, three spatial summarizations of the 5-variable weather fields (`tas`, `sfcWind`, `sfcWindmax`, `pr`, `psl`) are evaluated:
1. **Full Grid PCA**: Joint PCA over the full 5-variable grid (15 principal components + 5 France-wide spatial means).
2. **Regional Barycenters**: Fields sampled at 12 French regional load barycenters (15 regional principal components).
3. **Regional + Renewables**: Per-variable barycenters (solar-weighted temperature, wind-weighted wind, load-weighted precipitation/pressure).

## Evaluation Metrics

Induced net-load distributions over the 2000–2020 evaluation window are compared against the ERA5 reanalysis reference using:
- **KS (Kolmogorov-Smirnov Statistic)**: Overall distributional distance.
- **Wasserstein Distance (MW)**: Earth-mover distance measuring absolute distribution shift.
- **KL Divergence**: Relative entropy between induced probability distributions.
- **AE@q01 (MW)**: Absolute error at the 1st lower percentile (low demand / surplus generation).
- **AE@q99 (MW)**: Absolute error at the 99th upper percentile (high demand / peak load stress).

## Streamlit Dashboard Integration

The Streamlit dashboard includes a dedicated **"⚡ Load Forecasting (TAM)"** tab featuring:
- Interactive spatial representation selection.
- Performance metrics tables with automatic highlighting of the best results per metric.
- Interactive Plotly error comparisons, tail quantile profiles, CDF curves, and seasonal/wind-level breakdown charts.
