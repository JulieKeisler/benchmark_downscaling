# Spec — Metrics

Two metric families, computed against ERA5 (distributional) and against the GCM
(climate signal). Definitions live in `src/metrics/`.

## vs ERA5 (`src/metrics/scalar_metrics_era5.py`, `vs_obs` block)

| Key | Meaning |
| --- | --- |
| `bias_mae` | spatial mean of \|temporal-mean bias\| |
| `std_error` | error in temporal standard deviation |
| `ks_full` | Kolmogorov–Smirnov statistic, full distribution |
| `ks_q95` | KS statistic restricted to the upper tail (q > 0.95) |
| `seasonal_mae` | mean absolute error of the seasonal cycle |
| `extreme_freq` | bias in extreme-event frequency |
| `extreme_intensity` | bias in extreme-event intensity |
| `spectral_mae` | spatial-spectrum MAE (`src/metrics/spatial_spectrum_metrics.py`) |

## vs GCM (`src/metrics/scalar_metrics_gcm.py`, `vs_gcm` block)

| Key | Meaning |
| --- | --- |
| `temporal_corr` | daily temporal correlation vs the GCM |
| `temporal_corr_annual` | annual-mean temporal correlation vs the GCM |
| `annual_anomaly_bias` | bias in annual anomalies vs the GCM |
| `lag1_autocorr_mae` | MAE of lag-1 autocorrelation vs the GCM |

## Δ signal (`_delta` block)

`delta_change_score(period)` = error between the method's and the GCM's relative
climate-change signal over `period ∈ {2040_2060, 2080_2100}`. The dashboard shows
these as `Δ signal 2040–2060` / `Δ signal 2080–2100` and aggregates them in the
"Delta change scores — heatmap". A method with predictions through 2100 must
produce finite values for every shipped variable (see `CONTRACT.md §4`).

## Inter-variable (`_intervar` block)

`intervar_corr`: MAE of pointwise temporal correlation between variable pairs,
prediction vs ERA5. Unitless.

## Acceptance

For every published method × region, `metrics.json` contains all of the above
blocks with finite values for every shipped variable. (Audited benchmark-wide;
see `fix_plan.md`.)
