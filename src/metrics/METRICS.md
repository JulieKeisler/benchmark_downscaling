# Metrics

All metrics are computed per variable over the evaluation period (2000–2020 vs ERA5;
2000–2100 vs GCM). Scalars reported in `metrics.json`; spatial maps and curve data
in `plot_data.json`.

Variables: `tas` (K), `pr` (mm day⁻¹), `sfcWind` (m s⁻¹), `sfcWindmax` (m s⁻¹), `psl` (Pa).

Notation: `ŷ` = method prediction, `y` = ERA5 reference, `x` = GCM, `T` = number
of timesteps, `N` = number of grid points, `ȳ` = temporal mean.

---

## 1. vs ERA5 — distribution quality

These metrics compare method predictions to ERA5 over the historical period
(2000–2020) where ERA5 observations are available.

### 1.1 Bias MAE (`bias_mae`)

Mean absolute temporal-mean bias across all grid points:

```
bias_mae = (1/N) Σ_{i=1}^{N} | ȳ_i - ŷ̄_i |
```

where `ȳ_i = (1/T) Σ_t y_{t,i}` is the ERA5 time mean at grid point `i`.
Lower is better.

### 1.2 Standard deviation error (`std_error`)

Mean absolute error of temporal standard deviation across all grid points:

```
std_error = (1/N) Σ_{i=1}^{N} | σ(ŷ_i) - σ(y_i) |
```

where `σ(·)` is the temporal standard deviation. Measures how well the method
reproduces daily variability amplitude. Lower is better.

### 1.3 KS statistics (`ks_full`, `ks_q95`, `ks_q05`)

Kolmogorov–Smirnov distance between flattened (all grid points × all timesteps)
empirical distributions:

```
KS(F_y, F_ŷ) = sup_x | F_y(x) − F_ŷ(x) |
```

Three variants:
- `ks_full`: full marginal distribution.
- `ks_q95`: upper tail — both distributions restricted to values above the
  ERA5 global 95th percentile.
- `ks_q05`: lower tail — both distributions restricted to values at or below
  the ERA5 global 5th percentile. Returns NaN for `pr` when the 5th percentile
  is negative (CDS rounding artifact makes the lower tail empty).

Lower is better. Computed via `scipy.stats.ks_2samp`.

### 1.4 Seasonal MAE (`seasonal_mae`)

MAE of the 12-month climatology averaged over space and months:

```
seasonal_mae = (1/12) Σ_{m=1}^{12} (1/N) Σ_{i=1}^{N} | μ̂_{m,i} − μ_{m,i} |
```

where `μ_{m,i} = mean_{t: month(t)=m} y_{t,i}` is the ERA5 monthly climatology.
Measures the ability to reproduce the seasonal cycle. Lower is better.

### 1.5 Extreme frequency bias (`extreme_freq`)

Bias in the fraction of days exceeding the ERA5 global 95th percentile `q₀.₉₅`:

```
extreme_freq = (1/N) Σ_{i=1}^{N} [ P(ŷ_{t,i} > q₀.₉₅) − P(y_{t,i} > q₀.₉₅) ]
```

The threshold `q₀.₉₅` is a single global quantile over all grid points and
timesteps. A positive value means the method generates too many extreme events;
negative means too few.

### 1.6 Extreme intensity bias (`extreme_intensity`)

Bias in the mean intensity of events exceeding `q₀.₉₅`:

```
extreme_intensity = (1/N) Σ_{i=1}^{N} [ E[ŷ_{t,i} | ŷ_{t,i} > q₀.₉₅]
                                         − E[y_{t,i} | y_{t,i} > q₀.₉₅] ]
```

Uses the same global threshold as `extreme_freq`.

### 1.7 Spectral MAE (`spectral_mae`)

Normalised RMSE of the time-averaged isotropic power spectral density (PSD):

```
spectral_mae = sqrt( mean_{k valid} [ (PSD_ŷ(k) − PSD_y(k))² / PSD_y(k)² ] )
```

**PSD computation.** For each timestep `t`, the anomaly field (mean removed) is
Fourier-transformed with `np.fft.fft2`. The 2D power spectrum is binned into `N=40`
log-spaced radial wavenumber bins. The radially-averaged PSD is the mean power
per bin. The time average gives `PSD_y(k)` and `PSD_ŷ(k)`.

Only bins where both ERA5 and prediction PSD are finite and positive (`valid`) enter
the score. Wavenumber centers are stored in `k_centers_km_valid` as wavelengths in km
for plotting. Lower is better; a perfect score is 0.

---

## 2. vs GCM — climate change consistency

These metrics compare method output to the GCM over the full 2000–2100 period
to assess whether climate change signals are preserved.

### 2.1 Daily temporal correlation (`temporal_corr`)

Spatial mean of the Pearson correlation between GCM and prediction over time,
computed at each grid point independently:

```
temporal_corr = (1/N) Σ_{i=1}^{N} r( x_{:,i}, ŷ_{:,i} )
```

where `r(·,·)` is Pearson's r over the time axis. For free-running GCMs (not
time-aligned with ERA5), this tests whether the method follows the GCM's daily
evolution. Deterministic methods that are direct functions of the GCM score ~1;
stochastic methods that generate independent samples score near 0. Returns 0.0
when all grid points have zero variance in predictions (e.g. a stochastic model
that collapses to monthly climatology).

### 2.2 Annual temporal correlation (`temporal_corr_annual`)

Same as above but computed on annual-mean fields:

```
temporal_corr_annual = (1/N) Σ_{i=1}^{N} r( x̄ᵧ_{:,i}, ŷ̄ᵧ_{:,i} )
```

where `x̄ᵧ` denotes annual mean. More informative than daily correlation for
stochastic methods: tests preservation of interannual variability. Higher is better.

### 2.3 Annual anomaly bias (`annual_anomaly_bias`)

MAE between annual-mean anomalies of GCM and prediction:

```
annual_anomaly_bias = (1/Y) (1/N) Σ_{y,i} | (ŷ̄ᵧ_{y,i} − ŷ̄_{i}) − (x̄ᵧ_{y,i} − x̄_{i}) |
```

Measures whether the interannual variability amplitude is preserved. Lower is better.

### 2.4 Lag-1 autocorrelation MAE (`lag1_autocorr_mae`)

MAE of the lag-1 temporal autocorrelation map:

```
lag1_autocorr_mae = (1/N) Σ_{i=1}^{N} | ρ₁(ŷ_{:,i}) − ρ₁(x_{:,i}) |
```

where `ρ₁(z) = r(z_{1:T-1}, z_{2:T})` is the Pearson correlation between
successive daily values. Tests day-to-day persistence structure. Lower is better.

### 2.5 Delta change score (`delta_change_score`)

Preservation of the GCM's climate change signal (lower = better):

```
Δ_method(i) = (ŷ̄_fut(i) − ŷ̄_hist(i)) / |ŷ̄_hist(i)| × 100
Δ_gcm(i)    = (x̄_fut(i)  − x̄_hist(i))  / |x̄_hist(i)|  × 100

delta_change_score = mean_{i: |ŷ̄_hist(i)| > ε} |Δ_method(i) − Δ_gcm(i)|
```

Grid points where the historical mean is negligible (`|ŷ̄_hist| ≤ 1e-8`) are
excluded. Historical period: 2000–2039; future period: 2060–2100 (defined in
`evaluate.py`). The score measures how well the method's relative change matches
the GCM's relative change; a method that merely bias-corrects without preserving
the GCM trend would score badly here.

---

## 3. Inter-variable consistency

### 3.1 Inter-variable correlation MAE (`intervar_corr_mae`)

MAE of the pointwise temporal Pearson correlation between two variables:

```
C_y(i) = r( y^{(a)}_{:,i}, y^{(b)}_{:,i} )      (ERA5 pair)
C_ŷ(i) = r( ŷ^{(a)}_{:,i}, ŷ^{(b)}_{:,i} )     (prediction pair)

intervar_corr_mae = (1/N) Σ_{i=1}^{N} |C_ŷ(i) − C_y(i)|
```

Computed for variable pairs `(tas, pr)`, `(tas, sfcWind)`, `(pr, sfcWind)` etc.
Tests multivariate realism: whether co-occurrence patterns (e.g. warm-dry or
cold-wet days) are reproduced. Lower is better.

---

## Notes

- All spatial aggregations are arithmetic means over the France domain grid points.
- All temporal aggregations cover the full evaluation period without overlap removal.
- For `pr`, extreme and tail metrics use raw daily precipitation sums (mm day⁻¹)
  after unit conversion; the 5th-percentile tail test (`ks_q05`) may return NaN
  when the ERA5 5th percentile is ≤ 0 (CDS numerical rounding).
