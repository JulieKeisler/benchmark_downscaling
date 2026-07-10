# Metrics

All metrics are computed per variable over the evaluation period (2000–2020 vs ERA5;
2000–2100 vs GCM). Scalars reported in `metrics.json`; spatial maps and curve data
in `plot_data.json`.

Variables: `tas` (K), `pr` (mm day⁻¹), `sfcWind` (m s⁻¹), `sfcWindmax` (m s⁻¹), `psl` (Pa).

Notation: $\hat{y}$ = method prediction, $y$ = ERA5 reference, $x$ = GCM, $T$ = number
of timesteps, $N$ = number of grid points, $\bar{y}$ = temporal mean.

---

## 1. vs ERA5 — distribution quality

These metrics compare method predictions to ERA5 over the historical period
(2000–2020) where ERA5 observations are available.

### 1.1 Bias MAE (`bias_mae`)

Mean absolute temporal-mean bias across all grid points:

$$\text{bias\_mae} = \frac{1}{N} \sum_{i=1}^{N} \left| \bar{y}_i - \bar{\hat{y}}_i \right|$$

where $\bar{y}_i = \frac{1}{T} \sum_t y_{t,i}$ is the ERA5 time mean at grid point $i$.
Lower is better.

### 1.2 Standard deviation error (`std_error`)

Mean absolute error of temporal standard deviation across all grid points:

$$\text{std\_error} = \frac{1}{N} \sum_{i=1}^{N} \left| \sigma(\hat{y}_i) - \sigma(y_i) \right|$$

where $\sigma(\cdot)$ is the temporal standard deviation. Measures how well the method
reproduces daily variability amplitude. Lower is better.

### 1.3 KS statistics (`ks_full`, `ks_q95`, `ks_q05`)

Kolmogorov–Smirnov distance between flattened (all grid points × all timesteps)
empirical distributions:

$$\text{KS}(F_y, F_{\hat{y}}) = \sup_x \left| F_y(x) - F_{\hat{y}}(x) \right|$$

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

$$\text{seasonal\_mae} = \frac{1}{12} \sum_{m=1}^{12} \frac{1}{N} \sum_{i=1}^{N} \left| \hat{\mu}_{m,i} - \mu_{m,i} \right|$$

where $\mu_{m,i} = \text{mean}_{t:\,\text{month}(t)=m}\; y_{t,i}$ is the ERA5 monthly climatology.
Measures the ability to reproduce the seasonal cycle. Lower is better.

### 1.5 Extreme frequency bias (`extreme_freq`)

Bias in the fraction of days exceeding the ERA5 global 95th percentile $q_{0.95}$:

$$\text{extreme\_freq} = \frac{1}{N} \sum_{i=1}^{N} \left[ P(\hat{y}_{t,i} > q_{0.95}) - P(y_{t,i} > q_{0.95}) \right]$$

The threshold $q_{0.95}$ is a single global quantile over all grid points and
timesteps. A positive value means the method generates too many extreme events;
negative means too few.

### 1.6 Extreme intensity bias (`extreme_intensity`)

Bias in the mean intensity of events exceeding $q_{0.95}$:

$$\text{extreme\_intensity} = \frac{1}{N} \sum_{i=1}^{N} \left[ \mathbb{E}[\hat{y}_{t,i} \mid \hat{y}_{t,i} > q_{0.95}] - \mathbb{E}[y_{t,i} \mid y_{t,i} > q_{0.95}] \right]$$

Uses the same global threshold as `extreme_freq`.

### 1.7 Spectral MAE (`spectral_mae`)

Normalised RMSE of the time-averaged isotropic power spectral density (PSD):

$$\text{spectral\_mae} = \sqrt{ \text{mean}_{k \in \text{valid}} \left[ \frac{\left( \text{PSD}_{\hat{y}}(k) - \text{PSD}_y(k) \right)^2}{\text{PSD}_y(k)^2} \right] }$$

**PSD computation.** For each timestep $t$, the anomaly field (mean removed) is
Fourier-transformed with `np.fft.fft2`. The 2D power spectrum is binned into $N=40$
log-spaced radial wavenumber bins. The radially-averaged PSD is the mean power
per bin. The time average gives $\text{PSD}_y(k)$ and $\text{PSD}_{\hat{y}}(k)$.

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

$$\text{temporal\_corr} = \frac{1}{N} \sum_{i=1}^{N} r\!\left( x_{:,i},\; \hat{y}_{:,i} \right)$$

where $r(\cdot, \cdot)$ is Pearson's $r$ over the time axis. For free-running GCMs (not
time-aligned with ERA5), this tests whether the method follows the GCM's daily
evolution. Deterministic methods that are direct functions of the GCM score ~1;
stochastic methods that generate independent samples score near 0. Returns 0.0
when all grid points have zero variance in predictions (e.g. a stochastic model
that collapses to monthly climatology).

### 2.2 Annual temporal correlation (`temporal_corr_annual`)

Same as above but computed on annual-mean fields:

$$\text{temporal\_corr\_annual} = \frac{1}{N} \sum_{i=1}^{N} r\!\left( \bar{x}^{\text{yr}}_{:,i},\; \bar{\hat{y}}^{\text{yr}}_{:,i} \right)$$

where $\bar{x}^{\text{yr}}$ denotes annual mean. More informative than daily correlation for
stochastic methods: tests preservation of interannual variability. Higher is better.

### 2.3 Annual anomaly bias (`annual_anomaly_bias`)

MAE between annual-mean anomalies of GCM and prediction:

$$\text{annual\_anomaly\_bias} = \frac{1}{Y} \frac{1}{N} \sum_{y,i} \left| \left(\bar{\hat{y}}^{\text{yr}}_{y,i} - \bar{\hat{y}}_i\right) - \left(\bar{x}^{\text{yr}}_{y,i} - \bar{x}_i\right) \right|$$

Measures whether the interannual variability amplitude is preserved. Lower is better.

### 2.4 Lag-1 autocorrelation MAE (`lag1_autocorr_mae`)

MAE of the lag-1 temporal autocorrelation map:

$$\text{lag1\_autocorr\_mae} = \frac{1}{N} \sum_{i=1}^{N} \left| \rho_1(\hat{y}_{:,i}) - \rho_1(x_{:,i}) \right|$$

where $\rho_1(z) = r(z_{1:T-1},\; z_{2:T})$ is the Pearson correlation between
successive daily values. Tests day-to-day persistence structure. Lower is better.

### 2.5 Delta change score (`delta_change_score`)

Preservation of the GCM's climate change signal (lower = better):

$$\Delta_{\text{method}}(i) = \frac{\bar{\hat{y}}_{\text{fut}}(i) - \bar{\hat{y}}_{\text{hist}}(i)}{\left|\bar{\hat{y}}_{\text{hist}}(i)\right|} \times 100$$

$$\Delta_{\text{gcm}}(i) = \frac{\bar{x}_{\text{fut}}(i) - \bar{x}_{\text{hist}}(i)}{\left|\bar{x}_{\text{hist}}(i)\right|} \times 100$$

$$\text{delta\_change\_score} = \text{mean}_{i:\,|\bar{\hat{y}}_{\text{hist}}(i)| > \epsilon} \left| \Delta_{\text{method}}(i) - \Delta_{\text{gcm}}(i) \right|$$

Grid points where the historical mean is negligible ($|\bar{\hat{y}}_{\text{hist}}| \leq 10^{-8}$) are
excluded. Historical period: 2000–2039; future period: 2060–2100 (defined in
`evaluate.py`). The score measures how well the method's relative change matches
the GCM's relative change; a method that merely bias-corrects without preserving
the GCM trend would score badly here.

---

## 3. Inter-variable consistency

### 3.1 Inter-variable correlation MAE (`intervar_corr_mae`)

MAE of the pointwise temporal Pearson correlation between two variables:

$$C_y(i) = r\!\left( y^{(a)}_{:,i},\; y^{(b)}_{:,i} \right) \qquad \text{(ERA5 pair)}$$

$$C_{\hat{y}}(i) = r\!\left( \hat{y}^{(a)}_{:,i},\; \hat{y}^{(b)}_{:,i} \right) \qquad \text{(prediction pair)}$$

$$\text{intervar\_corr\_mae} = \frac{1}{N} \sum_{i=1}^{N} \left| C_{\hat{y}}(i) - C_y(i) \right|$$

Computed for variable pairs (`tas`, `pr`), (`tas`, `sfcWind`), (`pr`, `sfcWind`) etc.
Tests multivariate realism: whether co-occurrence patterns (e.g. warm-dry or
cold-wet days) are reproduced. Lower is better.

---

## Notes

- All spatial aggregations are arithmetic means over the France domain grid points.
- All temporal aggregations cover the full evaluation period without overlap removal.
- For `pr`, extreme and tail metrics use raw daily precipitation sums (mm day⁻¹)
  after unit conversion; the 5th-percentile tail test (`ks_q05`) may return NaN
  when the ERA5 5th percentile is $\leq 0$ (CDS numerical rounding).