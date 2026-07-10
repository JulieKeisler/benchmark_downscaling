# Baselines

All methods produce daily predictions on the ERA5 grid (57×65, ~0.25°) for
variables `tas`, `pr`, `sfcWind`, `sfcWindmax`, `psl` over 2000–2100.
Calibration period: ERA5 1980–1999. GCM (for now): IPSL-CM6A-LR (historical + ssp585).

---

## 1. Bicubic (`bicubic`)

**Type:** deterministic regridding, no calibration.

Cubic spline interpolation of the GCM field to the ERA5 grid, independently
at each timestep:

$$\hat{y}_t = \text{bicubic\_interp}( x_t,\; \text{lat}_{\text{ERA5}},\; \text{lon}_{\text{ERA5}} )$$

Implemented using the Keys cubic convolution kernel ($a = -0.5$), applied
separably in latitude and longitude — equivalent to CDO `remapbic`. No bias
correction. Precipitation is clipped to zero after regridding. Serves as the
lower-bound baseline: any method that cannot beat bicubic on spatial metrics
adds no value.

---

## 2. BCSD (`bcsd`)

**Type:** empirical quantile mapping + spatial disaggregation (Wood et al. 2002).

**Pipeline:**

**Step 1 — Coarsen ERA5 to GCM grid.**
Bilinear interpolation of ERA5 calibration data onto GCM lat/lon.

**Step 2 — Monthly quantile mapping at GCM resolution.**
For each calendar month $m \in \{1,\dots,12\}$ and each GCM grid cell $(i,j)$,
fit an empirical quantile map from GCM calibration monthly means to
ERA5-coarse calibration monthly means using 200 quantile levels:

$$\text{QM}_m : F_{x,m}^{-1} \;\longrightarrow\; F_{y_{\text{coarse}},m}^{-1}$$

Apply the map to GCM evaluation monthly means to obtain bias-corrected monthly
means $\text{bc\_mon}$.

**Step 3 — Daily temporal disaggregation.**
Scale daily GCM values by the monthly correction factor. For precipitation
(multiplicative) and other variables (additive):

$$\text{pr:}\quad \hat{y}_{\text{daily}} = x_{\text{daily}} \times \text{clip}\!\left(\frac{\text{bc\_mon}}{x_{\text{mon}}},\; 0,\; 10\right)$$

$$\text{other:}\quad \hat{y}_{\text{daily}} = x_{\text{daily}} + \left(\text{bc\_mon} - x_{\text{mon}}\right)$$

**Step 4 — Spatial interpolation and fine-scale detail.**
Bicubic interpolation to the ERA5 grid. Add fine-scale ERA5 spatial
detail from the calibration climatology:

$$\text{pr:}\quad \hat{y}_{\text{fine}} \mathrel{*}= \text{clip}\!\left(\frac{\text{clim}\_{\text{ERA5,fine}}}{\text{clim}\_{\text{ERA5,coarse→fine}}},\; 0.1,\; 10\right)$$

$$\text{other:}\quad \hat{y}_{\text{fine}} \mathrel{+}= \left(\text{clim}\_{\text{ERA5,fine}} - \text{clim}\_{\text{ERA5,coarse→fine}}\right)$$

where $\text{clim}\_{\text{ERA5,fine}}$ and $\text{clim}\_{\text{ERA5,coarse→fine}}$ are the ERA5 monthly
climatologies at fine and GCM resolution (the latter regridded to fine
resolution), respectively.

BCSD preserves the GCM's large-scale climate change signal via the additive/
multiplicative monthly delta, while correcting the distribution and adding
observed spatial detail from the calibration period.

---

## 3. CDF-t (`cdft`)

**Type:** empirical CDF transform, per grid cell, per calendar month
(Michelangeli et al. 2009).

For each grid cell and calendar month, CDF-t applies the mapping:

$$F_{R_f}(x) = F_{R_p}\!\left( F_{G_p}^{-1}\!\left( F_{G_f}(x) \right) \right)$$

where:
- $F_{G_p}$ = empirical CDF of GCM over the **calibration** period (historical)
- $F_{G_f}$ = empirical CDF of GCM over the **evaluation** period (may be future)
- $F_{R_p}$ = empirical CDF of ERA5 over the **calibration** period
- $F_{R_f}$ = target corrected CDF for the evaluation period

The key insight vs plain quantile mapping: by composing $F_{G_p}^{-1}(F_{G_f}(\cdot))$
before applying $F_{R_p}$, CDF-t explicitly accounts for a shift in the GCM
distribution between calibration and evaluation periods. This makes it more
appropriate for future projections.

**Support expansion.** The target range for evaluation is extended beyond the
calibration range to handle future out-of-sample values. For additive variables
($\text{scale}=1$, e.g. `tas`, `psl`):

$$\bigl[m - \text{dev} \cdot a,\;\; M + \text{dev} \cdot a\bigr], \qquad a = \bigl|\text{mean}(G_f) - \text{mean}(G_p)\bigr|$$

For multiplicative variables ($\text{scale}=0$, e.g. `pr`, `sfcWind`):

$$\bigl[m \cdot a / \text{dev},\;\; M \cdot \text{dev} \cdot a\bigr], \qquad a = \bigl|\text{mean}(G_f) / \text{mean}(G_p)\bigr|$$

with $\text{dev}=2$ and tail-correction heuristics to ensure $F_{R_f}$ reaches 0 and 1.

**Dry-day handling for precipitation.** Wet-day frequency adaptation
(hard threshold): GCM days below the GCM quantile matching the ERA5 wet-day
frequency are set to zero; CDF-t is fit only on wet days.

**Pre-processing.** GCM is bicubic-regridded to the ERA5 grid before CDF-t.
CDF-t is applied in ERA5 grid space (each of the 57×65 cells independently).

**Hyperparameters** were optimised with Optuna on the France domain to minimise
per-month KS statistic:

| Variable   | `scale` | `pui` | `npas` | `etype` | Notes              |
|------------|---------|-------|--------|---------|---------------------|
| `tas`      | 1       | 0.5   | 100    | 4       | additive            |
| `pr`       | 0       | 0.5   | 100    | 4       | dry-day correction  |
| `sfcWind`  | 0       | 0.5   | 100    | 4       | clip to 0           |
| `sfcWindmax`| 0      | 0.5   | 100    | 4       | clip to 0           |
| `rsds`     | 0       | 1.0   | 100    | 4       | clip to 0           |
| `psl`      | 1       | 1.0   | 100    | 4       | additive            |

---

## 4. iDownscale (`idownscale`)

**Type:** supervised U-Net regression with perfect-prognosis training
(Soares et al. 2024 / CERFACS approach).

**Training (ERA5 only, 1980–1999).**
Pairs are constructed from ERA5 alone by subsampling (stride 4) and regridding:

$$x_t = \text{bicubic\_interp}\bigl( \text{ERA5}_t[::4,\; ::4],\; \text{lat}_{\text{ERA5}},\; \text{lon}_{\text{ERA5}} \bigr) \qquad \text{(coarsened)}$$

$$y_t = \text{ERA5}_t \qquad \text{(fine target)}$$

A plain U-Net (no attention, no time conditioning) is trained with MSE loss:

$$\mathcal{L} = \mathbb{E}\!\left[\left\| f_\theta\bigl( \text{encode}(x_t),\; \text{month}_t \bigr) - \text{encode}(y_t) \right\|^2\right]$$

All variables are normalised to $[-1,1]$ jointly before training (`encode`/`decode`;
see §Shared normalisation below). The "perfect-prognosis" assumption is that the
model trained on coarsened ERA5 will generalise to real GCM inputs.

**Inference.**
GCM fields are first bias-corrected at coarse (4×-subsampled ERA5) resolution
using CDF-t, then bicubic-upsampled to the ERA5 grid, then fed to the U-Net:

$$\tilde{x}_t = \text{CDF-t}\bigl( \text{GCM}_t \to \text{ERA5},\; \text{at coarse grid} \bigr)$$

$$\hat{y}_t = \text{decode}\!\left( f_\theta\bigl( \text{encode}( \text{bicubic}(\tilde{x}_t) ),\; \text{month}_t \bigr) \right)$$

The CDF-t step at coarse resolution removes the GCM mean bias before the
neural network sees the input, which prevents large distribution shifts from
degrading neural network predictions (the network was trained on ERA5 statistics).

**Architecture.** Plain `UNet` (no attention), `init_features=32`. Input:
$[N_{\text{VARS}} + 2,\; H,\; W]$ (5 variables + sin/cos month). Output: $[N_{\text{VARS}},\; H,\; W]$.

---

## 5. SerpentFlow (`serpentflow`)

**Type:** flow-matching generative model with spectral scale separation
(Keisler et al. 2026).

### Scale separation

Each ERA5 field is decomposed into a shared large-scale component (low-pass)
and a domain-specific fine-scale component (residual). The low-pass filter is
a 2D Fourier circular mask with cutoff radius $r_{\text{cut}}=1$ pixel in frequency space:

$$x_{\text{lp}} = \text{IFFT}\!\left( \text{FFT}(x) \cdot \mathbb{1}\!\left[\sqrt{f_x^2 + f_y^2} \leq r_{\text{cut}}\right] \right)$$

With $r_{\text{cut}}=1$, only the DC component and first-order harmonics are retained —
essentially the spatial mean and very large-scale gradient. The model must learn
to generate all spatial structure from this minimal conditioning signal.

### Training

Pseudo-pairs $(x_0, x_1)$ are constructed from ERA5 1980–1999:
- $x_0 = x_{\text{lp}} + \varepsilon$, $\;\varepsilon \sim \mathcal{N}(0,\; \sigma^2_{\text{HF}} \cdot I)$ with $\sigma_{\text{HF}}=0.1$ (stochastic source)
- $x_1 = x$ (full ERA5 target, in normalised space)

The flow model $v_\theta$ learns the conditional velocity field via linear flow
matching with an **EDM-skewed time schedule** (Karras et al. 2022):

$$\sigma = \exp\!\left( \mathcal{N}(\mu_P,\; \sigma_P^2) \right), \quad \mu_P = -1.2, \quad \sigma_P = 1.2$$

$$t = \frac{1}{1 + \sigma}, \quad t \in (0, 1)$$

This concentrates training samples near $t \approx 0.77$ (fine-scale generation
regime where the interpolated field $x_t = (1-t)\,x_0 + t\,x_1$ is close to the
target but still has structure to resolve).

The MSE loss on the velocity field:

$$x_t = (1-t)\,x_0 + t\,x_1$$

$$\mathcal{L} = \mathbb{E}\!\left[\left\| v_\theta( x_t,\; \text{month}_t,\; t ) - (x_1 - x_0) \right\|^2\right]$$

### Inference

Starting from the bicubic-regridded GCM after spectral low-pass ($r_{\text{cut}}=1$)
plus HF noise:

$$x_0 = \text{encode}\!\left( \text{spectral\_lowpass}( \text{GCM},\; r_{\text{cut}}=1 ) \right) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,\; \sigma^2_{\text{HF}} \cdot I)$$

$$\hat{y} = \text{ODESolve}\!\left( \frac{dx}{dt} = v_\theta(x,\; \text{month},\; t),\;\; x(0)=x_0,\;\; t: 0 \to 1 \right)$$

ODE integration uses `dopri5` (Dormand–Prince adaptive RK45) via `torchdiffeq`
when available, falling back to 50-step fixed Euler otherwise.

**Architecture.** `UNetAttention` (self-attention at two lowest spatial resolutions),
`init_features=32`. Input: $[N_{\text{VARS}} + 3,\; H,\; W]$ (5 variables + sin/cos month + $t$).
Output: $[N_{\text{VARS}},\; H,\; W]$.

---

## 6. Consistency Model (`cm`)

**Type:** consistency distillation with stroke-guidance inference
(Hess et al. 2025 / adapted).

### Training

Consistency distillation operates on ERA5 fields (not paired with GCM).
A student network $f_\theta$ and EMA teacher $f_{\bar{\theta}}$ are trained to satisfy the
**self-consistency property**: for any point on a PF-ODE trajectory,
$f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ for all $t, t'$ on the same trajectory.

Time is discretised into $N$ bins (curriculum: $N$ increases from $N_{\min}=10$
to $N_{\max}=150$ over training). The loss is:

$$\mathcal{L} = \mathbb{E}\!\left[\left\| f_\theta(x_{t_n},\; t_n) - f_{\bar{\theta}}(x_{t_{n+1}},\; t_{n+1}) \right\|^2\right]$$

where $t_n < t_{n+1}$ are adjacent bin boundaries sampled by the EDM schedule
($T_{\min}=0.002$, $T_{\max}=2.0$), $x_{t_n} = y + \varepsilon \cdot t_n$ with $\varepsilon \sim \mathcal{N}(0, I)$ and
$y$ = ERA5. The teacher $f_{\bar{\theta}}$ is an EMA copy of the student.

The skip-connection parameterisation ensures boundary condition $f(x, T_{\min}) = x$:

$$f_\theta(x, t) = c_{\text{skip}}(t) \cdot x + c_{\text{out}}(t) \cdot U_\theta(x, t)$$

$$c_{\text{skip}}(t) = \frac{\sigma^2_{\text{data}}}{(t - T_{\min})^2 + \sigma^2_{\text{data}}}$$

$$c_{\text{out}}(t) = \frac{\sigma_{\text{data}} \cdot t}{\sqrt{t^2 + \sigma^2_{\text{data}}}}, \qquad \sigma_{\text{data}} = 0.5$$

The EMA decay is adapted to the current curriculum width:

$$\mu = \exp\!\left( \frac{N_{\min} \cdot \ln(\mu_0)}{N} \right), \qquad \mu_0 = 0.9$$

### Inference — stroke guidance

At inference time, the GCM is used as a conditioning anchor. The bicubic-regridded
GCM is encoded and perturbed at noise level $t^* = 0.5$:

$$x^* = \text{encode}\!\left( \text{bicubic}( \text{GCM}_t ) \right) + \varepsilon \cdot t^*, \qquad \varepsilon \sim \mathcal{N}(0, I)$$

$$\hat{y} = f_\theta(x^*,\; t^*)$$

A single denoising step projects the perturbed GCM onto the learned ERA5 manifold.
The noise level $t^* = 0.5$ (relative to $T_{\max}=2.0$) controls the strength of
GCM conditioning: larger $t^*$ → more GCM structure is preserved but more ERA5
distribution information injected.

**Architecture.** `UNetAttention`, `init_features=32`. Input: $[N_{\text{VARS}} + 3,\; H,\; W]$
(5 variables + sin/cos month + $t$). Output: $[N_{\text{VARS}},\; H,\; W]$.

### Implementation differences from Hess et al. 2025

Our CM implementation follows the training and inference principles of Hess et al.
but differs in several respects beyond the multivariate output:

| Aspect | Hess et al. 2025 | This work |
|--------|------------------|-----------|
| **Architecture** | Diffusers `UNet2DModel`; sinusoidal time embedding + FiLM conditioning; 128/128/256/256 channels; ~27M params | Custom `UNetAttention`; $t$ broadcast as a scalar spatial channel; 32/64/128/256/512 channels; ~3M params |
| **Time conditioning** | FiLM (Feature-wise Linear Modulation) per residual block | Scalar $t$ concatenated as an additional input channel at full resolution |
| **$T_{\max}$** | Image-scale (large-noise regime suited for pixel intensities) | 2.0 — calibrated so peak noise = 4× the data std (~0.5 in normalized space) |
| **Score pre-training** | Score network trained first (200 epochs), CM distilled from it (150 epochs) | CM trained directly without a pre-trained score network |
| **Batch size** | 1 (single field per step) | 128 |
| **Variables** | Univariate (single variable at a time) | Multivariate (5 variables jointly) |
| **Attention** | Single attention layer at the U-Net bottleneck | Multi-head self-attention at the two lowest spatial resolutions |

The smaller architecture and direct CM training are chosen to keep training
feasible on a single GPU for a small (57×65) domain. The $T_{\max}=2.0$ choice
matches the consistency model literature (Song et al. 2023) for normalized data.

---

## Shared normalisation

All three neural baselines (iDownscale, SerpentFlow, CM) use the same
normalisation codec, fitted on ERA5 1980–1999:

**Standard variables** (linear min-max to $[-1, 1]$):

$$\text{encode}(x) = 2 \cdot \frac{x - x_{\min}}{x_{\max} - x_{\min}} - 1$$

$$\text{decode}(z) = \frac{z + 1}{2} \cdot (x_{\max} - x_{\min}) + x_{\min}$$

**Log variables** (`pr` only — log-transform before min-max to compress heavy tail):

$$\text{encode}(x) = \text{linear\_minmax}\!\left( \ln(\text{clip}(x, 0) + \epsilon) - \ln(\epsilon) \right)$$

$$\text{decode}(z) = \exp\!\left( \text{linear\_minmax}^{-1}(z) + \ln(\epsilon) \right) - \epsilon, \qquad \epsilon = 10^{-6}$$

Statistics ($x_{\min}$, $x_{\max}$) are computed in the transformed space over the
full ERA5 calibration set. Checkpoints store the stats dict; a different calibration
period produces a different codec and incompatible checkpoints.

---

## Shared architecture: UNet / UNetAttention

Both architectures are 4-level encoder–decoder U-Nets with skip connections.
`init_features=32` doubles at each level: 32→64→128→256 (UNet) or with an
additional 512-channel bottleneck (UNetAttention).

`UNetAttention` adds multi-head self-attention ($n_{\text{heads}}=4$) at the two lowest
spatial resolutions (channels 256 and 512), implemented via `nn.MultiheadAttention`
with `GroupNorm` pre-norm. Attention is applied over spatial tokens ($H \times W$ flattened),
enabling global spatial reasoning at low cost given the small domain (57×65 → 4×5
at the bottleneck).

Month is encoded as two channels $[\sin(2\pi m/12),\; \cos(2\pi m/12)]$ broadcast to the
full spatial resolution and concatenated with the input.

For SerpentFlow and CM, the flow time $t$ is broadcast as a scalar constant spatial
channel (simpler than sinusoidal embedding; unified across modes).

---

## Bibliography

- **Wood et al. (2002).** *Hydrologic implications of dynamical and statistical approaches to downscaling climate model outputs.* Climatic Change, 62(1–3), 189–216.
  → Foundation of BCSD: monthly quantile mapping at GCM resolution + spatial disaggregation.

- **Michelangeli et al. (2009).** *Probabilistic downscaling approaches: Application to wind cumulative distribution functions.* Geophysical Research Letters, 36(11).
  → CDF-t method: accounting for distribution shift between calibration and projection periods.

- **Soares et al. (2024).** *iDownscale: A machine learning-based hybrid downscaling approach for climate projections.* CERFACS technical report / Geoscientific Model Development (submitted).
  → iDownscale: perfect-prognosis U-Net regression with coarse CDF-t pre-correction.

- **Song et al. (2023).** *Consistency Models.* arXiv:2303.01469.
  → Consistency distillation framework: self-consistency property, curriculum $N$, EMA teacher, skip parameterisation.

- **Karras et al. (2022).** *Elucidating the Design Space of Diffusion-Based Generative Models (EDM).* NeurIPS 2022.
  → EDM time schedule ($P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1.2$), $c_{\text{skip}}$ / $c_{\text{out}}$ parameterisation, $\sigma_{\text{data}} = 0.5$.

- **Hess et al. (2025).** *Physically consistent downscaling of climate model output using consistency models.* Nature Machine Intelligence.
  → CM applied to univariate climate downscaling; stroke guidance for GCM conditioning; training and evaluation protocol.

- **Keisler et al. (2026).** *SerpentFlow: Spectral-domain flow matching for multivariate statistical downscaling.* (in preparation).
  → SerpentFlow: spectral LP scale separation, EDM-skewed flow matching, multivariate joint generation.