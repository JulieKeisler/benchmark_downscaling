# Benchmark Tutorials

Quick reference for common extension tasks. All paths are relative to the
project root unless noted.

---

## 0 — Hackathon quick-start: evaluate any predictions

If you have daily downscaled fields (from any tool, language, or workflow),
this section explains how to package them so the benchmark's evaluation
pipeline can score your method and display it in the dashboard without
writing any framework code.

### What you need to produce

A directory containing **five files**:

| File | Shape / dtype | Description |
|---|---|---|
| `predictions.pt` | `[T, C, H, W]` float32 | Downscaled fields — PyTorch tensor, **physical units** (see table below) |
| `variables.npy` | `[C]` str | Variable names in channel order (CMIP6 short names) |
| `eval_dates.npy` | `[T]` str | ISO date strings: `"2000-01-01"`, `"2000-01-02"`, … |
| `lat.npy` | `[H]` float64 | ERA5 latitude values (north → south) |
| `lon.npy` | `[W]` float64 | ERA5 longitude values (west → east) |

**Grid** — France domain, 0.25° ERA5 grid:
- `H = 57`, lat from **55.0 to 41.0** (north first)
- `W = 65`, lon from **-6.0 to 10.0**

**Time** — Cover at minimum **2000-01-01 to 2020-12-31** (ERA5 evaluation
period). Extending to **2100-12-31** enables future-climate delta scores.
The calendar is the same as CMIP6 — no leap-year stripping needed.

**Physical units** — match ERA5 internal units after `data_processing.py`
conversions. Not normalised.

| Variable | Unit | Notes |
|---|---|---|
| `tas` | K | Daily mean 2-m temperature |
| `pr` | mm day⁻¹ | Daily total precipitation — **not** kg m⁻² s⁻¹ |
| `sfcWind` | m s⁻¹ | Daily mean near-surface wind speed |
| `sfcWindmax` | m s⁻¹ | Daily maximum near-surface wind speed |
| `psl` | Pa | Daily mean sea-level pressure |

You can predict a **subset of variables** — just list only those in
`variables.npy` and set `C` accordingly. The scorer uses whatever is present.

### Create the files from Python

```python
import numpy as np
import torch

# Assume your predictions are in a dict: var → np.ndarray [T, H, W]
preds = {
    "tas":       my_tas,    # [T, 57, 65] float32, kelvin
    "pr":        my_pr,     # [T, 57, 65] float32, mm/day
    "sfcWind":   my_wind,   # [T, 57, 65] float32, m/s
    "sfcWindmax": my_wmax,  # [T, 57, 65] float32, m/s
    "psl":       my_psl,    # [T, 57, 65] float32, Pa
}

variables = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
out_dir = "results/my_method"

import os; os.makedirs(out_dir, exist_ok=True)

# Stack channels: [T, C, H, W]
tensor = torch.stack([torch.tensor(preds[v], dtype=torch.float32)
                      for v in variables], dim=1)

torch.save(tensor, f"{out_dir}/predictions.pt")
np.save(f"{out_dir}/variables.npy",   np.array(variables))
np.save(f"{out_dir}/eval_dates.npy",  np.array(my_dates_iso))   # ["2000-01-01", ...]
np.save(f"{out_dir}/lat.npy",         np.linspace(55.0, 41.0, 57))
np.save(f"{out_dir}/lon.npy",         np.linspace(-6.0, 10.0, 65))
```

If your predictions are already on a different grid, regrid to the ERA5
0.25° France grid first (e.g. with `xarray.interp` or `scipy.interpolate`).

### Run the evaluation

```bash
python src/evaluation/evaluate.py \
    --era5-dir /path/to/era5 \
    --gcm-dir  /path/to/cmip6/ipsl_cm6a_lr \
    --method   my_method \
    --pred-dir results/my_method \
    --output-dir results/ \
    --start    2000 \
    --end      2020
```

Pass `--gcm-end 2100` if your predictions extend to 2100, to unlock the
delta-change scores (warming signal preservation metrics).

Output lands in `results/my_method/`:
- `metrics.json` — all scalar scores
- `plots/` — bias maps, temporal correlation, CDF, seasonal cycle, radar

### See it in the dashboard

The dashboard auto-discovers any subdirectory of `results/` that contains
`metrics.json`. No registration needed.

```bash
streamlit run src/dashboard/dashboard.py
```

Your method appears in all tabs alongside the official baselines.
The display name is the `--method` string you passed (e.g. `my_method`).

> **On Cleps:** copy your prediction bundle to
> `/scratch/jkeisler/downscaling_benchmark/results/my_method/`, then submit:
> ```bash
> METHOD=my_method PRED_DIR=/scratch/jkeisler/downscaling_benchmark/results/my_method \
>     sbatch cleps/jobs/job_evaluate.sh
> ```
> The job skips inference if `predictions.pt` already exists and goes straight
> to evaluation.

---

## 1 — Add a new baseline and run it

A baseline needs: an inference script, a registration in the method registry,
and optionally a training script for neural methods.
The evaluation pipeline handles metrics and plots automatically.

### Step 1 — Training (neural methods only)

For stateless methods (bicubic, BCSD, CDF-t), skip this step.
For learned methods, create `src/baselines/train_mymethod.py`:

```python
# Minimal skeleton
from pathlib import Path
import torch
# ...
parser.add_argument("--era5-dir", required=True)
parser.add_argument("--out-dir",  required=True)
# train ... save checkpoint
torch.save(model.state_dict(), Path(args.out_dir) / "mymethod_best.pth")
```

### Step 2 — Inference

Create `src/baselines/inference_mymethod.py`. Use `inference_bicubic.py` as a
template for stateless methods, or `inference_idownscale.py` for
checkpoint-based (neural) ones.
The script should:
- Load ERA5 and GCM via `data_processing.load_era5` / `load_gcm`
- Run the downscaling logic
- Save `predictions.pt` with shape `[T, C, H, W]` in physical units (same as ERA5)
- Save `eval_dates.npy`, `lat.npy`, `lon.npy` alongside

Existing inference scripts for reference:
- `src/baselines/inference_bicubic.py` — stateless, cubic spline
- `src/baselines/inference_bcsd.py` — quantile mapping + spatial disaggregation
- `src/baselines/inference_cdft.py` — CDF-t (per-cell quantile mapping at fine grid)
- `src/baselines/inference_cm.py` — consistency model, loads checkpoint

### Step 3 — Register in the method registry

In `src/baselines/methods.py`, add your function (see the docstring there).
Then register it in `src/baselines/__init__.py`:

```python
from .methods import ..., mymethod_downscale

METHOD_REGISTRY: dict = {
    ...,
    "mymethod": mymethod_downscale,
}

METHOD_LABELS: dict[str, str] = {
    ...,
    "mymethod": "MyMethod",  # display label for plots and dashboard
}
```

That's all — `evaluate.py` dispatches through `METHOD_REGISTRY` automatically,
and the dashboard reads `METHOD_LABELS` from `__init__.py` (not from `dashboard.py`).

For checkpoint-based methods that need GPU pre-loading (like CM or SF), you
instead produce `predictions.pt` from the inference script and pass
`--pred-dir` to `evaluate.py` — no code changes needed in `evaluate.py`.

### Step 4 — Run

```bash
# Inference
python src/baselines/inference_mymethod.py \
    --era5-dir /path/to/era5 \
    --gcm-dir  /path/to/cmip6/ipsl_cm6a_lr \
    --out-dir  results/mymethod

# Evaluate
python src/evaluation/evaluate.py \
    --config configs/benchmark_local.yaml \
    --method mymethod \
    --pred-dir results/mymethod \
    --output-dir results/mymethod

# Dashboard
streamlit run src/dashboard/dashboard.py
```

> **On a SLURM cluster:** copy `cleps/jobs/job_evaluate.sh` as a template,
> adapt the paths, and submit with `sbatch`. The `cleps/` directory contains
> ready-made scripts for the Cleps cluster.

---

## 2 — Change the GCM

The benchmark is designed so the GCM is a parameter, not hardcoded.

### Step 1 — Download data

```bash
# CDS (CNRM-CM6-1, CESM2, IPSL-CM6A-LR):
python src/data_download/cds/download_cmip6.py \
    --config configs/data/cmip6.yaml --models cnrm_cm6_1

# ESGF (for variables not on CDS, e.g. psl for IPSL):
python src/data_download/esgf/download_cmip6_esgf.py \
    --model CNRM-CM6-1 --variables psl
```

### Step 2 — Run evaluate.py pointing at the new GCM

Pass `--gcm-dir` to the inference and evaluate scripts:
```bash
python src/baselines/inference_bicubic.py \
    --era5-dir /path/to/era5 \
    --gcm-dir  /path/to/cmip6/cnrm_cm6_1 \
    --out-dir  results/cnrm/bicubic

python src/evaluation/evaluate.py \
    --config configs/benchmark_local.yaml \
    --era5-dir /path/to/era5 \
    --gcm-dir  /path/to/cmip6/cnrm_cm6_1 \
    --method   bicubic \
    --pred-dir results/cnrm/bicubic \
    --output-dir results/cnrm/bicubic
```

---

## 3 — Change the variables

### Step 1 — Add the variable to the config

Edit `configs/benchmark.yaml`:
```yaml
variables:
  - tas
  - pr
  - sfcWind
  - sfcWindmax
  - psl
  - rsds   # ← add here
```

### Step 2 — Download ERA5 data for the new variable

```bash
python src/data_download/cds/download_era5.py \
    --variables rsds --config configs/data/era5.yaml
```

### Step 3 — Download CMIP6 data

```bash
python src/data_download/cds/download_cmip6.py \
    --variables rsds --config configs/data/cmip6.yaml
```

### Step 4 — Re-run evaluation

Add `rsds` to `--variables` in the job scripts and resubmit.

> Unit trap: ERA5 `rsds` from the daily-stats pipeline gives mean of hourly
> accumulations (J m⁻² per hour). Check unit conventions in `data_processing.py:load_era5`.

---

## 4 — Add a metric

### Step 1 — Write the function

Add to the appropriate file in `src/metrics/`:

```python
# src/metrics/scalar_metrics_era5.py  (vs obs metrics)
# src/metrics/scalar_metrics_gcm.py   (vs GCM metrics)

def my_metric(obs: xr.DataArray, pred: xr.DataArray) -> float:
    """One-line description."""
    ...
    return float(result)
```

### Step 2 — Export from the package

In `src/metrics/__init__.py`:
```python
from .scalar_metrics_era5 import (
    ...,
    my_metric,   # ← add
)
```

### Step 3 — Call it in evaluate.py

In `src/evaluation/evaluate.py`, find `compute_all_metrics()` and add:

```python
# Inside the vs_obs block:
metrics_vs_obs["my_metric"] = my_metric(obs, pred)
```

The value will automatically appear in `metrics.json` and the dashboard's
radar/bar charts if it's in the `vs_obs` dict.

---

## 5 — Add a plot

### Step 1 — Write the function

Add to the appropriate file in `src/plots/`:

```python
# src/plots/spatial_plots.py  — map plots
# src/plots/line_plots.py     — line/distribution plots
# src/plots/radar_plots.py    — radar charts

def plot_my_diagnostic(obs: xr.DataArray, pred: xr.DataArray,
                       variable: str, method: str, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # ...
    fig.savefig(plot_dir / f"my_diagnostic_{variable}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved my_diagnostic_{variable}.png")
```

### Step 2 — Export from the package

In `src/plots/__init__.py`:
```python
from .line_plots import (
    ...,
    plot_my_diagnostic,   # ← add
)
```

### Step 3 — Call it in evaluate.py

In `src/evaluation/evaluate.py`, find the per-variable plot block
(search for `plot_bias_map`) and add:

```python
plot_my_diagnostic(obs, pred, variable, method, plot_dir)
```

The PNG will be picked up by `sync_results.sh` and shown in the dashboard
under that variable's plots section.
