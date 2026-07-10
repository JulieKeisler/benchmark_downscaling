# Contributing

This document explains how to add a new downscaling method to the benchmark.
The pipeline is intentionally thin: you implement one function, register it,
add an inference script, and you're done.

---

## Architecture overview

```
src/baselines/methods.py          ← all per-variable downscaling logic lives here
src/baselines/__init__.py         ← METHOD_REGISTRY + METHOD_LABELS
src/baselines/inference_*.py      ← standalone scripts that produce predictions.pt
src/evaluation/evaluate.py        ← dispatches through METHOD_REGISTRY at eval time
src/evaluation/data_processing.py ← shared data loading and I/O helpers
```

**Standard predictions format** — all inference scripts must produce:
```
out_dir/
  predictions.pt    torch.Tensor [T, C, H, W], float32, physical units
  variables.npy     np.array of variable names, length C
  eval_dates.npy    np.array of "YYYY-MM-DD" strings, length T
  lat.npy           ERA5 latitude grid, length H
  lon.npy           ERA5 longitude grid, length W
```
Call `save_predictions(tensor, variables, out_dir, ref_da)` — it writes all five files.

---

## Reusable helpers

Before implementing from scratch, check these helpers in `methods.py` and
`data_processing.py`:

| Helper | Location | Signature | What it does |
|--------|----------|-----------|-------------|
| `bicubic_regrid` | `data_processing.py` | `(gcm, target_lat, target_lon)` | Sequential 1D cubic spline, GCM grid → ERA5 grid |
| `_qmap_1d` | `methods.py` | `(src, ref, apply)` | Fit empirical CDF src→ref, apply quantile mapping to `apply` |
| `align_time` | `data_processing.py` | `({"key": da, ...})` | Intersect time axes across multiple DataArrays |
| `save_predictions` | `data_processing.py` | `(tensor, variables, out_dir, ref_da)` | Write predictions.pt + metadata |
| `load_predictions` | `data_processing.py` | `(pred_dir, variables)` | Load predictions.pt → dict of DataArrays |

### After calling `bicubic_regrid` — always re-stamp the time axis

`bicubic_regrid` preserves the GCM's original time coordinate, which may be
cftime-typed and will cause silent failures in `align_time` and `save_predictions`.
Always overwrite it with the ERA5 time coordinate immediately after upsampling:

```python
gcm_fine = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
gcm_fine = gcm_fine.assign_coords(time=obs.time)  # required — prevents cftime/datetime64 mismatch
```

---

## Input units inside method functions

Both `gcm` and `obs` arrive at your method function already in ERA5 units.
The loaders (`load_era5`, `load_gcm` in `data_processing.py`) handle unit
conversion transparently before any method is called:

| Variable | Unit inside method | Conversion done by loader |
|----------|--------------------|--------------------------|
| `tas`    | K                  | none |
| `pr`     | mm/day             | ERA5 m→mm/day × 1000; CMIP6 kg m⁻² s⁻¹ × 86400 |
| `sfcWind`, `sfcWindmax` | m/s   | none |
| `psl`    | Pa                 | none |

You do **not** need unit conversion inside your downscaling function. If you
write your own loader (e.g. for a custom data format), make sure it outputs the
same units as the table above.

---

## A note on calibration vs evaluation period

Statistical baselines (BCSD, CDF-t, PSM) use the full ERA5 record as **both
calibration and evaluation data** — there is no train/test split. This is the
standard practice for climatological bias correction: the calibration period and
evaluation period are the same historical record. It is not data leakage in the
traditional ML sense: the methods correct GCM distributions rather than predicting
future sequences from past ones. Evaluation metrics measure spatial structure and
distribution fidelity, not temporal generalisation.

---

## Why `sys.path.insert` instead of package imports

The inference scripts are designed to be run directly:
```bash
python src/baselines/inference_mymethod.py --era5-dir ...
```
They are not part of an installed package, so `from baselines.methods import ...`
would fail without first putting `src/` on the path. Every inference script starts
with this two-line header — copy it verbatim:

```python
_HERE = Path(__file__).resolve().parent              # = src/baselines/
sys.path.insert(0, str(_HERE))                       # lets `from methods import ...` work
sys.path.insert(0, str(_HERE.parent / "evaluation")) # lets `from data_processing import ...` work
```

---

## Adding a per-variable method (e.g. `mymethod`)

### 1. Implement the downscaling function in `src/baselines/methods.py`

```python
def mymethod_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                       variable: str) -> xr.DataArray:
    """One-line description.

    Args:
        gcm: GCM DataArray on its native (coarse) grid, dims [time, lat, lon]
        obs: ERA5 DataArray on the fine grid, dims [time, lat, lon]
        variable: CMIP6 short name ("tas", "pr", ...)
    Returns:
        DataArray on the ERA5 grid, same time axis as obs.
    """
    # Upsample GCM to ERA5 grid
    gcm_fine = bicubic_regrid(gcm, obs.lat.values, obs.lon.values)
    gcm_fine = gcm_fine.assign_coords(time=obs.time)  # required after bicubic_regrid

    # ... your algorithm on gcm_fine and obs ...

    if variable == "pr":
        out.values[:] = np.clip(out.values, 0.0, None)  # pr must be non-negative
    return out
```

The function must not do any I/O. Time axes are pre-aligned before the call.

### 2. Register in `src/baselines/__init__.py`

```python
from .methods import ..., mymethod_downscale

METHOD_REGISTRY: dict = {
    ...,
    "mymethod": mymethod_downscale,
}

METHOD_LABELS: dict[str, str] = {
    ...,
    "mymethod": "MyMethod",  # display label used in plots and dashboard
}
```

### 3. Add a standalone inference script

Copy `src/baselines/inference_bcsd.py` as a template. The import header must be:

```python
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))
from data_processing import DEFAULT_VARIABLES, load_era5, load_gcm, align_time, save_predictions
from methods import mymethod_downscale
```

The `run()` function should:
1. Load ERA5 and GCM for each variable with `load_era5` / `load_gcm`.
2. Align time axes with `align_time()`.
3. Call `mymethod_downscale(gcm[var], era5[var], var)` per variable.
4. Stack into a `[T, C, H, W]` tensor and call `save_predictions()`.

### 4. Document hyperparameters in `configs/methods.yaml`

Add a key for your method. Use `{}` if it's stateless:

```yaml
mymethod: {}           # stateless, no hyperparameters

# or, if tunable:
mymethod:
  n_neighbors: 10      # brief description of each parameter
```

### 5. Add a SLURM job (optional, for Cleps)

Copy `cleps/jobs/job_evaluate.sh` as a template and adapt the paths.
The `cleps/paths.sh` file defines all cluster-specific variables; adapt it for your own HPC.

### 6. Test it

```bash
# Quick local test on a small period
python src/baselines/inference_mymethod.py \
    --era5-dir /path/to/era5 \
    --gcm-dir  /path/to/gcm \
    --out-dir  /tmp/mymethod_test \
    --eval-start 2010 --eval-end 2011

python src/evaluation/evaluate.py \
    --config configs/benchmark_local.yaml \
    --method mymethod \
    --pred-dir /tmp/mymethod_test \
    --output-dir /tmp/mymethod_eval

# Run the test suite (no real data needed)
python3 -m pytest tests/ -v
```

---

## Adding a multivariate method

Multivariate methods (e.g. consistency model, SerpentFlow) process all
variables jointly through a neural network. They are handled separately in
`evaluate.py` rather than through `METHOD_REGISTRY`.

1. Add the inference logic to a new `inference_<name>.py` (see `inference_cm.py`
   for the pattern — it loads all 5 channels, pads, applies the network, decodes).
2. The script must still produce the standard `predictions.pt` format.
3. Pass `--pred-dir` to `evaluate.py` — no changes to `evaluate.py` needed.

---

## Adding a new metric

1. Implement in the appropriate file under `src/metrics/`:
   - ERA5-referenced metrics → `scalar_metrics_era5.py`
   - GCM-correlation metrics → `scalar_metrics_gcm.py`
   - Inter-variable metrics  → `inter_variables_metrics.py`
   - Spectral metrics        → `spatial_spectrum_metrics.py`
2. Export from `src/metrics/__init__.py`.
3. Call the metric and store the result in the `metrics` dict inside `evaluate.py`'s per-variable loop.

Note: `scalar_metrics_gcm.py` functions expect `gcm` to already be on the ERA5 grid.
`evaluate.py` always passes `gcm_on_era5` (bicubic-regridded GCM), not the raw GCM.

---

## Adding a new plot

1. Implement in `src/plots/` (spatial, line, or radar).
2. Export from `src/plots/__init__.py`.
3. Call inside `evaluate.py` after metrics are computed.

---

## Data format notes

- All variables use **CMIP6 short names**: `tas`, `pr`, `sfcWind`, `sfcWindmax`, `psl`.
- `pr` unit in `predictions.pt`: **mm/day** (loaders convert from raw ERA5/CMIP6 units).
- ERA5 grid: 57 × 65 cells over France (0.5° resolution, `[55.0, -6.0, 41.0, 10.0]`).
- Time dimension name must be `time`; lat/lon must be named `lat`/`lon`.

---

## Code style

- No new files for trivial helpers — extend an existing module.
- No comments that restate what the code does — only WHY comments.
- Docstrings on public functions, one-liner max on private helpers.
- No error handling for impossible scenarios — trust xarray/numpy.
- If your method requires a heavy optional dependency (e.g. sklearn), import it
  lazily inside the function rather than at module level. This keeps
  `from baselines import METHOD_REGISTRY` fast for methods that don't need it.
