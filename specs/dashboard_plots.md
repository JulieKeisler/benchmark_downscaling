# Spec — Dashboard plots and units

Applies to `src/dashboard/dashboard.py` and `src/dashboard/plot_meta.py`.

## Requirements

1. **Units on every plot.** Every Line-plots axis and every Spatial-maps
   colorbar names the physical unit of the plotted variable, using
   `plot_meta.vlabel(var)` (e.g. `tas (K)`). Change maps show `Δ <var> (<unit>)`.
   Unitless quantities (temporal-correlation r, lag-1 autocorrelation) are
   labelled `(unitless)`.

2. **A caption per plot.** Each selectable plot type has a plain-language
   caption of a few sentences explaining what the figure shows and in which
   unit. Captions live in `plot_meta.line_help(var, line_type)` and
   `plot_meta.map_help(var, plot_type)`.

3. **Single source of truth.** Units and captions exist only in
   `plot_meta.py`; the dashboard imports them. The dashboard MUST NOT redefine
   `VAR_UNITS` or embed unit strings in axis labels.

4. **Tab coverage.**
   - *Line plots*: `seasonal`, `doy`, `annual`, `cdf`, `spectra`.
   - *Spatial maps*: `bias_map`, `temporal_corr`, `autocorr`, `samples`,
     `delta_2040_2060`, `delta_2080_2100`.
   - *Inter-variable*: correlation MAE table/bar, pointwise correlation maps,
     energy-event bars. Correlation quantities are unitless; event counts/days
     use `Events`/`Days`.

5. **Unknown plot type fails loudly.** `line_help`/`map_help`/`colorbar_label`
   raise `KeyError` for an unknown type (a typo is a programming error, not a
   data condition).

## Acceptance

`python3 -m pytest tests/test_plot_meta.py -q` passes. The dashboard additionally
loads end-to-end without exception — currently checked with a manual AppTest
driver (see `fix_plan.md`; a committed AppTest test is still TODO).

## Implementation notes / gotchas

- The dashboard is executed via `runpy.run_path` by both deploy entry points
  (`deploy/streamlit_cloud/app.py`, `deploy/hf_space/app.py`). `run_path` does
  **not** put the script's directory on `sys.path`, so `dashboard.py` inserts
  `Path(__file__).resolve().parent` before `from plot_meta import ...`.
- Do not import numpy/streamlit/plotly from `plot_meta.py`; it must stay pure so
  the tests run fast and offline.
