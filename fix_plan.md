# fix_plan.md — Ralph ledger

Prioritized list of what remains. Ralph re-reads this every loop, does **one
item**, runs the relevant test (backpressure), then updates this file.

Legend: `[ ]` open · `[~]` in progress · `[x]` done (kept briefly, then pruned).

## Done (recent)

- [x] **Benchmark-wide data audit.** Every published method × region
  (`bicubic, bcsd, cdft, dotc, idownscale, corrdiff_qm, cm, serpentflowclim` ×
  `france, australia, us_central`) has complete `metrics.json` (`vs_obs`,
  `vs_gcm`, `_delta` for 2040_2060 & 2080_2100), `plots/**/*.npz`,
  `plot_data.json`, and `_intervar`. `us_central` lacks `sfcWindmax` as
  expected (CESM2). **SerpentFlow-CLIM delta values are present** (e.g. france
  tas: 0.061 / 0.131). Conclusion: the empty heatmap cells observed live were
  from a stale deployed bundle/cache — no data is missing.
- [x] **Plot captions + units.** Every Line-plots type (`seasonal, doy, annual,
  cdf, spectra`) and every Spatial-maps type (`bias_map, temporal_corr,
  autocorr, samples, delta_*`) now has a few-sentence caption naming its unit;
  every axis/colorbar shows the unit. Extracted to `src/dashboard/plot_meta.py`
  (single source of truth; the dashboard selectboxes are built from its tuples)
  + `tests/test_plot_meta.py` (18 passing, asserts the *actual* unit appears).
- [x] **Agent harness established** (legoESM / Ralph): `CONTRACT.md`, `specs/`,
  `fix_plan.md`, `PROMPT.md`, `ralph.sh`. Adversarial review (independent agent)
  performed; its findings fixed: contract now lists `rsds` as non-negative;
  plot-type tuples single-sourced; captions rendered + unit-checked end-to-end
  (AppTest), not just asserted on the word "Unit".

## Open (priority order)

1. [ ] **(high) Confirm the deployed bundle is fresh.** The re-run of
   `tools/deploy_hf.py` staged `.deploy_build/results.tar.gz` with
   serpentflowclim `_delta`, but the running Streamlit app caches data in
   `/tmp/benchmark_downscaling` behind a `.ready` marker — it will NOT
   re-download unless the container restarts. Action: reboot the Streamlit app
   (Manage app → Reboot) and re-check the "Delta change scores — heatmap" row for
   SerpentFlow-CLIM.
2. [ ] **(medium) Replace deprecated `use_container_width`.** Streamlit ≥ 1.58
   warns it is removed after 2025-12-31; migrate `use_container_width=True` →
   `width="stretch"` (and `False` → `width="content"`) across
   `src/dashboard/dashboard.py`. Backpressure: `py_compile` + AppTest.
3. [ ] **(medium) Add AppTest coverage for the Line-plots and Spatial-maps
   tabs.** Assert the new captions render and contain a unit, for at least one
   region. Backpressure: new `tests/` (or `tools/`) AppTest driver.
4. [ ] **(low) Unit-label the inter-variable pointwise maps.** They are still
   served as pre-rendered PNGs; the numeric value plotted is a (unitless)
   correlation. Either label the PNG panels in the generator or migrate to the
   npz-driven draw path used by the other maps.
5. [ ] **(low) Deploy check for `plot_meta.py`.** Both entry points clone the
   repo then `runpy.run_path(src/dashboard/dashboard.py)`, so the sibling module
   ships automatically — but re-verify after the next deploy. Note: the import is
   currently **unguarded** (fails loudly if missing) whereas the adjacent
   `baselines` import degrades gracefully; keep it unguarded so an unlabeled
   dashboard can never ship silently.
6. [ ] **(medium) Reconcile `rsds` non-negativity clipping.** `evaluate.py:253`
   defines `PHYS_MIN = {pr, sfcWind, sfcWindmax}` (no `rsds`), while the method
   inference scripts (`cm`, `corrdiff`, `idownscale`, `serpentflowclim`) clip
   `rsds` too. `CONTRACT.md §3` now lists `rsds` as non-negative. Decide whether
   `evaluate.py` should also clip `rsds` and make all methods consistent.

## Rules of engagement

- One item per loop. Prefer the top open item unless a human redirects.
- Run the test for the touched unit before claiming done.
- Update this file (add findings, prune completed items) at the end of every turn.
- Commit + push + bump the patch tag only when the backpressure test passes.
