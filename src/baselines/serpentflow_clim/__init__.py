"""SerpentFlow-CLIM: climatology-aware multivariate generative downscaling.

Cleaned, deduplicated extraction of the logic that was previously scattered
and repeated (with drift) across the project notebooks:
Data_Preparation.ipynb, Training-Inference.ipynb, and
Visualization_and_Metrics_{3vars,5vars,clean}.ipynb.

Notable differences from the original notebook code:
- Duplicate/superseded function definitions were collapsed to a single
  version (the notebooks defined several metric functions twice per file,
  with only the later cell's version actually live at call time).
- The precipitation ensemble-metrics report had a column-shift bug (the
  n_ratio value was silently dropped from the printed table, shifting every
  later column left). Fixed in `reporting.print_ensemble_table`.
- `stack_variables_as_channels` (data_utils.py) replaces ad hoc
  `torch.cat([...], dim=1)` calls that assumed same-length per-variable
  tensors without checking; it now asserts the source time coordinates
  actually match before stacking.
- The alternative SDE / stochastic-interpolant training path
  (`SDEPath`, `train_one_epoch_si_sde`, `quick_si_sde_sample`) present in the
  notebooks is NOT reproduced here: it is not the path described in the
  paper (which uses ODE flow matching, Algorithms 1-2), no checkpoint in
  this repo was trained with it, and keeping it would have doubled the
  training/inference surface for no result any figure or table depends on.
"""
