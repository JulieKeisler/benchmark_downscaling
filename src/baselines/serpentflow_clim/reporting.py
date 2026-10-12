"""Ensemble aggregation and table printing.

Contains the fix for the precipitation ensemble-metrics table bug found in
the original notebooks: the printed table declared 4 columns
(KS, n_ratio, W1(log1p), W1(raw)) but the print statement only interpolated
3 values (`ks, nr, wlog, wraw = mean_metrics` then printed `ks, wlog, wraw`
-- `nr` silently dropped). Every column after KS was shifted left by one:
the number under "n_ratio" was actually W1(log1p), the number under
"W1(log1p)" was actually W1(raw), and "W1(raw)" printed nothing. This did
NOT corrupt the underlying stored metrics (only the console formatting),
but anyone reading the printed table off by column name would misattribute
every value after KS.

Fixed by keying the whole path by metric name instead of tuple position:
`metrics.compute_pr_metrics` returns a dict, and `print_pr_ensemble_table`
here reads it by key against `PR_TABLE_COLUMNS`, so a value can no longer
land under the wrong header regardless of ordering.
"""

import numpy as np
import xarray as xr


def compare_metrics_across_methods(var_name, pred_list, labels, era5_da, gcm_da, show=True,
                                    relative_change=False, time_slice=("2000-01-01", "2019-12-31"), mask=None):
    from . import metrics as M

    if len(pred_list) != len(labels):
        raise ValueError("pred_list and labels must have the same length")

    preds = [(p[var_name].sel(time=time_slice) if isinstance(p, xr.Dataset) else p.sel(time=time_slice))
             for p in pred_list]
    results = {label: {"era5": None, "gcm": None} for label in labels}

    if show:
        obs = era5_da[var_name].sel(time=time_slice) if isinstance(era5_da, xr.Dataset) else era5_da.sel(time=time_slice)
        for label, pred in zip(labels, preds):
            m, s = M.mean_std_differences(pred=pred, obs=obs, mask=mask)
            results[label]["era5"] = {
                "mean_diff": m, "std_diff": s,
                "ks": M.ks_statistic(pred=pred, obs=obs, mask=mask),
                "spatial_spearman": M.spatial_spearman_structure(pred=pred, obs=obs, mask=mask),
                "spatial_power_spec": M.spatial_power_spectrum_metric(pred=pred, obs=obs, mask=mask),
            }
    else:
        gcm = gcm_da[var_name].sel(time=time_slice) if isinstance(gcm_da, xr.Dataset) else gcm_da.sel(time=time_slice)
        for label, pred in zip(labels, preds):
            results[label]["gcm"] = {
                "temp_corr": M.temporal_correlation_with_gcm(pred=pred, gcm=gcm, mask=mask),
                "annual_anomaly_mae": M.global_annual_anomalies(pred=pred, gcm=gcm, mask=mask),
                "relative_change": None,
            }
        if relative_change:
            for label, pred in zip(labels, preds):
                results[label]["gcm"]["relative_change"] = {
                    "non_seasonal": M.relative_climate_change_signal(pred, gcm, seasonal=False, mask=mask),
                    "seasonal": M.relative_climate_change_signal(pred, gcm, seasonal=True, mask=mask),
                }
    return results


def _flatten(d: dict, parent: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def average_ensemble(runs: list, branch: str = "era5"):
    """runs: list of per-ensemble-member result dicts -> (mean, std) dicts."""
    labels = list(runs[0].keys())
    metrics = list(_flatten(runs[0][labels[0]][branch]).keys())
    mean = {lab: {} for lab in labels}
    std = {lab: {} for lab in labels}
    for lab in labels:
        flats = [_flatten(r[lab][branch]) for r in runs]
        for met in metrics:
            vals = np.array([f[met] for f in flats], dtype=float)
            mean[lab][met] = float(np.nanmean(vals))
            std[lab][met] = float(np.nanstd(vals))
    return mean, std


def print_ensemble_metrics(res: dict, branch: str, n: int = 8) -> None:
    for var in res:
        mean, std = average_ensemble(res[var], branch=branch)
        labels = list(mean.keys())
        metrics = list(mean[labels[0]].keys())
        label_w = max(len(l) for l in labels) + 2

        print(f"\n{'=' * 90}\n ENSEMBLE-AVERAGED {branch.upper()} METRICS | VARIABLE: {var.upper()} (n={n})\n{'=' * 90}")
        for met in metrics:
            print(f"\n{met}\n{'-' * 70}")
            for lab in labels:
                print(f"  {lab:<{label_w}} = {mean[lab][met]:>10.5f}  +/- {std[lab][met]:<9.5f}")
        print()


# (metric key -> (header label, column width)), in print order. Printing by
# key instead of tuple position is what actually forecloses the original
# bug (a positionally-unpacked value silently landing under the wrong
# header) -- adding/reordering a metric here can't misalign the others.
PR_TABLE_COLUMNS = (
    ("ks", "KS(q>0.9)", 12),
    ("n_ratio", "n_ratio", 10),
    ("w1_log1p", "W1(log1p)", 12),
    ("w1_raw", "W1(raw)", 10),
)


def print_pr_ensemble_table(results: dict) -> None:
    """Precipitation ensemble table: label -> list of per-member metrics.dicts
    (as returned by `metrics.compute_pr_metrics`)."""
    print(f"\n{'=' * 78}\n ENSEMBLE-AVERAGED PRECIPITATION DISTRIBUTION METRICS\n{'=' * 78}")
    print(f"{'method':<12} " + " ".join(f"{label:>{width}}" for _, label, width in PR_TABLE_COLUMNS))
    print("-" * 78)
    for label, members in results.items():
        row = f"{label:<12}"
        for key, _, width in PR_TABLE_COLUMNS:
            avg = float(np.nanmean([m[key] for m in members]))
            row += f" {avg:>{width}.4f}"
        print(row)
