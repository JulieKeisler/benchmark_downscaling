"""Plot metadata for the benchmark dashboard — units and captions.

This module is the single source of truth for:
  * the physical unit of every CMIP6 short-name variable, and
  * the plain-language caption and axis/colorbar label of every plot the
    dashboard draws (Line plots, Spatial maps tabs).

Rationale: the dashboard
script (``dashboard.py``) runs top-to-bottom under Streamlit and cannot be
imported by a unit test. Keeping the pure, dependency-free metadata here makes
it directly testable — so a wrong or missing unit fails CI instead of silently
shipping. ``dashboard.py`` adds its own directory to ``sys.path`` and imports
these helpers.

Rules (enforced by ``tests/test_plot_meta.py``):
  * every variable used by the benchmark has exactly one unit;
  * every plot type has a non-empty caption that names its unit;
  * colorbar labels for change maps are prefixed with "Δ".

Keep this file free of numpy/plotly/streamlit imports.
"""
from __future__ import annotations

# Physical units per CMIP6 short name. Must match the loader conventions in
# src/configs/benchmark.yaml (units are handled at load time, never inside
# method functions).
VAR_UNITS: dict[str, str] = {
    "tas":        "K",
    "pr":         "mm/day",
    "psl":        "Pa",
    "sfcWind":    "m/s",
    "sfcWindmax": "m/s",
    "rsds":       "W/m²",
}

# Plot-type vocabularies, in display order. The dashboard builds its selectboxes
# from these tuples, so a new plot type must be declared (and captioned) here.
LINE_PLOT_TYPES = ("seasonal", "doy", "annual", "cdf", "spectra")
MAP_PLOT_TYPES = (
    "bias_map",
    "temporal_corr",
    "acc_daily",
    "autocorr",
    "samples",
    "delta_2040_2060",
    "delta_2080_2099",
    "extreme_q99",
    "extreme_q01",
    "extreme_freq",
    "sal_freq",
    "sal_area",
    "sal_flat",
)
# Map types shown next to an ERA5 reference panel (fields, or the difference to ERA5).
REFERENCE_MAP_TYPES = ("extreme_q99", "extreme_q01", "extreme_freq", "sal_freq", "sal_area", "sal_flat")


def vlabel(var: str) -> str:
    """Return ``'tas (K)'``; fall back to the bare name if the unit is unknown.

    A missing unit must never raise (the dashboard discovers variables from the
    results directory, which may contain variables not listed above), but it
    must also never silently invent one — hence the bare-name fallback.
    """
    unit = VAR_UNITS.get(var)
    return f"{var} ({unit})" if unit else var


def line_help(var: str, line_type: str) -> str:
    """Caption (a few sentences) explaining the selected Line-plots figure.

    Every caption names the variable's unit so a reader never has to guess the
    ordinate scale. Raises ``KeyError`` on an unknown plot type — an unknown
    type is a programming error, not a runtime data condition.
    """
    u = VAR_UNITS.get(var, "physical units")
    helps = {
        "seasonal": (
            "Mean annual cycle over the evaluation period: the 12-monthly "
            "climatology of the region mean, for ERA5 (black) and each selected "
            "method. Averaging over many days removes weather noise and isolates "
            "the seasonal amplitude and phase, so a method that mis-times or "
            f"flattens the cycle stands out. Units: {vlabel(var)}."
        ),
        "doy": (
            "Daily climatology over the evaluation period (365 bins, 29 Feb "
            "removed) of the region-wide mean or of a single grid point. The "
            "lower panel shows the absolute day-to-day change: spikes on month "
            "boundaries flag methods that correct each calendar month "
            f"independently (BCSD, CDF-t, ...). Units: upper panel {vlabel(var)}; "
            f"lower panel {u} per day."
        ),
        "annual": (
            "Region-mean yearly means: ERA5 over 2000–2020 (grey, dotted), the "
            "bicubic-interpolated driving GCM (black) and each method (dashed) "
            "over 2000–2099. It shows whether a method keeps the observed level "
            "today and follows the GCM's long-term change; the GCM's individual "
            "years are not synchronised with observed years, so they are not a target. "
            f"Units: {vlabel(var)}."
        ),
        "cdf": (
            "Empirical cumulative distribution function (CDF) of all daily "
            "values: for a value x on the abscissa, the ordinate is the fraction "
            "of days at or below x. Comparing curves reveals whether a method "
            f"matches the observed spread and tails, not just the mean. Units: x = "
            f"{vlabel(var)}; y = cumulative probability (0–1, unitless)."
        ),
        "spectra": (
            "Isotropic spatial power spectral density, averaged over the "
            "evaluation days: how much variance sits at each spatial scale "
            "(abscissa = wavelength in km, log-spaced, decreasing to the right). "
            "Only the large scales scored by the spectral metric are shown "
            "(≈220–1000 km); wavelengths assume 0.25° ≈ 27.8 km in both "
            "directions (no cos(latitude) factor), so they overstate east–west "
            "scales at mid-latitudes. ERA5 (black) is the target. Units: x = km; "
            "y = PSD (a.u., same normalisation across all curves)."
        ),
    }
    return helps[line_type]


def map_help(var: str, plot_type: str) -> str:
    """Caption (a few sentences) explaining the selected Spatial-map figure."""
    helps = {
        "bias_map": (
            "Mean spatial bias over the evaluation period (mean(pred) − "
            "mean(ERA5)), one map per selected method. Red = overestimate, blue "
            "= underestimate; a white field is unbiased. The colour scale is "
            "shared across methods so panels are directly comparable. "
            f"Units: {vlabel(var)}."
        ),
        "temporal_corr": (
            "Pointwise temporal correlation (annual means) between each method "
            "and the GCM at every grid cell — how faithfully the method "
            "reproduces the GCM's year-to-year evolution, rather than its mean "
            "state. Units: correlation coefficient (unitless, −1 to +1; "
            "1 = perfect)."
        ),
        "acc_daily": (
            "Anomaly correlation (ACC) of daily values with the bicubic GCM at "
            "every grid cell, 2000–2099: each series' own daily climatology, "
            "refitted every 20 years, is removed first, so neither the seasonal "
            "cycle nor the forced trend counts. It shows how much of the GCM's "
            "day-to-day weather the method keeps. Units: correlation "
            "coefficient (unitless, 1 = same anomalies as the GCM)."
        ),
        "extreme_q99": (
            "99th percentile of daily values at every grid cell, 2000–2020: "
            "where the high extremes are. Leftmost panel is ERA5; switch to "
            "'difference to ERA5' to see method − ERA5 on a shared scale. "
            f"Units: {vlabel(var)}."
        ),
        "extreme_q01": (
            "1st percentile of daily values at every grid cell, 2000–2020: "
            "where the low extremes are (cold days, calm winds, low pressure, "
            "dark days). Not drawn for precipitation, whose low percentiles "
            "are 0 everywhere. Leftmost panel is ERA5. "
            f"Units: {vlabel(var)}."
        ),
        "extreme_freq": (
            "Fraction of days above the ERA5 95th percentile of the whole "
            "region (one threshold for all cells), 2000–2020: where the "
            "region's extreme days fall. The extreme-frequency score is the "
            "mean of |method − ERA5| over this map. Units: fraction of days "
            "(unitless, 0–1)."
        ),
        "sal_freq": (
            "Precipitation only. Fraction of days, 2000–2020, on which the cell "
            "lies inside a rain object (connected cells above 1/15 of the day's "
            "95th percentile of wet cells, Wernli et al. 2008): where it rains "
            "as part of an organised area. Units: fraction of days (unitless, 0–1)."
        ),
        "sal_area": (
            "Precipitation only. Median area of the rain objects covering the "
            "cell, over the days it is inside one, 2000–2020: whether rain "
            "areas have the right size where they occur. Units: km²."
        ),
        "sal_flat": (
            "Precipitation only. Median flatness (mean / maximum rain inside "
            "the object) of the rain objects covering the cell, 2000–2020: "
            "close to 1 for uniform rain areas, small for peaked cells. "
            "Units: ratio (unitless, 0–1)."
        ),
        "autocorr": (
            "Lag-1 autocorrelation difference (pred − GCM): how the method "
            "alters the GCM's day-to-day persistence. Positive = smoother/more "
            "persistent than the GCM, negative = more intermittent. "
            "Units: Δ autocorrelation (unitless)."
        ),
        "samples": (
            "Five randomly drawn days (same dates for every panel). Rows = GCM, "
            "ERA5 and each method; columns = the 5 dates. Compare daily spatial "
            "fields, not time-means, to spot where and when a method departs "
            f"from the reference. Units: {vlabel(var)}."
        ),
        "delta_2040_2060": (
            "Climate signal 2040–2060 vs 2000–2020: Δ = mean(future) − "
            "mean(hist) at every grid cell. Leftmost panel is the GCM reference; "
            "the others are per-method signals on a shared scale, so a method "
            f"that damps or amplifies the GCM's change is visible. Units: Δ {vlabel(var)}."
        ),
        "delta_2080_2099": (
            "Climate signal 2080–2099 vs 2000–2020: Δ = mean(future) − "
            "mean(hist) at every grid cell. Leftmost panel is the GCM reference; "
            "the others are per-method signals on a shared scale, so a method "
            f"that damps or amplifies the GCM's change is visible. Units: Δ {vlabel(var)}."
        ),
    }
    return helps[plot_type]


def colorbar_label(var: str, plot_type: str) -> str:
    """Unit-aware colorbar label for a Spatial-map figure.

    Change maps are prefixed with "Δ" because they show a difference (future −
    historical); correlation maps are unitless.
    """
    labels = {
        "bias_map":        vlabel(var),
        "temporal_corr":   "Temporal correlation r (unitless)",
        "autocorr":        "Δ lag-1 autocorr (unitless)",
        "acc_daily":       "ACC of daily anomalies (unitless)",
        "extreme_q99":     f"q99 {vlabel(var)}",
        "extreme_q01":     f"q01 {vlabel(var)}",
        "extreme_freq":    "Fraction of days > regional q95 (unitless)",
        "sal_freq":        "Fraction of days in a rain object (unitless)",
        "sal_area":        "Median rain-object area (km²)",
        "sal_flat":        "Median rain-object flatness (unitless)",
        "samples":         vlabel(var),
        "delta_2040_2060": f"Δ {vlabel(var)}",
        "delta_2080_2099": f"Δ {vlabel(var)}",
    }
    return labels[plot_type]
