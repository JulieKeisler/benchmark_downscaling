"""Plot metadata for the benchmark dashboard — units and captions.

This module is the single source of truth for:
  * the physical unit of every CMIP6 short-name variable, and
  * the plain-language caption and axis/colorbar label of every plot the
    dashboard draws (Line plots, Spatial maps tabs).

Rationale (legoESM "scientific contract" + Ralph backpressure): the dashboard
script (``dashboard.py``) runs top-to-bottom under Streamlit and cannot be
imported by a unit test. Keeping the pure, dependency-free metadata here makes
it directly testable — so a wrong or missing unit fails CI instead of silently
shipping. ``dashboard.py`` adds its own directory to ``sys.path`` and imports
these helpers.

Contract (enforced by ``tests/test_plot_meta.py`` and ``CONTRACT.md``):
  * every variable used by the benchmark has exactly one unit;
  * every plot type has a non-empty caption that names its unit;
  * colorbar labels for change maps are prefixed with "Δ".

Keep this file free of numpy/plotly/streamlit imports.
"""
from __future__ import annotations

# Physical units per CMIP6 short name. Must match CONTRACT.md §Units and the
# loader conventions in src/configs/benchmark.yaml (units are handled at load
# time, never inside method functions — see AGENTS.md).
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
    "autocorr",
    "samples",
    "delta_2040_2060",
    "delta_2080_2100",
)


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
            "Yearly means over the evaluation period for ERA5 (grey, dotted), "
            "the driving GCM where available (black) and each method (dashed). "
            "It checks whether a method tracks the observed year-to-year level "
            "rather than drifting; the GCM line is the free-running input, not a "
            f"target. Units: {vlabel(var)}."
        ),
        "cdf": (
            "Empirical cumulative distribution function (CDF) of all daily "
            "values: for a value x on the abscissa, the ordinate is the fraction "
            "of days at or below x. Comparing curves reveals whether a method "
            f"matches the observed spread and tails, not just the mean. Units: x = "
            f"{vlabel(var)}; y = cumulative probability (0–1, unitless)."
        ),
        "spectra": (
            "Isotropic spatial power spectral density: how much variance sits at "
            "each spatial scale (abscissa = wavelength in km, log-spaced, "
            "decreasing to the right). ERA5 (black) is the observational target on "
            "the 0.25° grid; methods that lose small-scale variance flatten at "
            "high frequency (small scales). Units: x = km; y = PSD (a.u., same "
            "normalisation across all curves)."
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
        "delta_2080_2100": (
            "Climate signal 2080–2100 vs 2000–2020: Δ = mean(future) − "
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
        "samples":         vlabel(var),
        "delta_2040_2060": f"Δ {vlabel(var)}",
        "delta_2080_2100": f"Δ {vlabel(var)}",
    }
    return labels[plot_type]
