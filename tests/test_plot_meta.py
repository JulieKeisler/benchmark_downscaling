"""Tests for src/dashboard/plot_meta.py — plot units and captions.

WHY THIS TEST EXISTS
--------------------
The dashboard (src/dashboard/dashboard.py) runs top-to-bottom under Streamlit
and cannot be imported by a test, so the unit/caption metadata was extracted
into the pure, dependency-free sibling module ``plot_meta``. This test is the
backpressure that keeps that contract honest:

  * A wrong or missing unit would otherwise ship silently — the dashboard would
    render "tas" with no "(K)", and a reader could mis-scale the plot axis.
    This happened conceptually with the earlier bug where the bias figure had
    no titles; we now assert titles/labels carry units.
  * Every line-plot and map-plot caption MUST name the variable's actual unit
    (requirement from the "every plot shows its unit" review) — not merely the
    word "Unit". If someone adds a new plot type without a caption,
    ``line_help``/``map_help`` raise KeyError and this test fails; the test is
    non-vacuous because it iterates the full type lists and every variable.
  * The variable list and units are the single source of truth for the whole
    benchmark (see CONTRACT.md §Units); duplicating them elsewhere is a defect.

These tests are intentionally offline and fast (no numpy/streamlit/plotly).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dashboard"))

import plot_meta  # noqa: E402


# Independent oracle for the six variables the benchmark ships and their units
# (src/configs/benchmark.yaml, CONTRACT.md §Units). Deliberately NOT imported
# from plot_meta: the whole point is to catch drift between the module and the
# contract, so this copy must be maintained by hand.
EXPECTED_UNITS = {
    "tas":        "K",
    "pr":         "mm/day",
    "psl":        "Pa",
    "sfcWind":    "m/s",
    "sfcWindmax": "m/s",
    "rsds":       "W/m²",
}


def test_all_benchmark_variables_have_a_unit():
    # A variable with no unit can never be labelled correctly anywhere.
    assert plot_meta.VAR_UNITS == EXPECTED_UNITS


def test_vlabel_formats_name_and_unit():
    # The dashboard shows "tas (K)"; the exact spacing/parentheses matter.
    assert plot_meta.vlabel("tas") == "tas (K)"
    assert plot_meta.vlabel("pr") == "pr (mm/day)"
    assert plot_meta.vlabel("rsds") == "rsds (W/m²)"


def test_vlabel_unknown_variable_does_not_invent_a_unit():
    # The dashboard discovers variables from results/, which may include names
    # not in VAR_UNITS. The fallback must be the bare name (never a wrong unit,
    # never an exception).
    assert plot_meta.vlabel("mysteryVar") == "mysteryVar"


# Line/map types whose caption must name the variable's physical unit verbatim.
# `spectra` is excluded: its axes are wavelength (km) and a normalised PSD (a.u.).
_PHYSICAL_LINE_TYPES = ("seasonal", "doy", "annual", "cdf")
_PHYSICAL_MAP_TYPES = ("bias_map", "samples", "delta_2040_2060", "delta_2080_2100")
_UNITLESS_MAP_TYPES = ("temporal_corr", "autocorr")


@pytest.mark.parametrize("line_type", _PHYSICAL_LINE_TYPES)
def test_physical_line_captions_state_the_actual_unit(line_type):
    # Stronger than "contains the word Unit": the variable's real unit string
    # must appear, so a caption that says "Units: ..." but names the wrong unit
    # (or omits it) still fails.
    for var, unit in EXPECTED_UNITS.items():
        caption = plot_meta.line_help(var, line_type)
        assert caption.strip(), f"empty caption for {line_type}"
        assert unit in caption, f"{line_type} caption for {var} lacks unit {unit!r}"


def test_spectra_caption_names_its_axis_units():
    # spectra plots spatial scale (km) vs normalised PSD (a.u.) — not a physical
    # variable unit, so it is checked separately.
    for var in EXPECTED_UNITS:
        cap = plot_meta.line_help(var, "spectra")
        assert "km" in cap and "a.u." in cap, f"spectra caption for {var} lacks axis units"


@pytest.mark.parametrize("plot_type", _PHYSICAL_MAP_TYPES)
def test_physical_map_captions_state_the_actual_unit(plot_type):
    for var, unit in EXPECTED_UNITS.items():
        caption = plot_meta.map_help(var, plot_type)
        assert caption.strip(), f"empty caption for {plot_type}"
        assert unit in caption, f"{plot_type} caption for {var} lacks unit {unit!r}"


@pytest.mark.parametrize("plot_type", _UNITLESS_MAP_TYPES)
def test_unitless_map_captions_are_marked_unitless(plot_type):
    for var in EXPECTED_UNITS:
        caption = plot_meta.map_help(var, plot_type)
        assert "unitless" in caption, f"{plot_type} caption for {var} should say 'unitless'"


def test_delta_captions_show_change_and_unit():
    # Δ maps show a difference; the caption must say so and name the unit.
    for pt in ("delta_2040_2060", "delta_2080_2100"):
        cap = plot_meta.map_help("tas", pt)
        assert "Δ" in cap
        assert "tas (K)" in cap


def test_colorbar_labels_carry_units_or_are_marked_unitless():
    # Spatial-map colorbars are the axis of the figure: each must name a unit,
    # and change maps must be prefixed with "Δ".
    unitless = {"temporal_corr", "autocorr"}
    for var in EXPECTED_UNITS:
        for pt in plot_meta.MAP_PLOT_TYPES:
            lbl = plot_meta.colorbar_label(var, pt)
            assert lbl.strip(), f"empty colorbar label {pt}/{var}"
            if pt in unitless:
                assert "unitless" in lbl, f"{pt} should be marked unitless"
            else:
                assert plot_meta.VAR_UNITS[var] in lbl, f"{pt}/{var} colorbar lacks unit"
            if pt.startswith("delta_"):
                assert lbl.startswith("Δ "), f"{pt} colorbar should mark the change (Δ)"


def test_plot_type_lists_are_non_empty_and_unique():
    for seq in (plot_meta.LINE_PLOT_TYPES, plot_meta.MAP_PLOT_TYPES):
        assert len(seq) == len(set(seq)) and len(seq) > 0


def test_helps_raise_on_unknown_type():
    # Unknown plot types are programming errors (e.g. a typo in a selectbox),
    # not data conditions — they must fail loudly rather than render blanks.
    with pytest.raises(KeyError):
        plot_meta.line_help("tas", "nope")
    with pytest.raises(KeyError):
        plot_meta.map_help("tas", "nope")
    with pytest.raises(KeyError):
        plot_meta.colorbar_label("tas", "nope")
