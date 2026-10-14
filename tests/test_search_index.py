"""Tests for src/dashboard/search_index.py — the sidebar full-text search."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dashboard"))

import search_index as si  # noqa: E402

SRC = '''
DOCS = {"m": {"name": "Méthode", "simple": "Uses SSR for dry days and a Hann taper."}}
st.title("Downscaling Benchmark")
with tab_a:
    st.subheader("Energy events")
    st.caption("Dunkelflaute spells … see the channels below")
    x = st.selectbox("Plot type", opts, key="some_widget_key")
with tab_b:
    if ww_mode == "Methods":
        st.markdown(DOCS["m"]["simple"])
    st.markdown("#### Spectra")
    st.caption(f"Fine scales below {km} km")

def line_help(v):
    return "Seasonal cycle of the monthly means"
'''
TABS = {"tab_a": "Inter-variable", "tab_b": "Welcome"}


def _idx():
    return si.build_index(SRC, TABS, {})


def test_locations_and_sections():
    hits = [(h["tab"], h["page"], h["section"]) for h in si.search(_idx(), "dunkelflaute")]
    assert hits == [("Inter-variable", "", "Energy events")]


def test_module_level_text_takes_location_of_first_use():
    (h,) = si.search(_idx(), "SSR")
    assert (h["tab"], h["page"]) == ("Welcome", "Methods")


def test_word_start_matching_and_bold_span():
    hits = si.search(_idx(), "hann")
    assert len(hits) == 1 and "**Hann**" in hits[0]["snippet"]   # not "c**hann**els"


def test_accents_and_all_words():
    assert si.search(_idx(), "methode")                      # accent-insensitive
    assert si.search(_idx(), "fine km") and not si.search(_idx(), "fine dunkelflaute")


def test_function_locations():
    idx = si.build_index(SRC, TABS, {"line_help": ("Line plots", "")})
    (h,) = si.search(idx, "monthly means")
    assert h["tab"] == "Line plots"


def test_query_cleaned_like_text():
    idx = si.build_index('with tab_a:\n    st.caption("see fine_spectrum_metric in metrics")\n', TABS, {})
    assert si.search(idx, "fine_spectrum_metric")


def test_code_strings_not_indexed():
    texts = [e["text"] for e in _idx()]
    assert "some_widget_key" not in texts and "simple" not in texts


def test_real_dashboard_index():
    root = Path(__file__).resolve().parent.parent / "src" / "dashboard"
    idx = si.build_index((root / "dashboard.py").read_text(),
                         {"tab_maps": "Spatial maps", "tab_welcome": "Welcome"},
                         {"METHOD_DOCS": ("Welcome", "Methods")})
    assert len(idx) > 300
    assert any(h["tab"] == "Welcome" for h in si.search(idx, "Wasserstein"))
    assert si.search(idx, "quantile delta mapping")
