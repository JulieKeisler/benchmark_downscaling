#!/usr/bin/env python3
"""Benchmark dashboard — compare downscaling methods.

Run with:
    pip install streamlit plotly
    streamlit run src/dashboard/dashboard.py
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
# Data location is overridable for non-local (e.g. hosted) deployments, where
# results/ is fetched from an external store at startup. See deploy/hf_space/.
RESULTS_ROOT = Path(os.environ["RESULTS_ROOT"]) if os.environ.get("RESULTS_ROOT") \
    else ROOT / "results"
STATIC_DIR = Path(__file__).resolve().parent / "static"
# Downstream TAM lens data (outside this repo). Overridable for hosted deploys,
# which bundle a small copy alongside the app.
TAM_DIR = Path(os.environ.get("TAM_DIR", "/home/jkeisler/Documents/Code/TAM_downscaling"))

# Experiment/region labels — each is a top-level subdir of results/
EXPERIMENT_LABELS = {
    "france":     "France",
    "australia":  "Australia",
    "us_central": "US central",
}

REGION_SHORT = {
    "france":     "France",
    "australia":  "Australia",
    "us_central": "US central",
}

# Boxes drawn on the world map. `area` is CDS order [North, West, South, East].
REGION_META = {
    "france":     {"area": [55.0, -6.0, 41.0, 10.0],  "color": "#4C72B0"},
    "australia":  {"area": [-24.0, 136.0, -38.0, 152.0], "color": "#55A868"},
    "us_central": {"area": [48.0, -105.0, 34.0, -89.0], "color": "#C44E52"},
}

# Native coupling per region: methods live directly under results/{region}/
NATIVE_GCM = {
    "france":     "ipsl_cm6a_lr",
    "australia":  "cnrm_cm6_1",
    "us_central": "cesm2",
}

# Display scenario per GCM (used in labels and the welcome protocol text)
GCM_SCEN = {
    "ipsl_cm6a_lr": "SSP3-7.0",
    "cnrm_cm6_1":   "SSP3-7.0",
    "cesm2":        "SSP2-4.5",
}

GCM_LABELS = {
    "ipsl_cm6a_lr": "IPSL-CM6A-LR",
    "cnrm_cm6_1":   "CNRM-CM6-1",
    "cesm2":        "CESM2",
}

GCM_GRIDS = {
    "ipsl_cm6a_lr": "≈1.3° × 2.5° (11×7)",
    "cnrm_cm6_1":   "≈1.4° (10×12)",
    "cesm2":        "≈0.9° × 1.25° (15×13)",
}

# Coarse mesh (rows × cols) per GCM, for the recipe box
GCM_MESH = {
    "ipsl_cm6a_lr": "11×7",
    "cnrm_cm6_1":   "10×12",
    "cesm2":        "15×13",
}

IMG_RESOLUTION_GAP = STATIC_DIR / "resolution_gap.png"
IMG_BIAS_COMPARISON = STATIC_DIR / "bias_comparison.png"
IMG_METHOD_SPECTRUM = STATIC_DIR / "method_spectrum.png"

# HTML pipeline diagram for the Welcome tab (self-contained design)
_PIPELINE_HTML = """
<div style="display:flex; align-items:stretch; justify-content:center; flex-wrap:wrap; gap:4px; font-family:sans-serif; margin:12px 0">
  <div style="background:#1f77b4; color:#fff; border-radius:8px; padding:10px 14px; width:180px; text-align:center">
    <div style="font-size:20px; font-weight:bold">GCM</div>
    <div style="font-size:12px; opacity:.9">IPSL-CM6A-LR<br>≈2.5° · coarse grid</div>
  </div>
  <div style="display:flex; align-items:center; color:#555; font-size:22px; font-weight:bold">→</div>
  <div style="background:#ff7f0e; color:#fff; border-radius:8px; padding:10px 14px; width:190px; text-align:center">
    <div style="font-size:20px; font-weight:bold">Baseline</div>
    <div style="font-size:12px; opacity:.95">Bicubic · BCSD · CDF-t · DOTC<br>iDownscale · CorrDiff-QM · CM · SerpentFlow-CLIM</div>
  </div>
  <div style="display:flex; align-items:center; color:#555; font-size:22px; font-weight:bold">→</div>
  <div style="background:#2ca02c; color:#fff; border-radius:8px; padding:10px 14px; width:180px; text-align:center">
    <div style="font-size:20px; font-weight:bold">Predictions</div>
    <div style="font-size:12px; opacity:.95">≈0.25° · 57×65<br>2000–2100</div>
  </div>
  <div style="display:flex; align-items:center; color:#555; font-size:22px; font-weight:bold">→</div>
  <div style="background:#9467bd; color:#fff; border-radius:8px; padding:10px 14px; width:200px; text-align:center">
    <div style="font-size:20px; font-weight:bold">Evaluation</div>
    <div style="font-size:12px; opacity:.95">accuracy vs ERA5<br>fidelity vs GCM</div>
  </div>
  <div style="display:flex; align-items:center; color:#555; font-size:22px; font-weight:bold">→</div>
  <div style="background:#d62728; color:#fff; border-radius:8px; padding:10px 14px; width:170px; text-align:center">
    <div style="font-size:20px; font-weight:bold">You are here</div>
    <div style="font-size:12px; opacity:.95">interactive dashboard</div>
  </div>
</div>
"""

# Import METHOD_LABELS from the canonical location (src/baselines/__init__.py)
# so adding a method there automatically surfaces in the dashboard.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
# Own directory: so the sibling plot_meta module (units + plot captions) is
# importable when this file is run via runpy.run_path() by the deploy
# entry points — run_path does NOT add the script's dir to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Plot metadata (units, per-plot captions, plot-type lists) — single source of
# truth, unit-tested. The selectboxes below are built from LINE_PLOT_TYPES /
# MAP_PLOT_TYPES so a new type must be declared (and captioned) here.
from plot_meta import (  # noqa: E402
    VAR_UNITS, vlabel, line_help, map_help, colorbar_label,
    LINE_PLOT_TYPES, MAP_PLOT_TYPES,
)
try:
    from baselines import METHOD_LABELS, is_unpublished
except ImportError:  # tolerate a stale checkout where baselines pre-dates the helper
    from baselines import METHOD_LABELS

    def is_unpublished(_name) -> bool:
        return False
# Extended color palette (tab20)
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5",
]

GITHUB_URL = "https://github.com/JulieKeisler/benchmark_downscaling"

# Developer/maintainer instructions are hidden from the public deployment unless
# SHOW_DEV_NOTES=1 is set (see deploy/streamlit_cloud/app.py).
DEV_NOTES = os.environ.get("SHOW_DEV_NOTES") == "1"

st.set_page_config(page_title="Downscaling Benchmark for the Electricity System", layout="wide")
st.title("Downscaling Benchmark for the Electricity System")

# ── Experiment / region + GCM selection ────────────────────────────────────────
def _has_methods(d: Path) -> bool:
    """True if `d` has method subdirs directly under it OR a single GCM level."""
    if any((d / m / "metrics.json").exists()
           for m in d.iterdir() if (d / m).is_dir() and not is_unpublished(m.name)):
        return True
    # factorial layout: results/{region}/{gcm}/{method}
    return any((d / g / m / "metrics.json").exists()
               for g in d.iterdir() if (d / g).is_dir()
               for m in (d / g).iterdir()
               if (d / g / m).is_dir() and not is_unpublished(m.name))

def _coupling_dir(region: str, gcm: str) -> Path:
    """Directory holding the method subdirs for a (region, gcm) coupling.

    Native couplings store methods directly under results/{region}/; factorial
    couplings (any GCM other than the region's native one) store them under
    results/{region}/{gcm}/.
    """
    if gcm == NATIVE_GCM.get(region):
        return RESULTS_ROOT / region
    return RESULTS_ROOT / region / gcm

def _available_gcms(region: str) -> list[str]:
    """GCMs with results for a region: the native one (if present) plus any
    factorial GCM subdirs of results/{region}/."""
    d = RESULTS_ROOT / region
    gcms: list[str] = []
    if not d.exists():
        return gcms
    native = NATIVE_GCM.get(region)
    if native and any((d / m / "metrics.json").exists()
                      for m in d.iterdir()
                      if (d / m).is_dir() and not is_unpublished(m.name)):
        gcms.append(native)
    for g in sorted(x for x in d.iterdir() if x.is_dir()):
        if g.name == native:
            continue
        if any((g / m / "metrics.json").exists()
               for m in g.iterdir() if (g / m).is_dir() and not is_unpublished(m.name)):
            gcms.append(g.name)
    return gcms

def _gcm_label(gcm: str) -> str:
    g = GCM_LABELS.get(gcm, gcm.upper())
    scen = GCM_SCEN.get(gcm, "")
    return f"{g} ({scen})" if scen else g

# Region order follows NATIVE_GCM (France first): the welcome copy and the
# static figures describe France, so it is the landing region.
experiments = [
    r for r in NATIVE_GCM
    if (RESULTS_ROOT / r).is_dir() and _has_methods(RESULTS_ROOT / r)
] if RESULTS_ROOT.exists() else []
if not experiments:
    st.error(f"No experiment subdirs found in {RESULTS_ROOT}")
    st.stop()
# A pending region selection from the welcome-page map click. Applied here —
# before the widget is instantiated — because widget keys can't be modified
# after their widget is created.
_pending_region = st.session_state.pop("_pending_region", None)
if _pending_region in experiments:
    st.session_state["experiment"] = _pending_region
_selected = st.selectbox(
    "Experiment / Region", experiments,
    format_func=lambda e: EXPERIMENT_LABELS.get(e, e),
    key="experiment",
)

# GCM selector — dependent on the chosen region.
_available_gcms_region = _available_gcms(_selected)
if not _available_gcms_region:
    st.error(f"No GCM results found under {RESULTS_ROOT / _selected}")
    st.stop()
# Reset the GCM choice to the region's native GCM when the region changes.
# The widget key is "gcm": keeping its stale value would silently show a
# factorial coupling (e.g. France × CNRM-CM6-1) whenever the previous region's
# GCM also exists as a factorial subdir of the new region.
if st.session_state.get("_gcm_region") != _selected:
    st.session_state.pop("gcm", None)
st.session_state["_gcm_region"] = _selected
_selected_gcm = st.selectbox(
    "GCM", _available_gcms_region,
    format_func=lambda g: _gcm_label(g),
    key="gcm",
)
RESULTS_DIR = _coupling_dir(_selected, _selected_gcm)

# ── Discover methods ──────────────────────────────────────────────────────────
all_method_dirs = sorted(
    d.name for d in RESULTS_DIR.iterdir()
    if d.is_dir() and (d / "metrics.json").exists() and not is_unpublished(d.name)
) if RESULTS_DIR.exists() else []

if not all_method_dirs:
    st.error(f"No method results found in {RESULTS_DIR}")
    st.stop()

raw_metrics: dict[str, dict] = {}
for m in all_method_dirs:
    with open(RESULTS_DIR / m / "metrics.json") as f:
        raw_metrics[m] = json.load(f)

# Load plot_data.json if available (generated by updated evaluate.py)
plot_data: dict[str, dict] = {}
for m in all_method_dirs:
    pf = RESULTS_DIR / m / "plot_data.json"
    if pf.exists():
        with open(pf) as f:
            plot_data[m] = json.load(f)

def mlabel(m: str) -> str:
    return METHOD_LABELS.get(m, m.upper())

def _build_method_colors(methods: list[str]) -> dict[str, str]:
    """Deterministic method-name -> tab20 color, unique within this region.

    Uses md5(method-name) to pick a palette slot, probing to the next free slot
    on collision so no two methods ever share a color. The same method name
    therefore gets a stable color regardless of how many methods are present.
    """
    n = len(COLORS)
    owner: dict[int, str] = {}
    mapping: dict[str, str] = {}
    for name in methods:
        h = int(hashlib.md5(name.encode()).hexdigest(), 16)
        idx = h % n
        start = idx
        while idx in owner:
            idx = (idx + 1) % n
            if idx == start:
                break
        if idx in owner:  # pool exhausted (should not happen: 20 colors, <20 methods)
            idx = mapping[name] if name in mapping else 0
        owner[idx] = name
        mapping[name] = COLORS[idx]
    return mapping


METHOD_COLORS = _build_method_colors(all_method_dirs)


def _method_color(m: str) -> str:
    return METHOD_COLORS.get(m, "#333333")

def _blob_links(paths_str: str) -> str:
    """Turn a space|`+`-separated list of repo paths into GitHub blob links."""
    parts = [p.strip() for p in paths_str.replace("+", " ").split() if p.strip()]
    return " · ".join(
        f"[{p}]({GITHUB_URL}/blob/main/{p})" for p in parts
    ) if parts else ""

def _render_3panel_plotly(m: str, ref_title: str, ref_arr: np.ndarray, pred_title: str, pred_arr: np.ndarray, diff_title: str, diff_arr: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, subplot_titles=[ref_title, pred_title, diff_title], horizontal_spacing=0.04)
    fig.add_trace(go.Heatmap(z=ref_arr, x=lon, y=lat, colorscale="Viridis", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=pred_arr, x=lon, y=lat, colorscale="Viridis", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=diff_arr, x=lon, y=lat, colorscale="RdBu_r", zmid=0, showscale=True), row=1, col=3)
    fig.update_layout(height=320, margin=dict(t=40, b=10, l=10, r=10), title_text=mlabel(m), title_font_size=12)
    st.plotly_chart(fig, width="stretch", key=f"3panel_{m}")


# ── Pedagogic documentation (Welcome tab + search) ────────────────────────────

METHOD_DOCS = {
    "bicubic": {
        "name": "Bicubic",
        "type": "Deterministic · interpolation",
        "family": "statistical",
        "simple": "Smoothly interpolates the coarse GCM field onto the ERA5 grid. No learning, no bias correction.",
        "detailed": "Bicubic convolution interpolation (Keys kernel, a = −0.5) applied separably in latitude then longitude, similar to CDO remapbic. Precipitation and radiation are clipped to zero. Serves as the lower-bound baseline for accuracy — and, because every vs-GCM metric compares against this same interpolated GCM, it is the reference (perfect score) for the fidelity metrics.",
        "eq": r"\hat{y}_t = \mathcal{I}\left( x_t,\ \mathrm{lat}_{\mathrm{ERA5}},\ \mathrm{lon}_{\mathrm{ERA5}} \right)",
        "grading": "Deterministic, zero-parameter.",
    },
    "bcsd": {
        "name": "BCSD",
        "type": "Statistical · QM + spatial disaggregation",
        "family": "statistical",
        "simple": "Monthly quantile mapping at GCM resolution, applied to the GCM's own days as a monthly delta, then fine-scale ERA5 climatological detail added back.",
        "detailed": "After Wood et al. (2004). Interpolate ERA5 to the GCM grid, fit a per-cell, per-calendar-month empirical quantile map from GCM to ERA5 monthly means over 1980–1999 (20 values per cell-month), apply it to the GCM monthly means, and shift the GCM's own daily values by the monthly correction (additive; multiplicative for pr with the ratio clipped to [0, 5]). The result is cubic-interpolated to the ERA5 grid and the observed fine-scale monthly climatology is added back. The empirical map holds values flat outside the calibration range, so future GCM values beyond the 1980–1999 range are capped — this removes a large part of the warming signal.",
        "eq": r"\hat{y}_{\mathrm{daily}} = x_{\mathrm{daily}} \times \mathrm{clip}\!\left(\tfrac{\mathrm{bc}_{\mathrm{mon}}}{x_{\mathrm{mon}}},\ 0,\ 5\right)\ \ (\mathrm{pr})",
        "grading": "Deterministic; designed for climatological correction. Not trend-preserving: it distorts the year-to-year signal and caps the long-term change of every variable whose future values leave the calibration range (e.g. tas).",
    },
    "cdft": {
        "name": "CDF-t",
        "type": "Statistical · empirical CDF transform",
        "family": "statistical",
        "simple": "Per-grid-cell, per-month CDF transform that accounts for the shift of the GCM distribution between the calibration period and each 20-year projection window.",
        "detailed": "Michelangeli et al. 2009, via SBCK (CDFt, version 2). The GCM historical and scenario runs are bicubic-regridded to the ERA5 grid; for each grid cell and calendar month the calibration-period (1980–1999) GCM and ERA5 CDFs define the transfer, applied to 20-year sliding projection windows (step 10 years, central 10 years kept). Non-negative variables are clipped at zero; no separate wet-day threshold is applied.",
        "eq": r"F_{R_f}(x) = F_{R_p}\!\left( F_{G_p}^{-1}\!\left( F_{G_f}(x) \right) \right)",
        "grading": "Deterministic; better for future projections than plain QM.",
    },
    "dotc": {
        "name": "DOTC",
        "type": "Statistical · CDF-t with linear detrending",
        "family": "statistical",
        "simple": "The CDF-t pipeline applied to linearly detrended series, with the calibration-period GCM slope added back. Despite its label it is univariate — not the multivariate dOTC of Robin et al. (2019).",
        "detailed": "Same windows and grid as CDF-t (SBCK, per cell and calendar month, 20-year windows, step 10 y). Within each window a linear trend is fitted against the sample index on the 1980–1999 GCM and ERA5 series, removed, CDF-t is applied, and the calibration-period GCM slope is re-added. Because the index restarts in every window, the re-added term acts as a near-constant offset of about twenty years of the 1980–1999 GCM slope rather than the GCM's 2000–2100 trend. Clipping happens before the trend is re-added, so outputs can be slightly negative.",
        "eq": r"\hat{y} = \mathrm{CDFt}\!\left( x_{\mathrm{det}}\right) + \beta_{G,\,1980\text{–}1999}\,(t - \bar{t}_{\mathrm{cal}})",
        "grading": "Deterministic; transmits the GCM change like CDF-t, plus a calibration-slope offset.",
    },
    "idownscale": {
        "name": "iDownscale",
        "type": "Neural · U-Net regression (perfect prognosis)",
        "family": "neural",
        "simple": "A U-Net is trained on ERA5 alone (coarsened → fine pairs) to learn a super-resolution mapping; at inference the GCM is first CDF-t-corrected against ERA5 at coarse resolution (a hybrid of perfect prognosis and bias correction).",
        "detailed": "CERFACS iDownscale (EGU26 short course). Training pairs: ERA5 1980–1999 resized to 14×16 and upsampled back (coarse) → ERA5 (fine). MSE loss, month as sin/cos input channels, variables scaled to [−1, 1] (log for pr). At inference the GCM is bicubic-regridded to a stride-4 ERA5 grid (15×17), CDF-t-corrected against ERA5 with the GCM historical run 1980–1999 (20-year windows), upsampled and fed to the network. Plain UNet, init_features=32. Note the coarse grid differs between training and inference.",
        "eq": r"\mathcal{L} = \mathbb{E}\!\left[ \left\| f_\theta\!\left( \phi(x_t),\ m_t \right) - \phi(y_t) \right\|^2 \right]",
        "grading": "Deterministic neural; learns observed spatial structure.",
    },
    "corrdiff_qm": {
        "name": "CorrDiff-QM",
        "type": "Neural · stochastic residual diffusion",
        "family": "neural",
        "simple": "A regression U-Net predicts the mean; a DDPM learns the residual distribution around it. The output is the mean of a 10-member ensemble, which shrinks the residual variance.",
        "detailed": "Mardani et al. 2025 (adapted). Stage 1: plain U-Net predicts the conditional mean from the coarsened ERA5 input + month. Stage 2: a DiffusersUNet denoises the residual with a cosine DDPM (T=1000). Both trained on ERA5 1980–1999. Inference: daily per-cell, per-month empirical quantile mapping of the GCM onto ERA5 on the coarse grid (fit 1980–1999; values outside the calibration range are held flat, so the mapping is not trend-preserving), bicubic upsampling, strided reverse sampling (50 steps), mean of 10 members; pr, sfcWind, sfcWindmax and rsds clipped at 0.",
        "eq": r"\hat{y}^{\,(n)}_t = \mu_t + r^{(n)}_t",
        "grading": "Stochastic model, evaluated as a 10-member mean (deterministic in effect, reduced variance).",
    },
    "cm": {
        "name": "CM",
        "type": "Neural · consistency model (single-step generative)",
        "family": "neural",
        "simple": "A consistency model, distilled from a diffusion model trained on ERA5 alone, maps a noised, quantile-delta-mapped GCM field back onto the ERA5 manifold in a single denoising step.",
        "detailed": "Hess et al. 2025 / Song et al. 2023 (adapted). An EDM score network (DiffusersUNet) is trained on ERA5 1980–1999 (month as the only conditioning; no coarse input), then distilled into a consistency model (EMA teacher, curriculum N 10→150, T_max = 80). Inference: the bicubic-regridded GCM is bias-corrected by quantile delta mapping against ERA5 (GCM historical 1980–1999, per cell and month), noised at t* = 1.0 and projected onto the learned ERA5 manifold with one denoising step. All variables jointly (6; 5 over the central US). One random draw per day.",
        "eq": r"\hat{y} = f_\theta\!\left( \phi(\mathrm{QDM}(\mathcal{I}(\mathrm{GCM}_t))) + \varepsilon\, t^{*},\ t^{*} \right)",
        "grading": "Stochastic-generative; one-step (fast); a single unseeded draw.",
    },
    "serpentflowclim": {
        "name": "SerpentFlow-CLIM",
        "type": "Generative · conditional flow matching (climatology-aware)",
        "family": "neural",
        "simple": "A flow-matching model generates fine fields from a climatology-aware conditioning: the GCM's Fourier low-pass plus a CDF-t-corrected seasonal climatology. An 8-member ODE ensemble is averaged.",
        "detailed": "Yousef, Keisler & Charantonis (in prep.), building on SerpentFlow. The model is trained on ERA5 only (1980–1999): a Fourier low-pass of ERA5 plus high-frequency noise is mapped to ERA5. At inference the conditioning is the low-pass GCM whose 20-year day-of-year climatology is replaced by a CDF-t-corrected one (calibrated 1980–1999). A UNetFlow (6 variables, 5 over the central US; 4 static channels; day-of-year embedding) integrates an 8-member dopri5 ODE ensemble whose mean is the output; the corrected GCM climatology is re-injected for tas. Non-negative variables are clipped at zero.",
        "eq": r"\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t\sim\mathcal{U}(0,1)}\!\left[\, \left\| v_\theta(\phi_t, t) - u \right\|^2 \,\right]",
        "grading": "Stochastic-generative; climatology-aware conditioning; evaluated as an 8-member mean (reduced variance).",
    },
}

METRIC_DOCS = {
    "bias_mae": {
        "name": "Bias MAE",
        "group": "vs ERA5",
        "simple": "Spatial mean of absolute temporal-mean bias (pred − ERA5). Captures systematic offsets.",
        "detailed": "Mean over grid points of |mean_t(pred) − mean_t(ERA5)|, evaluation period 2000–2020. Lower is better. Units are physical units of the variable (K, mm/day, m/s, Pa, W/m²).",
        "eq": r"\mathrm{MAE}\!\left[ \overline{pred}_s - \overline{era5}_s \right]",
    },
    "std_error": {
        "name": "Std error",
        "group": "vs ERA5",
        "simple": "Mean absolute error of the temporal standard deviation. Captures underestimated/overestimated variability.",
        "detailed": "MAE over grid points of |std(pred) − std(ERA5)|. A value near 0 means the method reproduces the observed day-to-day variability.",
        "eq": r"\mathrm{MAE}\!\left[ \sigma_s(pred) - \sigma_s(era5) \right]",
    },
    "ks_full": {
        "name": "KS full distribution",
        "group": "vs ERA5",
        "simple": "Kolmogorov–Smirnov distance between the flattened distributions of predictions and ERA5.",
        "detailed": "Supremum difference between the two empirical CDFs over all pooled grid points and days. Lower is better; < 0.05 is generally excellent.",
        "eq": r"\sup_x |F_{pred}(x) - F_{era5}(x)|",
    },
    "ks_q95": {
        "name": "KS upper tail (q > 0.95)",
        "group": "vs ERA5",
        "simple": "KS distance restricted to values above the 95th ERA5 quantile — how well extremes are reproduced.",
        "detailed": "KS statistic computed only on the upper tail. Catches under-estimated extreme intensities common in interpolation-only methods.",
        "eq": "",
    },
    "seasonal_mae": {
        "name": "Seasonal MAE",
        "group": "vs ERA5",
        "simple": "MAE of the 12-month climatology. Captures the shape of the seasonal cycle.",
        "detailed": "Mean over grid points and months of |mean(pred by month) − mean(ERA5 by month)|.",
        "eq": "",
    },
    "extreme_freq": {
        "name": "Extreme frequency bias",
        "group": "vs ERA5",
        "simple": "Bias in the fraction of days above the 95th ERA5 percentile.",
        "detailed": "Spatial mean of |frac_pred − frac_era5| with a common ERA5-based threshold. Lower = better extreme frequency.",
        "eq": "",
    },
    "extreme_intensity": {
        "name": "Extreme intensity bias",
        "group": "vs ERA5",
        "simple": "Bias in the mean intensity of days above the 95th ERA5 percentile.",
        "detailed": "Spatial mean of |difference| in mean intensity above the threshold. Captures how extreme the extremes are.",
        "eq": "",
    },
    "spectral_mae": {
        "name": "Spectral MAE",
        "group": "vs ERA5",
        "simple": "Normalised RMSE between the time-mean isotropic spatial power spectra of predictions and ERA5, restricted to the large scales (wavelengths ≈ 220–1000 km).",
        "detailed": "Radially averaged 2D Fourier power spectra (log-spaced bins in 10⁻⁶–10⁻⁴ cycles/m; grid spacing taken as 0.25° × 111 km without a cos(latitude) factor), averaged over days; the score is RMSE(pred PSD − ERA5 PSD) / mean(ERA5 PSD) over the lowest 30 % of bins only. Despite its name it is not an MAE, and it does not test the fine scales (< ≈200 km) that downscaling adds — it measures whether the large-scale variance is right.",
        "eq": "",
    },
    "temporal_corr_annual": {
        "name": "Temporal corr annual",
        "group": "vs GCM",
        "simple": "Pearson correlation of annual-mean anomalies vs the GCM — how well the forced year-to-year signal is preserved. Higher is better.",
        "detailed": "Spatial mean of per-cell correlations between annual means of the prediction and of the bicubic-interpolated GCM over 2000–2100 (so it includes the forced trend). Bicubic scores 1 by construction.",
        "eq": r"r\!\left( \mathrm{anom}_{pred},\ \mathrm{anom}_{GCM} \right)",
    },
    "temporal_corr": {
        "name": "Temporal corr daily",
        "group": "vs GCM",
        "simple": "Pearson daily temporal correlation vs the bicubic-interpolated GCM over 2000–2100 (1 for bicubic by construction).",
        "detailed": "Monotone per-cell corrections keep it close to 1; methods that inject noise or generate fine-scale detail lower it. A low value is not by itself an error, since the GCM's daily weather is not the target — prefer temporal corr annual for signal preservation.",
        "eq": "",
    },
    "annual_anomaly_bias": {
        "name": "Annual anomaly bias",
        "group": "vs GCM",
        "simple": "MAE of annual-mean anomalies (trend preservation).",
        "detailed": "Mean absolute error between the method's and GCM's year-to-year anomalies.",
        "eq": "",
    },
    "lag1_autocorr_mae": {
        "name": "Lag-1 autocorr MAE",
        "group": "vs GCM",
        "simple": "MAE of lag-1 temporal autocorrelation maps — does day-to-day persistence match the GCM?",
        "detailed": "Compare ac_pred vs ac_gcm per grid point; MAE over the map.",
        "eq": "",
    },
    "Δ signal 2040–2060": {
        "name": "Δ signal 2040–2060",
        "group": "vs GCM",
        "simple": "Mean absolute difference between the method's and the GCM's relative change, 2040–2060 vs 2000–2020, in % of the 2000–2020 mean.",
        "detailed": "Per grid cell, Δ = (future mean − 2000–2020 mean) / |2000–2020 mean| × 100. For tas (K) and psl (Pa) the percentage is taken of an absolute temperature or pressure, so values are small (0.1 % of tas ≈ 0.3 K); compare methods within a variable, not across variables.",
        "eq": "",
    },
    "Δ signal 2080–2100": {
        "name": "Δ signal 2080–2100",
        "group": "vs GCM",
        "simple": "MAE between the method's and GCM's relative climate change signal at end-of-century.",
        "detailed": "Same as the 2040–2060 signal but for 2080–2100 vs 2000–2020 (in %, per grid cell).",
        "eq": "",
    },
}

def _method_doc(m: str) -> dict | None:
    """Return METHOD_DOCS entry for a method key, resolving aliases."""
    key = {"consistency": "cm", "cm": "cm", "corrdiff_qm": "corrdiff_qm"}.get(m, m)
    return METHOD_DOCS.get(key)


# ── World map with clickable region boxes (region navigation) ────────────────

def _rounded_square(n, w, s, e, r):
    """Sample the perimeter of a rounded square (rounded rect) in lat/lon space.

    `n, w, s, e` = North, West, South, East; `r` = corner radius (degrees).
    Returns (lon, lat) arrays tracing the outline, starting at the bottom
    midpoint and going counter-clockwise.  `(e - w)` is assumed == `(n - s)`
    so the box is a true square in degree space.
    """
    lon, lat = [], []
    # bottom edge (BL → BR), then BR corner
    for t in np.linspace(180, 270, 16):
        a = np.deg2rad(t)
        lon.append((w + r) + r * np.cos(a)); lat.append((s + r) + r * np.sin(a))
    for x in np.linspace(w + r, e - r, 24):
        lon.append(x); lat.append(s)
    for t in np.linspace(270, 360, 16):
        a = np.deg2rad(t)
        lon.append((e - r) + r * np.cos(a)); lat.append((s + r) + r * np.sin(a))
    # right edge (BR → TR), then TR corner
    for y in np.linspace(s + r, n - r, 24):
        lon.append(e); lat.append(y)
    for t in np.linspace(0, 90, 16):
        a = np.deg2rad(t)
        lon.append((e - r) + r * np.cos(a)); lat.append((n - r) + r * np.sin(a))
    # top edge (TR → TL), then TL corner
    for x in np.linspace(e - r, w + r, 24):
        lon.append(x); lat.append(n)
    for t in np.linspace(90, 180, 16):
        a = np.deg2rad(t)
        lon.append((w + r) + r * np.cos(a)); lat.append((n - r) + r * np.sin(a))
    # left edge (TL → BL), then BL corner (closes the loop)
    for y in np.linspace(n - r, s + r, 24):
        lon.append(w); lat.append(y)
    return np.asarray(lon), np.asarray(lat)


def _region_square(key):
    """Bounding box squared up around the region centre (16° × 16°)."""
    n, w, s, e = REGION_META[key]["area"]
    clat, clon = 0.5 * (n + s), 0.5 * (w + e)
    half = max(0.5 * (n - s), 0.5 * (e - w))     # guarantee a square
    return clat + half, clon - half, clat - half, clon + half


def build_region_map(region_keys, selected=None):
    """Interactive white world map with one rounded square per benchmark region.

    Canvas is plain white with black coastlines/country borders (from Natural
    Earth data bundled in `static/world_lines.npz`, projected equirectangularly
    on a cartesian axis — plotly's geo fills are unreliable at world range, but
    cartesian fills are exact). Each region is a rounded square filled solid
    with its app colour (a light pastel version when not active), outlined in
    black (thicker than the map's own lines). Invisible markers tile each square
    so any click inside selects that region (via Streamlit's plotly selection
    event, whose `customdata` carries the region key). The currently active
    region is drawn full-strength with a thicker border.
    """
    traces = []
    # Basemap: coastlines and country borders from cached Natural Earth data.
    coast_x, coast_y, border_x, border_y = _load_world_lines()
    traces.append(go.Scatter(x=coast_x, y=coast_y, mode="lines", name="coast",
                             line=dict(color="#000000", width=0.6),
                             hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter(x=border_x, y=border_y, mode="lines", name="borders",
                             line=dict(color="#000000", width=0.4),
                             hoverinfo="skip", showlegend=False))

    for key in region_keys:
        if key not in REGION_META:
            continue
        n, w, s, e = _region_square(key)
        color = REGION_META[key]["color"]
        active = (key == selected)

        r = 3.0
        px, py = _rounded_square(n, w, s, e, r)
        c = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)], dtype=float)
        if not active:
            c = c * 0.80 + 255 * 0.20
        fill_color = "rgba(%d,%d,%d,%.2f)" % (
            int(c[0]), int(c[1]), int(c[2]), 0.75 if active else 0.55)
        box = go.Scatter(
            x=np.append(px, px[0]),
            y=np.append(py, py[0]),
            mode="lines",
            fill="toself",
            fillcolor=fill_color,
            line=dict(color="#000000", width=3.5 if active else 2.5),
            hovertemplate=f"<b>{EXPERIMENT_LABELS[key]}</b><br>click to select<extra></extra>",
            name=key,
            showlegend=False,
        )
        # Invisible markers tiling the square so any click inside selects it.
        glon, glat = np.meshgrid(np.linspace(w, e, 16), np.linspace(s, n, 16))
        marker_tile = go.Scatter(
            x=glon.ravel().tolist(),
            y=glat.ravel().tolist(),
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.0),
            customdata=[key] * glon.size,
            hovertemplate=f"<b>{EXPERIMENT_LABELS[key]}</b> — click to open<extra></extra>",
            name=key,
            showlegend=False,
        )
        traces += [box, marker_tile]

    fig = go.Figure(data=traces)

    # Pill labels for each region (cartesian annotations support background).
    annotations = []
    for key in region_keys:
        if key not in REGION_META:
            continue
        n, w, s, e = _region_square(key)
        color = REGION_META[key]["color"]
        active = (key == selected)
        annotations.append(dict(
            x=0.5 * (w + e),
            y=n + 1.5,
            xref="x",
            yref="y",
            text=REGION_SHORT.get(key, key) + (" ✓" if active else ""),
            showarrow=False,
            bgcolor=color if active else "rgba(255,255,255,0.94)",
            bordercolor="#000000",
            borderwidth=1.2,
            borderpad=4,
            font=dict(size=12, color="#FFFFFF" if active else "#111111"),
        ))

    fig.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=8, b=0),
        annotations=annotations,
        clickmode="event+select",
        hovermode="closest",
        xaxis=dict(
            range=[-140, 175],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-52, 68],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True,
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


@functools.lru_cache(maxsize=1)
def _load_world_lines():
    """Return (coast_x, coast_y, border_x, border_y) polylines with NaN breaks."""
    _npz = STATIC_DIR / "world_lines.npz"
    _d = np.load(_npz)
    return _d["coast_x"], _d["coast_y"], _d["border_x"], _d["border_y"]

# ── Publications / reference code / implementation links ─────────────────────
# pub  → published paper or public reference implementation of the method
# impl → implementation used inside this benchmark
METHOD_LINKS = {
    "bicubic":     {"pub": "Keys (1981) cubic convolution interpolation (no downscaling paper; standard regridding)", "impl": "src/baselines/inference_bicubic.py"},
    "bcsd":        {"pub": "Wood et al. (2004), Climatic Change 62:189–216", "impl": "src/baselines/inference_bcsd.py"},
    "cdft":        {"pub": "Michelangeli et al. (2009), GRL 36:L11708 — via SBCK (`pip install sbck`)", "impl": "src/baselines/inference_cdft.py"},
    "dotc":        {"pub": "Benchmark-specific: CDF-t (Michelangeli et al. 2009, via SBCK) with linear detrending — not the multivariate dOTC of Robin et al. (2019), HESS 23:773", "impl": "src/baselines/inference_dotc.py"},
    "idownscale":  {"pub": "CERFACS iDownscale EGU26 short course — https://github.com/cerfacs-globc/idownscale/tree/master/docs/egu26_short_course", "impl": "src/baselines/inference_idownscale.py"},
    "corrdiff_qm": {"pub": "Mardani et al. (2025), Communications Earth & Environment 6:124 (arXiv:2309.15214) — CorrDiff residual corrective diffusion", "impl": "src/baselines/inference_corrdiff.py"},
    "cm":          {"pub": "Hess et al. (2025) Nature Machine Intelligence 7:363 — consistency-model downscaling; distills Song et al. (2023) arXiv:2303.01469 — ref. code: https://github.com/p-hss/consistency-climate-downscaling", "impl": "src/baselines/inference_cm.py + train_consistency.py"},
    "serpentflowclim": {"pub": "SerpentFlow — Keisler et al., arXiv:2601.01979 (conditional flow matching); climatology-aware CLIM variant: Yousef, Keisler & Charantonis (in prep.)", "impl": "src/baselines/inference_serpentflowclim.py + serpentflow_clim/"},
}

METRIC_IMPL = {
    "bias_mae":             "src/metrics/scalar_metrics_era5.py — spatial_mean_abs_bias",
    "std_error":            "src/metrics/scalar_metrics_era5.py — std_error",
    "ks_full":              "src/metrics/scalar_metrics_era5.py — ks_stat",
    "ks_q95":               "src/metrics/scalar_metrics_era5.py — ks_stat(q=0.95)",
    "seasonal_mae":         "src/metrics/scalar_metrics_era5.py — seasonal_mae",
    "extreme_freq":         "src/metrics/scalar_metrics_era5.py — extreme_freq_bias",
    "extreme_intensity":    "src/metrics/scalar_metrics_era5.py — extreme_intensity_bias",
    "spectral_mae":         "src/metrics/spatial_spectrum_metrics.py — spatial_spectrum_metric",
    "temporal_corr_annual": "src/metrics/scalar_metrics_gcm.py — mean_temporal_corr_annual",
    "temporal_corr":        "src/metrics/scalar_metrics_gcm.py — mean_temporal_corr",
    "annual_anomaly_bias":  "src/metrics/scalar_metrics_gcm.py — annual_anomaly_bias",
    "lag1_autocorr_mae":    "src/metrics/scalar_metrics_gcm.py — lag1_autocorr_mae",
    "Δ signal 2040–2060":   "src/metrics/scalar_metrics_gcm.py — delta_change_score (2040_2060)",
    "Δ signal 2080–2100":   "src/metrics/scalar_metrics_gcm.py — delta_change_score (2080_2100)",
}

# Existing downscaling / bias-correction benchmark initiatives this benchmark
# builds on and differs from (used by the Welcome → Overview tab).
# Each entry: (short cite, domain/resolution, protocol, GCMs, variables, metrics).
BENCHMARK_CONTEXT = {
    "VALUE — <a href='https://doi.org/10.1002/2014EF000259' target='_blank'>Maraun et al. (2015), Earth's Future</a>": (
        "<b>Protocol:</b> the reference for standardised <b>perfect-prognosis</b> evaluation in "
        "Europe — inter-compares statistical downscaling methods on common benchmarks. "
        "<b>Settings:</b> fixed European stations/grid at daily scale across several GCMs and "
        "reanalyses; scores predictands vs observations. "
        "<b>What we take:</b> the accuracy-vs-ERA5 view. "
        "<b>What differs:</b> we also demand fidelity vs the GCM out to 2100 (not just skill on "
        "a reanalysis), benchmark the 5 electricity-relevant variables as multivariate fields "
        "rather than station series, and evaluate on the 57×65 France grid instead of stations."
    ),
    "CORDEX / EURO-CORDEX — <a href='https://doi.org/10.1007/s10113-013-0499-2' target='_blank'>Jacob et al. (2014)</a>": (
        "<b>Protocol:</b> dynamical downscaling via fixed regional models (RCMs) nested in GCMs, "
        "producing full spatio-temporal fields. "
        "<b>Settings:</b> ~12 km (EURO-CORDEX) over whole Europe; many variables; a handful of "
        "GCM-RCM chains. "
        "<b>Metrics:</b> mostly mean-climate and extremes diagnostics. "
        "<b>What differs:</b> an RCM is calendar-aligned to its driver (reanalysis or GCM), "
        "so it enables paired, supervised RCM-emulation benchmarks; our fine truth is observed "
        "ERA5, which is not aligned to the free-running GCM, so we evaluate unpaired "
        "(distribution metrics + fidelity vs the GCM). We also inter-compare statistical and "
        "machine-learning methods on a <i>single</i> GCM instead of RCM chains — cheaper, and "
        "isolates the baseline effect, but we lose RCM process realism and multi-GCM spread."
    ),
    "ISIMIP — <a href='https://doi.org/10.5194/gmd-10-4321-2017' target='_blank'>Frieler et al. (2017)</a>": (
        "<b>Protocol:</b> harmonised, trend-preserving <b>bias adjustment</b> (ISIMIP3BASD, Lange 2019) of GCM output feeding "
        "sectoral (incl. energy) impact models. "
        "<b>Settings:</b> a handful of GCMs per round, global coverage, RCP/SSP scenarios, daily "
        "output on a common 0.5° grid. "
        "<b>Metrics:</b> downstream impact-model outcomes. "
        "<b>What differs:</b> ISIMIP adjusts one product with one method at 0.5° (its statistical "
        "downscaling step is part of that method); we go to ~25 km, score both against observations <i>and</i> against the GCM's own future signal, and "
        "compare many methods against each other (not a single correction)."
    ),
    "WeatherBench / WeatherBench 2 — <a href='https://doi.org/10.1029/2020MS002203' target='_blank'>Rasp et al. (2020)</a>, <a href='https://doi.org/10.1029/2023MS004019' target='_blank'>Rasp et al. (2024)</a>": (
        "<b>Protocol:</b> standardised ML benchmark for short-range <i>weather forecasting</i> at "
        "native ~1.4° resolution. "
        "<b>Metrics:</b> RMSE/ACC vs reanalysis over lead times. "
        "<b>What differs:</b> we benchmark <i>downscaling</i> — turning coarse climate-model fields into "
        "fine (≈0.25°) ones over multi-decadal climate horizons — not forecasting from initial "
        "conditions. Lead-time RMSE is replaced by distribution, correlation and climate-signal "
        "metrics."
    ),
}

# Research references underpinning the electricity-system framing (DOI-linked, markdown links).
ELECTRICITY_REFS = {
    "Temperature → demand": (
        "[Auffhammer, M., Baylis, P., & Hausman, C. H. (2017), PNAS 114(8):1886–1891]"
        "(https://doi.org/10.1073/pnas.1613193114) — climate change raises peak-electricity "
        "demand frequency and intensity."
    ),
    "Wind resource → generation": (
        "[Pryor, S. C., & Barthelmie, R. J. (2010), Renewable & Sustainable Energy Reviews "
        "14(1):430–437](https://doi.org/10.1016/j.rser.2009.07.028) — review of climate-change "
        "impacts on wind-energy resource."
    ),
    "Solar resource → generation": (
        "[Jerez, S., et al. (2015), Nature Communications 6:10014](https://doi.org/10.1038/ncomms10014)"
        " — climate-change impact on photovoltaic power potential across Europe."
    ),
    "Hydrology → hydro & cooling": (
        "[van Vliet, M. T. H., et al. (2016), Nature Climate Change 6:375–380]"
        "(https://doi.org/10.1038/nclimate2903) — power-plant availability and hydro output "
        "under warmer, drier conditions."
    ),
    "System-wide synthesis": (
        "[Yalew, S. G., et al. (2020), Nature Energy 5:794–802](https://doi.org/10.1038/s41560-020-0664-z)"
        " — synthesis of climate-change impacts across demand, renewables and supply."
    ),
    "Weather as the driver of supply & demand": (
        "[Staffell, I., & Pfenninger, S. (2018), Energy 145:65–78]"
        "(https://doi.org/10.1016/j.energy.2017.12.051) — the increasing impact of weather on "
        "electricity supply and demand."
    ),
    "Climate projections tailored for the energy sector": (
        "[Bartók, B., et al. (2019), Climate Services 16:100138]"
        "(https://doi.org/10.1016/j.cliser.2019.100138) — EURO-CORDEX, bias-corrected, "
        "energy-relevant projection dataset: addresses resolution, biases, ensemble "
        "consistency and user guidance."
    ),
    "Weather & climate as input to grids": (
        "[Bloomfield, H. C., et al. (2021), Bulletin of the American Meteorological Society "
        "102:E159–E167](https://doi.org/10.1175/BAMS-D-20-0256.1) — workshop on next-generation "
        "challenges at the weather–climate–electricity-system interface."
    ),
}

# Perfect-prognosis vs model-output-stats vs delta-change: short primer shown in the
# Welcome → Overview tab.
PROTOCOL_PRIMER = (
    "**Perfect prognosis (PP)** learns the coarse→fine downscaling map entirely from "
    "*reanalysis* pairs (GCM-like coarse predictors, observed fine predictand, both from ERA5), "
    "then applies it to real GCM fields. It never corrects the GCM against observations, so it "
    "keeps the GCM's own climate-change signal — at the cost of assuming the GCM 'sees' like the "
    "reanalysis (the perfect-prognosis assumption). "
    "**Model output statistics / bias correction (MOS/BC)** instead learns the mapping between "
    "a *specific GCM*'s output and observations, which removes that GCM's bias but can also "
    "distort or 'freeze' its future signal (e.g. delta/quantile matching). "
    "**Our benchmark is NOT a pure perfect-prognosis study** — and that is exactly what "
    "distinguishes it from classic PP (VALUE) and pure-BC (ISIMIP) inter-comparisons. The "
    "baselines are calibrated in their *own* way on the selected GCM: "
    "bicubic interpolates with no training at all; BCSD, CDF-t and DOTC are "
    "MOS/bias-correction methods fit directly against the GCM historical output (their "
    "calibration *is* GCM-vs-ERA5, not reanalysis-vs-reanalysis); the neural/generative "
    "baselines (CorrDiff-QM, iDownscale, CM, SerpentFlow-CLIM) are hybrids — their networks are "
    "trained on ERA5 alone in the PP spirit, but at inference the GCM is first bias-corrected "
    "against ERA5 (quantile mapping, CDF-t or quantile delta mapping fitted on the GCM historical "
    "run 1980–1999) before the network sees it. "
    "Because calibration philosophies therefore differ *within* the benchmark, we hold the GCM, "
    "grid, period and train/test split fixed for all baselines and score each one on BOTH "
    "(i) reproduction of observed distributions and (ii) preservation of the GCM's "
    "climate signal to 2100 — the two things any of these philosophies can silently get wrong."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    sel_methods = st.multiselect(
        "Methods", all_method_dirs, default=all_method_dirs,
        format_func=mlabel, key="methods",
    )
    if not sel_methods:
        st.warning("Select at least one method.")
        st.stop()

    # Union across methods: taking the first method's keys hid variables (e.g. rsds)
    # that older submissions lack; missing (method, var) cells render as NaN.
    all_vars = list(dict.fromkeys(v for m in raw_metrics.values() for v in m
                                  if not v.startswith("_")))
    sel_vars = st.multiselect("Variables", all_vars, default=all_vars, key="vars")

    st.markdown("---")
    exp_label = EXPERIMENT_LABELS.get(_selected, _selected)
    st.caption(
        f"**Experiment:** {exp_label} · {_gcm_label(_selected_gcm)}  \n"
        "**ERA5:** pseudo-obs (1979–2022)  \n"
        "**Evaluation period:** 2000–2020 against ERA5 and 2000–2100 for GCM related metrics"
    )

    st.markdown("---")
    st.markdown(
        f"🔗 [Benchmark GitHub]({GITHUB_URL})\n\n"
        "For the electricity system: demand (tas), wind (sfcWind), "
        "solar/rad, hydro (pr), regimes (psl)."
    )

    st.markdown("---")
    search_query = st.text_input("🔎 Search", key="search_box",
                                 placeholder="e.g. bias, BCSD, extremes, trend...")
    if search_query.strip():
        q = search_query.strip().lower()
        hit_methods = [m for m in all_method_dirs
                       if q in mlabel(m).lower() or q in m.lower()
                       or q in (_method_doc(m) or {}).get("simple", "").lower()
                       or q in (_method_doc(m) or {}).get("type", "").lower()]
        hit_metrics = [k for k, d in METRIC_DOCS.items()
                       if q in d["name"].lower() or q in d["simple"].lower()]
        if hit_methods or hit_metrics:
            st.success(f"**{len(hit_methods)} method(s) · {len(hit_metrics)} metric(s) found**")
            if hit_methods:
                st.markdown("— *Methods:* " + ", ".join(mlabel(m) for m in hit_methods))
            if hit_metrics:
                st.markdown("— *Metrics:* " + ", ".join(METRIC_DOCS[k]["name"] for k in hit_metrics))
        else:
            st.caption("No matches — try a method or metric name.")

# ── Metric groups ─────────────────────────────────────────────────────────────
OBS_METRICS = [
    "bias_mae", "std_error",
    "ks_full", "ks_q95",
    "seasonal_mae", "extreme_freq", "extreme_intensity",
    "spectral_mae",
]
GCM_METRICS = [
    "temporal_corr_annual", "temporal_corr",
    "annual_anomaly_bias", "lag1_autocorr_mae",
    "Δ signal 2040–2060", "Δ signal 2080–2100",
]

# Maps delta metric key -> _delta period key in metrics.json
DELTA_PERIOD_MAP = {
    "Δ signal 2040–2060": "2040_2060",
    "Δ signal 2080–2100": "2080_2100",
}

METRIC_LABELS = {
    "bias_mae":             "Bias MAE (vs ERA5)",
    "std_error":            "Std error (vs ERA5)",
    "ks_full":              "KS full dist (vs ERA5)",
    "ks_q95":               "KS upper tail q>0.95",
    "seasonal_mae":         "Seasonal MAE (vs ERA5)",
    "extreme_freq":         "Extreme freq bias",
    "extreme_intensity":    "Extreme intensity bias",
    "spectral_mae":         "Spectral MAE (vs ERA5)",
    "intervar_corr":        "Inter-var corr MAE (vs ERA5)",
    "temporal_corr_annual": "Temporal corr annual (vs GCM)",
    "temporal_corr":        "Temporal corr daily (vs GCM)",
    "annual_anomaly_bias":  "Annual anomaly bias (vs GCM)",
    "lag1_autocorr_mae":    "Lag-1 autocorr MAE (vs GCM)",
    "Δ signal 2040–2060":   "Δ signal 2040–2060 (vs GCM)",
    "Δ signal 2080–2100":   "Δ signal 2080–2100 (vs GCM)",
}
HIGHER_BETTER = {"temporal_corr_annual", "temporal_corr"}

# VAR_UNITS / vlabel / per-plot captions live in the sibling plot_meta module
# (imported above) so they are unit-testable without a Streamlit runtime.


# ── Helpers ───────────────────────────────────────────────────────────────────
def _agg(method: str, key: str, axis: str = "vs_obs") -> float:
    """Aggregate a metric across selected variables. Handles delta metrics specially."""
    if key in DELTA_PERIOD_MAP:
        period = DELTA_PERIOD_MAP[key]
        vals = [
            raw_metrics[method].get("_delta", {}).get(v, {}).get(period, float("nan"))
            for v in sel_vars
        ]
    else:
        vals = [
            raw_metrics[method][v][axis][key]
            for v in sel_vars
            if v in raw_metrics[method] and key in raw_metrics[method][v].get(axis, {})
        ]
    return float(np.nanmean(vals)) if vals else float("nan")


def _get_metric_value(method: str, var: str, key: str) -> float:
    """Get a single metric value for a method/variable/key combo."""
    if key in DELTA_PERIOD_MAP:
        period = DELTA_PERIOD_MAP[key]
        return raw_metrics[method].get("_delta", {}).get(var, {}).get(period, float("nan"))
    else:
        return raw_metrics[method].get(var, {}).get("vs_gcm", {}).get(key, float("nan"))


def _center_image(path, max_width=720):
    """Display an image centred, never cropped, scaled down to fit its container."""
    import base64
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    st.markdown(
        f"<div style='display:flex;justify-content:center;width:100%;overflow:hidden'>"
        f"<img src='data:image/png;base64,{b64}' "
        f"style='max-width:{max_width}px;width:100%;height:auto;border-radius:6px;"
        f"object-fit:contain'/>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
# Tab count/changes with selected region: widget states for removed tabs are
# dropped automatically, so it's safe to build the tab list conditionally.
if _selected == "france":
    _tab_titles = [
        "🎯 Welcome", "📊 Metrics table", "🕸 Radar", "📈 Line plots",
        "🗺 Spatial maps", "🔗 Inter-variable", "⚡ Load Forecasting (TAM)",
    ]
elif _selected == "us_central":
    _tab_titles = [
        "🎯 Welcome", "📊 Metrics table", "🕸 Radar", "📈 Line plots",
        "🗺 Spatial maps", "🔗 Inter-variable", "⚡ Load Forecasting (TAM)",
    ]
elif _selected == "australia":
    _tab_titles = [
        "🎯 Welcome", "📊 Metrics table", "🕸 Radar", "📈 Line plots",
        "🗺 Spatial maps", "🔗 Inter-variable", "⚡ Load Forecasting (TAM)",
    ]
else:
    _tab_titles = [
        "🎯 Welcome", "📊 Metrics table", "🕸 Radar", "📈 Line plots",
        "🗺 Spatial maps", "🔗 Inter-variable",
    ]
tab_welcome, tab_table, tab_radar, tab_lines, tab_maps, tab_intervar, *tab_tail = st.tabs(_tab_titles)
tab_load = tab_tail[0] if tab_tail else None

# ══════════════════════════════════════════════════════════════════════════════
# Tab 0 — Welcome (pedagogic)
# ══════════════════════════════════════════════════════════════════════════════
with tab_welcome:
    st.markdown(
        "### 🎯 Welcome to the Downscaling Benchmark for the Electricity System\n"
        "**The one-sentence version.** Climate impact studies for the energy sector need "
        "fine-resolution, bias-corrected weather that global climate models do not provide — "
        "**statistical downscaling** bridges that gap by learning a map from coarse GCM "
        "fields to fine-resolution reanalysis-like fields. This dashboard is a benchmark "
        "asking *how well different downscaling methods do it* — on the map and, crucially, "
        "through the decisions the weather feeds in the power system."
    )
    st.markdown(
        f"📦 Benchmark code, data recipes and this dashboard: "
        f"[**github.com/JulieKeisler/benchmark_downscaling**]({GITHUB_URL})"
    )

    # ── Region navigation map ──────────────────────────────────────────────
    st.markdown("### 🌍 Regions — click a box to explore")
    st.caption(
        "Each square marks the ERA5 reference box of one benchmark region with its "
        "paired GCM. Click a region on the map (or use the *Experiment / Region* "
        "dropdown at the top, which stays in sync) to load its results."
    )
    region_fig = build_region_map(experiments, selected=_selected)
    region_chart = st.plotly_chart(
        region_fig,
        on_select="rerun",
        selection_mode="points",
        key="region_map",
        width="stretch",
    )

    # Interpret plotly selection events (Streamlit ≥ 1.35) into experiment selection.
    try:
        _sel = region_chart.selection
        _points = _sel["points"] if isinstance(_sel, dict) else getattr(_sel, "points", None)
    except (TypeError, AttributeError, KeyError):
        _points = None
    if _points:
        _clicked = []
        for p in _points:
            pd_ = p.get("customdata") if isinstance(p, dict) else getattr(p, "customdata", None)
            if isinstance(pd_, (list, tuple)):
                pd_ = pd_[0] if pd_ else None
            if isinstance(pd_, str) and pd_ in EXPERIMENT_LABELS:
                _clicked.append(pd_)
        if _clicked and _clicked[0] != _selected:
            st.session_state["_pending_region"] = _clicked[0]
            st.rerun()
    st.markdown(
        f"**Currently viewing:** {EXPERIMENT_LABELS.get(_selected, _selected)} · "
        f"{_gcm_label(_selected_gcm)} "
        f"· GCM grid {GCM_GRIDS.get(_selected_gcm, '')}"
    )
    st.markdown("---")

    ww_mode = st.radio(
        "Explore",
        ["Overview", "Methods", "Metrics", "Tour"],
        horizontal=True,
        key="welcome_mode",
    )

    # ── Welcome: Overview ────────────────────────────────────────────────────
    if ww_mode == "Overview":
        _wl_region = EXPERIMENT_LABELS.get(_selected, _selected)
        _wl_gcm = _gcm_label(_selected_gcm)
        _wl_mesh = GCM_MESH.get(_selected_gcm, "")
        _wl_era5 = "1979–2020" if _selected == "france" else "1979–2022"
        st.markdown(
            "Use the links below to jump between sections. Each section first defines the "
            "concept it needs, then explains what it means for this benchmark."
        )
        st.markdown(
            "**Contents** · "
            "[1. What is a GCM?](#ww-sec1) · "
            "[2. ERA5 reanalysis](#ww-sec2) · "
            "[3. Why the electricity system needs fine weather](#ww-sec3) · "
            "[4. What is statistical downscaling?](#ww-sec4) · "
            "[5. What this benchmark does](#ww-sec5) · "
            "[6. Why distribution metrics](#ww-sec6) · "
            "[7. Why not pure perfect prognosis](#ww-sec7) · "
            "[8. Downstream test: net-load forecasting](#ww-sec8) · "
            "[References](#ww-refs) · "
            "[How to read this dashboard](#ww-how)"
        )

        # ── Section 1: GCMs ─────────────────────────────────────────────────
        st.markdown('<a id="ww-sec1"></a>', unsafe_allow_html=True)
        st.markdown(
            "## 1 · What is a climate model, and why is it so coarse?\n"
            "A **Global Climate Model (GCM)** simulates the coupled atmosphere–ocean–land "
            "system by solving the equations of physics on a three-dimensional grid that "
            "covers the whole Earth. To produce a century or more of simulation, it must be "
            "computationally tractable, so each grid cell is huge — typically ≈1–2.5° of "
            "latitude/longitude, i.e. ~100–250 km across. Over a country-sized region "
            "(France is ~1000 km across), that is only a handful to a dozen cells; a single "
            "cell lumps together a city, its surrounding farmland and a mountain range. Each "
"cell holds one value *per variable per day* — 2-m temperature, near-surface wind, "
             "precipitation, sea-level pressure, surface radiation.\n\n"
             "Three properties of these coarse, free-running fields are what make "
             "downscaling necessary for the electricity system:\n"
             "- **Coarse output.** At ~250 km there is essentially no local detail: no "
             "valley-scale heatwave, no gusts at a specific wind-farm site, no convective "
             "storm cell — yet installed capacity and demand sit in places the grid cannot "
             "resolve.\n"
             "- **No calendar.** A GCM is not a weather forecast; it is a physics-based "
             "simulation that, due to the chaotic nature of the system, evolves unconstrained "
             "by observations, so it is not tied to the calendar. It produces long sequences "
             "of physically plausible weather, but there is no real \"15 January 2005\" that "
             "its 15 January 2005 corresponds to. This dictates how we must evaluate "
             "downscaling methods (see [6. Why distribution metrics](#ww-sec6)).\n"
             "- **Systematic biases.** Unresolved processes are approximated, so the outputs "
             "carry systematic biases — smoothed variability, shifted means, distorted "
             "extremes."
        )

        # ── Section 2: ERA5 reanalysis ──────────────────────────────────────
        st.markdown('<a id="ww-sec2"></a>', unsafe_allow_html=True)
        st.markdown(
            "## 2 · ERA5 reanalysis: our fine-resolution \"ground truth\"\n"
            "To build and test downscaling we need a *high-resolution reference of what the "
            "weather actually was*. Observations alone are sparse and irregular, but "
            "reanalysis fixes this: ERA5 (from ECMWF) blends decades of real "
            "observations (satellites, stations, radiosondes) into a numerical weather "
            "model, producing a physically consistent, gap-free historical record at "
            "≈0.25° (~25 km) resolution, daily since 1940 (we use the 1979–2022 crop). It "
            "covers any study region on a **57×65** grid (~3700 cells) with all the variables "
            "we use — availability follows the GCM, so e.g. CESM2 has no daily `sfcWindmax`.\n"
            "ERA5 is not perfect, but it is the best available \"ground truth\" at the "
            "resolution energy applications need, and it is the standard reference for "
            "downscaling development. Throughout this dashboard, \"the observations\" / "
            "\"pseudo-observations\" always mean this reanalysis."
        )

        # ── Section 3: why the energy sector needs fine weather ─────────────
        st.markdown('<a id="ww-sec3"></a>', unsafe_allow_html=True)
        st.markdown(
            "## 3 · Why does the electricity system need fine weather?\n"
            "Unlike the GCM's ~250 km cells, the power system is planned and operated at "
            "the scale of cities, wind farms and transmission corridors. And weather is an "
            "*input* to almost every part of its operation:\n"
            "- **Demand (the load)** scales with temperature — winter heating and summer "
            "cooling course through the same grids.\n"
            "- **Wind generation** depends on the local wind resource; **solar generation** "
            "depends on local radiation.\n"
            "- **Hydrology** bounds hydro output and the availability of cooling water for "
            "thermal plants; **pressure regimes** set broad-scale weather patterns.\n\n"
            "### Net load — the quantity that matters\n"
            "The single quantity that concentrates all these sensitivities is net "
            "load: `NetDemand = load − wind − solar`. It is what a system operator must "
            "balance in real time and what capacity planning must size for. Because it "
            "compounds the temperature sensitivity of demand *and* the wind/solar "
            "sensitivity of generation, it is exactly the object through which this "
            "benchmark reveals how errors in the downscaled weather translate into "
            "operationally relevant errors (see [8. Downstream test](#ww-sec8)).\n\n"
            "### The practical barriers\n"
            "Using climate projections for such work is routinely blocked by documented, "
            "practical barriers: the fields are too coarse for site-level analysis, they "
            "carry systematic biases that mislead impact models, different models and "
            "scenarios disagree, daily-mean outputs miss the sub-daily and extremes "
            "that stress grids, and the datasets are unwieldy for non-specialists "
            "([Bartók et al., 2019](https://doi.org/10.1016/j.cliser.2019.100138); "
            "[Staffell & Pfenninger, 2018](https://doi.org/10.1016/j.energy.2017.12.051); "
            "[Auffhammer et al., 2017](https://doi.org/10.1073/pnas.1613193114))."
        )

        # ── Section 4: what downscaling does ────────────────────────────────
        st.markdown('<a id="ww-sec4"></a>', unsafe_allow_html=True)
        st.markdown(
            "## 4 · What is statistical downscaling?\n"
            "**Downscaling** turns a coarse GCM field into a fine-resolution weather-like "
            "field on the grid, variables and daily scale an impact study needs, while "
            "respecting both the GCM's climate signal and the observed climate. Two "
            "philosophies scaffold the literature "
            "([Maraun et al., 2010](https://doi.org/10.1029/2009rg000314)):\n"
            "- **Perfect prognosis (PP).** Learns the coarse→fine mapping from *reanalysis "
            "pairs only* (coarse ERA5 degraded to GCM-like resolution → fine ERA5), then "
            "applies the learned map to real GCM fields. It keeps the GCM's signal by "
            "assuming the GCM \"sees\" like the reanalysis.\n"
            "- **Model output statistics / bias correction (MOS/BC).** Learns the mapping "
            "between a *specific GCM's* output and observations, removing that GCM's bias "
            "directly but risking distortion of its future signal.\n\n"
            "### The method spectrum\n"
            "Practical methods span the range: pure **interpolation** (bicubic), "
            "**distribution-based / quantile** methods (BCSD, CDF-t, DOTC), **learned "
            "regression** methods (iDownscale), and **generative** methods "
            "(CorrDiff, consistency models, SerpentFlow-CLIM). The `Methods` tab documents "
            "the eight baselines used here. Crucially, **our benchmark is not a pure "
            "perfect-prognosis study** — the "
            "baselines are calibrated in their *own* way on the selected GCM, and "
            "[7. Why not pure perfect prognosis](#ww-sec7) explains why this matters when "
            "comparing against existing benchmarks."
        )
        if IMG_RESOLUTION_GAP.exists():
            _center_image(str(IMG_RESOLUTION_GAP), max_width=720)
            st.caption(
                "The core problem in one picture (France, IPSL-CM6A-LR). Same January day, same "
                "variable: the raw GCM (right, ≈1.3° × 2.5°, 11×7 cells) collapses the whole region into a handful of "
                "blocky values; ERA5 (left, ≈0.25°, 57×65 cells) resolves real regional "
                "detail. Downscaling must invent realistic fine detail that is consistent "
                "with *both* the GCM's climate signal and the observed climate."
            )
        else:
            if DEV_NOTES:
                st.info("Generate `src/dashboard/static/resolution_gap.png` with `generate_welcome_images.py`.")
            else:
                st.caption("_Resolution-gap figure not included in this build._")

        # ── Setup: what this benchmark does ──────────────────────────────────
        st.markdown('<a id="ww-sec5"></a>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown(
                "## 5 · What this benchmark does\n"
                "We fix *every source of variation except the downscaling method*, then "
                "compare all methods fairly on the same inputs, the same grid, and the same "
                "train/test split:\n"
                f"- **Input (the GCM):** {_wl_gcm}, daily, {GCM_GRIDS.get(_selected_gcm, '')} "
                "coarse grid; projections evaluated 2000–2100.\n"
                "- **Reference (ground truth):** ERA5 reanalysis, 57×65 grid, "
                "1979–2022 (France: 1979–2020).\n"
                "- **Variables:** the electricity drivers — `tas` (2-m temperature, drives "
                "demand), `sfcWind` / `sfcWindmax` (10-m wind speed and gust, drive wind "
                "generation), `rsds` (surface solar radiation, drives PV), `pr` "
                "(precipitation, drives hydrology), `psl` (sea-level pressure, sets weather "
                "regimes). Availability follows the GCM.\n"
                "- **Calibration / evaluation split:** fitted baselines calibrate on "
                "1980–1999 and are evaluated on the disjoint 2000–2020 window (the "
                "raw, interpolation-only baseline needs no calibration).\n\n"
                "Each baseline therefore has a single job: turn the GCM's coarse fields "
                "into 57×65 daily fields whose *distribution* matches ERA5 (accuracy) while "
                "preserving the *GCM's own climate-change signal* through 2100 (fidelity). "
                "Because the GCM is free-running, we cannot align its days to historical "
                "dates — so pointwise scores like MSE/MAE vs ERA5 are not even well defined. "
                "The only fair comparison is between distributions — and "
                "[6. Why distribution metrics](#ww-sec6) explains what this forces us to "
                "measure."
            )
        with c2:
            st.markdown("#### Recipe")
            for step, txt in [
                ("1", f"GCM input: {_wl_gcm}, {_wl_mesh} grid (2000–2100)"),
                ("2", "Downscale with each baseline to 57×65 (~25 km)"),
                ("3", "Score vs ERA5 pseudo-observations (accuracy)"),
                ("4", "Score vs GCM (does it keep the climate signal?)"),
                ("5", "Feed into a net-load forecast (⚠️ Load Forecasting tab)"),
            ]:
                st.markdown(
                    f"<div style='background:#f0f2f6;border-radius:8px;padding:6px 10px;"
                    f"margin:4px 0'><b>{step}.</b> {txt}</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Protocol at a glance")
        st.markdown(
            "| Item | Setting |\n"
            "|---|---|\n"
            f"| **Region box** | {_wl_region} — 57×65 cells at ≈0.25° |\n"
            f"| **Reference** | ERA5 pseudo-observations; daily {_wl_era5} |\n"
            f"| **GCM** | {_wl_gcm}; daily; IPSL-CM6A-LR & CNRM-CM6-1 under SSP3-7.0, CESM2 under SSP2-4.5 |\n"
            "| **Calibration** | 1980–1999 — distribution baselines fit on the selected GCM; neural baselines trained on ERA5, with a GCM-vs-ERA5 pre-correction at inference |\n"
            "| **Accuracy evaluation** | 2000–2020 vs ERA5 |\n"
            "| **Fidelity evaluation** | 2000–2100 vs the GCM; Δ-signal 2040–2060 & 2080–2100 |\n"
            "| **Downstream (TAM)** | trained 2012–2021 on ERA5 + load; applied unchanged 2000–2022 |\n"
            "| **Units** | tas K · pr mm/day · sfcWind / sfcWindmax m/s · psl Pa · rsds W/m² |"
        )

        # ── Section 6: distribution metrics ──────────────────────────────────
        st.markdown('<a id="ww-sec6"></a>', unsafe_allow_html=True)
        st.markdown(
            "## 6 · Why distribution metrics — not MSE/MAE\n"
            "It is tempting to score downscaling with pointwise errors (MSE/MAE vs ERA5). "
            "Three distinct problems close that route; together they set the "
            "evaluation design:\n"
            "1. **No fair pointwise truth exists.** GCM days (raw or downscaled) are not "
            "aligned to historical calendar dates, so there is no \"correct\" ERA5 value for "
            "a given GCM day to be compared against. Only *distributions* of many days are "
            "comparable. That alone is why we use distribution metrics.\n"
            "2. **Point errors miss what planning fears.** Even with an aligned truth, "
            "MSE/MAE are dominated by the bulk of the distribution — blind to the **tails** "
            "(heatwaves, droughts, gusts), the **variability/autocorrelation** (does a bad "
            "day persist into a bad week?), the **spatial structure** (wind correlated "
            "across a whole region), and the **inter-variable links** (cold, calm nights "
            "push peak demand when wind and solar both dip). Two series can share an "
            "identical RMSE and have totally different tails "
            "([Maraun et al., 2010](https://doi.org/10.1029/2009rg000314); "
            "[Gneiting & Raftery, 2007](https://doi.org/10.1198/016214506000001437)). This "
            "is why the distribution comparison isolates these aspects one at a time rather "
            "than collapsing them into a single number.\n"
            "3. **The full target is a giant joint distribution.** Each method yields a "
            "5-variable, 57×65, 100-year field — tens of millions of values. Distance "
            "measures between such high-dimensional multivariate distributions (KS, energy "
            "distance…) converge slowly and depend on how you flatten space and time "
            "([Székely & Rizzo, 2016](https://doi.org/10.1002/wics.1375)). This is why we "
            "do not score the whole object with a single distance.\n\n"
            "So we use two complementary views: **(a)** *monotone summary metrics* that "
            "isolate one distributional aspect at a time (KS statistics, extreme "
            "frequencies, autocorrelation, spatial spectra), computed separately against the "
            "observed distribution (accuracy) and against the GCM (signal preservation); "
            "and **(b)** a *downstream shortcut*: pass each scenario through a **fixed** "
            "net-load forecaster and compare the resulting forecast-error distribution "
            "(see [8. Downstream test](#ww-sec8)) — the engineering view that a forecast is "
            "only as good as the operational decision it informs "
            "([Hong & Fan, 2016](https://doi.org/10.1016/j.ijforecast.2015.11.011))."
        )

        # ── Residual bias image ─────────────────────────────────────────────
        st.markdown("### Why interpolation is not enough")
        if IMG_BIAS_COMPARISON.exists():
            _center_image(str(IMG_BIAS_COMPARISON), max_width=1400)
            st.caption(
                "Climatological mean bias (Prediction − ERA5) for three baselines, with "
                "coastlines. Bicubic merely interpolates the GCM, so its systematic "
                "cold/warm biases survive. The distribution-based (DOTC) and generative "
                "(CM) baselines explicitly correct against ERA5 and remove most of it. The "
                "effect is large: over France (2000–2020), the temperature bias-MAE vs ERA5 "
                "is 0.88 K for bicubic against 0.12 K for the best-corrected baseline; for "
                "10-m wind it is 0.66 vs 0.08 m/s. The "
                "bias that remains is exactly what the accuracy metrics on the other tabs "
                "quantify — and it is precisely the kind of error that feeds straight into a "
                "power-system model's demand forecasts."
            )
        else:
            if DEV_NOTES:
                st.info("Generate `src/dashboard/static/bias_comparison.png` with `generate_welcome_images.py`.")
            else:
                st.caption("_Bias-comparison figure not included in this build._")

        # ── Section 7: Protocol: not pure perfect prognosis ─────────────────
        st.markdown('<a id="ww-sec7"></a>', unsafe_allow_html=True)
        st.markdown("## 7 · Protocol — why this is not a pure perfect-prognosis study")
        st.markdown(PROTOCOL_PRIMER)

        # ── vs existing benchmarks (subsection of 7) ─────────────────────────
        st.markdown("### This benchmark vs existing downscaling/bias-correction benchmarks")
        st.markdown(
            "Existing community benchmarks come from different traditions and were not "
            "built for the energy sector: VALUE (station-based, perfect-prognosis), "
            "EURO-CORDEX (dynamical downscaling of RCMs), ISIMIP (bias-corrected GCMs for "
            "impact studies), and WeatherBench (ML benchmarks on raw reanalysis fields, "
            "not GCM projections)."
        )
        for title, note in BENCHMARK_CONTEXT.items():
            st.markdown(
                f"<div style='border-left:4px solid #4C72B0;background:#f8f9fb;"
                f"padding:6px 12px;margin:5px 0;border-radius:4px'>"
                f"<b>{title}.</b> {note}</div>",
                unsafe_allow_html=True,
            )
        st.caption(
            "Compared with those initiatives, this benchmark is: **electricity-oriented** "
            "(demand/wind/solar variables + a load-forecasting downstream task), **transparent** "
            "(every baseline, metric and plot links to its implementation below), "
            "**projection-aware** (all methods evaluated on the same GCM out to 2100), and "
            f"**open** (code released under [this GitHub repository]({GITHUB_URL}))."
        )

        # ── Evaluation pipeline (subsection of 7) ───────────────────────────
        st.markdown("#### The evaluation pipeline")
        st.markdown(_PIPELINE_HTML, unsafe_allow_html=True)

        # ── Section 8: Load forecasting downstream ──────────────────────────
        st.markdown('<a id="ww-sec8"></a>', unsafe_allow_html=True)
        st.markdown("## 8 · A decision-relevant downstream test — net-load forecasting")
        st.markdown(
            "Distributional errors only matter operationally if they propagate into "
            "decisions, so the ⚡ Load Forecasting tab pushes each downscaled scenario "
            "through a **fixed**, already-trained net-load forecaster and compares what "
            "breaks.\n\n"
            f"**The target.** **Net load** for {_wl_region} — `NetDemand = load − wind − solar`, "
            "a daily series. It is the quantity a system operator must "
            "balance in real time, and it is where errors in `tas` (demand), "
            "`sfcWind`/`sfcWindmax` (wind), and `psl`/`pr` (weather regimes, hydro) "
            "interact.\n\n"
            "**The model.** A **Time series Additive Model (TAM)**: a regression spline in "
            "season (`s(toy)`) interacting with temperature (`te(toy, rbf(temperature))`), "
            "plus day-of-week, school/bank/Christmas-holiday dummies, and a smooth nonlinear "
            "response to the weather inputs driving demand. Three spatial representations "
            "are compared: **(1) Full Grid PCA** — the 15 leading PCA components of the "
            "daily spatial field (57×65 grid, ≈80% of variance); **(2) Regional "
            "Barycenters** — population-weighted weather for 12 sub-regions (15 regional PCs, ≈93% of variance); "
            "**(3) Regional + Renewables** — the regional representation plus explicit wind "
            "and solar resource indices. (The multi-region lenses apply to France; the other "
            "regions ship the full-grid PCA.)\n\n"
            "**The weather modeling.** The same TAM is trained once on **ERA5 reanalysis "
            "weather (2012–2021)** plus the region's observed net demand (central US: from "
            "July 2018, when its record starts). For evaluation it is never "
            "retrained: each downscaled scenario simply substitutes the method's own "
            "predictions (next-day temperature, wind, precipitation from the downscaled "
            "`tas`, `sfcWind`, `sfcWindmax`, `pr`, `psl` fields, lagged one day) into the "
            "already-fit model and produces a daily net-load forecast distribution. Any "
            "downscaling error is therefore propagated through unchanged model weights — an "
            "honest, apples-to-apples test of how the *input* weather drives the downstream "
            "error.\n\n"
            "**What we report.** For each method we compare its forecast-error *distribution* "
            "to the ERA5-truth one with **KS distance, Wasserstein distance (MW), KL "
            "divergence, and absolute errors at the extreme quantiles AE@q01 / AE@q99 (MW)** — "
            "i.e. how badly a method misses the most costly (very low / very high net-load) "
            "days. The tab offers error-bar "
            "comparisons, quantile-distortion plots, a radar per method, net-demand CDFs, and "
            "breakdowns by season and by wind level."
        )

        # ── References ──────────────────────────────────────────────────────
        st.markdown('<a id="ww-refs"></a>', unsafe_allow_html=True)
        with st.expander("📚 References — electricity-system climate sensitivity"):
            for area, cite in ELECTRICITY_REFS.items():
                st.markdown(f"- **{area}.** {cite}")

        # ── Cite & data ─────────────────────────────────────────────────────
        with st.expander("📖 Cite this benchmark, and where the data lives"):
            st.markdown(
                "- **Paper.** *In preparation* — a DOI/preprint link will be added here when "
                "available (author list to be confirmed).\n"
                f"- **Code.** Open source under the MIT licence: "
                f"[this GitHub repository]({GITHUB_URL}).\n"
                "- **Results data.** Downscaled predictions, metrics and TAM archives are "
                "published as a public dataset: `Jkeisler75/benchmark-downscaling-results` "
                "(Hugging Face Datasets). A Zenodo DOI archive is planned.\n\n"
                "If this dashboard or benchmark supports your work, please cite the paper "
                "(once published) and link this repository. Code is under MIT; the data "
                "follows the terms of the underlying ERA5 (Copernicus) and CMIP6 datasets."
            )

        # ── How to read ─────────────────────────────────────────────────────
        st.markdown('<a id="ww-how"></a>', unsafe_allow_html=True)
        st.markdown("## How to read this dashboard")
        for tab_name, desc in [
            ("📊 Metrics table", "every method × metric as a table, best highlighted in green"),
            ("🕸 Radar", "methods as radar polygons, each axis = one normalised metric"),
            ("📈 Line plots", "seasonal cycles, annual means, distributions & spatial spectra"),
            ("🗺 Spatial maps", "bias / correlation / climate-signal maps per method"),
            ("🔗 Inter-variable", "do methods keep observed links between variables?"),
            ("⚡ Load Forecasting", "forecasting net electricity load with the fixed downstream model"),
        ]:
            st.markdown(f"- **{tab_name}** — {desc}")

    # ── Welcome: Methods ─────────────────────────────────────────────────────
    elif ww_mode == "Methods":
        st.markdown("#### The baselines: from interpolation to generative models")
        m_detailed = st.toggle("Show detailed documentation (equations, hyperparameters)", key="welcome_methods_detailed")
        if IMG_METHOD_SPECTRUM.exists():
            st.image(str(IMG_METHOD_SPECTRUM), width="stretch")
        for m in ["bicubic", "bcsd", "cdft", "dotc", "idownscale", "corrdiff_qm", "cm", "serpentflowclim"]:
            doc = METHOD_DOCS[m]
            link = METHOD_LINKS.get(m, {})
            links_md = ""
            if link:
                links_md = "\n\n"
                if link.get("pub"):
                    links_md += f"📖 **Publication/ref:** {link['pub']}"
                if link.get("impl"):
                    links_md += f"\n⚙️ **Implementation:** {_blob_links(link['impl'])}"
            if m_detailed:
                with st.expander(f"{doc['name']} — {doc['type']}", expanded=False):
                    st.markdown(f"> **{doc['simple']}**")
                    if doc["eq"]:
                        st.latex(doc["eq"])
                    st.markdown(f"**Detail.** {doc['detailed']}")
                    st.caption(f"*{doc['grading']}*")
                    if links_md:
                        st.markdown(links_md)
            else:
                st.markdown(
                    f"<div style='border-left:4px solid #4C72B0;background:#f8f9fb;"
                    f"padding:8px 12px;margin:6px 0;border-radius:4px'>"
                    f"<b>{doc['name']}</b> <span style='color:#666'>· {doc['type']}</span><br>"
                    f"{doc['simple']}</div>",
                    unsafe_allow_html=True,
                )
                if link.get("impl"):
                    st.caption(f"📖 {link.get('pub','')} · ⚙️ Implementation: {_blob_links(link['impl'])}")

    # ── Welcome: Metrics ─────────────────────────────────────────────────────
    elif ww_mode == "Metrics":
        st.markdown("#### What we measure — and why")
        m_detailed = st.toggle("Show formulas (detailed)", key="welcome_metrics_detailed")
        g1, g2 = st.columns(2)
        for col, group in [(g1, "vs ERA5"), (g2, "vs GCM")]:
            with col:
                st.markdown(f"##### {group}")
                for k, doc in METRIC_DOCS.items():
                    if doc["group"] != group:
                        continue
                    impl = METRIC_IMPL.get(k, "")
                    if m_detailed:
                        with st.expander(doc["name"], expanded=False):
                            st.markdown(doc["simple"])
                            if doc["eq"]:
                                st.latex(doc["eq"])
                            st.caption(doc["detailed"])
                            if impl:
                                st.caption(f"⚙️ Implementation: {_blob_links(impl)}")
                    else:
                        st.markdown(
                            f"<div style='background:#f8f9fb;border-radius:6px;padding:6px 10px;"
                            f"margin:4px 0'><b>{doc['name']}</b> — {doc['simple']}</div>",
                            unsafe_allow_html=True,
                        )
                        if impl:
                            st.caption(f"⚙️ Implementation: {_blob_links(impl)}")
                if group == "vs GCM":
                    st.caption(
                        "**Why two views?** Accuracy (vs ERA5) shows how well a method matches "
                        "observed *today*; fidelity (vs GCM) shows whether it preserves the "
                        "model's *future* signal — both matter for climate change impact studies."
                    )

    # ── Welcome: Tour ────────────────────────────────────────────────────────
    elif ww_mode == "Tour":
        st.markdown("#### A quick tour")
        tour_steps = [
            ("01 · Metrics table", "Start here. Pick variables & metric groups in the sidebar or on the tab. Green = best per row. Read the captions for caveats (e.g. daily temporal corr is trivially ~1)."),
            ("02 · Radar", "Zoom out. Each method is one polygon; every axis is a metric normalised so 1.0 = mean across methods. Points *inside* the bounding polygon are better."),
            ("03 · Line plots", "Drill into one variable. 'seasonal' shows the climatology, 'annual' the long-term evolution (watch GCM vs ERA5 vs methods), 'cdf' the distribution, 'spectra' the spatial structure."),
            ("04 · Spatial maps", "See *where* methods succeed or fail — bias, temporal correlation, and future climate-signal maps over France."),
            ("05 · Inter-variable", "Check whether methods preserve observed links, e.g. tas–pr correlation. A correct marginal distribution can still break physical relationships."),
            ("06 · Load Forecasting (TAM)", "A real downstream payoff: feed each downscaled scenario into a fixed net-load forecaster. Small distribution errors translate into MW of forecast error."),
        ]
        for title, body in tour_steps:
            st.markdown(
                f"<div style='border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:8px 0'>"
                f"<b>{title}</b><br>{body}</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            "#### Suggested reading path\n"
            "**1 min:** Overview → the two images.  \n"
            "**5 min:** Metrics table for *tas* + Fidelity vs *Δ signal*.  \n"
            "**15 min:** Spatial maps + Line spectra.  \n"
            "**Deep dive:** Load Forecasting → read about TAM in Method docs."
        )

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1 — Metrics table (pivot: one column per method, highlight best)
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    col_l, col_r = st.columns([3, 1])
    with col_l:
        show_obs = st.multiselect(
            "vs ERA5 metrics", OBS_METRICS,
            default=["bias_mae", "std_error", "ks_full", "seasonal_mae", "spectral_mae"],
            format_func=lambda k: METRIC_LABELS.get(k, k),
        )
    with col_r:
        show_gcm = st.multiselect(
            "vs GCM metrics", GCM_METRICS,
            default=["temporal_corr_annual", "annual_anomaly_bias",
                     "Δ signal 2040–2060", "Δ signal 2080–2100"],
            format_func=lambda k: METRIC_LABELS.get(k, k),
        )

    rows = []
    key_list: list[str] = []
    for var in sel_vars:
        for k in show_obs:
            row = {"Variable": var, "Metric": METRIC_LABELS.get(k, k)}
            for m in sel_methods:
                row[mlabel(m)] = raw_metrics[m].get(var, {}).get("vs_obs", {}).get(k, float("nan"))
            rows.append(row)
            key_list.append(k)
        for k in show_gcm:
            row = {"Variable": var, "Metric": METRIC_LABELS.get(k, k)}
            if k in DELTA_PERIOD_MAP:
                # Delta metrics live under _delta in metrics.json
                period = DELTA_PERIOD_MAP[k]
                vals = {m: raw_metrics[m].get("_delta", {}).get(var, {}).get(period, float("nan"))
                        for m in sel_methods}
                # Only add row if at least one method has data
                if not any(np.isfinite(v) for v in vals.values()):
                    continue
                for m in sel_methods:
                    row[mlabel(m)] = vals[m]
            else:
                for m in sel_methods:
                    row[mlabel(m)] = raw_metrics[m].get(var, {}).get("vs_gcm", {}).get(k, float("nan"))
            rows.append(row)
            key_list.append(k)

    if rows:
        df = pd.DataFrame(rows).set_index(["Variable", "Metric"])
        method_cols_present = [mlabel(m) for m in sel_methods if mlabel(m) in df.columns]

        def _style_row(row: pd.Series) -> list[str]:
            pos = list(df.index).index(row.name)
            key = key_list[pos]
            vals = pd.to_numeric(row, errors="coerce")
            is_higher = key in HIGHER_BETTER
            best = vals.idxmax() if is_higher else vals.idxmin()
            styles = [""] * len(row)
            if pd.notna(best) and best in row.index:
                styles[list(row.index).index(best)] = (
                    "background-color: #c6efce; color: #276221; font-weight: bold"
                )
            return styles

        styled = df.style.apply(_style_row, axis=1).format("{:.4f}", na_rep="—")
        st.dataframe(styled, width="stretch")
        st.caption(
            "One column per method — green = best per row. "
            "All metrics lower-is-better except temporal corr (higher = better).  \n"
            "**vs-GCM metrics are measured against the bicubic-interpolated GCM**, so "
            "bicubic scores perfectly on them by construction. Daily temporal correlation is "
            "lowered by any method that injects noise or generates detail; prefer *temporal "
            "corr annual* for signal preservation.  \n"
            "**BCSD**: its empirical quantile map holds values flat outside the 1980–1999 "
            "range, so it distorts the year-to-year evolution and caps the long-term change "
            "(largest annual-anomaly and Δ-signal errors for tas and psl)."
        )
    else:
        st.info("No data for the selected filters.")

    # Delta heatmap visual — drawn from the methods shown above rather than a
    # pre-rendered figure, so it always matches the published method set.
    st.markdown("---")
    _dh_periods = list(DELTA_PERIOD_MAP.items())
    _dh_methods = [m for m in sel_methods if raw_metrics[m].get("_delta")]
    _dh_vars = [v for v in sel_vars
                if any(v in raw_metrics[m]["_delta"] for m in _dh_methods)]
    if _dh_methods and _dh_vars:
        from plotly.subplots import make_subplots
        st.subheader("Delta change scores — heatmap")
        st.caption(
            "Mean absolute error between the method's and the GCM's relative climate change "
            "signal (% change of the future window vs 2000–2020, per grid cell). Lower = better. "
            "For tas (K) and psl (Pa) the % is of an absolute value, so scores are small "
            "(0.1 % of tas ≈ 0.3 K): compare methods within a column, not across columns. "
            "Only methods with predictions through 2100 produce these scores (see Δ signal rows above)."
        )
        _fig_dh = make_subplots(rows=1, cols=len(_dh_periods),
                                subplot_titles=[lbl for lbl, _ in _dh_periods],
                                horizontal_spacing=0.18)
        for _c, (_, _p) in enumerate(_dh_periods, start=1):
            _z = np.array([[raw_metrics[m]["_delta"].get(v, {}).get(_p, np.nan)
                            for v in _dh_vars] for m in _dh_methods], dtype=float)
            _fig_dh.add_trace(go.Heatmap(
                z=_z, x=_dh_vars, y=[mlabel(m) for m in _dh_methods],
                colorscale="YlOrRd", zmin=0,
                zmax=float(np.nanmax(_z)) if np.isfinite(_z).any() else 1.0,
                text=[[f"{x:.2f}" if np.isfinite(x) else "" for x in row] for row in _z],
                texttemplate="%{text}",
                colorbar=dict(x=0.41 if _c == 1 else 1.0, len=0.9, thickness=12),
                hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>",
            ), row=1, col=_c)
        _fig_dh.update_yaxes(autorange="reversed")
        _fig_dh.update_layout(height=max(300, 45 * len(_dh_methods) + 120),
                               margin=dict(t=60, b=40))
        st.plotly_chart(_fig_dh, width="stretch", key="dh_plotly")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Radar
# ══════════════════════════════════════════════════════════════════════════════
with tab_radar:
    st.subheader("Radar plot")
    radar_mode = st.radio(
        "Mode",
        ["All variables combined", "By variable"],
        horizontal=True,
        key="radar_mode",
    )
    
    radar_obs_keys = st.multiselect(
        "vs ERA5 metrics on radar", OBS_METRICS + ["intervar_corr"],
        default=OBS_METRICS + ["intervar_corr"],
        format_func=lambda k: METRIC_LABELS.get(k, k),
        key="radar_obs",
    )
    radar_gcm_keys = st.multiselect(
        "vs GCM metrics on radar", GCM_METRICS,
        default=GCM_METRICS,
        format_func=lambda k: METRIC_LABELS.get(k, k),
        key="radar_gcm",
    )
    radar_keys = radar_obs_keys + radar_gcm_keys

    if len(radar_keys) < 3:
        st.info("Select at least 3 metrics for a radar plot.")
    else:
        def _get_val(method: str, key: str, variable: str | None = None) -> float:
            if key == "intervar_corr":
                if variable is not None:
                    return float("nan")
                iv = raw_metrics[method].get("_intervar", {})
                vals = [v for v in iv.values() if np.isfinite(v)]
                return float(np.nanmean(vals)) if vals else float("nan")
            if key in DELTA_PERIOD_MAP:
                if variable is None:
                    # Average across all selected variables for delta metrics
                    period = DELTA_PERIOD_MAP[key]
                    delta_vals = []
                    for v in sel_vars:
                        val = raw_metrics[method].get("_delta", {}).get(v, {}).get(period, float("nan"))
                        if np.isfinite(val):
                            delta_vals.append(val)
                    return float(np.nanmean(delta_vals)) if delta_vals else float("nan")
                period = DELTA_PERIOD_MAP[key]
                return raw_metrics[method].get("_delta", {}).get(variable, {}).get(period, float("nan"))
            axis = "vs_obs" if key in OBS_METRICS else "vs_gcm"
            if variable is None:
                val = _agg(method, key, axis)
                return (1 - val) if key in HIGHER_BETTER else val
            v = raw_metrics[method].get(variable, {}).get(axis, {}).get(key, float("nan"))
            return (1 - v) if key in HIGHER_BETTER else v

        if radar_mode == "All variables combined":
            norm_base = {}
            for k in radar_keys:
                vals = [_get_val(m, k) for m in sel_methods]
                finite = [v for v in vals if np.isfinite(v)]
                norm_base[k] = float(np.nanmean(finite)) if finite else 1.0
                if norm_base[k] == 0.0:
                    norm_base[k] = 1.0           # avoid div-by-zero → NaN axis

            fig = go.Figure()
            theta = [METRIC_LABELS.get(k, k) for k in radar_keys]
            for i, m in enumerate(sel_methods):
                r_vals = [
                    _get_val(m, k) / norm_base[k] if norm_base[k] else float("nan")
                    for k in radar_keys
                ]
                fig.add_trace(go.Scatterpolar(
                    r=r_vals + [r_vals[0]],
                    theta=theta + [theta[0]],
                    fill="toself", opacity=0.5, name=mlabel(m),
                    line=dict(color=_method_color(m), width=2),
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                title="Normalised metrics (all vars) — lower = better (1.0 = mean)",
                height=520,
                legend=dict(orientation="h", yanchor="bottom", y=-0.18),
            )
            st.plotly_chart(fig, width="stretch", key="radar_all_vars")
            st.caption("Each axis normalised by mean across methods. temporal_corr* inverted (1−r).")

        else:  # By variable
            per_var_keys = [k for k in radar_keys if k != "intervar_corr"]
            if len(per_var_keys) < 3:
                st.info("Select at least 3 per-variable metrics (inter-var corr is not per-variable).")
            if len(per_var_keys) >= 3:
                for var in all_vars:
                    if var not in sel_vars:
                        continue
                    norm_base = {}
                    for k in per_var_keys:
                        vals = [_get_val(m, k, var) for m in sel_methods]
                        finite = [v for v in vals if np.isfinite(v)]
                        norm_base[k] = float(np.nanmean(finite)) if finite else 1.0
                        if norm_base[k] == 0.0:
                            norm_base[k] = 1.0        # avoid div-by-zero → NaN axis
    
                    fig = go.Figure()
                    theta = [METRIC_LABELS.get(k, k) for k in per_var_keys]
                    for i, m in enumerate(sel_methods):
                        r_vals = [
                            _get_val(m, k, var) / norm_base[k] if norm_base[k] else float("nan")
                            for k in per_var_keys
                        ]
                        fig.add_trace(go.Scatterpolar(
                            r=r_vals + [r_vals[0]],
                            theta=theta + [theta[0]],
                            fill="toself", opacity=0.5, name=mlabel(m),
                            line=dict(color=_method_color(m), width=2),
                        ))

                    # One chart per variable, after all methods are added: drawing it
                    # inside the method loop re-used key radar_{var} and crashed.
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                        title=f"Normalised metrics — {var}",
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    )
                    st.plotly_chart(fig, width="stretch", key=f"radar_{var}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Line plots (plotly, all methods on one figure)
# ══════════════════════════════════════════════════════════════════════════════
with tab_lines:
    has_plot_data = bool(plot_data)

    line_type = st.selectbox(
        "Plot type", list(LINE_PLOT_TYPES), key="ln_type",
    )
    sel_var_ln = st.selectbox("Variable", all_vars, key="ln_var")

    # Plain-language explainer for the selected line plot (a few sentences each),
    # shown above the figure so the reader knows what is being plotted and in
    # which unit. Text lives in plot_meta so it is unit-tested.
    st.caption(line_help(sel_var_ln, line_type))

    if has_plot_data:
        fig = go.Figure()

        if line_type == "seasonal":
            era5_done = False
            for m in sel_methods:
                d = plot_data.get(m, {}).get(sel_var_ln, {}).get("seasonal")
                if not d:
                    continue
                if not era5_done:
                    fig.add_trace(go.Scatter(
                        x=d["months"], y=d["era5"],
                        name="ERA5", line=dict(color="black", width=2.5),
                    ))
                    era5_done = True
                fig.add_trace(go.Scatter(
                    x=d["months"], y=d["pred"],
                    name=mlabel(m), line=dict(color=_method_color(m), width=2),
                ))
            fig.update_layout(
                title=f"Seasonal cycle — {vlabel(sel_var_ln)}",
                xaxis=dict(title="Month", tickvals=list(range(1, 13)),
                           ticktext=list("JFMAMJJASOND")),
                yaxis_title=vlabel(sel_var_ln), height=430,
                legend=dict(orientation="h", yanchor="bottom", y=-0.28),
            )

        elif line_type == "doy":
            # Daily climatology per calendar day (365 bins, 29 Feb removed):
            # region-wide mean or a single grid point (energy sites), with an
            # optional day-to-day change panel to spot month-boundary jumps.
            from energy_views import site_catalog
            from plotly.subplots import make_subplots

            _doy_by_m = {
                m: (plot_data.get(m, {}).get(sel_var_ln) or {}).get("doy") or {}
                for m in sel_methods
            }
            _avail = {k for d in _doy_by_m.values() for k in d}
            _cat = site_catalog(_selected)
            _sites = (
                (["global"] if "global" in _avail else [])
                + [k for k in _cat if k in _avail and k != "global"]
                + sorted(k for k in _avail if k != "global" and k not in _cat)
            )

            if not _sites:
                if DEV_NOTES:
                    st.info(
                        f"No day-of-year data for {sel_var_ln} yet — run "
                        "`python src/evaluation/backfill_energy.py` or re-run evaluate.py."
                    )
                else:
                    st.info(f"No day-of-year data for {sel_var_ln} in this selection.")
            else:
                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    _mode = st.radio("Spatial reduction", ["Region mean", "Site"],
                                     horizontal=True, key="ln_doy_mode")
                _site_opts = [k for k in _sites if k != "global"] or _sites
                with c2:
                    if _mode == "Site":
                        sel_site = st.selectbox(
                            "Site", _site_opts, key="ln_doy_site",
                            format_func=lambda k: (
                                f"{_cat[k]['label']} ({_cat[k]['lat']:.2f}°, "
                                f"{_cat[k]['lon']:.2f}°)" if k in _cat else k),
                        )
                    else:
                        sel_site = "global" if "global" in _avail else _sites[0]
                with c3:
                    show_diff = st.checkbox("Show day-to-day change |Δ|",
                                            value=True, key="ln_doy_diff")

                rows = 2 if show_diff else 1
                fig = make_subplots(
                    rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                    row_heights=[0.64, 0.36] if rows == 2 else [1.0],
                    subplot_titles=(["", "Day-to-day change |Δ|"] if rows == 2 else ()),
                )
                x = list(range(1, 366))
                era5_done = gcm_done = False
                for m in sel_methods:
                    d = _doy_by_m[m].get(sel_site)
                    if not d:
                        continue
                    for kind, vals, color, width, dash in (
                        ("era5", d.get("era5"), "black", 2.5, None),
                        ("gcm",  d.get("gcm"),  "#999999", 2, "dash"),
                        ("pred", d.get("pred"), _method_color(m), 2, None),
                    ):
                        if not vals:
                            continue
                        if kind == "era5" and era5_done:
                            continue
                        if kind == "gcm" and gcm_done:
                            continue
                        name = {"era5": "ERA5", "gcm": "GCM (regridded)"}.get(kind, mlabel(m))
                        line = dict(color=color, width=width)
                        if dash:
                            line["dash"] = dash
                        # Flat legend: one independent group per baseline so a
                        # legend click toggles only that method (raw + delta).
                        group = kind if kind != "pred" else f"pred::{m}"
                        fig.add_trace(go.Scatter(x=x, y=vals, name=name, line=line,
                                                 legendgroup=group), row=1, col=1)
                        if show_diff:
                            delta = np.abs(np.diff(np.asarray(vals, dtype=float)))
                            fig.add_trace(go.Scatter(x=x[1:], y=delta.tolist(), name=name,
                                                     line=line, legendgroup=group,
                                                     showlegend=False,
                                                     hoverinfo="skip"), row=2, col=1)
                        if kind == "era5":
                            era5_done = True
                        if kind == "gcm":
                            gcm_done = True

                _mid = [int(s + n // 2) for s, n in zip(
                    np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]),
                    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])]
                _site_lbl = (
                    f" — {_cat[sel_site]['label']}" if sel_site in _cat
                    else f" — region mean" if sel_site == "global"
                    else f" — {sel_site}"
                )
                fig.update_xaxes(tickmode="array",
                                 tickvals=[i + 1 for i in _mid],
                                 ticktext=list("JFMAMJJASOND"))
                fig.update_xaxes(title_text="", row=1, col=1)
                fig.update_xaxes(title_text="Day of year", row=rows, col=1)
                fig.update_yaxes(title_text=vlabel(sel_var_ln), row=1, col=1)
                if show_diff:
                    fig.update_yaxes(
                        title_text=f"|Δ| ({VAR_UNITS.get(sel_var_ln, 'units')}) / day",
                        row=2, col=1)
                    fig.update_layout(
                        title=f"Day-of-year cycle — {vlabel(sel_var_ln)}{_site_lbl}",
                        height=620,
                        # Keep the legend on top: a wrapped legend below the
                        # figure stacks over the small delta panel.
                        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                                    xanchor="left", x=0),
                        margin=dict(t=130),
                        hovermode="x unified",
                    )
                else:
                    fig.update_layout(
                        title=f"Day-of-year cycle — {vlabel(sel_var_ln)}{_site_lbl}",
                        height=430,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
                        hovermode="x unified",
                    )
                
                # Add vertical lines for month boundaries
                for i, x_pos in enumerate(_mid):
                    fig.add_vline(x=x_pos, line_width=0.5, line_dash="dot", 
                                line_color="gray", row=1, col=1)
                    if show_diff:
                        fig.add_vline(x=x_pos, line_width=0.5, line_dash="dot", 
                                    line_color="gray", row=2, col=1)

        elif line_type == "annual":
            gcm_done = False
            era5_done = False
            for m in sel_methods:
                d = plot_data.get(m, {}).get(sel_var_ln, {}).get("annual")
                if not d:
                    continue
                if not gcm_done:
                    fig.add_trace(go.Scatter(
                        x=d["years"], y=d["gcm"],
                        name="GCM", line=dict(color="black", width=2.5),
                    ))
                    gcm_done = True
                if not era5_done and "era5" in d:
                    fig.add_trace(go.Scatter(
                        x=d.get("years_era5", d["years"]), y=d["era5"],
                        name="ERA5", line=dict(color="#888888", width=2, dash="dot"),
                    ))
                    era5_done = True
                years_pred = d.get("years_pred", d["years"])
                fig.add_trace(go.Scatter(
                    x=years_pred, y=d["pred"],
                    name=mlabel(m),
                    line=dict(color=_method_color(m), width=2, dash="dash"),
                ))
            fig.update_layout(
                title=f"Annual means — {vlabel(sel_var_ln)}",
                xaxis_title="Year", yaxis_title=vlabel(sel_var_ln), height=430,
                legend=dict(orientation="h", yanchor="bottom", y=-0.28),
            )

        elif line_type == "cdf":
            era5_done = False
            for m in sel_methods:
                d = plot_data.get(m, {}).get(sel_var_ln, {}).get("cdf")
                if not d:
                    continue
                n_e, n_p = len(d["era5_x"]), len(d["pred_x"])
                if not era5_done:
                    fig.add_trace(go.Scatter(
                        x=d["era5_x"], y=np.linspace(0, 1, n_e).tolist(),
                        name="ERA5", line=dict(color="black", width=2.5),
                    ))
                    era5_done = True
                fig.add_trace(go.Scatter(
                    x=d["pred_x"], y=np.linspace(0, 1, n_p).tolist(),
                    name=mlabel(m), line=dict(color=_method_color(m), width=2),
                ))
            fig.update_layout(
                title=f"CDF — {vlabel(sel_var_ln)}",
                xaxis_title=vlabel(sel_var_ln),
                yaxis_title="Cumulative probability (0–1)",
                height=430,
                legend=dict(orientation="h", yanchor="bottom", y=-0.28),
            )

        elif line_type == "spectra":
            era5_done = False
            for m in sel_methods:
                sp = plot_data.get(m, {}).get(sel_var_ln, {}).get("spectra")
                if not sp or "k_centers_km" not in sp:
                    continue
                k_km = sp["k_centers_km"]
                if not era5_done:
                    fig.add_trace(go.Scatter(
                        x=k_km, y=sp["era5"],
                        name="ERA5", line=dict(color="black", width=2.5),
                    ))
                    era5_done = True
                fig.add_trace(go.Scatter(
                    x=k_km, y=sp["pred"],
                    name=mlabel(m), line=dict(color=_method_color(m), width=2),
                ))
            fig.update_layout(
                title=f"Isotropic spatial power spectrum — {vlabel(sel_var_ln)}",
                xaxis=dict(title="Spatial scale (km)", type="log", autorange="reversed"),
                yaxis=dict(title="PSD (a.u.)", type="log"),
                height=430,
                legend=dict(orientation="h", yanchor="bottom", y=-0.28),
            )

        if fig.data:
            st.plotly_chart(fig, width="stretch", key=f"lineplot_{sel_var_ln}_{line_type}")
        elif line_type != "doy":   # doy already shows its own hint above
            st.info(f"No plot data for {sel_var_ln} / {line_type}." + (" Re-run evaluate.py." if DEV_NOTES else ""))

    else:
        if DEV_NOTES:
            st.info(
                "**Interactive plots not yet available.**  \n"
                "Re-run `evaluate.py` with the updated script to generate `plot_data.json`. "
                "Showing static per-method PNGs for now."
            )
        else:
            st.info("Showing static per-method PNGs.")
        type_to_prefix = {"seasonal": "seasonal", "annual": "annual", "cdf": "cdf"}
        prefix = type_to_prefix.get(line_type)
        if prefix:
            cols_per_row = 2
            for i in range(0, len(sel_methods), cols_per_row):
                row_methods = sel_methods[i:i + cols_per_row]
                img_cols = st.columns(cols_per_row)
                for col, m in zip(img_cols, row_methods):
                    with col:
                        st.markdown(f"**{mlabel(m)}**")
                        img = RESULTS_DIR / m / "plots" / sel_var_ln / f"{prefix}_{sel_var_ln}.png"
                        if img.exists():
                            st.image(str(img))
                        else:
                            st.caption("No image")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Spatial maps
# ══════════════════════════════════════════════════════════════════════════════
with tab_maps:
    col1, col2 = st.columns(2)
    with col1:
        plot_type = st.selectbox(
            "Plot type",
            list(MAP_PLOT_TYPES),
            format_func=lambda x: {
                "bias_map": "Bias map",
                "temporal_corr": "Temporal correlation",
                "autocorr": "Autocorrelation",
                "samples": "Sample days",
                "delta_2040_2060": "Δ climate signal 2040–2060",
                "delta_2080_2100": "Δ climate signal 2080–2100",
            }.get(x, x),
            key="map_type",
        )
    with col2:
        sel_var_map = st.selectbox("Variable", all_vars, key="mv")

    # Caption and colorbar label (unit-aware) come from plot_meta — unit-tested.
    st.caption(map_help(sel_var_map, plot_type))
    _cb_label = colorbar_label(sel_var_map, plot_type)

    # ── Shared-colorbar spatial maps ───────────────────────────────────────
    # Render each map from the per-method npz data with one color scale shared
    # across all selected methods (scale may still differ between variables).

    def _load_map_data(m: str, variable: str, ptype: str) -> dict | None:
        m_dir = RESULTS_DIR / m / "plots" / variable
        npz_map = {
            "bias_map":        f"bias_data_{variable}.npz",
            "temporal_corr":   f"temporal_corr_data_{variable}.npz",
            "autocorr":        f"autocorr_data_{variable}.npz",
            "samples":         f"samples_data_{variable}.npz",
            "delta_2040_2060": f"delta_data_{variable}_2040_2060.npz",
            "delta_2080_2100": f"delta_data_{variable}_2080_2100.npz",
        }
        field_map = {
            "bias_map":        "mean_bias",
            "temporal_corr":   "corr",
            "autocorr":        ("ac_pred", "ac_gcm"),  # bias = ac_pred − ac_gcm
            "delta_2040_2060": "pred_delta",
            "delta_2080_2100": "pred_delta",
        }
        f = m_dir / npz_map[ptype]
        if not f.exists():
            return None
        try:
            d = np.load(f, allow_pickle=True)
            if ptype == "samples":
                return {
                    "pred":  d["pred"], "era5": d["era5"], "gcm": d["gcm"],
                    "dates": list(d["dates"]),
                    "lat":   d["lat"], "lon":  d["lon"],
                    "gcm_delta": None,
                }
            spec = field_map[ptype]
            if isinstance(spec, tuple):
                arr = d[spec[0]] - d[spec[1]]   # bias toward GCM
            else:
                arr = d[spec]
            out = {"arr": arr, "lat": d["lat"], "lon": d["lon"], "gcm": None}
            if ptype in ("delta_2040_2060", "delta_2080_2100") and "gcm_delta" in d.files:
                out["gcm"] = d["gcm_delta"]     # GCM reference overlay
            return out
        except (KeyError, OSError, ValueError):
            return None

    maps_data = {m: _load_map_data(m, sel_var_map, plot_type) for m in sel_methods}
    if plot_type == "samples":
        maps_data = {m: d for m, d in maps_data.items()
                     if d is not None and d["pred"] is not None and d["gcm"] is not None}
    else:
        maps_data = {m: d for m, d in maps_data.items() if d is not None and d["arr"] is not None}

    if not maps_data:
        st.info(f"No map data available for {sel_var_map} / {plot_type}.")
    elif plot_type == "samples":
        # ── Sample days: one grid per variable. ─────────────────────────────
        # Rows = GCM, ERA5, then each method; columns = the 5 sample dates.
        # Same dates across methods (generated with a variable-seeded RNG), so
        # the daily fields can be compared column-wise under one shared scale.
        def _render_samples_grid(mdata):
            import matplotlib.pyplot as plt
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
            except ImportError:
                ccrs, cfeature = None, None

            ref = next(iter(mdata.values()))
            dates = list(ref["dates"])
            n_days = len(dates)
            rows = [("GCM", ref["gcm"]), ("ERA5", ref["era5"])]
            for m in mdata:
                rows.append((mlabel(m), mdata[m]["pred"]))

            all_vals = np.concatenate([np.ravel(a) for _, a in rows if a is not None])
            finite = all_vals[np.isfinite(all_vals)]
            lo = float(np.percentile(finite, 1.0)) if finite.size else 0.0
            hi = float(np.percentile(finite, 99.0)) if finite.size else lo + 1.0
            if hi - lo < 1e-12:
                hi = lo + 1.0

            proj = ccrs.PlateCarree() if ccrs is not None else None
            fig, axes = plt.subplots(
                len(rows), n_days, figsize=(1.9 * n_days + 1.2, 1.7 * len(rows) + 1.3),
                subplot_kw={"projection": proj} if proj else {},
                squeeze=False, layout="constrained",
            )
            tr = ccrs.PlateCarree() if ccrs is not None else None
            im = None
            for r, (label, arr) in enumerate(rows):
                for c, dte in enumerate(dates):
                    ax = axes[r, c]
                    if arr is None:
                        ax.axis("off")
                        continue
                    z = np.asarray(arr[c], dtype=float)
                    im = ax.pcolormesh(ref["lon"], ref["lat"], z, cmap="viridis",
                                       vmin=lo, vmax=hi,
                                       **({"transform": tr} if tr is not None else {}))
                    if ccrs is not None:
                        ax.coastlines(linewidth=0.4)
                        ax.set_extent([ref["lon"].min(), ref["lon"].max(),
                                       ref["lat"].min(), ref["lat"].max()],
                                      crs=ccrs.PlateCarree())
                    ax.set_title(f"{dte}\n{label}", fontsize=7)
            if im is not None:
                ax_flat = [a for row in axes.tolist() for a in row]
                fig.colorbar(im, ax=ax_flat, shrink=0.6, pad=0.02,
                             label=_cb_label, format="%.3g")
            st.pyplot(fig)
            plt.close(fig)

        _render_samples_grid(maps_data)
        st.caption("Same 5 dates for every row. Rows: GCM input, ERA5 reference, "
                   f"per-method prediction. Dates shown in each panel title "
                   f"({vlabel(sel_var_map)}).")
    else:
        # Shared color scale across all selected methods (per variable), including
        # the GCM reference so the Δ maps are directly comparable with it.
        if plot_type in ("delta_2040_2060", "delta_2080_2100") and maps_data:
            ref = next(iter(maps_data.values()))
            if ref["gcm"] is not None:
                maps_data = {"GCM": {"arr": ref["gcm"], "lat": ref["lat"],
                                     "lon": ref["lon"]}, **maps_data}
        stacked = [d["arr"] for d in maps_data.values()]
        all_arr = np.stack(stacked)
        finite  = all_arr[np.isfinite(all_arr)]
        if plot_type in ("bias_map", "autocorr", "delta_2040_2060", "delta_2080_2100"):
            vmax    = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
            vmax    = max(vmax, 1e-9)
            zmin, zmax, cmap, zmid = -vmax, vmax, "RdBu_r", 0.0
        else:  # temporal_corr: scale driven by min/max across methods
            if finite.size:
                lo, hi = float(np.min(finite)), float(np.max(finite))
            else:
                lo, hi = 0.0, 1.0
            if hi - lo < 1e-12:
                hi = lo + 1.0
            zmin, zmax, cmap, zmid = lo, hi, "RdYlGn", None

        def _render_cartopy_row(row_methods):
            import matplotlib.pyplot as plt
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
            except ImportError:
                ccrs, cfeature = None, None

            def _add_geo(ax, lon, lat):
                if ccrs is not None:
                    ax.coastlines(linewidth=0.6)
                    ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
                    ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.5, linestyle=":")
                    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()],
                                  crs=ccrs.PlateCarree())

            # Fixed 2-slot row so every map has the same panel size; an unpaired
            # trailing map keeps slot width (second slot just hidden).
            n_slots = 2
            proj = ccrs.PlateCarree() if ccrs is not None else None
            fig, axes = plt.subplots(
                1, n_slots, figsize=(5.2 * n_slots + 0.8, 4.0),
                subplot_kw={"projection": proj} if proj else {},
                squeeze=False, layout="none",
            )
            fig.subplots_adjust(wspace=0.1)
            if len(row_methods) < n_slots:
                axes[0, len(row_methods)].axis("off")
            tr = ccrs.PlateCarree() if ccrs is not None else None
            im = None
            shown = []
            for ax, m in zip(axes[0], row_methods):
                d = maps_data[m]
                z = np.asarray(d["arr"], dtype=float)
                if not np.isfinite(z).all():
                    z = np.where(np.isfinite(z), z, np.nan)
                im = ax.pcolormesh(d["lon"], d["lat"], z, cmap=cmap,
                                   vmin=zmin, vmax=zmax,
                                   **({"transform": tr} if tr is not None else {}))
                _add_geo(ax, d["lon"], d["lat"])
                ax.set_title(mlabel(m), fontsize=10)
                shown.append(ax)
            if im is not None:
                fig.colorbar(im, ax=shown, shrink=0.85, pad=0.02,
                             label=_cb_label, format="%.3g")
            st.pyplot(fig)
            plt.close(fig)

        methods = list(maps_data.keys())
        for i in range(0, len(methods), 2):
            _render_cartopy_row(methods[i:i + 2])

        if plot_type in ("delta_2040_2060", "delta_2080_2100"):
            st.caption("Δ signal = future − historical. Leftmost panel: GCM Δ reference. "
                       "Other panels: per-method Δ signal.")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 5 — Inter-variable correlations
# ══════════════════════════════════════════════════════════════════════════════
with tab_intervar:
    st.subheader("Inter-variable correlation MAE (pred vs ERA5)")
    st.caption(
        "Mean absolute error of pointwise temporal correlation between variable pairs. "
        "Lower = better preservation of observed cross-variable dependencies. "
        "Units: correlation coefficient (unitless, 0 = identical dependencies; "
        "a perfect miss on an anti-correlated pair would be 2)."
    )

    # Scalar table (one column per method)
    intervar_rows = []
    for m in sel_methods:
        iv = raw_metrics[m].get("_intervar", {})
        if iv:
            intervar_rows.append({"Method": mlabel(m), **{k: round(v, 4) for k, v in iv.items()}})

    if intervar_rows:
        iv_df = pd.DataFrame(intervar_rows).set_index("Method")

        def _iv_style_col(col: pd.Series) -> list[str]:
            best = col.idxmin()
            return [
                "background-color: #c6efce; color: #276221; font-weight: bold"
                if idx == best else ""
                for idx in col.index
            ]

        st.dataframe(
            iv_df.style.apply(_iv_style_col, axis=0).format("{:.4f}"),
            width="stretch",
        )

        # Bar chart
        pairs = list(iv_df.columns)
        fig_iv = go.Figure()
        for m in sel_methods:
            lbl = mlabel(m)
            if lbl in iv_df.index:
                fig_iv.add_trace(go.Bar(
                    name=lbl, x=pairs,
                    y=[float(iv_df.loc[lbl, p]) if p in iv_df.columns else float("nan")
                       for p in pairs],
                    marker_color=_method_color(m),
                ))
        fig_iv.update_layout(
            barmode="group",
            title="Inter-variable correlation MAE by pair",
            yaxis_title="MAE of temporal correlation (unitless)",
            height=370,
            legend=dict(orientation="h", yanchor="bottom", y=-0.35),
        )
        st.plotly_chart(fig_iv, width="stretch", key="intervar_plotly")
    else:
        st.info("No inter-variable scalar metrics found for this selection" + (" — run evaluate.py first." if DEV_NOTES else "."))

    st.markdown("---")
    st.subheader("Pointwise correlation maps")
    st.caption(
        "Spatial maps of per-grid-point temporal correlation between variable pairs, "
        "where each panel shows one variable pair and distinguishes the ERA5 reference "
        "from the prediction. Separate ERA5 / prediction / bias panels are shown where "
        "available. Units: correlation coefficient (unitless, −1 to +1; 1 = perfectly "
        "in phase)."
    )

    # Discover available pairs from per-method PNGs
    all_pairs: list[str] = []
    for m in all_method_dirs:
        pdir = RESULTS_DIR / m / "plots"
        for p in sorted(pdir.glob("intervar_corr_*_*.png")):
            parts = p.stem.split("_")
            if len(parts) >= 4:
                pair = f"{parts[2]}_{parts[3]}"
                if pair not in all_pairs:
                    all_pairs.append(pair)

    if not all_pairs:
        st.info("No inter-variable correlation maps found for this selection" + (" — run evaluate.py first." if DEV_NOTES else "."))
    else:
        sel_pair = st.selectbox("Variable pair", all_pairs, key="iv_pair")
        v1, v2 = sel_pair.split("_", 1)

        st.markdown("### ERA5 Reference (Ground Truth)")
        for m in all_method_dirs:
            era5_img = RESULTS_DIR / m / "plots" / f"intervar_corr_{v1}_{v2}_era5.png"
            if era5_img.exists():
                col_ref1, col_ref2, col_ref3 = st.columns([1, 2, 1])
                with col_ref2:
                    st.image(str(era5_img), width="stretch")
                break

        st.markdown("---")
        st.subheader("Model Predictions")

        cols_per_row = 2
        for i in range(0, len(sel_methods), cols_per_row):
            row_methods = sel_methods[i:i + cols_per_row]
            # Fixed 2-slot row so every map keeps the same width; a lone trailing
            # map stays half-width (second slot just stays empty).
            cols = st.columns(cols_per_row)
            for col, m in zip(cols, row_methods):
                with col:
                    st.markdown(f"**{mlabel(m)}**")
                    img = RESULTS_DIR / m / "plots" / f"intervar_corr_{v1}_{v2}.png"
                    if img.exists():
                        st.image(str(img), width="stretch")
                    else:
                        st.info("No image available")

    # ── Energy events (Dunkelflaute / cold–calm spells) ──────────────────────
    st.markdown("---")
    st.subheader("Energy events — Dunkelflaute & cold–calm spells")
    st.caption(
        "Days when the region mean is low in wind AND low in solar (Dunkelflaute), "
        "or cold AND calm (heating demand meets weak wind generation). "
        "Thresholds are the 20th percentile of the ERA5 daily region mean, applied "
        "unchanged to the GCM and to every downscaled product so the three are "
        "classified with the same absolute cutoffs; a spell counts as an event "
        "from 2 consecutive days on."
    )
    _ev_pairs = [(m, (plot_data.get(m) or {}).get("_events")) for m in sel_methods]
    _ev_pairs = [(m, e) for m, e in _ev_pairs if e]
    if not _ev_pairs:
        if DEV_NOTES:
            st.info(
                "No energy-event data yet — run "
                "`python src/evaluation/backfill_energy.py` or re-run evaluate.py."
            )
        else:
            st.info("No energy-event data for this selection.")
    else:
        _base = _ev_pairs[0][1]
        _meta = _base.get("meta", {})
        _thr = _meta.get("thresholds", {})
        _per = _meta.get("period")
        st.caption(
            f"Thresholds (ERA5 region mean): wind < **{_thr.get('sfcWind', float('nan')):.2f} m/s**, "
            f"solar < **{_thr.get('rsds', float('nan')):.1f} W/m²**, "
            f"tas < **{_thr.get('tas', float('nan')):.1f} K**"
            + (f" · period **{_per[0]}–{_per[1]}**" if _per else "")
            + f" · minimum spell length **{_meta.get('min_days', 2)} days**"
        )
        _cats = ["ERA5", "GCM"] + [mlabel(m) for m, _ in _ev_pairs]
        _colors = ["black", "#999999"] + [_method_color(m) for m, _ in _ev_pairs]
        st.caption(
            "Click a legend entry to show or hide that baseline; "
            "ERA5 and the raw GCM toggle the same way."
        )

        def _ev_vals(event: str, stat: str) -> list:
            base = _base.get(event, {}).get(stat, {})
            return [base.get("era5"), base.get("gcm")] + [
                e.get(event, {}).get(stat, {}).get("pred") for _, e in _ev_pairs]

        _panels = [
            ("dunkelflaute",  "count",         "Dunkelflaute — number of events"),
            ("dunkelflaute",  "mean_duration", "Dunkelflaute — mean duration (days)"),
            ("cold_low_wind", "count",         "Cold & calm — number of events"),
            ("cold_low_wind", "mean_duration", "Cold & calm — mean duration (days)"),
        ]
        _cols = st.columns(2)
        for _i, (_evname, _stat, _title) in enumerate(_panels):
            with _cols[_i % 2]:
                _vals = [v if v is not None else np.nan
                         for v in _ev_vals(_evname, _stat)]
                _f = go.Figure(data=[
                    go.Bar(name=cat, x=[cat], y=[val], marker_color=color)
                    for cat, val, color in zip(_cats, _vals, _colors)
                ])
                _f.update_layout(
                    title=_title,
                    yaxis_title="Events" if _stat == "count" else "Days",
                    height=380, margin=dict(t=50, b=140, l=10, r=10),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.30,
                                font=dict(size=11)),
                )
                st.plotly_chart(_f, width="stretch", key=f"event_{_evname}_{_stat}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 6 — Load Forecasting (TAM). Region-aware: france has the full set of
# spatial-representation fixture files; us_central currently ships the
# full-grid-PCA lens only (net_lens_region.py output).
# ══════════════════════════════════════════════════════════════════════════════
if tab_load is not None:
    with tab_load:
        st.subheader("Evaluating Climate Downscaling Through a Net-Load Lens (TAM)")
        st.caption(
            "Downstream evaluation of climate downscaling via a fixed Time series Additive Model (TAM) "
            "forecasting daily net load: NetDemand = load − wind − solar. "
            "The model is trained on ERA5 reanalysis (2012–2021, restricted to days with observed net demand — from July 2018 for the central US) and applied unchanged to each downscaled scenario (2000–2022)."
        )

        tam_dir = TAM_DIR
        _tam_gcm_native = _selected_gcm == NATIVE_GCM.get(_selected)
        # Match the suffix emitted by net_lens_region.py: native coupling keeps
        # the legacy region-only files; factorial couplings get _region_gcm.
        region_sfx = "" if (_selected == "france" and _tam_gcm_native) else f"_{_selected}"
        if not _tam_gcm_native:
            region_sfx = f"_{_selected}_{_selected_gcm}"
        rep_opts = ["Full Grid PCA", "Regional Barycenters", "Regional + Renewables"]
        if _selected != "france":
            rep_opts = ["Full Grid PCA"]
        rep_choice = st.selectbox(
            "Spatial Representation",
            rep_opts,
            key="tam_rep",
        )

        json_map = {
            "Full Grid PCA": (f"metrics_results{region_sfx}.json",
                              f"cdf_results{region_sfx}.npz"),
            "Regional Barycenters": (f"metrics_regional{region_sfx}.json",
                                     f"cdf_regional{region_sfx}.npz"),
            "Regional + Renewables": (f"metrics_regional_full{region_sfx}.json",
                                      f"cdf_regional_full{region_sfx}.npz"),
        }

        jpath = tam_dir / json_map[rep_choice][0]
        cdf_path = tam_dir / json_map[rep_choice][1]
        if not jpath.exists() and _selected != "france":
            jpath = tam_dir / f"metrics_results{region_sfx}.json"
            cdf_path = tam_dir / f"cdf_results{region_sfx}.npz"

        def _tam_label(k: str) -> str:
            key = k.strip().lower()
            if key in METHOD_LABELS:
                return METHOD_LABELS[key]
            return {"era5": "ERA5", "ipsl": "IPSL-CM6A-LR (raw)",
                    "cesm2": "CESM2 (raw)", "ipsl_cm6a_lr": "IPSL-CM6A-LR (raw)",
                    "cnrm_cm6_1": "CNRM-CM6-1 (raw)"}.get(key, key)
    
        if jpath.exists():
            with open(jpath) as f:
                tam_data = json.load(f)
            # The lens scores every local result dir; keep unpublished methods out.
            tam_data = {k: v for k, v in tam_data.items() if not is_unpublished(k)}
            for _bk in ("by_season", "by_wind"):
                if isinstance(tam_data.get(_bk), dict):
                    tam_data[_bk] = {k: v for k, v in tam_data[_bk].items()
                                     if not is_unpublished(k)}

            # — Load forecasting results (ERA5 → observed): RMSE of the fixed TAM
            # trained on ERA5 weather and scored against observed net demand.
            # The observed record starts in July 2018 for the central US.
            _obs0 = "2018-07" if _selected == "us_central" else "2012"
            rmse_keys = {}
            for _sfx, _tag in [
                ("_train", f"Train ({_obs0}–2021)"),
                ("_2012_2022", f"All ({_obs0}–2022)"),
                ("_2022", "Hold-out 2022"),
            ]:
                for _data_k, _data_v in tam_data.items():
                    if _data_k.endswith(_sfx) and isinstance(_data_v, (int, float)):
                        rmse_keys[_tag] = float(_data_v)
            if rmse_keys:
                st.markdown(
                    "#### Load Forecasting Accuracy — ERA5 → observed (daily net demand RMSE, MW)")
                _cols = st.columns(len(rmse_keys))
                for _c, (_tag, _val) in zip(_cols, rmse_keys.items()):
                    _c.metric(_tag, f"{_val:,.0f} MW")
                st.caption(
                    "The same TAM (excluding downscaled scenarios) forecasts observed "
                    "net demand from ERA5 weather only — this is the 'ceiling' accuracy "
                    "of the fixed downstream model. Distributional metrics below compare each "
                    "downscaled scenario against this ERA5-reference forecast."
                )
                st.markdown("---")

            # — TAM forecasts vs ERA5: daily net-demand forecast series (2000–2022)
            # driven by each weather source through the same fixed downstream TAM.
            fc_map = {
                "Full Grid PCA": f"tam_forecasts_results{region_sfx}.npz",
                "Regional Barycenters": f"tam_forecasts_regional{region_sfx}.npz",
                "Regional + Renewables": f"tam_forecasts_regional_full{region_sfx}.npz",
            }
            fc_path = tam_dir / fc_map[rep_choice]
            if not fc_path.exists() and _selected != "france":
                fc_path = tam_dir / f"tam_forecasts_results{region_sfx}.npz"
            if fc_path.exists():
                _fc = np.load(fc_path, allow_pickle=True)
                _fc_keep = [i for i, n in enumerate(_fc["names"]) if not is_unpublished(n)]
                _fc_names = [str(_fc["names"][i]) for i in _fc_keep]
                _fc_dates = pd.to_datetime(_fc["dates"])
                _fc_pred = np.asarray(_fc["pred"], dtype=float)[:, _fc_keep]
                _fc_sel = st.multiselect(
                    "Weather sources (TAM forecast series)", _fc_names,
                    default=_fc_names, key=f"tam_fc_src_{rep_choice}",
                    help="Net-demand forecast from the same fixed TAM driven by each "
                         "weather source. Actual = observed net demand; "
                         "ERA5 = reference; methods = downscaled inputs.",
                )
                _fc_win = st.selectbox(
                    "Time window", 
                    ["Test period (2022)", "Full range (2000–2022)",
                     "Training (2012–2021)", "2012–2022"],
                    key=f"tam_fc_win_{rep_choice}",
                )
                _fc_ranges = {
                    "Test period (2022)": ("2022-01-01", "2022-12-31"),
                    "Full range (2000–2022)": ("2000-01-01", "2022-12-31"),
                    "Training (2012–2021)": ("2012-01-01", "2021-12-31"),
                    "2012–2022": ("2012-01-01", "2022-12-31"),
                }
                _x0, _x1 = _fc_ranges[_fc_win]
                _fig_fc = go.Figure()
                for _i, _nm in enumerate(_fc_names):
                    if _nm not in _fc_sel:
                        continue
                    _is_actual = _nm.startswith("Actual")
                    _fig_fc.add_trace(go.Scatter(
                        x=_fc_dates, y=_fc_pred[:, _i], name=_tam_label(_nm),
                        mode="lines",
                        line=dict(width=1.5, color="black") if _is_actual
                             else dict(width=1.0),
                        hovertemplate="%{x|%Y-%m-%d} · %{y:,.0f} MW<extra>"
                                      + _tam_label(_nm) + "</extra>",
                    ))
                _fig_fc.update_layout(
                    title=f"TAM Net-Demand Forecasts (2000–2022) — {rep_choice}",
                    xaxis_title="Date", yaxis_title="Net demand (MW)",
                    height=500,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.35,
                                font=dict(size=10)),
                    hovermode="x unified",
                )
                _fig_fc.update_xaxes(
                    range=[_x0, _x1],
                    rangeslider_visible=True,
                    rangeselector=dict(
                        buttons=[
                             dict(count=1, label="1Y", step="year", stepmode="backward"),
                             dict(count=3, label="3Y", step="year", stepmode="backward"),
                             dict(count=5, label="5Y", step="year", stepmode="backward"),
                             dict(count=10, label="10Y", step="year", stepmode="backward"),
                             dict(step="all", label="Full"),
                         ]),
                )
                st.plotly_chart(_fig_fc, width="stretch", key="fullcycle_plotly")
                st.markdown("---")

            rows = []
            for k, v in tam_data.items():
                if isinstance(v, dict) and "ks" in v:
                    rows.append({
                        "Method": _tam_label(k),
                        "KS stat": v.get("ks", float("nan")),
                        "Wasserstein (MW)": v.get("wasserstein", float("nan")),
                        "KL Div": v.get("kl", float("nan")),
                        "AE@q01 (MW)": v.get("ae_q01", float("nan")),
                        "AE@q99 (MW)": v.get("ae_q99", float("nan")),
                    })
            if rows:
                df_tam = pd.DataFrame(rows).set_index("Method")
    
                def _highlight_best_tam(col: pd.Series) -> list[str]:
                    try:
                        best = col.idxmin()
                        return [
                            "font-weight: bold; color: black; background-color: #e2f0d9"
                            if idx == best else ""
                            for idx in col.index
                        ]
                    except Exception:
                        return [""] * len(col)
    
                st.dataframe(
                    df_tam.style.apply(_highlight_best_tam, axis=0).format("{:.4f}"),
                    width="stretch",
                )
    
                # Interactive Plotly Line / Comparison Plots (same setup as line plot panel)
                st.markdown("---")
                st.subheader("Interactive Net-Load Performance Plots")
    
                tam_plot_type = st.selectbox(
                    "Plot type",
                    [
                        "Error Metrics Comparison",
                        "Quantile Distortion (AE@q01 & AE@q99)",
                        "Load-Forecasting Radar",
                        "Net-Demand CDF",
                        "Errors by Season (DJF, MAM, JJA, SON)",
                        "Errors by Wind Level (Low, Mid, High)",
                    ],
                    key="tam_plot_type",
                )
    
                if tam_plot_type == "Net-Demand CDF" and cdf_path.exists():
                    tam_cdf_view = st.radio(
                        "CDF view",
                        ["Full range", "Lower tail (q≤20%)", "Upper tail (q≥80%)"],
                        horizontal=True, key="tam_cdf_view",
                    )
                else:
                    tam_cdf_view = "Full range"
    
                fig_tam_plots = go.Figure()
                methods_list = [r["Method"] for r in rows]
                wass_vals = [r["Wasserstein (MW)"] for r in rows]
                ae99_vals = [r["AE@q99 (MW)"] for r in rows]
                ae01_vals = [r["AE@q01 (MW)"] for r in rows]
                ks_vals = [r["KS stat"] for r in rows]
    
                if tam_plot_type == "Error Metrics Comparison":
                    fig_tam_plots.add_trace(go.Bar(name="Wasserstein Distance (MW)", x=methods_list, y=wass_vals, marker_color="#2ca02c"))
                    fig_tam_plots.add_trace(go.Bar(name="Upper Tail Error AE@q99 (MW)", x=methods_list, y=ae99_vals, marker_color="#d62728"))
                    fig_tam_plots.add_trace(go.Bar(name="Lower Tail Error AE@q01 (MW)", x=methods_list, y=ae01_vals, marker_color="#1f77b4"))
                    fig_tam_plots.update_layout(
                        barmode="group",
                        title=f"Net-Demand Error Metrics Comparison — {rep_choice}",
                        xaxis_title="Method",
                        yaxis_title="Error (MW)",
                        height=430,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    )
                elif tam_plot_type == "Quantile Distortion (AE@q01 & AE@q99)":
                    fig_tam_plots.add_trace(go.Bar(
                        name="Upper Tail (AE@q99)", x=methods_list, y=ae99_vals,
                        marker_color="#d62728",
                    ))
                    fig_tam_plots.add_trace(go.Bar(
                        name="Lower Tail (AE@q01)", x=methods_list, y=ae01_vals,
                        marker_color="#1f77b4",
                    ))
                    fig_tam_plots.update_layout(
                        barmode="group",
                        title=f"Tail Quantile Absolute Error Profiles — {rep_choice}",
                        xaxis_title="Method",
                        yaxis_title="Absolute Error (MW)",
                        height=430,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    )
                elif tam_plot_type == "Load-Forecasting Radar":
                    radar_metrics = [
                        ("KS stat", "KS stat"),
                        ("Wasserstein (MW)", "Wasserstein (MW)"),
                        ("KL Div", "KL Div"),
                        ("AE@q01 (MW)", "AE@q01 (MW)"),
                        ("AE@q99 (MW)", "AE@q99 (MW)"),
                    ]
                    norm_base = {}
                    for _rk, rcol in radar_metrics:
                        vals = [r.get(rcol, float("nan")) for r in rows]
                        finite = [v for v in vals if np.isfinite(v)]
                        norm_base[rcol] = float(np.nanmean(finite)) if finite else 1.0
                        if norm_base[rcol] == 0.0:
                            norm_base[rcol] = 1.0
                    theta = [rk for rk, _ in radar_metrics]
                    for i, r in enumerate(rows):
                        r_vals = [
                            abs(r.get(rcol, float("nan"))) / norm_base[rcol]
                            if norm_base[rcol] else float("nan")
                            for _rk, rcol in radar_metrics
                        ]
                        fig_tam_plots.add_trace(go.Scatterpolar(
                            r=r_vals + [r_vals[0]],
                            theta=theta + [theta[0]],
                            fill="toself", opacity=0.5,
                            name=r["Method"],
                            line=dict(width=2),
                        ))
                    fig_tam_plots.update_layout(
                        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
                        title=f"Normalised Load-Forecasting Metrics (all lower = better, 1.0 = mean) — {rep_choice}",
                        height=520,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.18),
                    )
                elif tam_plot_type == "Errors by Season (DJF, MAM, JJA, SON)":
                    seasons = ["DJF", "MAM", "JJA", "SON"]
                    # Measured values only: no hard-coded fallback, so a missing
                    # breakdown shows as missing rather than as invented numbers.
                    season_data = {}
                    raw_season = tam_data.get("by_season")
                    if isinstance(raw_season, dict) and raw_season:
                        season_data = {
                            _tam_label(k): [float(x) for x in v]
                            for k, v in raw_season.items()
                            if isinstance(v, (list, tuple)) and len(v) == len(seasons)
                        }
                    if not season_data:
                        st.info("No seasonal breakdown in this TAM metrics file.")
                    for m_name, vals in season_data.items():
                        fig_tam_plots.add_trace(go.Bar(name=m_name, x=seasons, y=vals))
                    fig_tam_plots.update_layout(
                        barmode="group",
                        title=f"KS Error by Season — {rep_choice}",
                        xaxis_title="Season",
                        yaxis_title="Kolmogorov-Smirnov Statistic (KS)",
                        height=430,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    )
                elif tam_plot_type == "Errors by Wind Level (Low, Mid, High)":
                    wind_levels = ["Low wind", "Mid wind", "High wind"]
                    wind_data = {}
                    raw_wind = tam_data.get("by_wind")
                    if isinstance(raw_wind, dict) and raw_wind:
                        wind_data = {
                            _tam_label(k): [float(x) for x in v]
                            for k, v in raw_wind.items()
                            if isinstance(v, (list, tuple)) and len(v) == len(wind_levels)
                        }
                    if not wind_data:
                        st.info("No wind-level breakdown in this TAM metrics file.")
                    for m_name, vals in wind_data.items():
                        fig_tam_plots.add_trace(go.Bar(name=m_name, x=wind_levels, y=vals))
                    fig_tam_plots.update_layout(
                        barmode="group",
                        title=f"KS Error by Wind Level — {rep_choice}",
                        xaxis_title="Wind Level",
                        yaxis_title="Kolmogorov-Smirnov Statistic (KS)",
                        height=430,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    )
                else:
                    # Empirical CDF (real tails) from exported quantile curves.
                    if cdf_path.exists():
                        try:
                            cdf_npz = np.load(cdf_path, allow_pickle=True)
                            _cdf_keep = [i for i, n in enumerate(cdf_npz["names"])
                                         if not is_unpublished(n)]
                            cdf_names = [cdf_npz["names"][i] for i in _cdf_keep]
                            qgrid = cdf_npz["qgrid"]
                            curves = cdf_npz["curves"][_cdf_keep]
                            if tam_cdf_view == "Lower tail (q≤20%)":
                                mask = qgrid <= 0.20
                            elif tam_cdf_view == "Upper tail (q≥80%)":
                                mask = qgrid >= 0.80
                            else:
                                mask = np.ones_like(qgrid, dtype=bool)
                            for i, name in enumerate(cdf_names):
                                fig_tam_plots.add_trace(go.Scatter(
                                    x=curves[i][mask], y=qgrid[mask], name=_tam_label(name),
                                    mode="lines", line=dict(width=2)
                                ))
                            fig_tam_plots.update_layout(
                                title=f"Empirical CDF of Net Demand (2000–2022) — {rep_choice} — {tam_cdf_view}",
                                xaxis_title="Net Demand (MW)",
                                yaxis_title="Cumulative Probability",
                                height=430,
                                legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                            )
                        except (OSError, KeyError, ValueError):
                            st.warning("CDF data file exists but could not be read.")
                            fig_tam_plots = go.Figure()
                    else:
                        # Never draw a synthetic CDF: curves reconstructed from
                        # summary scores would look like measured results.
                        st.info("Empirical net-demand CDF curves are not available "
                                "for this selection.")

                if fig_tam_plots.data:
                    st.plotly_chart(
                        fig_tam_plots,
                        width="stretch",
                        key=f"tam_plot_{tam_plot_type}",
                    )
    
            else:
                st.info("No method metrics found in JSON.")
        else:
            if DEV_NOTES:
                st.warning(
                    f"TAM metrics file not found at {jpath}\n\n"
                    f"Resolved TAM_DIR: `{tam_dir}` (from env var "
                    f"`{'set' if os.environ.get('TAM_DIR') else 'unset — using default'}`).\n"
                    f"Directory exists: {tam_dir.exists()}; files present (up to 10): "
                    f"{[p.name for p in tam_dir.glob('*')][:10] if tam_dir.exists() else '—'}"
                )
            else:
                st.info("No TAM metrics available for this selection.")

# ── Footer (rendered below every tab) ────────────────────────────────────────
st.divider()
st.markdown(
    "#### AI use\n"
    "This dashboard was developed and is maintained with the help of AI coding "
    "assistants (Claude / OpenCode), which were also used to help reproduce most "
    "of the baseline results shown here."
)