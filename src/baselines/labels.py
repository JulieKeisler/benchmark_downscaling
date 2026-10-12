"""Human-readable display labels for baselines and discovered result dirs.

Kept dependency-free (no torch/xarray/sbck) so lightweight consumers such as the
dashboard can import it without pulling in the whole registry — see
``src/baselines/__init__.py``.
"""

METHOD_LABELS: dict[str, str] = {
    "bicubic":            "Bicubic",
    "bcsd":               "BCSD",
    "cdft":               "CDF-t",
    "dotc":               "DOTC",
    "corrdiff_qm":        "CorrDiff-QM",
    "cordiff":            "CorDiff",
    "idownscale":         "iDownscale",
    "consistency":        "CM",
    "cm":                 "CM",
    "automated_degradation":  "Automated Degradation",
    # Discovered-mu alignment + conditional flow matching (this project).
    # mu is a searched DAG of interpretable climate operators rather than the
    # fixed Fourier low-pass SerpentFlow uses -- hence "chimera", an assembly
    # of unlike parts. Supersedes the "automated_degradation" entry, which was
    # produced before the GCM latitude-orientation fix and is excluded from the
    # dashboard.
    "ChimeraFlow":        "ChimeraFlow",
    "chimeraflow":        "ChimeraFlow",
    "SerpentflowClim":    "SerpentFlow-CLIM",
    "serpentflow":        "SerpentFlow",
    "serpentflow_clim":   "SerpentFlow-CLIM",
    "serpentflowclim":    "SerpentFlow-CLIM",
}
