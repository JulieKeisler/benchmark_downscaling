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
    "SerpentflowClim":    "SerpentFlow-CLIM",
    "serpentflow":        "SerpentFlow",
    "serpentflow_clim":   "SerpentFlow-CLIM",
    "serpentflowclim":    "SerpentFlow-CLIM",
}

# Unpublished methods: hidden from the dashboard and never staged for deploy.
# Matched case-insensitively against result-dir names and net-load (TAM) keys.
UNPUBLISHED_PREFIXES: tuple[str, ...] = ("chimeraflow", "cf_", "automated_degradation")


def is_unpublished(name: str) -> bool:
    """True if `name` (a result dir or TAM key) belongs to an unpublished method."""
    return str(name).strip().lower().startswith(UNPUBLISHED_PREFIXES)
