

from __future__ import annotations

from pathlib import Path

import numpy as np

def plot_radar(metrics_vs_obs: dict, metrics_vs_gcm: dict,
               method: str, plot_dir: Path) -> None:
    """Radar plot with available metrics (lower = better, except temporal_corr*)."""
    import matplotlib.pyplot as plt

    all_metrics: dict[str, float] = {}
    for k, v in metrics_vs_obs.items():
        all_metrics[k] = float(v) if np.isfinite(v) else np.nan
    for k, v in metrics_vs_gcm.items():
        all_metrics[k] = float(1 - v) if k.startswith("temporal_corr") else float(v)

    labels = list(all_metrics.keys())
    vals = np.array([all_metrics[k] for k in labels], dtype=float)

    if len(labels) < 3:
        print("  too few metrics for radar plot, skipping")
        return

    norm = np.nanmax(np.abs(vals)) or 1.0
    vals_norm = vals / norm

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_plot = np.concatenate([vals_norm, vals_norm[:1]])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles, vals_plot, "b-o", lw=2)
    ax.fill(angles, vals_plot, alpha=0.15, color="blue")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"Benchmark summary — {method}", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    out = plot_dir / "radar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")