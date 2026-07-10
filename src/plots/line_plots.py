from pathlib import Path

import xarray as xr
import numpy as np

def plot_cdf(obs: xr.DataArray, pred: xr.DataArray, variable: str,
             method: str, plot_dir: Path,
             xlim: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    configs = [
        ("Full distribution", None),
        ("Upper tail (q>0.95)", 0.95),
        ("Lower tail (q<0.05)", 0.05),
    ]
    o_all = obs.values.ravel();  o_all = o_all[np.isfinite(o_all)]
    p_all = pred.values.ravel(); p_all = p_all[np.isfinite(p_all)]

    for ax, (title, q) in zip(axes, configs):
        o, p = o_all.copy(), p_all.copy()
        if q is not None:
            thr = np.quantile(o, q)
            if q >= 0.5:
                o = o[o > thr]; p = p[p > thr]
            else:
                o = o[o < thr]; p = p[p < thr]
        for data, label, color in [(o, "ERA5", "black"), (p, method, "tab:blue")]:
            xs = np.sort(data)
            ys = np.linspace(0, 1, len(xs))
            ax.plot(xs, ys, label=label, color=color, lw=1.5)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlabel(variable)
        if q is None and xlim is not None:
            ax.set_xlim(xlim)

    fig.suptitle(f"CDF — {variable} — {method}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = plot_dir / f"cdf_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def plot_seasonal_cycle(obs: xr.DataArray, pred: xr.DataArray, variable: str,
                        method: str, plot_dir: Path,
                        ylim: tuple[float, float] | None = None) -> None:
    import matplotlib.pyplot as plt

    obs_cycle  = obs.groupby("time.month").mean("time").mean(["lat", "lon"])
    pred_cycle = pred.groupby("time.month").mean("time").mean(["lat", "lon"])
    months = np.arange(1, 13)

    # Expand ylim to include both ERA5 and pred values (caller ylim may be ERA5-only)
    all_vals = np.concatenate([obs_cycle.values, pred_cycle.values])
    val_range = float(np.nanmax(all_vals) - np.nanmin(all_vals))
    margin = 0.15 * max(val_range, 1e-6)
    auto_ylim = (float(np.nanmin(all_vals)) - margin, float(np.nanmax(all_vals)) + margin)
    if ylim is not None:
        ylim = (min(ylim[0], auto_ylim[0]), max(ylim[1], auto_ylim[1]))
    else:
        ylim = auto_ylim

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(months, obs_cycle.values,  "k-o", label="ERA5", lw=2, ms=5)
    ax.plot(months, pred_cycle.values, "b-s", label=method, lw=2, ms=5)
    ax.set_xticks(months)
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylim(ylim)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f"Seasonal cycle — {variable} — {method}", fontsize=12)
    plt.tight_layout()
    out = plot_dir / f"seasonal_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")

    
def plot_annual_anomalies(gcm: xr.DataArray, pred: xr.DataArray, variable: str,
                          method: str, plot_dir: Path,
                          obs: xr.DataArray | None = None) -> None:
    import matplotlib.pyplot as plt

    gcm_ann  = gcm.mean(["lat", "lon"]).groupby("time.year").mean("time")
    pred_ann = pred.mean(["lat", "lon"]).groupby("time.year").mean("time")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gcm_ann.year.values,  gcm_ann.values,  "k-",  label="GCM",   lw=1.5)
    ax.plot(pred_ann.year.values, pred_ann.values,  "b--", label=method,  lw=1.8)
    if obs is not None:
        obs_ann = obs.mean(["lat", "lon"]).groupby("time.year").mean("time")
        ax.plot(obs_ann.year.values, obs_ann.values, color="0.45", ls=":",
                label="ERA5", lw=1.8, zorder=5)
    ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlabel("Year"); ax.set_ylabel(variable)
    ax.set_title(f"Annual means — {variable} — {method}", fontsize=12)
    plt.tight_layout()
    out = plot_dir / f"annual_{variable}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")

