"""CorrDiff-QM — residual corrective diffusion downscaling (Mardani et al. 2025).

Two-stage generative upsampler trained on ERA5 only (perfect prognosis):

    Stage 1 (deterministic regression):
        ERA5_LR(t) = coarsen(ERA5_HR(t), stride)  →  UNet → mu_theta(ERA5_LR)
    Stage 2 (residual diffusion):
        denoise r = ERA5_HR - mu_theta(ERA5_LR) conditioned on
        (ERA5_LR, mu_theta, month) via a timestep-conditioned UNet.

Inference on a GCM:

    GCM_LR --per-cell-per-month QM (calibrated on historical holdout)-->
    GCM_LR~ ≈ ERA5_LR distribution --> Stage 1 --> mu --> Stage 2 (n_members
    stochastic residual draws) --> average --> decode --> clip (pr>=0).

This follows the repository's iDownscale/SF/CM conventions: physical units in,
normalise with encode/decode, re-stamp time, save_predictions for evaluate.py.

Checkpoints are written to <out_dir>/checkpoints/:
    corrdiff_reg.pt     — Stage 1 regression UNet + stats/variables/stride
    corrdiff_resdiff.pt — Stage 2 residual diffusion network + schedule
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "evaluation"))

from utils import UNet, DiffusersUNet, encode, decode, compute_stats, month_channels
from utils.dataset import _coarsen_upsample
from methods import _qmap_1d
from data_processing import (
    load_era5, load_gcm, bicubic_regrid, align_time, save_predictions,
)

VARIABLES   = ["tas", "pr", "sfcWind", "sfcWindmax", "psl"]
N_VARS      = len(VARIABLES)
STRIDE      = 4
PHYS_MIN    = {"pr": 0.0, "sfcWind": 0.0, "sfcWindmax": 0.0}
REG_ARCH    = "unet"            # Stage 1 — plain regression U-Net (idownscale-style)
DIFF_ARCH   = "diffusers"       # Stage 2 — timestep-FiLM conditioned U-Net
N_DIFF_T    = 1000              # DDPM schedule length
COSINE_S    = 0.008             # cosine schedule offset (Nichol & Dhariwal 2021)

# Stage-2 input channels: [x_lr | mu | x_t | month]
N_DIFF_IN   = N_VARS * 3 + 2


# ── DDPM (residual) diffusion schedule ───────────────────────────────────────

def cosine_alpha_bar(t: torch.Tensor, n_t: int = N_DIFF_T, s: float = COSINE_S) -> torch.Tensor:
    """Variance-preserving cosine noise schedule (Nichol & Dhariwal 2021).

    alpha_bar(t) = cos^2(((t/n_t + s) / (1 + s)) * pi/2), monotone in [0, 1],
    normalised so alpha_bar(0) = 1.  t is an integer index ∈ [0, n_t].
    """
    f = ((t.float() / n_t + s) / (1.0 + s))
    return torch.cos(f * (math.pi / 2.0)) ** 2


def _schedule(n_t: int = N_DIFF_T, s: float = COSINE_S) -> torch.Tensor:
    abar = cosine_alpha_bar(torch.arange(n_t + 1), n_t, s)
    return abar / abar[0]                       # alpha_bar(n_t + 1) = abar[0] → 1


def _to_datetime64(values) -> np.ndarray:
    """Convert a cftime or np.datetime64 array to plain np.datetime64[ns].

    bicubic_regrid preserves the GCM cftime axis (AGENTS.md gotcha #1), which
    breaks align_time and save_predictions; re-stamp before building refs.
    """
    vals = np.asarray(values)
    if isinstance(vals[0], np.datetime64):
        return vals.astype("datetime64[ns]")
    # cftime DatetimeGregorian → str → np.datetime64[ns]
    return np.array([str(v)[:10] for v in vals], dtype="datetime64[ns]")


def q_sample(x0: torch.Tensor, t: torch.Tensor, abar: torch.Tensor,
             eps: torch.Tensor) -> torch.Tensor:
    """Forward diffusion: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""
    a = abar[t][:, None, None, None]
    return a.sqrt() * x0 + (1.0 - a).sqrt() * eps


@torch.no_grad()
def p_sample_step(
    net:      nn.Module,
    x_lr:     torch.Tensor,   # [B, C, H, W]
    mu:       torch.Tensor,   # [B, C, H, W]
    month_ch: torch.Tensor,   # [B, 2, H, W]
    x_t:      torch.Tensor,   # [B, C, H, W]  (noisy residual)
    t:        torch.Tensor,   # [B] current schedule index
    t_next:   torch.Tensor,   # [B] previous (smaller) schedule index
    abar:     torch.Tensor,
    n_t:      int,
    noise:    torch.Tensor | None = None,
) -> torch.Tensor:
    """One reverse-diffusion step between consecutive (possibly strided) steps."""
    a_c = abar[t][:, None, None, None]
    a_n = abar[t_next][:, None, None, None]
    alpha  = a_c / a_n.clamp(min=1e-8)          # effective alpha over the stride
    beta   = (1.0 - alpha).clamp(min=0.0)
    eps    = net(torch.cat([x_lr, mu, x_t, month_ch], dim=1), t.float() / n_t * 1000.0)
    x0_hat = (x_t - (1.0 - a_c).sqrt() * eps) / a_c.sqrt().clamp(min=1e-8)
    x0_hat = x0_hat.clamp(-1.0, 1.0)            # data is normalised to [-1, 1]
    # Posterior mean as a convex combination of x0_hat and x_t (numerically
    # stable at the deep end where alpha≈0; avoids 1/sqrt(alpha) blow-up).
    denom      = (1.0 - a_c).clamp(min=1e-8)
    coef0      = a_n.sqrt() * beta / denom
    coef_x     = alpha.sqrt() * (1.0 - a_n) / denom
    mean       = coef0 * x0_hat + coef_x * x_t
    if noise is None:
        return mean
    post_var = (beta * (1.0 - a_n) / (1.0 - a_c)).clamp(min=0.0)
    return mean + post_var.sqrt() * noise


@torch.no_grad()
def sample_residual(
    net:         nn.Module,
    x_lr:        torch.Tensor,   # [B, C, H, W]
    mu:          torch.Tensor,   # [B, C, H, W]
    month_ch:    torch.Tensor,   # [B, 2, H, W]
    n_t:         int,
    n_steps:     int,
    abar:        torch.Tensor,
    device:      str,
) -> torch.Tensor:
    """Reverse-diffusion from pure noise over n_steps strided steps → residual."""
    idx = torch.linspace(n_t, 0, n_steps + 1).round().long().to(x_lr.device)
    B   = x_lr.shape[0]
    x   = torch.randn_like(mu)
    for i in range(n_steps):
        t_now  = idx[i].expand(B)
        t_next = idx[i + 1].expand(B)
        z      = None if i == n_steps - 1 else torch.randn_like(x)
        x      = p_sample_step(net, x_lr, mu, month_ch, x, t_now, t_next,
                               abar, n_t, noise=z)
    return x


# ── Training ────────────────────────────────────────────────────────────────────

def train_stage1(
    x_lr:      np.ndarray,     # [T, C, H, W] normalised coarsened ERA5
    y_hr:      np.ndarray,     # [T, C, H, W] normalised ERA5
    months:    np.ndarray,     # [T]
    n_epochs:  int   = 5,
    batch_size: int  = 16,
    lr:        float = 2e-4,
    device:    str  = "cpu",
    net:       nn.Module | None = None,
) -> nn.Module:
    """Stage 1 — deterministic regression UNet (mean prediction)."""
    T, _, H, W = y_hr.shape
    if net is None:
        net = UNet(in_channels=N_VARS + 2, out_channels=N_VARS)
    model = net.to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    mc     = torch.from_numpy(month_channels(months, (T, H, W)))
    x_lr_t = torch.from_numpy(x_lr)
    y_t    = torch.from_numpy(y_hr)

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(T)
        tot  = 0.0
        for i in range(0, T, batch_size):
            idx = perm[i:i + batch_size]
            b_in  = torch.cat([x_lr_t[idx], mc[idx]], dim=1).to(device)
            b_hr  = y_t[idx].to(device)
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(b_in), b_hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot  += loss.item() * len(idx)
        print(f"    [corrdiff] Stage1 epoch {epoch + 1}/{n_epochs} "
              f"loss={tot / T:.5f}", flush=True)
    return model


def train_stage2(
    x_lr:      np.ndarray,     # [T, C, H, W] normalised coarsened ERA5
    mu:        np.ndarray,     # [T, C, H, W] Stage-1 mean (frozen)
    res:       np.ndarray,     # [T, C, H, W] residual y - mu
    months:    np.ndarray,     # [T]
    n_epochs:  int   = 5,
    batch_size: int  = 16,
    lr:        float = 2e-4,
    device:    str  = "cpu",
    n_diff_t:  int  = N_DIFF_T,
    net:       nn.Module | None = None,
) -> nn.Module:
    """Stage 2 — residual diffusion: denoise r conditioned on (x_lr, mu, month)."""
    T, _, H, W = res.shape
    if net is None:
        net = DiffusersUNet(in_channels=N_DIFF_IN, out_channels=N_VARS)
    net = net.to(device)
    opt  = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95))
    mc   = torch.from_numpy(month_channels(months, (T, H, W)))
    x_lr_t = torch.from_numpy(x_lr)
    m_t  = torch.from_numpy(mu)
    r_t  = torch.from_numpy(res)
    abar = _schedule(n_diff_t).to(device)

    for epoch in range(n_epochs):
        net.train()
        perm = torch.randperm(T)
        tot  = 0.0
        for i in range(0, T, batch_size):
            idx = perm[i:i + batch_size]
            b_in  = torch.cat([x_lr_t[idx], m_t[idx]], dim=1).to(device)
            b_r   = r_t[idx].to(device)
            b_mc  = mc[idx].to(device)
            t     = torch.randint(1, n_diff_t + 1, (len(idx),), device=device)
            eps   = torch.randn_like(b_r)
            x_n   = q_sample(b_r, t, abar, eps)
            opt.zero_grad()
            eps_pred = net(torch.cat([b_in, x_n, b_mc], dim=1),
                           t.float() / n_diff_t * 1000.0)
            loss = torch.nn.functional.mse_loss(eps_pred, eps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        print(f"    [corrdiff] Stage2 epoch {epoch + 1}/{n_epochs} "
              f"loss={tot / T:.5f}", flush=True)
    return net


def train_corrdiff(
    raw:       np.ndarray,     # [T, C, H, W] physical ERA5
    months:    np.ndarray,     # [T]
    out_dir:   Path,
    n_epochs:  int   = 5,
    batch_size: int  = 16,
    lr:        float = 2e-4,
    device:    str  = "cpu",
    n_diff_t:  int  = N_DIFF_T,
    stage1_net: nn.Module | None = None,
    stage2_net: nn.Module | None = None,
) -> tuple[dict, nn.Module, nn.Module]:
    """Train both stages on ERA5 and write checkpoints."""
    T, _, H, W = raw.shape
    stats = compute_stats(raw, VARIABLES)
    norm  = np.empty((T, N_VARS, H, W), dtype=np.float32)
    for i, var in enumerate(VARIABLES):
        norm[:, i] = encode(raw[:, i], stats, var)

    x_lr = _coarsen_upsample(norm, STRIDE)

    reg  = train_stage1(x_lr, norm, months, n_epochs=n_epochs,
                        batch_size=batch_size, lr=lr, device=device,
                        net=stage1_net)
    reg.eval()
    with torch.no_grad():
        mc_all = torch.from_numpy(month_channels(months, (T, H, W))).to(device)
        mu_np  = np.empty((T, N_VARS, H, W), dtype=np.float32)
        for i in range(0, T, batch_size):
            b_in = torch.cat([torch.from_numpy(x_lr[i:i + batch_size]).to(device),
                              mc_all[i:i + batch_size]], dim=1)
            mu_np[i:i + batch_size] = reg(b_in).cpu().numpy()

    res = (norm - mu_np).astype(np.float32)

    diff = train_stage2(x_lr, mu_np, res, months, n_epochs=n_epochs,
                        batch_size=batch_size, lr=lr, device=device,
                        n_diff_t=n_diff_t, net=stage2_net)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": reg.state_dict(), "stats": stats, "variables": VARIABLES,
        "stride": STRIDE, "arch": REG_ARCH, "in_channels": N_VARS + 2,
        "out_channels": N_VARS,
    }, ckpt_dir / "corrdiff_reg.pt")
    torch.save({
        "model": diff.state_dict(), "stats": stats, "variables": VARIABLES,
        "stride": STRIDE, "arch": DIFF_ARCH, "in_channels": N_DIFF_IN,
        "out_channels": N_VARS, "n_diff_t": n_diff_t,
    }, ckpt_dir / "corrdiff_resdiff.pt")
    return stats, reg, diff


# ── Quantile mapping ────────────────────────────────────────────────────────────

def _qm_per_month(src_cal: np.ndarray, ref_cal: np.ndarray,
                  app_eval: np.ndarray, cal_mon: np.ndarray,
                  ref_mon: np.ndarray, app_mon: np.ndarray) -> np.ndarray:
    """Per-cell, per-calendar-month empirical QM on [T, H, W] arrays.

    For every grid cell, the historical GCM coarse distribution (src_cal) is
    mapped quantile-by-quantile to the ERA5 coarse distribution (ref_cal), then
    applied to the evaluation-period GCM field (app_eval).  Per-month fitting
    captures seasonality; the mapping is calibrated strictly on the historical
    holdout (src_cal, ref_cal) and never refit on the future.
    """
    out   = app_eval.copy()
    Hc, Wc = src_cal.shape[1], src_cal.shape[2]
    for m in range(1, 13):
        c_sel = np.where(cal_mon == m)[0]
        r_sel = np.where(ref_mon == m)[0]
        a_sel = np.where(app_mon == m)[0]
        if len(c_sel) == 0 or len(r_sel) == 0 or len(a_sel) == 0:
            continue
        for i in range(Hc):
            for j in range(Wc):
                out[a_sel, i, j] = _qmap_1d(
                    src_cal[c_sel, i, j], ref_cal[r_sel, i, j],
                    app_eval[a_sel, i, j],
                )
    return out


def _qm_to_era5_coarse(
    era5_dir:    Path,
    gcm_dir:     Path,
    coarse_lat:  np.ndarray,
    coarse_lon:  np.ndarray,
    calib_start: int,
    calib_end:   int,
    eval_start:  int,
    eval_end:    int,
    scenario:    str,
) -> list[xr.DataArray]:
    """Per-cell, per-month QM: GCM coarse (calib) → ERA5 coarse (calib) → apply.

    Bias-corrects the GCM coarse predictors to the ERA5 coarse distribution,
    calibrated strictly on the historical period and applied forward.
    """
    era5_cal = align_time({v: load_era5(era5_dir, v, calib_start, calib_end)
                           for v in VARIABLES})
    gcm_cal  = align_time({v: load_gcm(gcm_dir, v, calib_start, calib_end,
                                       "historical") for v in VARIABLES})
    gcm_eval = align_time({v: load_gcm(gcm_dir, v, eval_start, eval_end,
                                       scenario) for v in VARIABLES})

    out = []
    for var in VARIABLES:
        era5_c  = era5_cal[var].interp(lat=coarse_lat, lon=coarse_lon,
                                       method="linear")
        gcm_c_c = bicubic_regrid(gcm_cal[var], coarse_lat, coarse_lon)
        gcm_e_c = bicubic_regrid(gcm_eval[var], coarse_lat, coarse_lon)

        qm_app = _qm_per_month(
            gcm_c_c.values,
            era5_c.values,
            gcm_e_c.values,
            np.asarray(gcm_c_c.time.dt.month, dtype=int),
            np.asarray(era5_c.time.dt.month, dtype=int),
            np.asarray(gcm_e_c.time.dt.month, dtype=int),
        )
        # Re-stamp time to plain datetimes: bicubic_regrid preserves the GCM
        # cftime axis, which breaks align_time and save_predictions.
        eval_time = _to_datetime64(gcm_eval[var].time.values)
        out.append(xr.DataArray(
            qm_app, dims=["time", "lat", "lon"],
            coords={"time": eval_time, "lat": coarse_lat, "lon": coarse_lon},
        ))
    return out


# ── Inference ───────────────────────────────────────────────────────────────────

def run_inference(
    out_dir:     Path,
    era5_dir:    Path,
    gcm_dir:     Path,
    calib_start: int   = 1980,
    calib_end:   int   = 1999,
    eval_start:  int   = 2000,
    eval_end:    int   = 2100,
    scenario:    str   = "ssp370",
    n_members:   int   = 10,
    n_steps:     int   = 50,
    batch_size:  int   = 128,
    device:      str   = "cpu",
    seed:        int   = 0,
) -> None:
    """CorrDiff-QM inference on the GCM: QM → Stage 1 → Stage 2 → save."""
    torch.manual_seed(seed)
    ckpt_reg  = out_dir / "checkpoints" / "corrdiff_reg.pt"
    ckpt_diff = out_dir / "checkpoints" / "corrdiff_resdiff.pt"
    if not ckpt_reg.exists() or not ckpt_diff.exists():
        raise FileNotFoundError(
            f"No CorrDiff-QM checkpoints at {out_dir / 'checkpoints'}. "
            "Run `python src/baselines/inference_corrdiff.py --train` first.")

    reg_ckpt  = torch.load(ckpt_reg,  map_location=device, weights_only=False)
    diff_ckpt = torch.load(ckpt_diff, map_location=device, weights_only=False)
    stats     = reg_ckpt["stats"]
    n_diff_t  = diff_ckpt.get("n_diff_t", N_DIFF_T)

    if "ref_lat" not in reg_ckpt:
        grid_ck = out_dir / "checkpoints" / "corrdiff_grid.pt"
        if not grid_ck.exists():
            raise FileNotFoundError(
                "CorrDiff-QM checkpoints lack the ERA5 grid (ref_lat/ref_lon). "
                "Re-run --train.")
        grid = torch.load(grid_ck, map_location="cpu", weights_only=False)
        reg_ckpt["ref_lat"] = grid["ref_lat"]
        reg_ckpt["ref_lon"] = grid["ref_lon"]

    reg  = UNet(in_channels=reg_ckpt.get("in_channels", N_VARS + 2),
                out_channels=N_VARS).to(device)
    reg.load_state_dict(reg_ckpt["model"])
    diff = DiffusersUNet(in_channels=diff_ckpt.get("in_channels", N_DIFF_IN),
                         out_channels=N_VARS).to(device)
    diff.load_state_dict(diff_ckpt["model"])
    reg.eval(); diff.eval()

    target_lat = reg_ckpt["ref_lat"]
    target_lon = reg_ckpt["ref_lon"]
    coarse_lat = target_lat[::STRIDE]
    coarse_lon = target_lon[::STRIDE]

    print("  [corrdiff] QM: GCM coarse → ERA5 coarse (calib) ...", flush=True)
    qm_das = _qm_to_era5_coarse(era5_dir, gcm_dir, coarse_lat, coarse_lon,
                                calib_start, calib_end, eval_start, eval_end,
                                scenario)

    T_eval   = len(qm_das[0].time)
    H, W     = len(target_lat), len(target_lon)
    gcm_time = qm_das[0].time

    x_in = np.empty((T_eval, N_VARS, H, W), dtype=np.float32)
    for i, var in enumerate(VARIABLES):
        up = bicubic_regrid(qm_das[i], target_lat, target_lon)
        x_in[:, i] = encode(up.values.astype(np.float32), stats, var)

    months_e  = np.asarray(gcm_time.dt.month, dtype=int)
    abar      = _schedule(n_diff_t).to(device)

    print(f"  [corrdiff] sampling {n_members} members × {n_steps} steps ...",
          flush=True)
    preds_phys = np.empty((T_eval, N_VARS, H, W), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, T_eval, batch_size):
            stop = min(start + batch_size, T_eval)
            x_lr_b = torch.from_numpy(x_in[start:stop]).to(device)
            mc_b   = torch.from_numpy(month_channels(
                months_e[start:stop], (stop - start, H, W))).to(device)
            mu_b   = reg(torch.cat([x_lr_b, mc_b], dim=1))
            acc    = np.zeros((stop - start, N_VARS, H, W), dtype=np.float64)
            for _ in range(n_members):
                r_b  = sample_residual(diff, x_lr_b, mu_b, mc_b, n_diff_t,
                                       n_steps, abar, device)
                y_b  = mu_b + r_b
                for c_i, var in enumerate(VARIABLES):
                    acc[:, c_i] += decode(y_b[:, c_i].cpu().numpy(), stats, var)
            preds_phys[start:stop] = (acc / n_members).astype(np.float32)
            if (start // batch_size) % 25 == 0:
                print(f"    ~ chunk {start}-{stop}/{T_eval}", flush=True)

    for c_i, var in enumerate(VARIABLES):
        if var in PHYS_MIN:
            preds_phys[:, c_i] = np.clip(preds_phys[:, c_i], PHYS_MIN[var], None)

    ref_da = xr.DataArray(
        dims=["time", "lat", "lon"],
        coords={"time": gcm_time, "lat": target_lat, "lon": target_lon},
    )
    tensor = torch.from_numpy(preds_phys).float()
    save_predictions(tensor, VARIABLES, out_dir, ref_da)
    print(f"  [corrdiff] CorrDiff-QM predictions saved to {out_dir}", flush=True)


# ── Registry entry (pre-computed methods are loaded via --pred-dir) ───────────

def corrdiff_qm_downscale(gcm: xr.DataArray, obs: xr.DataArray,
                          variable: str) -> xr.DataArray:
    raise RuntimeError(
        "CorrDiff-QM is a checkpoint-based, pre-computed method. "
        "Run `python src/baselines/inference_corrdiff.py --train --predict ...` "
        "first, then evaluate with `--method corrdiff_qm --pred-dir <out_dir>`."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--era5-dir",      required=True)
    p.add_argument("--gcm-dir",       default=None)
    p.add_argument("--out-dir",       required=True)
    p.add_argument("--calib-start",   type=int,   default=1980)
    p.add_argument("--calib-end",     type=int,   default=1999)
    p.add_argument("--eval-start",    type=int,   default=2000)
    p.add_argument("--eval-end",      type=int,   default=2100)
    p.add_argument("--scenario",      default="ssp370")
    p.add_argument("--epochs",        type=int,   default=5)
    p.add_argument("--batch-size",    type=int,   default=16)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--n-members",     type=int,   default=10)
    p.add_argument("--sample-steps",  type=int,   default=50)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--train",         action="store_true")
    p.add_argument("--predict",       action="store_true")
    args = p.parse_args()

    out_dir  = Path(args.out_dir)
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CorrDiff-QM on {device}", flush=True)
    do_train   = args.train   or (not args.train and not args.predict)
    do_predict = args.predict or (not args.train and not args.predict)

    if do_train:
        raw, months, lat, lon = _load_era5_raw(out_dir, Path(args.era5_dir),
                                               args.calib_start, args.calib_end)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        # stash grid for inference ref_lat/ref_lon
        Path(out_dir / "checkpoints").mkdir(exist_ok=True)
        torch.save({"ref_lat": lat, "ref_lon": lon},
                   out_dir / "checkpoints" / "corrdiff_grid.pt")
        train_corrdiff(raw, months, out_dir, n_epochs=args.epochs,
                       batch_size=args.batch_size, lr=args.lr, device=device)

    if do_predict:
        if not args.gcm_dir:
            raise SystemExit("--gcm-dir is required for --predict")
        grid_ck = out_dir / "checkpoints" / "corrdiff_grid.pt"
        if not grid_ck.exists():
            raise FileNotFoundError(
                "No corrdiff_grid.pt — run --train first (stores ERA5 grid).")
        grid = torch.load(grid_ck, map_location="cpu", weights_only=False)
        # stash ref_lat/ref_lon into the regression checkpoint for inference
        ckpt_reg = out_dir / "checkpoints" / "corrdiff_reg.pt"
        obj = torch.load(ckpt_reg, map_location="cpu", weights_only=False)
        obj.setdefault("ref_lat", grid["ref_lat"])
        obj.setdefault("ref_lon", grid["ref_lon"])
        torch.save(obj, ckpt_reg)
        run_inference(out_dir, Path(args.era5_dir), Path(args.gcm_dir),
                      calib_start=args.calib_start, calib_end=args.calib_end,
                      eval_start=args.eval_start, eval_end=args.eval_end,
                      scenario=args.scenario, n_members=args.n_members,
                      n_steps=args.sample_steps, device=device, seed=args.seed)


def _load_era5_raw(out_dir: Path, era5_dir: Path,
                   calib_start: int, calib_end: int):
    das     = {v: load_era5(era5_dir, v, calib_start, calib_end) for v in VARIABLES}
    aligned = align_time(das)
    ref     = next(iter(aligned.values()))
    months  = ref.time.dt.month.values.astype(np.float32)
    raw     = np.stack([aligned[v].values.astype(np.float32) for v in VARIABLES],
                       axis=1)
    return raw, months, ref.lat.values, ref.lon.values


if __name__ == "__main__":
    main()