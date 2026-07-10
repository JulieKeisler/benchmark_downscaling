"""Unified training entry-point for all three neural baselines.

Shared outer loop; mode-specific inner step:

    train_neural_model(mode="idownscale",  ...)  → MSE regression
    train_neural_model(mode="serpentflow", ...)  → conditional flow matching
    train_score_model(...)                        → EDM denoising score matching
    train_neural_model(mode="consistency", ..., score_model=...)
                                                  → consistency distillation (CD)
                                                    from a pre-trained score model

CM training follows Hess et al. 2025 / Song et al. 2023 exactly:
  Phase 1 — score network with EDM preconditioning and log-normal σ sampling.
  Phase 2 — consistency distillation: one Euler step of the PF-ODE via the
             score network gives the target x_n from x_n+1.
"""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ema import EMA

# ── Consistency-model hyper-parameters (Song et al. 2023 / Hess 2025) ─────────
BINS_MIN    = 10
BINS_MAX    = 150
BINS_RHO    = 7.0
DATA_STD    = 0.5
T_MIN       = 0.002
T_MAX       = 80.0          # Matches Song et al. 2023 / Hess et al. 2025
INITIAL_EMA = 0.9

# EDM log-normal σ sampling (Karras et al. 2022)
P_MEAN      = -1.2
P_STD       = 1.2

# Precomputed for log-uniform timestep embedding [0, 1000]
_LOG_RATIO  = math.log(T_MAX / T_MIN)   # ≈ 10.6


def _n_in(mode: str, n_vars: int) -> int:
    """Number of input channels for each mode.

    idownscale:  [x_in | month]       → N_VARS + 2
    serpentflow: [x_t  | month | t]   → N_VARS + 3  (t as spatial channel)
    consistency: [x_t  | month]       → N_VARS + 2  (t via UNet2DModel timestep)
    score:       [x_noisy | month]    → N_VARS + 2  (σ via UNet2DModel timestep)
    """
    return n_vars + 3 if mode == "serpentflow" else n_vars + 2


def _t_emb(t: torch.Tensor) -> torch.Tensor:
    """Log-uniform mapping of t ∈ [T_MIN, T_MAX] → [0, 1000] for sinusoidal embedding."""
    return (t.clamp(T_MIN, T_MAX) / T_MIN).log() / _LOG_RATIO * 1000.0


# ── Time / bin helpers (consistency distillation only) ────────────────────────

def _timestep_to_time(n: torch.Tensor, N: int, device) -> torch.Tensor:
    n = n.to(device).float()
    return (
        T_MIN ** (1.0 / BINS_RHO)
        + n / (N - 1) * (T_MAX ** (1.0 / BINS_RHO) - T_MIN ** (1.0 / BINS_RHO))
    ).pow(BINS_RHO).clamp(T_MIN, T_MAX)


def _current_bins(step: int, total: int) -> int:
    frac = step / max(total, 1)
    return math.ceil(math.sqrt(frac * (BINS_MAX**2 - BINS_MIN**2) + BINS_MIN**2))


@torch.no_grad()
def _ema_update_params(online: nn.Module, target: nn.Module, N: int):
    decay = math.exp(BINS_MIN * math.log(INITIAL_EMA) / N)
    for p, p_ema in zip(online.parameters(), target.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ── Score network forward (EDM preconditioning) ────────────────────────────────

def score_forward(
    net:      nn.Module,
    x_noisy:  torch.Tensor,   # [B, N_VARS, H, W]
    month_ch: torch.Tensor,   # [B, 2, H, W]
    sigma:    torch.Tensor,   # [B]
) -> torch.Tensor:
    """Denoised estimate D_θ(x_noisy; σ) with EDM preconditioning.

    D_θ = c_skip·x_noisy + c_out·F_θ(c_in·x_noisy; c_noise(σ))

    Returns the denoised estimate [B, N_VARS, H, W].
    """
    s4       = sigma[:, None, None, None]
    c_skip   = DATA_STD**2 / (s4**2 + DATA_STD**2)
    c_out    = s4 * DATA_STD / (s4**2 + DATA_STD**2).sqrt()
    c_in     = (s4**2 + DATA_STD**2).rsqrt()
    inp      = torch.cat([c_in * x_noisy, month_ch], dim=1)
    t_emb    = _t_emb(sigma)
    F_pred   = net(inp, t_emb)
    return c_skip * x_noisy + c_out * F_pred


# ── Consistency model forward (shared by train & inference) ───────────────────

def consistency_forward(
    net:      nn.Module,
    x:        torch.Tensor,   # [B, N_VARS, H, W] — noisy ERA5 at level t
    month_ch: torch.Tensor,   # [B, 2, H, W]
    t:        torch.Tensor,   # [B]
    n_vars:   int,
) -> torch.Tensor:
    """One-step consistency denoiser f_θ(x, t) with skip connection.

    f_θ(x, t) = c_skip(t)·x + c_out(t)·F_θ(x; t)

    Boundary condition: at t = T_MIN, c_skip → 1 and c_out → 0, so f_θ(x, T_MIN) = x.
    """
    t4       = t[:, None, None, None]
    dt       = (t4 - T_MIN).clamp(min=0.0)
    c_skip   = DATA_STD**2 / (dt**2 + DATA_STD**2)
    c_out    = DATA_STD * dt / (dt**2 + DATA_STD**2).sqrt()
    inp      = torch.cat([x, month_ch], dim=1)
    t_emb    = _t_emb(t)
    F_pred   = net(inp, t_emb)
    return c_skip * x + c_out * F_pred


# ── Mode-specific inner training loops ────────────────────────────────────────

def _step_regression(model, batch, device):
    x_in     = batch["x_in"].to(device)
    x_target = batch["x_target"].to(device)
    mc       = batch["month_ch"].to(device)
    return F.mse_loss(model(torch.cat([x_in, mc], dim=1)), x_target)


def _step_flow(model, batch, device, hf_noise_std: float = 0.1):
    x_lp = batch["x_in"].to(device)
    x1   = batch["x_target"].to(device)
    mc   = batch["month_ch"].to(device)
    B, _, H, W = x_lp.shape
    x0   = x_lp + torch.randn_like(x_lp) * hf_noise_std
    sigma = (torch.randn(B, device=device) * P_STD + P_MEAN).exp()
    t     = (1.0 / (1.0 + sigma)).clamp(1e-4, 1.0)
    t4   = t[:, None, None, None]
    x_t  = (1 - t4) * x0 + t4 * x1
    t_ch = t4.expand(B, 1, H, W)
    return F.mse_loss(model(torch.cat([x_t, mc, t_ch], dim=1)), x1 - x0)


def _step_score(model, batch, device):
    """EDM denoising score matching (Karras et al. 2022)."""
    y  = batch["x_target"].to(device)
    mc = batch["month_ch"].to(device)
    B, _, H, W = y.shape
    log_sigma = torch.randn(B, device=device) * P_STD + P_MEAN
    sigma     = log_sigma.exp().clamp(T_MIN, T_MAX)
    eps       = torch.randn_like(y)
    x_noisy   = y + sigma[:, None, None, None] * eps
    D_pred    = score_forward(model, x_noisy, mc, sigma)
    weight    = (sigma**2 + DATA_STD**2) / (sigma * DATA_STD)**2
    return (weight[:, None, None, None] * (D_pred - y)**2).mean()


def _step_consistency_distill(
    student, teacher, score_net,
    batch, device, n_vars, step, total_steps,
):
    """One CD step: ODE Euler via score_net gives target x_n from x_n+1."""
    y  = batch["x_target"].to(device)
    mc = batch["month_ch"].to(device)
    B, _, H, W = y.shape
    N    = _current_bins(step, total_steps)
    n    = torch.randint(0, N - 1, (B,), device=device)
    t_n  = _timestep_to_time(n,     N, device)
    t_n1 = _timestep_to_time(n + 1, N, device)
    eps  = torch.randn(B, n_vars, H, W, device=device)
    x_n1 = y + eps * t_n1[:, None, None, None]

    with torch.no_grad():
        D_n1 = score_forward(score_net, x_n1, mc, t_n1)
        # PF-ODE Euler step: dx/dt = (x - D(x,t)) / t
        dt   = (t_n - t_n1)[:, None, None, None]
        x_n  = x_n1 + dt * (x_n1 - D_n1) / t_n1[:, None, None, None]

    f_n1 = consistency_forward(student, x_n1, mc, t_n1, n_vars)
    with torch.no_grad():
        f_n = consistency_forward(teacher, x_n, mc, t_n, n_vars)
    return ((f_n1 - f_n)**2).mean()


# ── Public training entry-points ──────────────────────────────────────────────

def _run_epoch(
    mode, model, loader, opt, ema, model_ema,
    device, n_vars, global_step, total_steps,
    hf_noise_std, score_net, dataset_len,
):
    model.train()
    total = 0.0
    for batch in loader:
        if mode == "idownscale":
            loss = _step_regression(model, batch, device)
        elif mode == "serpentflow":
            loss = _step_flow(model, batch, device, hf_noise_std=hf_noise_std)
        elif mode == "score":
            loss = _step_score(model, batch, device)
        else:   # consistency
            loss = _step_consistency_distill(
                model, model_ema, score_net, batch, device,
                n_vars, global_step, total_steps,
            )

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            global_step += 1
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        global_step += 1

        if ema is not None:
            ema.update()
        if model_ema is not None:
            _ema_update_params(model, model_ema, _current_bins(global_step, total_steps))

        total += loss_val * batch["x_target"].shape[0]

    return total / dataset_len, global_step


def train_score_model(
    dataset,
    model:      nn.Module,
    ckpt_path:  Path,
    n_epochs:   int   = 200,
    batch_size: int   = 32,
    lr:         float = 2e-4,
    device:     str | torch.device = "cuda",
    ema_decay:  float = 0.999,
) -> nn.Module:
    """Train the EDM score network (Phase 1 of CM pipeline)."""
    device    = torch.device(device)
    n_vars    = dataset.x_target.shape[1]
    model     = model.to(device)
    ema       = EMA(model, decay=ema_decay)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    num_workers = min(4, torch.multiprocessing.cpu_count() // 2)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True,
                           persistent_workers=(num_workers > 0))
    n_params  = sum(p.numel() for p in model.parameters())
    total_steps = len(loader) * n_epochs
    print(f"  [score] {n_params/1e6:.1f}M params  "
          f"{len(dataset)} samples  {n_epochs} epochs  ({device})", flush=True)
    best_loss   = math.inf
    global_step = 0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        avg, global_step = _run_epoch(
            "score", model, loader, opt, ema, None,
            device, n_vars, global_step, total_steps,
            0.0, None, len(dataset),
        )
        sched.step()
        if avg < best_loss:
            best_loss = avg
            _save_ckpt(model, ema, ckpt_path)
        print(f"    epoch {epoch:3d}/{n_epochs}  "
              f"loss={avg:.5f}  best={best_loss:.5f}  t={time.time()-t0:.0f}s", flush=True)
    if not ckpt_path.exists():
        _save_ckpt(model, ema, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ema.load_state_dict(ckpt["ema"])
    ema.apply_shadow()
    return model


def train_neural_model(
    mode:          str,
    dataset,
    model:         nn.Module,
    ckpt_path:     Path,
    n_epochs:      int   = 50,
    batch_size:    int   = 32,
    lr:            float = 2e-4,
    device:        str | torch.device = "cuda",
    ema_decay:     float = 0.999,
    hf_noise_std:  float = 0.1,
    score_model:   nn.Module | None = None,
) -> nn.Module:
    """Train a neural baseline (idownscale / serpentflow / consistency).

    For mode="consistency": score_model must be a pre-trained score network.
    The consistency model is trained via distillation (CD), not self-supervised CT.
    """
    if mode == "consistency" and score_model is None:
        raise ValueError("mode='consistency' requires score_model (pre-trained score network)")

    device    = torch.device(device)
    n_vars    = dataset.x_target.shape[1]
    model     = model.to(device)
    use_ema   = mode in ("serpentflow", "consistency")
    model_ema = copy.deepcopy(model).to(device) if mode == "consistency" else None
    ema       = EMA(model, decay=ema_decay)     if use_ema else None
    if score_model is not None:
        score_model = score_model.to(device).eval()

    opt   = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    num_workers = min(4, torch.multiprocessing.cpu_count() // 2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    n_params    = sum(p.numel() for p in model.parameters())
    total_steps = len(loader) * n_epochs
    print(f"  [{mode}] {n_params/1e6:.1f}M params  "
          f"{len(dataset)} samples  {n_epochs} epochs  ({device})", flush=True)

    best_loss   = math.inf
    global_step = 0
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        avg, global_step = _run_epoch(
            mode, model, loader, opt, ema, model_ema,
            device, n_vars, global_step, total_steps,
            hf_noise_std, score_model, len(dataset),
        )
        sched.step()
        if avg < best_loss:
            best_loss = avg
            _save_ckpt(model, ema, ckpt_path)
        print(f"    epoch {epoch:3d}/{n_epochs}  "
              f"loss={avg:.5f}  best={best_loss:.5f}  t={time.time()-t0:.0f}s", flush=True)

    if not ckpt_path.exists():
        _save_ckpt(model, ema, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if use_ema and ema is not None:
        ema.load_state_dict(ckpt["ema"])
        ema.apply_shadow()
    else:
        model.load_state_dict(ckpt["model"])
    return model


def _save_ckpt(model: nn.Module, ema: "EMA | None", path: Path) -> None:
    obj = {"model": model.state_dict()}
    if ema is not None:
        obj["ema"] = ema.state_dict()
    torch.save(obj, path)
