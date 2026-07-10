"""Shared inference utilities for all three neural baselines.

Each mode expects a different form of conditional input `x_in` (already
normalized), and uses a different decoding scheme:

    mode="idownscale"   → deterministic single forward pass
    mode="serpentflow"  → Euler ODE integration (t=0 → t=1)
    mode="consistency"  → stroke guidance: add noise, one-step denoise

The `consistency_forward` function is imported by train.py too (shared).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .training import consistency_forward   # shared with training loop


# ── Mode-specific inference steps ─────────────────────────────────────────────

@torch.no_grad()
def _infer_regression(
    model:     nn.Module,
    x_in:      torch.Tensor,   # [T, N_VARS, H, W]
    month_ch:  torch.Tensor,   # [T, 2, H, W]
    device:    torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    results = []
    for s in range(0, len(x_in), batch_size):
        xb = x_in[s:s+batch_size].to(device)
        mc = month_ch[s:s+batch_size].to(device)
        results.append(model(torch.cat([xb, mc], dim=1)).cpu().numpy())
    return np.concatenate(results, axis=0)


@torch.no_grad()
def _infer_ode(
    model:      nn.Module,
    x_in:       torch.Tensor,   # [T, N_VARS, H, W]  — ODE start (t=0)
    month_ch:   torch.Tensor,   # [T, 2, H, W]
    device:     torch.device,
    n_steps:    int   = 50,     # Euler fallback steps only
    batch_size: int   = 32,
    rtol:       float = 1e-4,   # dopri5 relative tolerance
    atol:       float = 1e-4,   # dopri5 absolute tolerance
) -> np.ndarray:
    """ODE integration of the learned velocity field from t=0 to t=1.

    Uses dopri5 (Dormand-Prince adaptive RK45) via torchdiffeq when available,
    falling back to fixed-step Euler otherwise.
    """
    try:
        from torchdiffeq import odeint as _odeint
        _use_dopri5 = True
    except ImportError:
        _use_dopri5 = False

    dt = 1.0 / n_steps
    results = []
    for s in range(0, len(x_in), batch_size):
        x0 = x_in[s:s+batch_size].to(device)
        mc = month_ch[s:s+batch_size].to(device)
        B, _, H, W = x0.shape

        if _use_dopri5:
            def _vel(t_scalar, x):
                t_ch = torch.full((B, 1, H, W), float(t_scalar),
                                  device=device, dtype=x.dtype)
                return model(torch.cat([x, mc, t_ch], dim=1))
            t_span = torch.tensor([0.0, 1.0], dtype=x0.dtype, device=device)
            # odeint returns [len(t_span), B, C, H, W]; take the last time point
            y = _odeint(_vel, x0, t_span, method="dopri5", rtol=rtol, atol=atol)
            results.append(y[-1].cpu().numpy())
        else:
            x = x0.clone()
            for step in range(n_steps):
                t_val = step * dt
                t_ch  = torch.full((B, 1, H, W), t_val, device=device, dtype=x.dtype)
                v     = model(torch.cat([x, mc, t_ch], dim=1))
                x     = x + v * dt
            results.append(x.cpu().numpy())
    return np.concatenate(results, axis=0)


@torch.no_grad()
def _infer_stroke(
    model:       nn.Module,
    x_in:        torch.Tensor,   # [T, N_VARS, H, W]  — upsampled GCM
    month_ch:    torch.Tensor,   # [T, 2, H, W]
    device:      torch.device,
    stroke_time: float = 0.468,
    n_members:   int   = 1,
    n_vars:      int   = 5,
    batch_size:  int   = 32,
) -> np.ndarray:
    """Stroke guidance: add noise at stroke_time, one-step consistency denoising.

    Multiple members are averaged to reduce stochastic variance.
    """
    results = []
    for s in range(0, len(x_in), batch_size):
        xb = x_in[s:s+batch_size].to(device)
        mc = month_ch[s:s+batch_size].to(device)
        B, _, H, W = xb.shape
        t = torch.full((B,), stroke_time, device=device)
        accum = torch.zeros_like(xb)
        for _ in range(n_members):
            noise   = torch.randn(B, n_vars, H, W, device=device)
            x_noisy = xb + noise * stroke_time
            accum  += consistency_forward(model, x_noisy, mc, t, n_vars)
        results.append((accum / n_members).cpu().numpy())
    return np.concatenate(results, axis=0)


# ── Public entry-point ─────────────────────────────────────────────────────────

@torch.no_grad()
def neural_predict(
    mode:        str,
    model:       nn.Module,
    x_in:        np.ndarray,    # [T, N_VARS, H, W] — normalized, mode-specific
    month_ch:    np.ndarray,    # [T, 2, H, W]
    device:      str | torch.device,
    n_steps:     int   = 50,    # Euler fallback steps (mode="serpentflow")
    rtol:        float = 1e-4,  # dopri5 rtol (mode="serpentflow")
    atol:        float = 1e-4,  # dopri5 atol (mode="serpentflow")
    stroke_time: float = 1.0,   # noise level (mode="consistency")
    n_members:   int   = 1,     # ensemble members (mode="consistency")
    n_vars:      int   = 5,
    batch_size:  int   = 32,
) -> np.ndarray:
    """Run mode-specific neural inference and return [T, N_VARS, H, W] normalized array."""
    device = torch.device(device)
    model  = model.to(device).eval()
    x_t    = torch.from_numpy(x_in)
    mc_t   = torch.from_numpy(month_ch)

    if mode == "idownscale":
        return _infer_regression(model, x_t, mc_t, device, batch_size)
    if mode == "serpentflow":
        return _infer_ode(model, x_t, mc_t, device, n_steps, batch_size, rtol=rtol, atol=atol)
    if mode == "consistency":
        return _infer_stroke(model, x_t, mc_t, device, stroke_time, n_members, n_vars, batch_size)
    raise ValueError(f"Unknown mode {mode!r}")
