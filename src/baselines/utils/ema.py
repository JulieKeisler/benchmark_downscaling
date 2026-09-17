"""Exponential moving average (EMA) of model parameters."""
from __future__ import annotations

import copy
import torch
import torch.nn as nn


class EMA:
    """Maintains shadow EMA weights alongside a live model.

    Call `update()` after each optimizer step. At inference, call
    `apply_shadow()` to load the averaged weights into the model, or use
    the returned shadow state dict directly via `state_dict()`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model       = model
        self.decay       = decay
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            n: p.data.clone().detach()
            for n, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self) -> None:
        # Warmup schedule matching original SF: min(decay, (1+n)/(10+n))
        # Starts at ~0.1, reaches 0.999 after ~10k steps
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        self.num_updates += 1
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(d).add_(param.data, alpha=1.0 - d)

    def apply_shadow(self) -> None:
        """Copy EMA weights into the live model (use before inference)."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore_live(self, live_state: dict[str, torch.Tensor]) -> None:
        """Restore live weights saved before apply_shadow()."""
        for name, param in self.model.named_parameters():
            param.data.copy_(live_state[name])

    def state_dict(self) -> dict:
        return {"shadow": {n: t.cpu() for n, t in self.shadow.items()},
                "num_updates": self.num_updates}

    def load_state_dict(self, state: dict) -> None:
        if isinstance(state, dict) and "shadow" in state:
            self.shadow      = {n: t.clone() for n, t in state["shadow"].items()}
            self.num_updates = state.get("num_updates", 0)
        else:
            # Legacy: state is the shadow dict directly
            self.shadow = {n: t.clone() for n, t in state.items()}
