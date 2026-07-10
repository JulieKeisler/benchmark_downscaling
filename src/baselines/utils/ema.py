"""Exponential moving average (EMA) of model parameters — used by SerpentFlow."""
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
        self.model  = model
        self.decay  = decay
        self.shadow: dict[str, torch.Tensor] = {
            n: p.data.clone().detach()
            for n, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Copy EMA weights into the live model (use before inference)."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore_live(self, live_state: dict[str, torch.Tensor]) -> None:
        """Restore live weights saved before apply_shadow()."""
        for name, param in self.model.named_parameters():
            param.data.copy_(live_state[name])

    def state_dict(self) -> dict:
        return {n: t.cpu() for n, t in self.shadow.items()}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = {n: t.clone() for n, t in state.items()}
