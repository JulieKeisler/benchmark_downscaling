"""Exponential Moving Average model wrapper."""

from typing import List

import torch
from torch import Tensor
from torch.nn import Module, Parameter, ParameterList


class EMA(Module):
    def __init__(self, model: Module, decay: float = 0.999):
        super().__init__()
        self.model = model
        self.decay = decay
        self.register_buffer("num_updates", torch.tensor(0))
        # cached once: update_ema/copy_to_model always operate on this same
        # (trainable-only) parameter list, computed fresh from
        # model.parameters() on every previous call
        self._trainable_params: List[Parameter] = [p for p in model.parameters() if p.requires_grad]
        self.shadow_params: ParameterList = ParameterList(
            [Parameter(p.clone().detach(), requires_grad=False) for p in self._trainable_params]
        )
        self.backup_params: List[torch.Tensor] = []

    def train(self, mode: bool = True):
        if self.training == mode:
            return super().train(mode)
        if not mode:
            self.backup()
            self.copy_to_model()
        else:
            self.restore_to_model()
        return super().train(mode)

    def update_ema(self) -> None:
        self.num_updates += 1
        n = self.num_updates.item()
        decay = min(self.decay, (1 + n) / (10 + n))
        with torch.no_grad():
            for shadow, param in zip(self.shadow_params, self._trainable_params):
                shadow.sub_((1 - decay) * (shadow - param))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model(*args, **kwargs)

    def copy_to_model(self) -> None:
        for shadow, param in zip(self.shadow_params, self._trainable_params):
            param.data.copy_(shadow.data)

    def backup(self) -> None:
        assert self.training, "Backup only allowed in train mode."
        if self.backup_params:
            for p, b in zip(self.model.parameters(), self.backup_params):
                b.data.copy_(p.data)
        else:
            self.backup_params = [param.clone() for param in self.model.parameters()]

    def restore_to_model(self) -> None:
        for param, backup in zip(self.model.parameters(), self.backup_params):
            param.data.copy_(backup.data)
