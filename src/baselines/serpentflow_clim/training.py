"""Flow-matching training loop (Clim-Fixed + DOY + static conditioning)."""

import math
import os
import time

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric

from .data_utils import maybe_to_device
from .ema import EMA


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """EDM-style skewed flow-time sampling (paper Eq. time-sampling)."""
    P_mean, P_std = -1.2, 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    t = 1 / (1 + sigma)
    return torch.clip(t, min=0.0001, max=1.0)


def vel_loss(z, z_hat, mask=None):
    loss = (z_hat - z).pow(2)
    if mask is not None:
        loss = loss * mask.unsqueeze(-1)
    return loss.mean()


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    if norm_type == torch.inf:
        return max(p.grad.detach().abs().max() for p in parameters)
    return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]), norm_type)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        norm = None
        if update_grad:
            self._scaler.unscale_(optimizer)
            norm = (torch.nn.utils.clip_grad_norm_(parameters, clip_grad) if clip_grad is not None
                    else get_grad_norm_(parameters))
            self._scaler.step(optimizer)
            self._scaler.update()
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def train_one_epoch(model, data_loader, optimizer, lr_schedule, device, epoch, loss_scaler, path,
                     loss_fn=vel_loss, skewed_timesteps=True, accum_iter=1, test_run=False, mask=None):
    model.train(True)
    epoch_loss = MeanMetric().to(device)

    for step, data in enumerate(data_loader):
        s_time = time.time()
        if step % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        samples = data["data"].to(device)
        noise = data.get("noisy", torch.randn_like(samples)).to(device)
        stats = maybe_to_device(data.get("stats"), device)
        doy = maybe_to_device(data.get("doy"), device)

        if mask is not None:
            noise = torch.nan_to_num(noise, nan=0.0) * mask
            samples = torch.nan_to_num(samples, nan=0.0) * mask

        t = skewed_timestep_sample(samples.shape[0], device=device) if skewed_timesteps \
            else torch.rand(samples.shape[0], device=device)

        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t, u_t = path_sample.x_t, path_sample.dx_t

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            if mask is not None:
                x_t = torch.cat([x_t, mask.repeat(x_t.shape[0], 1, 1, 1)], dim=1)
            pred_vel = model(x_t, t, stats=stats, doy=doy)
            loss = loss_fn(u_t.to(pred_vel.dtype), pred_vel, mask)

        loss = loss / accum_iter
        apply_update = (step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=apply_update)

        if apply_update:
            if isinstance(model, EMA):
                model.update_ema()
            elif isinstance(model, DistributedDataParallel) and isinstance(model.module, EMA):
                model.module.update_ema()
            if lr_schedule is not None:
                lr_schedule.step()

        epoch_loss.update(loss.item())
        if step % 100 == 0:
            print(f"[epoch {epoch}] step {step}/{len(data_loader)}, loss={loss.item():.4f}, "
                  f"time={time.time() - s_time:.2f}s")
        if test_run and step > 0:
            break

    return float(epoch_loss.compute().detach().cpu())


def train_flow_matching(model, dataset, name, ckpt=None, epochs=200, batch_size=64, lr=1e-4, accum_iter=1,
                         device="cuda", betas=(0.9, 0.95), save_dir="checkpoints", mask=None, path=None, **_):
    from flow_matching.path import CondOTProbPath
    path = path or CondOTProbPath()

    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, f"fm_{name}_best.pth")
    device = torch.device(device)
    model = model.to(device)
    model.train(True)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False,
                     num_workers=1, persistent_workers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)

    total_steps = epochs * len(dl) // accum_iter
    warmup_steps = 5000

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    loss_scaler = NativeScalerWithGradNormCount()

    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["loss"]
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch - 1)
    else:
        start_epoch, best_loss = 1, float("inf")
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience, no_improve = 20, 0
    for epoch in range(start_epoch, epochs + 1):
        epoch_loss = train_one_epoch(model, dl, optimizer, lr_schedule, device, epoch, loss_scaler, path, mask=mask,
                                      accum_iter=accum_iter)
        print(f"[Epoch {epoch:03d}/{epochs}] Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss, no_improve = epoch_loss, 0
            model.eval()
            torch.save({"ema": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "lr_schedule": lr_schedule.state_dict(), "epoch": epoch, "loss": best_loss}, best_ckpt)
            model.train(True)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[INFO] early stopping at epoch {epoch}")
                break

    return model
