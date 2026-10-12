"""ODE-integration sampling from a trained velocity field (Algorithm 2)."""

import os

import torch
from torchdiffeq import odeint

from .data_utils import maybe_to_device


@torch.no_grad()
def odeint_torchdiffeq_adaptive(model, x, stats=None, doy=None, t_span=(0.0, 1.0), device="cuda",
                                 args=None, method="dopri5", mask=None):
    """Integrate the learned velocity field from x (t=0) to the generated field (t=1).

    `doy` conditions the field and is fixed across the whole trajectory;
    static conditioning lives inside the model (see model.UNetFlow.set_static).
    """
    args = args or {"rtol": 1e-3, "atol": 1e-4}
    model = model.to(device).eval()
    x = x.to(device)
    if mask is not None:
        x = torch.nan_to_num(x) * mask
    stats = maybe_to_device(stats, device)
    doy = maybe_to_device(doy, device)

    def f(t, x_state):
        t = t.expand(x_state.size(0))
        xin = x_state
        if mask is not None:
            xin = torch.cat([torch.nan_to_num(xin) * mask, mask.repeat(xin.shape[0], 1, 1, 1)], dim=1)
        v = model(xin, t, stats=stats, doy=doy)
        return v * mask if mask is not None else v

    time_steps = torch.tensor(list(t_span), device=device)
    return odeint(f, x, time_steps, method=method, **args)[-1]


def integrate_and_store(dataloader, model_to, t_span_to=(0.0, 1.0), device="cuda", filename=None,
                         chunk_size=100, temp_dir="temp_results", method="dopri5", mask=None):
    """Generate samples via ODE integration and store to disk in chunks.

    `dataloader` MUST be un-shuffled so DOY stays aligned with sample order.
    """
    os.makedirs(temp_dir, exist_ok=True)
    model = model_to.model.to(device).eval()  # unwrap EMA

    buffer, temp_files = [], []
    file_idx = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x_state = data["noisy"].to(device)
            stats = maybe_to_device(data.get("stats"), device)
            doy = maybe_to_device(data.get("doy"), device)

            sample = odeint_torchdiffeq_adaptive(model, x_state, stats=stats, doy=doy, t_span=t_span_to,
                                                  device=device, method=method, mask=mask).cpu()
            buffer.append(sample)

            if sum(b.shape[0] for b in buffer) >= chunk_size:
                chunk = torch.cat(buffer, dim=0)
                tmp_path = os.path.join(temp_dir, f"chunk_{file_idx:05d}.pt")
                torch.save(chunk, tmp_path)
                print(f"[SerpentFlow] chunk saved: {tmp_path} ({chunk.shape[0]} samples) "
                      f"{batch_idx}/{len(dataloader)}")
                temp_files.append(tmp_path)
                file_idx += 1
                buffer = []

        if buffer:
            chunk = torch.cat(buffer, dim=0)
            tmp_path = os.path.join(temp_dir, f"chunk_{file_idx:05d}.pt")
            torch.save(chunk, tmp_path)
            temp_files.append(tmp_path)

    final_tensor = torch.cat([torch.load(p, map_location="cpu") for p in temp_files], dim=0)
    if filename is not None:
        torch.save(final_tensor, filename)
    for p in temp_files:
        os.remove(p)
    os.rmdir(temp_dir)

    print(f"[SerpentFlow] total samples generated: {final_tensor.shape[0]}")
    return final_tensor
