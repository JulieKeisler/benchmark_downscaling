"""Spatial low/high-pass filters (LF_s / HF_s in the paper's notation)."""

import math
from functools import lru_cache

import torch
import torch.nn.functional as F


def _unbatch(x: torch.Tensor) -> torch.Tensor:
    """Drop a size-1 leading batch dim (matches the original notebook's
    per-call `squeeze(0) if shape[0] == 1` convention -- including for a
    genuinely single-sample 4D input, not just inputs we unsqueezed
    ourselves)."""
    return x.squeeze(0) if x.shape[0] == 1 else x


@lru_cache(maxsize=32)
def _lowpass_mask(H: int, W: int, r_cut, high: bool, device: torch.device) -> torch.Tensor:
    """Circular Fourier-domain mask for (H, W); cached since it only depends
    on the grid size/cutoff/device, not on the data -- rebuilding it on every
    call (e.g. once per training sample in `SerpentFlowDataset.__getitem__`)
    would otherwise redo the same meshgrid/sqrt every time."""
    Y, X = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    cy, cx = H // 2, W // 2
    radius = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = (radius >= r_cut).float() if high else (radius <= r_cut).float()
    return mask[None, None, :, :]


def fourier_tensor_batch(imgs: torch.Tensor, r_cut, apply_noise: bool = False, high: bool = False):
    """Circular low-pass filter in Fourier space.

    Args:
        imgs: (N, C, H, W) or (C, H, W)
        r_cut: cutoff radius in the (shifted) Fourier domain
        apply_noise: also return standard-normal noise restricted to the
            complementary (high-frequency) band
        high: if True, keep frequencies >= r_cut instead of <= r_cut
    """
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)

    N, C, H, W = imgs.shape
    mask = _lowpass_mask(H, W, r_cut, high, imgs.device)

    fft = torch.fft.fftshift(torch.fft.fft2(imgs, dim=(-2, -1)), dim=(-2, -1))
    low_freq = torch.fft.ifft2(torch.fft.ifftshift(fft * mask, dim=(-2, -1)), dim=(-2, -1)).real

    if not apply_noise:
        return _unbatch(low_freq)

    noise = torch.randn_like(imgs)
    fft_noise = torch.fft.fftshift(torch.fft.fft2(noise, dim=(-2, -1)), dim=(-2, -1))
    high_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_noise * (1 - mask), dim=(-2, -1)), dim=(-2, -1)).real

    return _unbatch(low_freq), _unbatch(high_freq)


def gaussian_kernel_1d(sigma, max_size: int = 101, device: str = "cpu"):
    size = min(2 * math.ceil(3.0 * sigma) + 1, max_size)
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def low_pass_blur_ignore_nans(imgs: torch.Tensor, sigma: float, apply_noise: bool = False):
    """NaN-aware separable Gaussian blur (alternative to the Fourier filter)."""
    if imgs.ndim == 3:
        imgs = imgs.unsqueeze(0)
    N, C, H, W = imgs.shape
    device = imgs.device

    k_lat = gaussian_kernel_1d(sigma, device=device).view(1, 1, -1, 1)
    k_lon = gaussian_kernel_1d(sigma, device=device).view(1, 1, 1, -1)
    pad_lat = k_lat.shape[2] // 2
    pad_lon = k_lon.shape[3] // 2

    mask = (~torch.isnan(imgs)).float()
    imgs_zeroed = torch.nan_to_num(imgs, nan=0.0)

    imgs_reshape = imgs_zeroed.reshape(N * C, 1, H, W)
    mask_reshape = mask.reshape(N * C, 1, H, W)
    blurred = F.conv2d(F.pad(imgs_reshape, (0, 0, pad_lat, pad_lat)), k_lat)
    blurred = F.conv2d(F.pad(blurred, (pad_lon, pad_lon, 0, 0)), k_lon)

    mask_blur = F.conv2d(F.pad(mask_reshape, (0, 0, pad_lat, pad_lat)), k_lat)
    mask_blur = F.conv2d(F.pad(mask_blur, (pad_lon, pad_lon, 0, 0)), k_lon)
    mask_blur = torch.clamp(mask_blur, min=1e-6)

    blurred = (blurred / mask_blur).reshape(N, C, H, W)
    blurred[mask == 0] = float("nan")

    if not apply_noise:
        return _unbatch(blurred)

    noise = torch.randn_like(imgs)
    noise_reshape = noise.reshape(N * C, 1, H, W)
    noise_blurred = F.conv2d(F.pad(noise_reshape, (0, 0, pad_lat, pad_lat)), k_lat)
    noise_blurred = F.conv2d(F.pad(noise_blurred, (pad_lon, pad_lon, 0, 0)), k_lon).reshape(N, C, H, W)
    high_freq = noise - noise_blurred

    return _unbatch(blurred), _unbatch(high_freq)


def low_pass_tensor_batch(imgs: torch.Tensor, r_cut: int, apply_noise: bool = False,
                           method: str = "fourier", high: bool = False):
    if method == "fourier":
        return fourier_tensor_batch(imgs, r_cut, apply_noise, high=high)
    return low_pass_blur_ignore_nans(imgs, sigma=20 - r_cut, apply_noise=apply_noise)
