"""U-Net architectures shared by iDownscale, ConsistencyModel, and CorrDiff-QM.

Variants:
    UNet         — plain conv encoder-decoder (fast; good for regression)
    UNetAttention — same backbone + multi-head self-attention at the two lowest
                    spatial resolutions (bottleneck + enc4 level); better for
                    generative tasks (consistency distillation)
    DiffusersUNet — Hess et al. 2025 diffusers UNet2DModel (time via FiLM).

UNet / UNetAttention handle the 57×65 ERA5 grid internally: padding to 64×80
in forward(), cropped back to H_OUT×W_OUT before returning.

Factory:
    build_unet(arch, in_channels, out_channels, ...)
        arch = "unet"       → UNet
        arch = "unet_attn"  → UNetAttention
        arch = "diffusers"  → DiffusersUNet
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as TF

# Pad (left, right, top, bottom): 65→80, 57→64
PAD    = (0, 15, 0, 7)
H_OUT  = 57
W_OUT  = 65


# ── Shared building blocks ────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class _SelfAttn2d(nn.Module):
    """Multi-head self-attention over spatial positions."""
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        # GroupNorm instead of LayerNorm so we stay in [B, C, H, W] world
        n_groups = min(8, channels)
        self.norm = nn.GroupNorm(n_groups, channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)       # residual


# ── Plain U-Net ───────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """4-level U-Net — plain conv blocks, no attention."""

    def __init__(self, in_channels: int = 7, out_channels: int = 5,
                 init_features: int = 32):
        super().__init__()
        f = init_features
        self.enc1      = _conv_block(in_channels, f)
        self.enc2      = _conv_block(f,      f * 2)
        self.enc3      = _conv_block(f * 2,  f * 4)
        self.enc4      = _conv_block(f * 4,  f * 8)
        self.bottleneck = _conv_block(f * 8, f * 16)
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = _conv_block(f * 16, f * 8)
        self.up3  = nn.ConvTranspose2d(f * 8,  f * 4, 2, stride=2)
        self.dec3 = _conv_block(f * 8,  f * 4)
        self.up2  = nn.ConvTranspose2d(f * 4,  f * 2, 2, stride=2)
        self.dec2 = _conv_block(f * 4,  f * 2)
        self.up1  = nn.ConvTranspose2d(f * 2,  f,     2, stride=2)
        self.dec1 = _conv_block(f * 2,  f)
        self.out  = nn.Conv2d(f, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = TF.pad(x, PAD)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)[:, :, :H_OUT, :W_OUT]


# ── U-Net with self-attention ─────────────────────────────────────────────────

class UNetAttention(nn.Module):
    """4-level U-Net + multi-head self-attention at the two lowest resolutions.

    Attention is applied after the bottleneck and enc4 conv blocks.  At those
    depths the spatial size is 4×5 and 8×10 (padded grid), so attention is
    cheap while capturing long-range structure — important for generative
    objectives (consistency distillation).
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 5,
                 init_features: int = 32, n_attn_heads: int = 4):
        super().__init__()
        f = init_features
        self.enc1       = _conv_block(in_channels, f)
        self.enc2       = _conv_block(f,      f * 2)
        self.enc3       = _conv_block(f * 2,  f * 4)
        self.enc4       = _conv_block(f * 4,  f * 8)
        self.attn_e4    = _SelfAttn2d(f * 8,  n_attn_heads)
        self.bottleneck = _conv_block(f * 8,  f * 16)
        self.attn_b     = _SelfAttn2d(f * 16, n_attn_heads)
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = _conv_block(f * 16, f * 8)
        self.up3  = nn.ConvTranspose2d(f * 8,  f * 4, 2, stride=2)
        self.dec3 = _conv_block(f * 8,  f * 4)
        self.up2  = nn.ConvTranspose2d(f * 4,  f * 2, 2, stride=2)
        self.dec2 = _conv_block(f * 4,  f * 2)
        self.up1  = nn.ConvTranspose2d(f * 2,  f,     2, stride=2)
        self.dec1 = _conv_block(f * 2,  f)
        self.out  = nn.Conv2d(f, out_channels, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x  = TF.pad(x, PAD)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.attn_e4(self.enc4(self.pool(e3)))
        b  = self.attn_b(self.bottleneck(self.pool(e4)))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)[:, :, :H_OUT, :W_OUT]


# ── Diffusers U-Net (Hess et al. 2025) ───────────────────────────────────────

class DiffusersUNet(nn.Module):
    """U-Net matching Hess et al. 2025 via diffusers.UNet2DModel.

    Architecture: 128/128/256/256 channels, SiLU, GroupNorm, attention only
    at the bottleneck. Time conditioning via sinusoidal embedding + FiLM
    (built into UNet2DModel), not via channel concatenation. ~27M params.

    Forward signature: (x, timestep) → prediction
        x:         [B, in_channels, H, W]
        timestep:  [B] float tensor, log-uniform scaled to [0, 1000]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        try:
            from diffusers import UNet2DModel
        except ImportError as e:
            raise ImportError(
                "diffusers is required for DiffusersUNet: pip install diffusers"
            ) from e
        self._net = UNet2DModel(
            in_channels=in_channels,
            out_channels=out_channels,
            block_out_channels=(128, 128, 256, 256),
            layers_per_block=2,
            # Plain blocks at all levels; bottleneck self-attention comes from
            # the default UNetMidBlock2D (matching Hess et al. 2025 "attention
            # layer at the architecture bottleneck").
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
            ),
            norm_num_groups=32,
            act_fn="silu",
        )

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        x = TF.pad(x, PAD)
        out = self._net(x, timestep).sample
        return out[:, :, :H_OUT, :W_OUT]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_unet(
    arch:          str,
    in_channels:   int,
    out_channels:  int,
    init_features: int  = 32,
    n_attn_heads:  int  = 4,
) -> nn.Module:
    """Instantiate a UNet variant.

    Default arch per baseline:
        idownscale    → "unet"          (deterministic regression; fast)
        consistency   → "diffusers"     (Hess et al. 2025: UNet2DModel + FiLM)
    """
    if arch == "unet":
        return UNet(in_channels, out_channels, init_features)
    if arch == "unet_attn":
        return UNetAttention(in_channels, out_channels, init_features, n_attn_heads)
    if arch == "diffusers":
        return DiffusersUNet(in_channels, out_channels)
    raise ValueError(f"Unknown arch {arch!r}.")
