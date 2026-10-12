"""UNetFlow: the conditional flow-matching velocity network.

This is the DOY + static-conditioning version (matches the checkpoints
actually used for SerpentFlow-CLIM / clim5). An earlier, simpler variant
without DOY/static support was left duplicated in one of the visualization
notebooks' "Utils" cells; that variant is not used to produce any reported
result (those notebooks load precomputed prediction tensors rather than
re-instantiating the model) and is not reproduced here to avoid the
confusion of two competing model classes with the same name.
"""

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_utils import ensure_batched


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(ch: int) -> nn.GroupNorm:
    for g in (32, 16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class DOYEmbedding(nn.Module):
    """day-of-year (1..365) -> Fourier features -> MLP -> (B, out_dim)."""

    def __init__(self, out_dim: int, n_frequencies: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_frequencies, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        x = (doy.float() - 1.0) / 365.0
        freq = torch.arange(1, self.n_frequencies + 1, device=doy.device, dtype=x.dtype)
        angle = 2 * math.pi * x[:, None] * freq[None, :]
        fourier = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.mlp(fourier)


class TimeDOYFusion(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.time_norm = nn.LayerNorm(embed_dim)
        self.doy_norm = nn.LayerNorm(embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, time_emb: torch.Tensor, doy_emb: torch.Tensor) -> torch.Tensor:
        fused = self.fusion(torch.cat([self.time_norm(time_emb), self.doy_norm(doy_emb)], dim=-1))
        return time_emb + fused


class TimestepBlock(nn.Module):
    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, out_channels or channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv=True, out_channels=None):
        super().__init__()
        self.op = (nn.Conv2d(channels, out_channels or channels, 3, stride=2, padding=1)
                   if use_conv else nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout=0.0, out_channels=None,
                 use_scale_shift_norm=False, up=False, down=False):
        super().__init__()
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels), nn.SiLU(), nn.Conv2d(channels, self.out_channels, 3, padding=1)
        )
        if up:
            self.h_upd = Upsample(channels, use_conv=False)
            self.x_upd = Upsample(channels, use_conv=False)
        elif down:
            self.h_upd = Downsample(channels, use_conv=False)
            self.x_upd = Downsample(channels, use_conv=False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        self.skip_connection = (nn.Identity() if self.out_channels == channels
                                 else nn.Conv2d(channels, self.out_channels, 1))

    def forward(self, x, emb):
        h = self.in_layers(x)
        h_emb = self.emb_layers(emb).type(h.dtype)
        while len(h_emb.shape) < len(h.shape):
            h_emb = h_emb[..., None, None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(h_emb, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = self.out_layers(h + h_emb)
        return self.skip_connection(self.x_upd(x)) + h


class AttentionBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = normalization(ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_in = x
        x = x.view(b, c, -1)
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(c), dim=-1)
        out = (attn @ v.transpose(1, 2)).transpose(1, 2)
        return x_in + self.proj(out).view(b, c, out.shape[-1] // w, w)


class CBAM(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch, ch // reduction, 1), nn.SiLU(),
                                 nn.Conv2d(ch // reduction, ch, 1), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid())

    def forward(self, x):
        x = x * self.ca(x)
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * self.sa(torch.cat([avg, mx], 1))


class LiteGN_CBAM(nn.Module):
    def __init__(self, ch, groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(min(groups, ch), ch)
        self.conv1d = nn.Conv1d(1, 1, 3, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.gn(x).mean((2, 3))
        y = self.conv1d(y.unsqueeze(1)).sigmoid().squeeze(1)
        return x * y[:, :, None, None]


class CrossChannelAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = normalization(ch)
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)

    def forward(self, x):
        b, c, h, w = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_norm = self.norm(x_flat.permute(0, 2, 1)).permute(0, 2, 1)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.view(b, h * w, self.num_heads, c // self.num_heads).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c // self.num_heads), dim=-1)
        out = self.proj((attn @ v).transpose(1, 2).reshape(b, h * w, c))
        return x + out.reshape(b, h, w, c).permute(0, 3, 1, 2)


class LocalCrossAttention(nn.Module):
    def __init__(self, channels, patch_size=5, num_heads=2):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, h, w = x.shape
        ph, pw = self.patch_size, self.patch_size
        pad_h, pad_w = (ph - h % ph) % ph, (pw - w % pw) % pw
        x_padded = F.pad(x, (0, pad_w, 0, pad_h))
        H, W = x_padded.shape[2], x_padded.shape[3]

        x_patches = x_padded.unfold(2, ph, ph).unfold(3, pw, pw)
        nH, nW = x_patches.shape[2], x_patches.shape[3]
        x_patches = x_patches.permute(0, 2, 3, 4, 5, 1).contiguous().view(-1, ph * pw, c)

        q, k, v = self.qkv(self.norm(x_patches)).chunk(3, dim=-1)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, c // self.num_heads).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(c // self.num_heads), dim=-1)
        out = self.proj((attn @ v).transpose(1, 2).reshape(x_patches.shape[0], ph * pw, c))

        out = out.view(b, nH, nW, ph, pw, c).permute(0, 5, 1, 3, 2, 4).contiguous().view(b, c, H, W)
        H_min, W_min = min(h, out.shape[2]), min(w, out.shape[3])
        return x[:, :, :H_min, :W_min] + out[:, :, :H_min, :W_min]


_ATTENTION_TYPES = {
    "self": AttentionBlock,
    "cbam": CBAM,
    "lite_gn_cbam": LiteGN_CBAM,
    "cross_channel": CrossChannelAttention,
    "local_cross": lambda ch: LocalCrossAttention(ch, patch_size=5, num_heads=2),
}


def make_attention(ch, ds, attn_res, attn_type):
    if ds not in attn_res or attn_type == "none":
        return nn.Identity()
    if attn_type not in _ATTENTION_TYPES:
        raise ValueError(attn_type)
    return _ATTENTION_TYPES[attn_type](ch)


@dataclass(eq=False)
class UNetFlow(nn.Module):
    C_in: int
    C_out: int
    base_ch: int = 128
    num_res_blocks: int = 2
    attention_resolutions: Tuple[int] = (1, 2)
    dropout: float = 0.0
    ch_mult: Tuple[int] = (1, 2, 4)
    conv_resample: bool = True
    use_scale_shift_norm: bool = False
    resblock_updown: bool = False
    attention_type: str = "none"
    time_factor: int = 4
    stats_dim: int = 2
    static_dim: int = 4
    num_doy: int = 365
    doy_fourier_dim: int = 16
    doy_hidden_dim: int = 256
    use_doy: bool = True

    def __post_init__(self):
        super().__init__()

        self.time_embed_dim = self.base_ch * self.time_factor
        self.time_embed = nn.Sequential(
            nn.Linear(self.base_ch, self.time_embed_dim), nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        if self.use_doy:
            self.doy_embed = DOYEmbedding(self.time_embed_dim, self.doy_fourier_dim, self.doy_hidden_dim)
            self.time_doy_fusion = TimeDOYFusion(self.time_embed_dim)
        self.register_buffer("static_cond", None, persistent=True)

        ch = input_ch = self.ch_mult[0] * self.base_ch
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(self.C_in + self.static_dim, ch, 3, padding=1))]
        )
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.ch_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(ch, self.time_embed_dim, self.dropout, out_channels=mult * self.base_ch,
                             use_scale_shift_norm=self.use_scale_shift_norm),
                    make_attention(mult * self.base_ch, ds, self.attention_resolutions, self.attention_type),
                ]
                ch = mult * self.base_ch
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(self.ch_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, self.conv_resample, ch)))
                input_block_chans.append(ch)
                ds *= 2

        ch += self.stats_dim
        middle_layers = [ResBlock(ch, self.time_embed_dim, self.dropout,
                                   use_scale_shift_norm=self.use_scale_shift_norm)]
        if ds in self.attention_resolutions:
            middle_layers.append(AttentionBlock(ch))
        middle_layers.append(ResBlock(ch, self.time_embed_dim, self.dropout, out_channels=ch,
                                       use_scale_shift_norm=self.use_scale_shift_norm))
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        self.output_blocks = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(self.ch_mult))):
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, self.time_embed_dim, self.dropout, out_channels=mult * self.base_ch,
                             use_scale_shift_norm=self.use_scale_shift_norm),
                    make_attention(mult * self.base_ch, ds, self.attention_resolutions, self.attention_type),
                ]
                ch = mult * self.base_ch
                if level and i == self.num_res_blocks:
                    layers.append(Upsample(ch, self.conv_resample))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(normalization(ch), nn.SiLU(),
                                  zero_module(nn.Conv2d(input_ch, self.C_out, 3, padding=1)))
        if self.stats_dim > 0:
            self.stats_proj = nn.Linear(2 * self.C_in, self.stats_dim)

    def set_static(self, static: torch.Tensor):
        self.static_cond = ensure_batched(static.float(), expected_channels=self.static_dim, what="static")

    def forward(self, x, timesteps, stats=None, doy=None):
        if self.static_dim > 0:
            assert self.static_cond is not None, "call model.set_static(static) first"
            static = self.static_cond.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, static.expand(x.shape[0], -1, -1, -1)], dim=1)

        time_emb = self.time_embed(timestep_embedding(timesteps, self.base_ch).to(x))
        emb = self.time_doy_fusion(time_emb, self.doy_embed(doy)) if doy is not None else time_emb

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)

        if stats is not None and self.stats_dim > 0:
            B, C, _ = stats.shape
            stats_emb = self.stats_proj(stats.view(B, C * 2))[:, :, None, None].expand(-1, -1, h.shape[2], h.shape[3])
            h = torch.cat([h, stats_emb], dim=1)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            if h.shape[-2:] != hs[-1].shape[-2:]:
                h = F.interpolate(h, size=hs[-1].shape[-2:], mode="nearest")
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
