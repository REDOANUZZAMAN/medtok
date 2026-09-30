"""Lightweight clip autoencoder for sampled video frames.

A clip of `clip_len` frames (B, T, H, W, C uint8) is encoded with factorised
3D convolutions: `log2(patch_size)` spatial stride-2 stages, then
`log2(temporal_stride)` temporal stride-2 stages, giving a latent grid of
(T / temporal_stride, H / patch_size, W / patch_size) vectors. With the default
8 x 64 x 64 clip, temporal_stride 4 and patch_size 16, a clip is 2 x 4 x 4 = 32
tokens. Temporal feature aggregation is the temporal convolution stack; the
decoder mirrors the encoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import LatentAutoencoder, register


class ResBlock3d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(nn.GELU(), nn.Conv3d(c, c, 3, padding=1), nn.GELU(), nn.Conv3d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


def _log2(n: int, what: str) -> int:
    k = int(round(math.log2(n)))
    if 2 ** k != n:
        raise ValueError(f"{what} must be a power of two, got {n}")
    return k


@register
class VideoClipAutoencoder(LatentAutoencoder):
    def __init__(self, frame_size: int = 64, clip_len: int = 8, in_channels: int = 3, patch_size: int = 16,
                 temporal_stride: int = 4, dim: int = 32, hidden: int = 64, n_res: int = 1,
                 quantizer: dict | None = None):
        super().__init__(dim, quantizer)
        self.cfg = dict(frame_size=frame_size, clip_len=clip_len, in_channels=in_channels, patch_size=patch_size,
                        temporal_stride=temporal_stride, dim=dim, hidden=hidden, n_res=n_res, quantizer=quantizer)
        ns, nt = _log2(patch_size, "patch_size"), _log2(temporal_stride, "temporal_stride")
        if frame_size % patch_size or clip_len % temporal_stride:
            raise ValueError("patch_size must divide frame_size and temporal_stride must divide clip_len")
        self.grid = (clip_len // temporal_stride, frame_size // patch_size, frame_size // patch_size)
        self.clip_len, self.frame_size = clip_len, frame_size
        enc, c = [], in_channels
        for _ in range(ns):
            enc += [nn.Conv3d(c, hidden, (1, 4, 4), (1, 2, 2), (0, 1, 1)), nn.GELU()]
            c = hidden
        for _ in range(nt):
            enc += [nn.Conv3d(c, hidden, (4, 1, 1), (2, 1, 1), (1, 0, 0)), nn.GELU()]
            c = hidden
        self.enc = nn.Sequential(*enc, *[ResBlock3d(hidden) for _ in range(n_res)], nn.GELU(),
                                 nn.Conv3d(hidden, dim, 1))
        dec = [nn.Conv3d(dim, hidden, 3, padding=1), *[ResBlock3d(hidden) for _ in range(n_res)], nn.GELU()]
        for _ in range(nt):
            dec += [nn.ConvTranspose3d(hidden, hidden, (4, 1, 1), (2, 1, 1), (1, 0, 0)), nn.GELU()]
        for i in range(ns):
            last = i == ns - 1
            dec += [nn.ConvTranspose3d(hidden, in_channels if last else hidden, (1, 4, 4), (1, 2, 2), (0, 1, 1))]
            if not last:
                dec.append(nn.GELU())
        self.dec = nn.Sequential(*dec)

    @staticmethod
    def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
        """(B,T,H,W,C) uint8 -> (B,C,T,H,W) float in [-1, 1]."""
        return x_uint8.permute(0, 4, 1, 2, 3).float() / 127.5 - 1

    @staticmethod
    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 4, 1)

    def encode_latent(self, x):                 # (B,C,T,H,W) -> (B,t,h,w,dim)
        return self.enc(x).permute(0, 2, 3, 4, 1)

    def decode_latent(self, z):                 # (B,t,h,w,dim) -> (B,C,T,H,W)
        return self.dec(z.permute(0, 4, 1, 2, 3))
