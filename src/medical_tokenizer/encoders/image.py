"""Lightweight 2D image autoencoders.

patch_size p = side length (pixels) of the image region one latent vector /
token covers; an HxW image becomes an (H/p)x(W/p) token grid.

  arch="conv"   hierarchical: log2(p) stride-2 convolutions (p must be a power of 2)
  arch="patch"  ViT-style patch embedding: one p x p stride-p convolution (any p
                dividing the image size), then residual blocks on the grid.
                Used for the patch-size ablation, where every p shares one design.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import LatentAutoencoder, register


class ResBlock2d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.net = nn.Sequential(nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


@register
class ImageAutoencoder(LatentAutoencoder):
    def __init__(self, image_size: int = 28, in_channels: int = 3, patch_size: int = 4, dim: int = 32,
                 hidden: int = 96, n_res: int = 2, arch: str = "conv", quantizer: dict | None = None):
        super().__init__(dim, quantizer)
        self.cfg = dict(image_size=image_size, in_channels=in_channels, patch_size=patch_size, dim=dim,
                        hidden=hidden, n_res=n_res, arch=arch, quantizer=quantizer)
        if image_size % patch_size:
            raise ValueError(f"patch_size {patch_size} must divide image_size {image_size}")
        self.grid = image_size // patch_size
        res = [ResBlock2d(hidden) for _ in range(n_res)]
        res_d = [ResBlock2d(hidden) for _ in range(n_res)]
        if arch == "conv":
            n_down = int(round(math.log2(patch_size)))
            if 2 ** n_down != patch_size:
                raise ValueError("arch='conv' needs a power-of-two patch_size; use arch='patch'")
            down, c = [], in_channels
            for i in range(n_down):
                down += [nn.Conv2d(c, hidden, 4, 2, 1)] + ([nn.GELU()] if i < n_down - 1 else [])
                c = hidden
            if n_down == 0:
                down = [nn.Conv2d(in_channels, hidden, 3, padding=1)]
            up = []
            for i in range(n_down):
                last = i == n_down - 1
                up += [nn.ConvTranspose2d(hidden, in_channels if last else hidden, 4, 2, 1)] + ([] if last else [nn.GELU()])
            if n_down == 0:
                up = [nn.Conv2d(hidden, in_channels, 3, padding=1)]
            self.enc = nn.Sequential(*down, *res, nn.GELU(), nn.Conv2d(hidden, dim, 1))
            self.dec = nn.Sequential(nn.Conv2d(dim, hidden, 3, padding=1), *res_d, nn.GELU(), *up)
        elif arch == "patch":
            self.enc = nn.Sequential(nn.Conv2d(in_channels, hidden, patch_size, patch_size), *res, nn.GELU(),
                                     nn.Conv2d(hidden, dim, 1))
            self.dec = nn.Sequential(nn.Conv2d(dim, hidden, 3, padding=1), *res_d, nn.GELU(),
                                     nn.ConvTranspose2d(hidden, hidden, patch_size, patch_size), nn.GELU(),
                                     nn.Conv2d(hidden, in_channels, 3, padding=1))
        else:
            raise ValueError(f"unknown arch {arch!r}")

    @staticmethod
    def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
        """(B,H,W,C) uint8 -> (B,C,H,W) float in [-1, 1]."""
        return x_uint8.permute(0, 3, 1, 2).float() / 127.5 - 1

    @staticmethod
    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) float in [-1, 1] -> (B,H,W,C) uint8."""
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)

    def encode_latent(self, x):                 # (B,C,H,W) -> (B,g,g,dim)
        return self.enc(x).permute(0, 2, 3, 1)

    def decode_latent(self, z):                 # (B,g,g,dim) -> (B,C,H,W)
        return self.dec(z.permute(0, 3, 1, 2))
