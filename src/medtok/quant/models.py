"""Lightweight encoders / decoders for the discrete tokenizers.

ImageVQVAE   28x28x3 -> 7x7 grid of latent vectors -> 7x7 code indices.
TSPatchVQ    48 h x 37 var grid (+ mask) -> 12 patches of 4 h, each encoded to
             G latent vectors (product quantisation) -> 12*G code indices.

Both expose encode_latent (continuous, pre-quantisation), quantize, and
decode_latent so the same trained network serves the continuous-projection
and the VQ representation -- the comparison then isolates the effect of
discretisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vq import VectorQuantizer


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.GELU(), nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 1))

    def forward(self, x):
        return x + self.net(x)


class ImageVQVAE(nn.Module):
    def __init__(self, n_codes=512, dim=32, hidden=96):
        super().__init__()
        self.cfg = dict(n_codes=n_codes, dim=dim, hidden=hidden)
        self.enc = nn.Sequential(
            nn.Conv2d(3, hidden, 4, 2, 1), nn.GELU(),          # 14
            nn.Conv2d(hidden, hidden, 4, 2, 1),                # 7
            ResBlock(hidden), ResBlock(hidden), nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        self.vq = VectorQuantizer(n_codes, dim)
        self.dec = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1),
            ResBlock(hidden), ResBlock(hidden), nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(hidden, 3, 4, 2, 1),
        )

    @staticmethod
    def to_input(x_uint8: torch.Tensor) -> torch.Tensor:
        """(B,28,28,3) uint8 -> (B,3,28,28) float in [-1,1]."""
        return x_uint8.permute(0, 3, 1, 2).float() / 127.5 - 1

    @staticmethod
    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)

    def encode_latent(self, x):                 # -> (B,7,7,dim)
        return self.enc(x).permute(0, 2, 3, 1)

    def decode_latent(self, z):                 # (B,7,7,dim) -> (B,3,28,28)
        return self.dec(z.permute(0, 3, 1, 2))

    def forward(self, x):
        z = self.encode_latent(x)
        q, idx, commit, stats = self.vq(z)
        xr = self.decode_latent(q)
        return xr, idx, commit, stats

    def decode_codes(self, idx):                # (B,7,7) long
        return self.decode_latent(self.vq.lookup(idx))


class TSPatchVQ(nn.Module):
    """Masked patch autoencoder for the hourly ICU grid."""

    def __init__(self, n_vars=37, hours=48, patch=4, groups=4, dim=16, n_codes=512, hidden=256):
        super().__init__()
        self.cfg = dict(n_vars=n_vars, hours=hours, patch=patch, groups=groups, dim=dim,
                        n_codes=n_codes, hidden=hidden)
        self.n_vars, self.patch, self.groups, self.dim = n_vars, patch, groups, dim
        self.n_patches = hours // patch
        inp = patch * n_vars * 2  # values + mask
        self.pos = nn.Parameter(torch.zeros(self.n_patches, hidden))
        self.enc = nn.Sequential(nn.Linear(inp, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, groups * dim))
        self.vq = VectorQuantizer(n_codes, dim)
        self.dec = nn.Sequential(nn.Linear(groups * dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, patch * n_vars))
        self.dec_pos = nn.Parameter(torch.zeros(self.n_patches, groups * dim))

    def _patches(self, x, m):                   # (B,H,V) -> (B,P,patch*V*2)
        B = x.shape[0]
        xp = x.view(B, self.n_patches, self.patch * self.n_vars)
        mp = m.float().view(B, self.n_patches, self.patch * self.n_vars)
        return torch.cat([xp, mp], -1)

    def encode_latent(self, x, m):              # -> (B,P,G,dim)
        p = self._patches(x, m)
        h = self.enc[0](p) + self.pos
        h = self.enc[1:](h)
        return h.view(x.shape[0], self.n_patches, self.groups, self.dim)

    def decode_latent(self, z):                 # (B,P,G,dim) -> (B,H,V)
        B = z.shape[0]
        h = z.reshape(B, self.n_patches, self.groups * self.dim) + self.dec_pos
        return self.dec(h).view(B, self.n_patches * self.patch, self.n_vars)

    def forward(self, x, m):
        z = self.encode_latent(x, m)
        q, idx, commit, stats = self.vq(z)
        return self.decode_latent(q), idx, commit, stats

    def decode_codes(self, idx):                # (B,P,G)
        return self.decode_latent(self.vq.lookup(idx))


def masked_mse(xr, x, m):
    m = m.float()
    return ((xr - x) ** 2 * m).sum() / m.sum().clamp(min=1)


def save_model(model: nn.Module, path: str, **extra):
    torch.save({"cls": type(model).__name__, "cfg": model.cfg, "state": model.state_dict(), **extra}, path)


def load_model(path: str) -> tuple[nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cls = {"ImageVQVAE": ImageVQVAE, "TSPatchVQ": TSPatchVQ}[ck["cls"]]
    m = cls(**ck["cfg"])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck
