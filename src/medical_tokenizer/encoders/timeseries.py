"""Window autoencoder for regularly gridded, partially observed multivariate series.

Input: normalised grid x (B, T, V) and observation mask m (B, T, V). The series
is cut into windows of `window` steps every `stride` steps; each window
(values + mask, flattened) is encoded by an MLP into `groups` latent vectors of
size `dim` (product quantisation: one window -> `groups` tokens). Decoding maps
each window back and, when windows overlap (stride < window), averages the
overlapping predictions (overlap-add). Temporal order is kept by a learned
window-position embedding and by the token order.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LatentAutoencoder, register


@register
class TimeSeriesWindowAutoencoder(LatentAutoencoder):
    def __init__(self, n_vars: int = 37, steps: int = 48, window: int = 4, stride: int | None = None,
                 groups: int = 4, dim: int = 16, hidden: int = 256, quantizer: dict | None = None):
        super().__init__(dim, quantizer)
        stride = stride or window
        self.cfg = dict(n_vars=n_vars, steps=steps, window=window, stride=stride, groups=groups, dim=dim,
                        hidden=hidden, quantizer=quantizer)
        if (steps - window) % stride:
            raise ValueError(f"(steps - window) = {steps - window} must be divisible by stride {stride}")
        self.n_vars, self.steps, self.window, self.stride, self.groups = n_vars, steps, window, stride, groups
        self.n_windows = (steps - window) // stride + 1
        inp = window * n_vars * 2
        self.enc_in = nn.Linear(inp, hidden)
        self.pos = nn.Parameter(torch.zeros(self.n_windows, hidden))
        self.enc = nn.Sequential(nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, groups * dim))
        self.dec_pos = nn.Parameter(torch.zeros(self.n_windows, groups * dim))
        self.dec = nn.Sequential(nn.Linear(groups * dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, window * n_vars))

    @property
    def grid(self) -> tuple[int, int]:
        return self.n_windows, self.groups

    def _windows(self, a: torch.Tensor) -> torch.Tensor:          # (B,T,V) -> (B,W,window*V)
        w = a.unfold(1, self.window, self.stride)                 # (B,W,V,window)
        return w.permute(0, 1, 3, 2).reshape(a.shape[0], self.n_windows, self.window * self.n_vars)

    def encode_latent(self, xm) -> torch.Tensor:                  # (x, m) -> (B,W,G,dim)
        x, m = xm
        h = self.enc_in(torch.cat([self._windows(x), self._windows(m.float())], -1)) + self.pos
        return self.enc(h).view(x.shape[0], self.n_windows, self.groups, self.dim)

    def decode_latent(self, z) -> torch.Tensor:                   # (B,W,G,dim) -> (B,T,V)
        B = z.shape[0]
        out = self.dec(z.reshape(B, self.n_windows, self.groups * self.dim) + self.dec_pos)
        if self.stride == self.window:
            return out.view(B, self.steps, self.n_vars)
        cols = out.view(B, self.n_windows, self.window, self.n_vars).permute(0, 3, 2, 1)   # (B,V,window,W)
        cols = cols.reshape(B, self.n_vars * self.window, self.n_windows)
        y = F.fold(cols, (self.steps, 1), (self.window, 1), stride=(self.stride, 1))        # (B,V,T,1)
        ones = torch.ones(1, self.window, self.n_windows, device=z.device)
        cnt = F.fold(ones, (self.steps, 1), (self.window, 1), stride=(self.stride, 1))
        return (y / cnt).squeeze(-1).permute(0, 2, 1)


def masked_mse(xr: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    m = m.float()
    return ((xr - x) ** 2 * m).sum() / m.sum().clamp(min=1)
