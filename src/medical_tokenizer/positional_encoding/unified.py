"""Modality-aware positional encoding built from the protocol's side channels.

Every token carries (see protocol/schema.py)
    mod  modality index     t  minutes since reference (-1 = n/a)     pos  (3,) index (-1 = n/a)
with pos meaning, per modality:
    text (i,-,-)   image (row,col,-)   video (t',row,col)   time-series (window,group,-)
    point cloud (x,y,z)
The encoding is the sum of
    E_mod[mod]                                         which modality
    Time(t)        sinusoidal on log1p(t / time_unit)  absolute time, shared by all modalities
    Axis_k(pos_k)  k = 0,1,2, per-modality tables      spatial / structural index
and a learned "n/a" vector wherever a channel is -1. Per-modality axis tables
keep, e.g., an image row index and a video frame index from sharing weights.
kind="sinusoidal" uses fixed sinusoids for the axes (no length limit),
kind="learned" uses embedding tables of size max_pos.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..protocol.vocab import MODALITIES


def sinusoid(x: torch.Tensor, d: int, max_period: float = 10000.0) -> torch.Tensor:
    """(...,) float -> (..., d) sin/cos features."""
    freq = torch.exp(-math.log(max_period) * torch.arange(0, d, 2, device=x.device).float() / d)
    a = x[..., None].float() * freq
    return torch.cat([a.sin(), a.cos()], -1)[..., :d]


class UnifiedPositionalEncoding(nn.Module):
    def __init__(self, d: int, kind: str = "learned", max_pos: int = 256, time_unit_min: float = 1.0,
                 use_time: bool = True, use_axes: bool = True):
        super().__init__()
        if kind not in ("learned", "sinusoidal"):
            raise ValueError(kind)
        self.d, self.kind, self.max_pos, self.time_unit = d, kind, max_pos, time_unit_min
        self.use_time, self.use_axes = use_time, use_axes
        n_mod = len(MODALITIES) + 1                       # +1: structural (<BOS>/<EOS>)
        self.mod = nn.Embedding(n_mod, d)
        self.time_proj = nn.Linear(d, d)
        self.time_na = nn.Parameter(torch.zeros(d))
        if kind == "learned":
            self.axes = nn.ModuleList([nn.Embedding(n_mod * (max_pos + 1), d) for _ in range(3)])
        else:
            self.axis_proj = nn.ModuleList([nn.Linear(d, d) for _ in range(3)])
            self.axis_mod = nn.Embedding(n_mod * 3, d)
        self.axis_na = nn.Parameter(torch.zeros(3, d))

    def forward(self, mod: torch.Tensor, t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        m = mod.long() + 1                                 # -1 -> 0
        out = self.mod(m)
        if self.use_time:
            te = self.time_proj(sinusoid(torch.log1p(t.clamp(min=0) / self.time_unit) * 100, self.d))
            out = out + torch.where((t < 0)[..., None], self.time_na.expand_as(te), te)
        if self.use_axes:
            for k in range(3):
                p = pos[..., k].long()
                na = (p < 0)[..., None]
                if self.kind == "learned":
                    e = self.axes[k](m * (self.max_pos + 1) + p.clamp(0, self.max_pos))
                else:
                    e = self.axis_proj[k](sinusoid(p.clamp(min=0), self.d)) + self.axis_mod(m * 3 + k)
                out = out + torch.where(na, self.axis_na[k].expand_as(e), e)
        return out
