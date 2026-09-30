"""Vector quantisation (VQ-VAE style) and an identity stand-in for continuous baselines.

VectorQuantizer options, each switchable for ablations of codebook collapse:
  update="loss"   classic VQ-VAE (van den Oord et al., 2017): the codebook is a
                  parameter trained by the codebook loss ||sg(z) - e||^2
  update="ema"    codebook = exponential moving average of assigned encoder
                  outputs (same paper, appendix); no codebook loss
  normalize=True  L2-normalise inputs and codes before lookup (cosine codebook,
                  Yu et al., 2022, ViT-VQGAN)
  restart_dead    re-initialise codes whose running usage falls below
                  `restart_threshold` assignments per batch to random encoder
                  outputs of the current batch (Dhariwal et al., 2020, Jukebox)
Both modes add the commitment loss beta * ||z - sg(e)||^2 and pass gradients to
the encoder with the straight-through estimator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantizerOutput:
    z_q: torch.Tensor                    # quantised latents (straight-through), same shape as input
    indices: torch.Tensor | None         # (...,) code ids; None for the identity quantiser
    loss: torch.Tensor                   # total quantiser loss to add to the reconstruction loss
    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    perplexity: float                    # of this batch's code usage
    active_codes: int                    # distinct codes used in this batch
    restarted: int = 0                   # codes re-initialised in this step


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes: int, dim: int, update: str = "ema", beta: float = 0.25, decay: float = 0.99,
                 normalize: bool = True, restart_dead: bool = True, restart_threshold: float = 1.0,
                 init: str = "data"):
        super().__init__()
        if update not in ("ema", "loss"):
            raise ValueError(f"update must be 'ema' or 'loss', got {update!r}")
        self.n_codes, self.dim, self.update, self.beta, self.decay = n_codes, dim, update, beta, decay
        self.normalize, self.restart_dead, self.restart_threshold = normalize, restart_dead, restart_threshold
        self.init = init
        w = torch.randn(n_codes, dim) / (1 if normalize else dim ** 0.5)
        if normalize:
            w = F.normalize(w, dim=-1)
        self.codebook = nn.Parameter(w, requires_grad=(update == "loss"))
        self.register_buffer("ema_count", torch.ones(n_codes))     # running assignments per code per batch
        self.register_buffer("ema_sum", w.clone())
        self.register_buffer("initialised", torch.tensor(init != "data"))

    # ---------------------------------------------------------------- lookup
    def _prep(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1) if self.normalize else z

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        e = self.codebook
        return flat.pow(2).sum(1, keepdim=True) - 2 * flat @ e.T + e.pow(2).sum(1)[None]

    @torch.no_grad()
    def codes(self, z: torch.Tensor) -> torch.Tensor:
        """Nearest-code ids (token ids local to the codebook) for latents z (..., dim)."""
        flat = self._prep(z.reshape(-1, self.dim))
        return self._distances(flat).argmin(1).view(z.shape[:-1])

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        """Decode token ids to code vectors: (...,) -> (..., dim)."""
        return F.embedding(idx, self.codebook)

    # --------------------------------------------------------------- training
    @torch.no_grad()
    def _init_from(self, flat: torch.Tensor) -> None:
        idx = torch.randint(0, flat.shape[0], (self.n_codes,), device=flat.device)
        self.codebook.data.copy_(flat[idx])
        self.ema_sum.copy_(flat[idx])
        self.ema_count.fill_(1.0)
        self.initialised.fill_(True)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
        n = self.ema_count.sum()
        count = (self.ema_count + 1e-5) / (n + self.n_codes * 1e-5) * n     # Laplace smoothing
        w = self.ema_sum / count[:, None]
        self.codebook.data.copy_(F.normalize(w, dim=-1) if self.normalize else w)

    @torch.no_grad()
    def _restart(self, flat: torch.Tensor) -> int:
        dead = self.ema_count < self.restart_threshold
        k = int(dead.sum())
        if k:
            r = flat[torch.randint(0, flat.shape[0], (k,), device=flat.device)]
            self.codebook.data[dead] = r
            self.ema_sum[dead] = r
            self.ema_count[dead] = 1.0
        return k

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        shape = z.shape
        flat = self._prep(z.reshape(-1, self.dim))
        if self.training and not bool(self.initialised):
            self._init_from(flat.detach())
        idx = self._distances(flat.detach()).argmin(1)
        q = F.embedding(idx, self.codebook)
        restarted = 0
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                if self.update == "ema":
                    self._ema_update(flat.detach(), onehot)
                if self.restart_dead:
                    restarted = self._restart(flat.detach())
        commit = F.mse_loss(flat, q.detach())
        codebook_loss = F.mse_loss(q, flat.detach()) if self.update == "loss" else torch.zeros((), device=z.device)
        loss = self.beta * commit + codebook_loss
        q_st = flat + (q - flat).detach()
        with torch.no_grad():
            p = torch.bincount(idx, minlength=self.n_codes).float()
            p = p / p.sum()
            ppl = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
            active = int((p > 0).sum())
        return QuantizerOutput(q_st.view(shape), idx.view(shape[:-1]), loss, commit, codebook_loss, ppl, active,
                               restarted)


class IdentityQuantizer(nn.Module):
    """No quantisation: the continuous-projection baseline shares the autoencoder code path."""

    def __init__(self, dim: int, normalize: bool = False):
        super().__init__()
        self.dim, self.normalize, self.n_codes = dim, normalize, 0

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        zq = F.normalize(z, dim=-1) if self.normalize else z
        zero = torch.zeros((), device=z.device)
        return QuantizerOutput(zq, None, zero, zero, zero, float("nan"), 0)


def build_quantizer(cfg: dict | None, dim: int) -> nn.Module:
    """cfg None / {"type": "none"} -> IdentityQuantizer; {"type": "vq", ...} -> VectorQuantizer."""
    cfg = dict(cfg or {"type": "none"})
    kind = cfg.pop("type", "vq")
    if kind == "none":
        return IdentityQuantizer(dim, normalize=cfg.get("normalize", False))
    if kind != "vq":
        raise ValueError(f"unknown quantizer type {kind!r}")
    return VectorQuantizer(dim=dim, **cfg)
