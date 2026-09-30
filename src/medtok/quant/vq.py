"""Vector quantizer with EMA codebook updates and dead-code restarts.

Codebook collapse (a few codes absorb everything, the rest die) is the main
failure mode of discrete tokenizers. Two standard remedies are used:
  * EMA updates of code vectors (van den Oord et al. 2017, appendix) instead of
    a codebook loss -- more stable on small data;
  * codes whose EMA usage drops below `restart_thresh` hits per batch are
    re-initialised to random encoder outputs from the current batch
    (Jukebox, Dhariwal et al. 2020).
Inputs are L2-normalised before lookup (cosine codebook, Yu et al. 2022),
which further improves utilisation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes: int, dim: int, decay: float = 0.99, beta: float = 0.25,
                 restart_thresh: float = 1.0, normalize: bool = True):
        super().__init__()
        self.n_codes, self.dim, self.decay, self.beta = n_codes, dim, decay, beta
        self.restart_thresh, self.normalize = restart_thresh, normalize
        embed = torch.randn(n_codes, dim)
        if normalize:
            embed = F.normalize(embed, dim=-1)
        self.register_buffer("embed", embed)
        self.register_buffer("ema_count", torch.ones(n_codes))
        self.register_buffer("ema_sum", embed.clone())
        self.register_buffer("initted", torch.tensor(False))

    def _prep(self, z):
        return F.normalize(z, dim=-1) if self.normalize else z

    @torch.no_grad()
    def _init_from(self, flat):
        idx = torch.randint(0, flat.shape[0], (self.n_codes,), device=flat.device)
        self.embed.copy_(flat[idx])
        self.ema_sum.copy_(flat[idx])
        self.ema_count.fill_(1.0)
        self.initted.fill_(True)

    def codes(self, z: torch.Tensor) -> torch.Tensor:
        """Nearest-code indices for z (..., dim)."""
        flat = self._prep(z.reshape(-1, self.dim))
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.T + self.embed.pow(2).sum(1)[None]
        return d.argmin(1).view(z.shape[:-1])

    def lookup(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.embed)

    def forward(self, z: torch.Tensor):
        """Returns (z_q with straight-through gradient, indices, commitment loss, stats)."""
        shape = z.shape
        flat = self._prep(z.reshape(-1, self.dim))
        if self.training and not bool(self.initted):
            self._init_from(flat.detach())
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.T + self.embed.pow(2).sum(1)[None]
        idx = d.argmin(1)
        q = F.embedding(idx, self.embed)
        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.n_codes).type(flat.dtype)
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                n = self.ema_count.sum()
                count = (self.ema_count + 1e-5) / (n + self.n_codes * 1e-5) * n
                new = self.ema_sum / count[:, None]
                self.embed.copy_(F.normalize(new, dim=-1) if self.normalize else new)
                dead = self.ema_count < self.restart_thresh
                if dead.any():
                    r = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]
                    self.embed[dead] = r
                    self.ema_sum[dead] = r
                    self.ema_count[dead] = 1.0
        commit = F.mse_loss(flat, q.detach())
        q_st = flat + (q - flat).detach()
        probs = torch.bincount(idx, minlength=self.n_codes).float()
        probs = probs / probs.sum()
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        stats = {"perplexity": perplexity.item(), "used": int((probs > 0).sum())}
        return q_st.view(shape), idx.view(shape[:-1]), self.beta * commit, stats


def usage_stats(indices, n_codes: int) -> dict:
    """Codebook utilisation over a whole dataset of indices (any iterable of int arrays)."""
    import numpy as np
    counts = np.zeros(n_codes, np.int64)
    for a in indices:
        counts += np.bincount(np.asarray(a).ravel(), minlength=n_codes)
    p = counts / max(counts.sum(), 1)
    nz = p[p > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    return {"n_codes": n_codes, "used": int((counts > 0).sum()), "utilization": float((counts > 0).mean()),
            "perplexity": float(2 ** ent), "entropy_bits": ent, "counts": counts.tolist()}
