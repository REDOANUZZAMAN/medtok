"""UnifiedTokenTransformer: one encoder for every modality / representation mix.

    x_i = E_tok[id_i] + PE(mod_i, t_i, pos_i)
At <CONT> slots the token embedding is replaced by a per-modality linear
projection of the continuous vector (the "projection layer" of the continuous
baseline), so continuous and discrete inputs share the backbone. A pre-LN
Transformer encoder, masked mean pooling, and one linear head per task follow.
The model never inspects raw modality data.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..positional_encoding.unified import UnifiedPositionalEncoding
from ..protocol.vocab import CONT, MODALITIES, PAD, VOCAB_SIZE


class EncoderBlock(nn.Module):
    """Pre-LN Transformer block using F.scaled_dot_product_attention without attention dropout.

    With attention dropout, PyTorch's CPU attention falls back to a kernel that materialises the full
    (B, heads, L, L) matrix -- several GB for 800-token event sequences. Dropout is applied to the
    residual branches instead.
    """

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        self.h, self.drop = heads, dropout
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + F.dropout(self.out(a.transpose(1, 2).reshape(B, L, D)), self.drop, self.training)
        return x + F.dropout(self.ff(self.n2(x)), self.drop, self.training)


class UnifiedTokenTransformer(nn.Module):
    def __init__(self, tasks: dict[str, int], cont_dims: dict[str, int] | None = None, d_model: int = 128,
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1, pe: dict | None = None):
        super().__init__()
        self.cfg = dict(tasks=tasks, cont_dims=cont_dims, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                        dropout=dropout, pe=pe)
        self.tok = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD)
        self.pe = UnifiedPositionalEncoding(d_model, **(pe or {}))
        self.cont_proj = nn.ModuleDict({m: nn.Linear(k, d_model) for m, k in (cont_dims or {}).items() if k})
        self.blocks = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict({k: nn.Linear(d_model, n) for k, n in tasks.items()})

    def embed(self, b: dict) -> torch.Tensor:
        x = self.tok(b["ids"]) + self.pe(b["mod"], b["t"], b["pos"])
        if self.cont_proj and b["cont_mask"].any():
            x = x.clone()
            for m, proj in self.cont_proj.items():
                sel = b["cont_mask"] & (b["mod"] == MODALITIES.index(m))
                if sel.any():
                    x[sel] = x[sel] - self.tok.weight[CONT] + proj(b["cont"][sel][:, :proj.in_features])
        return x

    def encode(self, b: dict) -> torch.Tensor:
        """Pooled sequence representation (B, d_model)."""
        pad = b["ids"] == PAD
        h = self.embed(b)
        for blk in self.blocks:
            h = blk(h, ~pad)
        h = self.norm(h)
        keep = (~pad).float()[..., None]
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)

    def forward(self, b: dict, task: str) -> torch.Tensor:
        return self.heads[task](self.encode(b))
