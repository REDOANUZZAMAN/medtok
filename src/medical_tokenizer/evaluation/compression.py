"""Compression and sequence-length accounting.

Two notions, both reported (docs/experiments.md explains the choice):

1. Storage compression ratio = raw payload bits / token payload bits
     raw bits          defined per codec (Codec.raw_bits), e.g. H*W*C*8 for uint8 images,
                       (16+8+32) bits per (time, variable, value) observation for ICU records
     fixed-length bits discrete ids at log2(codec alphabet) each; continuous vectors at
                       16 bits per value (fp16); <CONT> placeholder ids not counted
     entropy-coded bits discrete ids at the empirical entropy of the id distribution
                       (a lower bound for any per-token code, e.g. Huffman / arithmetic)
2. Sequence length = number of Transformer positions for the payload span, compared
   with the number of raw elements (pixels, frames x pixels, observations) and
   between representations. Protocol delimiters are counted separately.
"""
from __future__ import annotations

import numpy as np


def entropy_bits(ids) -> float:
    ids = np.concatenate([np.asarray(a).ravel() for a in ids]) if isinstance(ids, (list, tuple)) else np.asarray(ids).ravel()
    if ids.size == 0:
        return 0.0
    _, c = np.unique(ids, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def compression_summary(raw_bits, payload_bits, n_tokens, raw_elements=None, discrete_ids=None,
                        cont_values_per_sample=None) -> dict:
    """Aggregate per-sample accounting into ratios (total raw / total token bits, i.e. size-weighted)."""
    raw_bits, payload_bits, n_tokens = map(lambda a: np.asarray(a, np.float64), (raw_bits, payload_bits, n_tokens))
    out = {"n_samples": int(len(raw_bits)),
           "raw_bits_per_sample": float(raw_bits.mean()),
           "token_bits_per_sample": float(payload_bits.mean()),
           "compression_ratio": float(raw_bits.sum() / max(payload_bits.sum(), 1e-9)),
           "tokens_per_sample_mean": float(n_tokens.mean()),
           "tokens_per_sample_median": float(np.median(n_tokens)),
           "tokens_per_sample_max": int(n_tokens.max()) if len(n_tokens) else 0}
    if raw_elements is not None:
        raw_elements = np.asarray(raw_elements, np.float64)
        out["raw_elements_per_sample"] = float(raw_elements.mean())
        out["sequence_reduction"] = float(raw_elements.sum() / max(n_tokens.sum(), 1e-9))
    if discrete_ids is not None:
        h = entropy_bits(discrete_ids)
        n_disc = sum(len(np.asarray(a)) for a in discrete_ids)
        cont_bits = 16.0 * float(np.sum(cont_values_per_sample)) if cont_values_per_sample is not None else 0.0
        out["token_entropy_bits"] = h
        out["entropy_coded_ratio"] = float(raw_bits.sum() / max(h * n_disc + cont_bits, 1e-9))
    return out


def transformer_input_bytes(n_tokens: float, d_model: int, bytes_per_value: int = 4) -> float:
    """Activation memory of the embedded input sequence for one sample (one layer's input)."""
    return float(n_tokens * d_model * bytes_per_value)
