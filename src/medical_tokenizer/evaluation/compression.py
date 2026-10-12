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
3. Ratio with protocol overhead = raw bits / (token payload bits + overhead bits), where the
   overhead of a span is its <MOD_START> <REP:*> <MOD_END> tokens at ceil(log2(VOCAB_SIZE))
   bits each plus the span metadata needed to decode it (shapes, fps, ...) at its compact
   JSON size in bytes x 8 (an upper bound; a binary header would be smaller).

Continuous spans: the <CONT> placeholders are Transformer positions, NOT vocabulary tokens.
They are reported as `continuous_vectors_per_sample`, never as discrete tokens.
"""
from __future__ import annotations

import json
import math

import numpy as np

SPAN_STRUCTURAL_TOKENS = 3       # <MOD_START> <REP:*> <MOD_END>


def protocol_overhead_bits(span) -> float:
    """Bits a span needs beyond its payload: structural tokens + decoding metadata."""
    from ..protocol.schema import json_default
    from ..protocol.vocab import VOCAB_SIZE
    meta = json.dumps(span.meta, separators=(",", ":"), default=json_default) if span.meta else ""
    return SPAN_STRUCTURAL_TOKENS * math.ceil(math.log2(VOCAB_SIZE)) + 8 * len(meta.encode())


def entropy_bits(ids) -> float:
    ids = np.concatenate([np.asarray(a).ravel() for a in ids]) if isinstance(ids, (list, tuple)) else np.asarray(ids).ravel()
    if ids.size == 0:
        return 0.0
    _, c = np.unique(ids, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def compression_summary(raw_bits, payload_bits, n_tokens, raw_elements=None, discrete_ids=None,
                        cont_values_per_sample=None, spans=None) -> dict:
    """Aggregate per-sample accounting into ratios (total raw / total token bits, i.e. size-weighted).

    `spans` (optional) adds protocol overhead and the discrete-token / continuous-vector split.
    """
    raw_bits, payload_bits, n_tokens = map(lambda a: np.asarray(a, np.float64), (raw_bits, payload_bits, n_tokens))
    out = {"n_samples": int(len(raw_bits)),
           "raw_bits_per_sample": float(raw_bits.mean()),
           "token_bits_per_sample": float(payload_bits.mean()),
           "compression_ratio": float(raw_bits.sum() / max(payload_bits.sum(), 1e-9)),
           "tokens_per_sample_mean": float(n_tokens.mean()),
           "tokens_per_sample_median": float(np.median(n_tokens)),
           "tokens_per_sample_max": int(n_tokens.max()) if len(n_tokens) else 0}
    if spans is not None:
        from ..protocol.vocab import CONT
        over = np.array([protocol_overhead_bits(sp) for sp in spans], np.float64)
        out["protocol_overhead_bits_per_sample"] = float(over.mean())
        out["compression_ratio_with_overhead"] = float(raw_bits.sum() / max(payload_bits.sum() + over.sum(), 1e-9))
        out["discrete_tokens_per_sample"] = float(np.mean([(sp.ids != CONT).sum() for sp in spans]))
        out["continuous_vectors_per_sample"] = float(np.mean([(sp.ids == CONT).sum() for sp in spans]))
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
