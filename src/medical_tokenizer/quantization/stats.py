"""Codebook / vocabulary utilisation statistics (the vocabulary-collapse diagnostics).

All statistics derive from a usage-count vector over the K entries:
    active_codes     entries used at least once
    utilization      active / K
    dead_codes       K - active;  dead_pct = 100 * dead / K
    entropy_bits     H(p) of the empirical usage distribution
    norm_entropy     H(p) / log2 K   (1 = perfectly uniform usage)
    perplexity       2^H: the effective number of equally used codes
    top10_share      fraction of all assignments taken by the 10% most used entries
"""
from __future__ import annotations

import numpy as np


class CodeUsageTracker:
    def __init__(self, n_codes: int):
        self.counts = np.zeros(n_codes, np.int64)

    def update(self, indices) -> None:
        a = np.asarray(indices.detach().cpu() if hasattr(indices, "detach") else indices).ravel()
        self.counts += np.bincount(a, minlength=len(self.counts))[:len(self.counts)]

    def stats(self) -> dict:
        return usage_stats(self.counts)


def usage_stats(counts) -> dict:
    counts = np.asarray(counts, np.int64)
    K = len(counts)
    total = int(counts.sum())
    if K == 0 or total == 0:
        return {"n_codes": K, "total_assignments": total, "active_codes": 0, "utilization": 0.0,
                "dead_codes": K, "dead_pct": 100.0, "entropy_bits": 0.0, "norm_entropy": 0.0,
                "perplexity": 0.0, "top10_share": 0.0}
    p = counts / total
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum())
    active = int((counts > 0).sum())
    top = np.sort(counts)[::-1][:max(1, int(np.ceil(0.1 * K)))]
    return {"n_codes": K, "total_assignments": total, "active_codes": active, "utilization": active / K,
            "dead_codes": K - active, "dead_pct": 100.0 * (K - active) / K, "entropy_bits": h,
            "norm_entropy": h / np.log2(K) if K > 1 else 0.0, "perplexity": float(2 ** h),
            "top10_share": float(top.sum() / total)}
