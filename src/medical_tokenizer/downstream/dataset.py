"""UnifiedTokenDataset: any list of TokenSequences + a label key -> padded model batches.

The model sees only protocol fields (ids, mod, t, pos, continuous payloads);
nothing modality-specific is needed to batch sequences, so mixed-modality and
mixed-representation batches work unchanged.
"""
from __future__ import annotations

import numpy as np
import torch

from ..protocol.schema import TokenSequence
from ..protocol.vocab import CONT, PAD


class UnifiedTokenDataset(torch.utils.data.Dataset):
    def __init__(self, seqs: list[TokenSequence], label_key: str, max_len: int = 2048):
        self.seqs, self.label_key, self.max_len = seqs, label_key, max_len
        missing = [s.sample_id for s in seqs if label_key not in s.labels]
        if missing:
            raise KeyError(f"{len(missing)} sequences lack label {label_key!r}, e.g. {missing[0]}")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]

    def labels(self) -> np.ndarray:
        return np.array([s.labels[self.label_key] for s in self.seqs])

    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.seqs])

    def collate(self, seqs: list[TokenSequence]) -> dict:
        return collate(seqs, self.label_key, self.max_len)


def cont_dim_of(seqs: list[TokenSequence]) -> int:
    for s in seqs:
        for c in s.cont:
            return int(c.shape[1])
    return 0


def collate(seqs: list[TokenSequence], label_key: str | None, max_len: int = 2048) -> dict:
    """Right-pads to the longest sequence (truncating at max_len, keeping <EOS> semantics out of scope)."""
    L = min(max(len(s) for s in seqs), max_len)
    B = len(seqs)
    ids = np.full((B, L), PAD, np.int64)
    mod = np.full((B, L), -1, np.int64)
    t = np.full((B, L), -1.0, np.float32)
    pos = np.full((B, L, 3), -1, np.int64)
    cd = max((c.shape[1] for s in seqs for c in s.cont), default=0)
    cont = np.zeros((B, L, cd), np.float32)
    for i, s in enumerate(seqs):
        n = min(len(s), L)
        ids[i, :n], mod[i, :n], t[i, :n], pos[i, :n] = s.ids[:n], s.mod[:n], s.t[:n], s.pos[:n]
        ci = 0
        for sp in s.spans:
            if sp["rep"] != "continuous":
                continue
            c = s.cont[ci].astype(np.float32)
            ci += 1
            where = np.flatnonzero(s.ids[sp["start"]:sp["end"]] == CONT) + sp["start"]
            keep = where < n
            cont[i, where[keep], :c.shape[1]] = c[:len(where)][keep]
    b = {"ids": torch.as_tensor(ids), "mod": torch.as_tensor(mod), "t": torch.as_tensor(t),
         "pos": torch.as_tensor(pos), "cont": torch.as_tensor(cont), "cont_mask": torch.as_tensor(ids == CONT)}
    if label_key is not None:
        b["y"] = torch.as_tensor(np.array([s.labels[label_key] for s in seqs]))
    return b


def length_bucketed_batches(lengths: np.ndarray, bs: int, rng: np.random.Generator, window: int = 20) -> list[list[int]]:
    """Shuffle, sort within windows of `window` batches by length, split into batches, shuffle batches."""
    perm = rng.permutation(len(lengths))
    out = []
    for w in range(0, len(perm), bs * window):
        chunk = sorted(perm[w:w + bs * window], key=lambda i: lengths[i])
        out += [list(chunk[i:i + bs]) for i in range(0, len(chunk), bs)]
    rng.shuffle(out)
    return out
