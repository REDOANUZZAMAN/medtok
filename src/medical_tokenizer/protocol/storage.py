"""On-disk token datasets: many TokenSequences in one compressed .npz shard.

Arrays of all sequences are concatenated and indexed by `offsets`
(len n+1); per-sequence headers (sample id, labels, spans) are one JSON blob;
continuous payloads are concatenated with their own offsets. No pickle, so a
shard is safe to load and readable from any language with an npz reader.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .schema import TokenSequence, check_version, json_default
from .vocab import PROTOCOL_VERSION


def save_shard(path: str, seqs: list[TokenSequence], meta: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lens = np.array([len(s) for s in seqs], np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    conts = [c for s in seqs for c in s.cont]
    n_cont = np.array([len(s.cont) for s in seqs], np.int64)
    cont_rows = np.array([len(c) for c in conts], np.int64)
    head = {"protocol_version": PROTOCOL_VERSION, "meta": meta or {},
            "seqs": [{"sample_id": s.sample_id, "labels": s.labels, "spans": s.spans} for s in seqs]}
    arrays = {
        "offsets": offsets,
        "ids": np.concatenate([s.ids for s in seqs]) if seqs else np.zeros(0, np.int64),
        "mod": np.concatenate([s.mod for s in seqs]) if seqs else np.zeros(0, np.int8),
        "t": np.concatenate([s.t for s in seqs]) if seqs else np.zeros(0, np.float32),
        "pos": np.concatenate([s.pos for s in seqs]) if seqs else np.zeros((0, 3), np.int32),
        "n_cont": n_cont, "cont_rows": cont_rows,
        "header": np.frombuffer(json.dumps(head, default=json_default).encode(), np.uint8),
    }
    if conts:
        arrays["cont"] = np.concatenate(conts).astype(np.float16)
    if any(s.conf is not None for s in seqs):
        arrays["conf"] = np.concatenate([s.conf if s.conf is not None else np.full(len(s), np.nan, np.float32)
                                         for s in seqs])
    np.savez_compressed(path, **arrays)


def load_shard(path: str) -> tuple[list[TokenSequence], dict]:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    head = json.loads(bytes(d["header"]).decode())
    check_version(head["protocol_version"])
    off = d["offsets"]
    cont_off = np.concatenate([[0], np.cumsum(d["cont_rows"])])
    seq_cont_off = np.concatenate([[0], np.cumsum(d["n_cont"])])
    seqs = []
    for i, h in enumerate(head["seqs"]):
        a, b = off[i], off[i + 1]
        cs = [d["cont"][cont_off[j]:cont_off[j + 1]] for j in range(seq_cont_off[i], seq_cont_off[i + 1])]
        seqs.append(TokenSequence(d["ids"][a:b], d["mod"][a:b], d["t"][a:b], d["pos"][a:b], h["spans"], cs,
                                  h["sample_id"], h["labels"], d["conf"][a:b] if "conf" in d else None))
    return seqs, head["meta"]
