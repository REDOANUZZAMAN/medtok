"""Decode token sequences back to payloads, or dump them as human-readable tokens.

    python -m medical_tokenizer.decode --run experiments/image_vq/latest \
        --shard experiments/image_vq/latest/tokens/bloodmnist/test.npz --index 0 --out recon.npz [--jsonl tokens.jsonl]

Only the spans whose modality the given run(s) can decode are reconstructed; pass
--run several times for multimodal sequences, and --text to decode text spans.
"""
from __future__ import annotations

import argparse

import numpy as np

from .protocol.serialize import split_spans
from .protocol.storage import load_shard
from .protocol.tokens import write_jsonl
from .toolkit import build_lexicon, load_text_tokenizer, load_tokenizer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Decode token sequences.")
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--text", action="store_true", help="also decode text spans (templated-note BPE)")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", help="npz with one array/str per decoded span")
    ap.add_argument("--jsonl", help="write the sequence as MedicalToken JSONL")
    args = ap.parse_args(argv)
    seqs, meta = load_shard(args.shard)
    seq = seqs[args.index]
    codecs = {}
    for r in args.run:
        c = load_tokenizer(r)
        codecs[c.modality] = c
    text = load_text_tokenizer() if args.text else None
    if text is not None:
        codecs["text"] = text
    out = {}
    for k, (sp, span) in enumerate(zip(seq.spans, split_spans(seq))):
        m = sp["modality"]
        if m not in codecs:
            print(f"span {k}: {m}/{sp['rep']} ({len(span.ids)} tokens) -- no codec given, skipped")
            continue
        payload = codecs[m].decode(span)
        print(f"span {k}: {m}/{sp['rep']} ({len(span.ids)} tokens) -> "
              f"{type(payload).__name__} {getattr(payload, 'shape', '') or (list(payload) if isinstance(payload, dict) else payload)}")
        if isinstance(payload, dict):
            for key, v in payload.items():
                if isinstance(v, np.ndarray):
                    out[f"span{k}_{key}"] = v
        else:
            out[f"span{k}"] = np.asarray(payload)
    if args.out:
        np.savez_compressed(args.out, **out)
        print("wrote", args.out)
    if args.jsonl:
        write_jsonl(seq, args.jsonl, build_lexicon(), text.piece if text else None)
        print("wrote", args.jsonl)


if __name__ == "__main__":
    main()
