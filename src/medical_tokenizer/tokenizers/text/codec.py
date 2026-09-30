"""Clinical-text tokenizer (interface modality). Payload: str.

Backend: a byte-level BPE (Hugging Face `tokenizers`) mapped into the protocol's
`text` id range. Byte-level BPE has no <UNK>, so decode(encode(s)) == s for any
UTF-8 string: the codec is exactly reversible.

Any standard sub-word tokenizer can be plugged in with `from_pretrained`
(a tokenizer.json file or a Hugging Face hub id when `transformers` is
installed), as long as its vocabulary fits the 8192-id text range.

NOTE: none of the project's open datasets contains free-text clinical notes
(MIMIC notes are credentialed-access). The text used in experiments is rendered
from structured fields by data/notes.py and is labelled as such.
"""
from __future__ import annotations

import json
import math

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ...codecs.base import Codec
from ...protocol.schema import Segment, TokenSpan
from ...protocol.vocab import RANGES

R = RANGES["text"]


class ClinicalTextTokenizer(Codec):
    modality, rep, reversibility = "text", "subword", "exact"

    def __init__(self, tokenizer: Tokenizer):
        if tokenizer.get_vocab_size() > R.size:
            raise ValueError(f"sub-word vocabulary {tokenizer.get_vocab_size()} exceeds the text range {R.size}")
        self.tok = tokenizer

    # ---- construction
    @classmethod
    def train(cls, texts, vocab_size: int = 1024) -> "ClinicalTextTokenizer":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=min(vocab_size, R.size), show_progress=False,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @classmethod
    def from_pretrained(cls, name_or_path: str) -> "ClinicalTextTokenizer":
        if name_or_path.endswith(".json"):
            return cls(Tokenizer.from_file(name_or_path))
        return cls(Tokenizer.from_pretrained(name_or_path))

    @classmethod
    def load(cls, path: str) -> "ClinicalTextTokenizer":
        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        self.tok.save(path)

    # ---- string-level API
    def tokenize(self, text: str) -> list[str]:
        """Sub-word pieces (byte-level, so spaces show as 'Ġ')."""
        return self.tok.encode(text).tokens

    def encode_ids(self, text: str) -> list[int]:
        """Global protocol ids."""
        return [R.start + i for i in self.tok.encode(text).ids]

    def decode_ids(self, ids) -> str:
        return self.tok.decode([R.to_local(int(i)) for i in ids])

    def serialize(self, text: str) -> str:
        """JSON record of the tokenisation (pieces + global ids), for inspection."""
        enc = self.tok.encode(text)
        return json.dumps({"text": text, "pieces": enc.tokens, "ids": [R.start + i for i in enc.ids]})

    def piece(self, local_id: int) -> str:
        return self.tok.id_to_token(local_id)

    # ---- Codec API
    def encode(self, seg: Segment) -> TokenSpan:
        ids = self.encode_ids(seg.data)
        n = len(ids)
        return TokenSpan.make(ids, "subword", t=[seg.t_start] * n, pos=[[i, -1, -1] for i in range(n)])

    def decode(self, span: TokenSpan) -> str:
        return self.decode_ids(span.ids)

    def alphabet_size(self) -> int:
        return self.tok.get_vocab_size()

    def raw_bits(self, seg: Segment) -> float:
        return float(len(seg.data.encode("utf-8")) * 8)

    def bits_per_token(self) -> float:
        return math.log2(self.alphabet_size())
