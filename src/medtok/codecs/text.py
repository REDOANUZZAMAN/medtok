"""Clinical-text codec: byte-level BPE into the shared `text` id range.

Byte-level BPE is exactly reversible for any UTF-8 string (no <unk>), so
decode(encode(s)) == s.
"""
from __future__ import annotations

import math

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from ..protocol.schema import Segment, TokenSpan
from ..protocol.vocab import RANGES
from .base import Codec


class TextBPECodec(Codec):
    modality, rep, reversible = "text", "bpe", True

    def __init__(self, tokenizer: Tokenizer):
        self.tok = tokenizer
        if tokenizer.get_vocab_size() > RANGES["text"].size:
            raise ValueError("BPE vocab larger than the protocol text range")

    @classmethod
    def train(cls, texts, vocab_size: int = 2048) -> "TextBPECodec":
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=min(vocab_size, RANGES["text"].size),
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                      show_progress=False)
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str) -> "TextBPECodec":
        return cls(Tokenizer.from_file(path))

    def save(self, path: str) -> None:
        self.tok.save(path)

    def encode(self, seg: Segment) -> TokenSpan:
        local = self.tok.encode(seg.data).ids
        ids = [RANGES["text"].to_global(i) for i in local]
        n = len(ids)
        pos = [[i, 0, 0] for i in range(n)]
        return TokenSpan.make(ids, "bpe", t=[seg.t_start] * n, pos=pos)

    def decode(self, span: TokenSpan) -> str:
        return self.tok.decode([RANGES["text"].to_local(int(i)) for i in span.ids])

    def piece(self, local_id: int) -> str:
        return self.tok.id_to_token(local_id)

    def raw_bits(self, seg: Segment) -> float:
        return len(seg.data.encode("utf-8")) * 8

    def alphabet_size(self) -> int:
        return self.tok.get_vocab_size()

    def bits_per_token(self) -> float:
        return math.log2(self.alphabet_size())
