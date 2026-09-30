"""Fit the non-neural parts of the protocol on the TRAINING split only.

Outputs (artifacts/): lexicon.json, text_bpe.json, ts_binner.json, ts_norm.json
and a parsed-stay cache in data/processed/stays.pkl.
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from medtok.codecs.text import TextBPECodec  # noqa: E402
from medtok.codecs.timeseries import ValueBinner  # noqa: E402
from medtok.data.notes import IMAGE_CONTEXT, stay_note  # noqa: E402
from medtok.data.physionet import Normalizer, load_stays  # noqa: E402
from medtok.toolkit import build_lexicon  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(ART, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

stays = load_stays(os.path.join(ROOT, "data", "raw", "physionet2012"))
print(f"{len(stays)} stays; splits:", {s: sum(x.split == s for x in stays) for s in ("train", "val", "test")},
      "mortality:", round(sum(x.death for x in stays) / len(stays), 3))
pickle.dump(stays, open(os.path.join(PROC, "stays.pkl"), "wb"))

lex = build_lexicon()
lex.save(os.path.join(ART, "lexicon.json"))

binner = ValueBinner(lex.n_value_bins).fit(stays)
json.dump(binner.to_json(), open(os.path.join(ART, "ts_binner.json"), "w"))
norm = Normalizer().fit(stays)
json.dump(norm.to_json(), open(os.path.join(ART, "ts_norm.json"), "w"))

corpus = [stay_note(s) for s in stays if s.split == "train"]
corpus += [t for opts in IMAGE_CONTEXT.values() for t in opts] * 50
bpe = TextBPECodec.train(corpus, vocab_size=1024)
bpe.save(os.path.join(ART, "text_bpe.json"))
print("BPE vocab:", bpe.alphabet_size())
print("example note:", corpus[0])
