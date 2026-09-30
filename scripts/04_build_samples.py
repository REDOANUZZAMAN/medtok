"""Tokenise the cross-modal sample set under every representation type.

Writes data/processed/seqs_<dataset>_<rep>.pkl (lists of TokenSequence, used
by evaluation) and exports a browsable slice of the sample set to
data/samples/ (npz per sample + manifest.jsonl). data/samples/ holds
patient-level derived data and is git-ignored.
"""
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from medtok.data.build import encode_icu, encode_images, icu_sample, image_sample  # noqa: E402
from medtok.data.medmnist_data import load_subset  # noqa: E402
from medtok.protocol.schema import sample_to_manifest  # noqa: E402
from medtok.protocol.serialize import validate  # noqa: E402
from medtok.toolkit import Toolkit  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
PROC = os.path.join(ROOT, "data", "processed")
SAMPLES = os.path.join(ROOT, "data", "samples")
IMAGE_SETS = ("bloodmnist", "pneumoniamnist")
os.makedirs(SAMPLES, exist_ok=True)

tk = Toolkit(os.path.join(ROOT, "artifacts"))
stays = pickle.load(open(os.path.join(PROC, "stays.pkl"), "rb"))

for rep in ("event", "vq", "cont"):
    t0 = time.time()
    seqs = encode_icu(stays, tk, rep)
    for s in seqs:
        validate(s)
    pickle.dump(seqs, open(os.path.join(PROC, f"seqs_physionet_{rep}.pkl"), "wb"))
    lens = [len(s) for s in seqs]
    print(f"physionet/{rep}: {len(seqs)} seqs, len mean {sum(lens) / len(lens):.0f} max {max(lens)}"
          f" ({time.time() - t0:.0f}s)", flush=True)

for subset in IMAGE_SETS:
    d = load_subset(os.path.join(ROOT, "data", "raw", "medmnist"), subset)
    for rep in ("vq", "cont"):
        seqs = []
        for split in ("train", "val", "test"):
            seqs += encode_images(subset, split, *d[split], tk, rep)
        for s in seqs[:100]:
            validate(s)
        pickle.dump(seqs, open(os.path.join(PROC, f"seqs_{subset}_{rep}.pkl"), "wb"))
        print(f"{subset}/{rep}: {len(seqs)} seqs, len {len(seqs[0])}", flush=True)

# ---- browsable export: 50 ICU + 50 image samples, VQ + event representations
with open(os.path.join(SAMPLES, "manifest.jsonl"), "w") as man:
    ev = pickle.load(open(os.path.join(PROC, "seqs_physionet_event.pkl"), "rb"))
    vq = pickle.load(open(os.path.join(PROC, "seqs_physionet_vq.pkl"), "rb"))
    for st, se, sv in list(zip(stays, ev, vq))[:50]:
        m = sample_to_manifest(icu_sample(st))
        for rep, seq in (("event", se), ("vq", sv)):
            fn = f"{m['sample_id'].replace('/', '_')}.{rep}.npz"
            seq.save(os.path.join(SAMPLES, fn))
            m[f"tokens_{rep}"] = fn
        man.write(json.dumps(m) + "\n")
    d = load_subset(os.path.join(ROOT, "data", "raw", "medmnist"), "bloodmnist")
    iv = pickle.load(open(os.path.join(PROC, "seqs_bloodmnist_vq.pkl"), "rb"))
    x, y = d["train"]
    for i in range(50):
        m = sample_to_manifest(image_sample("bloodmnist", "train", i, x[i], y[i]))
        fn = f"{m['sample_id'].replace('/', '_')}.vq.npz"
        iv[i].save(os.path.join(SAMPLES, fn))
        m["tokens_vq"] = fn
        man.write(json.dumps(m) + "\n")
print("exported sample slice to", SAMPLES)
