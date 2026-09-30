# MedTok: a medical multimodal tokenizer and unified token protocol

Final-year project: *Design of a Medical Multimodal-Specific Tokenizer and a Unified Data Protocol* (Task 1.1 Multimodal Data Semantic Integration, Task 1.2 Multi-dimensional Tokenizer).

MedTok maps clinical text, 2D medical images and physiological time-series (fully implemented), plus surgical-video events and 3D point clouds (interfaces with synthetic demos), into **one 13,056-id vocabulary and one sequence format**. On that format it compares three representation types: **continuous projection**, **vector quantization** and **event-based tokens**. The comparison covers compression, reconstruction and codebook utilization, and performance on two downstream tasks, in-hospital mortality prediction and MedMNIST classification. Both tasks are also solved together by a single shared model.

## Deliverables

| Deliverable (from the brief) | Where |
|---|---|
| Data protocol and lexicon specification | [`docs/PROTOCOL.md`](docs/PROTOCOL.md), `artifacts/lexicon.json` |
| Tokenizer toolkit | `src/medtok/` (codecs, protocol, quantizers, downstream model), tests in `tests/` |
| Cross-modal sample set | built by `scripts/04_build_samples.py` into `data/samples/` (git-ignored, patient-level); synthetic five-modality example in `data/samples_demo/` |
| Compression and downstream evaluation report | [`docs/REPORT.md`](docs/REPORT.md), raw numbers in `artifacts/results/*.json` |

## Layout

```
src/medtok/
  protocol/   vocab.py (id ranges, special/dt tokens, Lexicon)  schema.py (Sample/Segment/TokenSequence)  serialize.py
  codecs/     text.py  image.py  timeseries.py  video.py  pointcloud.py   (all implement codecs/base.py:Codec)
  quant/      vq.py (EMA vector quantizer + dead-code restart)  models.py (image VQ-VAE, time-series patch VQ)
  data/       physionet.py  medmnist_data.py  notes.py  build.py
  eval/       downstream.py (UnifiedTransformer over any TokenSequence)
  toolkit.py  load lexicon + all trained codecs in one call
scripts/      01..08 pipeline (below)
docs/         PROTOCOL.md  REPORT.md  figures/
```

## Reproduce (CPU only, about 3–4 h on 6 cores)

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e . pytest

# data: PhysioNet 2012 set-a + MedMNIST v2 (28 px)
mkdir -p data/raw/physionet2012 data/raw/medmnist
(cd data/raw/physionet2012 && for f in set-a.tar.gz Outcomes-a.txt; do curl -sSLO https://physionet.org/files/challenge-2012/1.0.0/$f; done && tar xzf set-a.tar.gz)
(cd data/raw/medmnist && for s in bloodmnist pneumoniamnist dermamnist; do curl -sSLO https://zenodo.org/records/10519652/files/$s.npz; done)

.venv/bin/python scripts/01_prepare.py            # lexicon, BPE, value bins, normaliser (train split only)
.venv/bin/python scripts/02_train_image_vq.py --epochs 10
.venv/bin/python scripts/03_train_ts_vq.py
.venv/bin/python scripts/04_build_samples.py      # tokenise the cross-modal sample set, every representation
.venv/bin/python scripts/05_eval_compression.py   # compression / reconstruction / utilisation
.venv/bin/python scripts/06_downstream.py         # downstream tasks, 3 seeds
.venv/bin/python scripts/07_interface_demos.py    # video + point-cloud interfaces, five-modality sample
.venv/bin/python scripts/08_figures.py
.venv/bin/python -m pytest -q
```

## Using the toolkit

```python
from medtok.toolkit import Toolkit
from medtok.protocol.schema import Sample, Segment
from medtok.protocol.serialize import serialize, deserialize

tk = Toolkit("artifacts")
codecs = tk.codec_set(image="vq", timeseries="event")
sample = Sample("demo", [Segment("text", "Chest X-ray, frontal view."), Segment("image", img_uint8_28x28x3)])
seq = serialize(sample, codecs)          # TokenSequence: ids + modality/time/position side channels
seq.save("demo.npz")
print([tk.lex.id_to_str(int(i), tk.text.piece) for i in seq.ids][:12])
restored = deserialize(seq, codecs)      # [(modality, payload), ...]
```

## Data usage

MIMIC is not used. PhysioNet 2012 is open data (ODC-By), but following the course's data-usage rules no patient-level raw or derived data is committed: `data/raw`, `data/processed` and `data/samples` are git-ignored. Cholec80 and SCARED require data requests, so their codecs read the official file formats and are demonstrated on synthetic data.
