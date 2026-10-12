# Medical Multimodal Tokenizer & Unified Token Protocol

Undergraduate research project: **Design of a Medical Multimodal-Specific Tokenizer and a Unified Data Protocol** (Task 1.2 Multi-dimensional Tokenizer; Task 1.1 Multimodal Data Semantic Integration).

The project converts heterogeneous medical data into **one discrete token vocabulary (13,824 ids) and one sequence format**, so that a single Transformer can consume any mix of modalities. On that format it compares three representation strategies:

- **A. continuous projection**;
- **B. vector quantisation**;
- **C. event-based tokens**.

The comparison measures compression ratio, reconstruction error, codebook utilisation, sequence length and downstream task performance (images: AUROC, macro-F1; ICU mortality: AUROC, AUPRC). Images and time-series are the two fully implemented modalities; the scope follows the six-week plan in `slides/WEEKLY_PLAN.md`.

| Modality | Data | Status |
|---|---|---|
| 2D medical images | MedMNIST v2 (Blood / Pneumonia / Derma) | full: VQ + continuous tokenizers, 3 classification tasks |
| Physiological time-series | PhysioNet 2012 (4,000 ICU stays) | full: event (2 variants) + VQ + continuous, mortality prediction |
| Surgical video | Cataract-101 (101 surgeries, 10 phases) — open stand-in for request-only Cholec80; results not comparable with Cholec80 | adaptation: VQ + continuous + unsupervised segment events, online phase recognition; annotation-event codec (encodes ground-truth labels, not recognition) |
| Clinical text | **synthetic** templated notes, versioned (no open clinical free text in scope) | protocol support: exact byte-level BPE codec, accepted on round-trip / field preservation only |
| 3D point clouds | synthetic surfaces (SCARED is request-only) | interface: normalise → voxel groups → features → codebook |

**Results** are generated from experiment outputs, never typed in:

- [`reports/RESULTS.md`](reports/RESULTS.md) — tables and figures;
- [`docs/experiments.md`](docs/experiments.md) — experiment protocol, findings and limitations.

## Documentation

| | |
|---|---|
| [`docs/token_protocol.md`](docs/token_protocol.md) | formal protocol / lexicon specification: id ranges, special tokens, layout, positions, reversibility, bit accounting |
| [`docs/architecture.md`](docs/architecture.md) | components, design decisions, CLI |
| [`docs/datasets.md`](docs/datasets.md) | datasets, licences, splits, data-handling rules |
| [`docs/experiments.md`](docs/experiments.md) | experiment matrix, ablations, results discussion |

## Quick start (CPU is enough)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[video,dev]"
pytest -q                                              # synthetic-data unit + end-to-end tests, no downloads

python -m medical_tokenizer.data.prepare --config configs/data.yaml     # MedMNIST + PhysioNet (+ Cataract-101 if placed)
python -m medical_tokenizer.train      --config configs/image_vq.yaml
python -m medical_tokenizer.evaluate   --run experiments/image_vq
python -m medical_tokenizer.downstream --config configs/downstream/bloodmnist_vq.yaml

# everything (main matrix, then ablations, then tables / figures)
python -m medical_tokenizer.pipeline --config configs/experiment_comparison.yaml --skip-existing
python -m medical_tokenizer.pipeline --config configs/ablations.yaml --skip-existing
python -m medical_tokenizer.report
```

## Demo app

```bash
.venv/bin/python demo/server.py --port 8093     # open http://<host>:8093
```

A browser demo (standard library server + one HTML page, `demo/`). It shows, on test-set samples, each modality
turned into tokens and decoded again (images, ICU charts, surgery video and phase events, synthetic text), one
mixed text + time-series token sequence with its JSONL form, and `reports/RESULTS.md`. You can also upload your own picture.

## GPU re-runs on Modal (optional)

Training loops use CUDA when available (`MEDTOK_DEVICE` overrides). `scripts/modal_rerun.py` runs the plans in
`configs/rerun_gpu/` as four parallel Modal jobs and merges the new runs back into `experiments/`.

## Using the toolkit

```python
from medical_tokenizer.toolkit import load_tokenizer, load_text_tokenizer, MultimodalCodec
from medical_tokenizer.protocol import Sample, Segment, to_tokens, write_jsonl

codec = MultimodalCodec({
    "text": load_text_tokenizer(),
    "image": load_tokenizer("experiments/image_vq"),
    "timeseries": load_tokenizer("experiments/timeseries_event"),
})
sample = Sample("case-1", [Segment("text", "Chest X-ray, frontal view."),
                           Segment("image", img_uint8_28x28x3, t_start=5.0)])
seq = codec.encode(sample)      # TokenSequence: ids + modality / time / position side channels
payloads = codec.decode(seq)    # [(modality, reconstruction), ...]
write_jsonl(seq, "case-1.jsonl")  # one human-readable MedicalToken per line
```

## Repository layout

```
configs/         YAML experiment configs (tokenizers, downstream, ablations, pipeline plans)
src/medical_tokenizer/
  protocol/      vocabulary, schema, serialisation, MedicalToken view, shards
  tokenizers/    image, video, timeseries, text, pointcloud codecs
  encoders/      lightweight autoencoders (image / time-series / video)
  quantization/  vector quantiser + utilisation statistics
  codecs/        Codec interface, MultimodalCodec
  positional_encoding/  modality-aware positional encoding
  downstream/    UnifiedTokenDataset, UnifiedTokenTransformer, tasks, training
  evaluation/    reconstruction and compression metrics
  data/          dataset loaders, adapters, prepare CLI
scripts/         thin wrappers around the CLI entry points
tests/           unit + integration tests on synthetic data
notebooks/       token exploration (reads real experiment outputs)
reports/         generated tables and figures (committed)
experiments/     run directories (git-ignored: checkpoints, codebooks, token shards)
```

## Data use

Only open datasets are downloaded automatically. Controlled or request-only data (MIMIC, Cholec80, SCARED) are never downloaded, and no raw or patient-level derived data is committed (see `.gitignore` and `docs/datasets.md`). Cataract-101 is CC BY-NC 4.0 (research use only). Cite Schoeffmann et al., MMSys 2018.

## Licence

Code: MIT (see `LICENSE`). Datasets keep their own licences.
