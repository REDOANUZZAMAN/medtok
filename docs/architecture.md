# Architecture

```
raw data ──► dataset adapter ──► Sample(segments…) ──► modality codec ──► TokenSpan ──► serializer ──► TokenSequence
 (data/)       (data/datasets.py)   (protocol/schema)     (tokenizers/)                  (protocol/serialize)    │
                                                          │                                                     ▼
                                    encoder ─► quantiser ─┘                                   token shards (protocol/storage)
                                   (encoders/) (quantization/)                                                  │
                                                                                                                ▼
                                                              UnifiedTokenTransformer (downstream/) ─► task heads
```

Each box can be replaced independently. The only contracts are:

- `Codec.encode/decode`;
- the `TokenSequence` fields;
- the `LatentAutoencoder` interface (`encode_latent`, `decode_latent`, quantiser plug-in).

## Package layout (`src/medical_tokenizer/`)

| Package | Contents |
|---|---|
| `protocol/` | `vocab.py` (id ranges, special tokens, dt buckets, `Lexicon`), `schema.py` (`Sample`, `Segment`, `TokenSpan`, `TokenSequence`), `serialize.py` (layout, `validate`, `drop_modality`), `tokens.py` (`MedicalToken`, `TokenType`, JSONL), `storage.py` (npz shards) |
| `codecs/` | `Codec` base class (`encode`, `decode`, bit accounting, reversibility flag), `MultimodalCodec` (sample-level, batched) |
| `encoders/` | `LatentAutoencoder` base; `ImageAutoencoder` (conv hierarchy or ViT-style patch embedding), `TimeSeriesWindowAutoencoder` (windowed MLP, product quantisation, overlap-add), `VideoClipAutoencoder` (factorised 3D conv) |
| `quantization/` | `VectorQuantizer` (EMA or codebook-loss updates, cosine lookup, dead-code restart, straight-through estimator), `IdentityQuantizer`, usage statistics |
| `tokenizers/<modality>/` | modality codecs (see `docs/token_protocol.md` §6) |
| `positional_encoding/` | `UnifiedPositionalEncoding` from the (mod, t, pos) side channels |
| `downstream/` | `UnifiedTokenDataset` + `collate`, `UnifiedTokenTransformer`, `DownstreamTask` registry + metrics, training loop, CLI |
| `evaluation/` | reconstruction metrics (MSE / MAE / PSNR / SSIM, masked z-errors), compression and sequence-length accounting |
| `data/` | MedMNIST, PhysioNet 2012, Cataract-101 / Cholec80 loaders, templated notes, adapters, `prepare` CLI |
| `utils/` | YAML config (inheritance + `--set`), experiment run directories, seeding, environment capture |
| `train.py`, `trainers.py`, `evaluate.py`, `encode.py`, `decode.py`, `pipeline.py`, `report.py` | command-line entry points |

## Design decisions

1. **One autoencoder, swappable bottleneck.** VQ and continuous tokenizers of a modality share the encoder / decoder architecture and differ only in the quantiser. Differences between A and B therefore come from discretisation, not from architecture.
2. **Side channels instead of positional tokens.** Time and spatial positions travel in `t` / `pos`, not as extra tokens. This keeps sequences short and lets every modality use the same encoding module.
3. **Fixed id ranges, append-only special tokens, versioned protocol.** Sequences from different tokenizer checkpoints stay mutually readable. Loading checks the version.
4. **Event tokens are generic unless annotated.** Measurement / change events (time-series) and appearance-state segments (video) are data-driven and carry no clinical meaning. Annotation-derived events live in a separate id range and are never used as model inputs for the task they annotate.
5. **The same static block in every time-series representation.** Comparisons isolate the temporal encoding.
6. **Continuous baseline = continuous latents + a learned linear projection** (`UnifiedTokenTransformer.cont_proj`) at `<CONT>` slots. This is the standard way multimodal LLMs attach encoders to a language backbone.
7. **Reproducibility.**
   - Every command takes `--config`.
   - Every run gets a fresh directory with the resolved config, environment, git commit, logs, checkpoint, codebook and metrics.
   - Seeds are fixed and deterministic algorithms are enabled (warn-only).
   - `latest` symlinks point at the newest run, and nothing is overwritten.
8. **CPU-first sizes.** Every model trains on a 6-core CPU in minutes to tens of minutes (image autoencoder ≈ 0.72 M parameters incl. a 1,024 × 32 codebook). No GPU is assumed.

## Command-line interface

```
python -m medical_tokenizer.data.prepare  --config configs/data.yaml
python -m medical_tokenizer.train         --config configs/image_vq.yaml [--set quantizer.n_codes=256]
python -m medical_tokenizer.evaluate      --run experiments/image_vq            # or --interfaces
python -m medical_tokenizer.encode        --run experiments/image_vq --task bloodmnist [--include-text]
python -m medical_tokenizer.decode        --run experiments/image_vq --shard <shard.npz> --index 0 --out x.npz --jsonl x.jsonl
python -m medical_tokenizer.downstream    --config configs/downstream/bloodmnist_vq.yaml
python -m medical_tokenizer.pipeline      --config configs/experiment_comparison.yaml --skip-existing
python -m medical_tokenizer.report
```

`scripts/*.py` are thin wrappers around the same entry points.
