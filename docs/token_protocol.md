# Unified Medical Token Protocol — specification v0.2

The reference implementation is `src/medical_tokenizer/protocol/`. If this document and the code disagree, the code wins and this document has a bug.

## 1. Purpose and scope

The protocol maps five medical modality categories into **one flat token-id space** and **one sequence format**. A single sequence model can then read any mix of modalities and representation types without knowing how each input was produced.

| Modality | Id | Payload (`Segment.data`) | Codecs (representation) | Status |
|---|---|---|---|---|
| `text` | 0 | UTF-8 `str` | `ClinicalTextTokenizer` (subword) | interface; exact |
| `image` | 1 | `uint8 (H, W, C)` | `ImageVQCodec` (vq), `ImageContinuousCodec` (continuous) | full (MedMNIST v2) |
| `timeseries` | 2 | `{"obs": float32 (n,3) [minute, variable, value], "static": {name: value}}` | `TSEventCodec` (event, `obs`/`change`), `TSVQCodec` (vq), `TSContinuousCodec` (continuous) | full (PhysioNet 2012) |
| `video` | 3 | `{"frames": uint8 (T,S,S,3), "fps": float}` or `{"phase": int (T,), "fps": float}` | `VideoVQCodec` (vq), `VideoContinuousCodec` (continuous), `VideoSegmentCodec` (event, unsupervised), `AnnotatedPhaseEventCodec` (event, from annotations) | full (Cataract-101); Cholec80 loaders need a local copy |
| `pointcloud` | 4 | `float32 (N, 3)` | `PointCloudTokenizer` (vq) | interface; synthetic data only |

## 2. Data model

```
Sample                       one case
  sample_id : str            e.g. "physionet2012/132539", "bloodmnist/test/17", "cataract101/269/40"
  segments  : list[Segment]
  labels    : dict           task labels, e.g. {"death": 0}, {"class": 3}, {"phase": 4}
  split     : "train" | "val" | "test"
  meta      : dict

Segment                      one modality instance
  modality  : one of the five modality names
  data      : payload (table above)
  t_start   : float          minutes relative to the sample's reference time
  source    : str            dataset / file of origin
  meta      : dict
```

A **codec** maps a `Segment` to a `TokenSpan` (payload tokens only). The **serializer** lays the spans of one `Sample` out as a `TokenSequence`.

## 3. Vocabulary

The id space has fixed, non-overlapping ranges (`protocol/vocab.py:RANGES`). Ranges are part of the protocol and are never learned. A codebook smaller than its range leaves the tail unused.

| Range | Ids | Size | Tokens |
|---|---|---|---|
| `special` | 0–255 | 256 | structural tokens (§3.1) |
| `dt` | 256–511 | 256 | `<DT:k>` time gaps (§3.2) |
| `var` | 512–767 | 256 | `<VAR:name>` time-series variables (41 used: 4 static + 37 temporal) |
| `vbin` | 768–1023 | 256 | `<VBIN:k>` per-variable quantile bins (16 used) |
| `event` | 1024–1279 | 256 | `<EV:dataset:kind:name>` events **derived from dataset annotations** |
| `segment` | 1280–1535 | 256 | `<SEG:k>` **unsupervised** segment states (no semantic meaning) |
| `text` | 1536–9727 | 8192 | sub-word pieces |
| `image_vq` | 9728–10751 | 1024 | image codebook entries |
| `ts_vq` | 10752–11775 | 1024 | time-series codebook entries |
| `video_vq` | 11776–12799 | 1024 | video codebook entries |
| `pc_vq` | 12800–13823 | 1024 | point-cloud codebook entries |

`VOCAB_SIZE = 13824`. The machine-readable lexicon (range table, special tokens, variable / event names, dt bucket values) is written by `Lexicon.save` to `lexicon.json` in every tokenizer run and in `data/processed/assets/`.

The `event` and `segment` ranges are kept apart on purpose. An `<EV:…>` token asserts something an expert annotated (for example, `cataract101:phase:Rhexis`). A `<SEG:k>` token only says that the encoder put this clip into appearance cluster *k*. Keeping the ranges separate makes that distinction visible in every sequence, so unsupervised states cannot be mistaken for clinical events.

### 3.1 Special tokens (ids are positional; the list is append-only)

| Id | Token | Role |
|---|---|---|
| 0–5 | `<PAD> <BOS> <EOS> <UNK> <SEP> <MASK>` | padding, sequence boundaries, reserved |
| 6–15 | `<TEXT_START> <TEXT_END> <IMAGE_START> <IMAGE_END> <TIME_SERIES_START> <TIME_SERIES_END> <VIDEO_START> <VIDEO_END> <POINT_CLOUD_START> <POINT_CLOUD_END>` | modality span boundaries |
| 16–19 | `<REP:SUBWORD> <REP:VQ> <REP:EVENT> <REP:CONTINUOUS>` | which representation produced the span |
| 20 | `<ROW>` | image: start of a row of latent patches |
| 21 | `<FRAME>` | video: start of a frame (reserved for frame-level codecs) |
| 22 | `<CLIP>` | video: start of a clip |
| 23 | `<PATCH>` | time-series: start of a temporal window |
| 24 | `<MISSING>` | time-series: window without any observation (when `skip_empty`) |
| 25 | `<STATIC>` | time-series: start of the static-attribute block |
| 26 | `<CONT>` | placeholder for one continuous vector |
| 27–255 | `<RESERVED:i>` | reserved |

### 3.2 Time-delta tokens

`<DT:k>` encodes a gap Δ (between consecutive spans, event groups, or as a segment duration):

| k | Δ | Resolution |
|---|---|---|
| 0 | 0 | exact |
| 1–59 | 1–59 s | exact to the second |
| 60–118 | 1–59 min | exact to the minute |
| 119–166 | 1–48 h | exact to the hour |
| 167–255 | 48 h – 30 days | log-spaced |

The `t` side channel always carries the exact time, so `<DT>` tokens give the model a coarse, discrete view of gaps and are never the only record of time.

## 4. Sequence layout (wire format)

```
<BOS>
  <TEXT_START> <REP:SUBWORD> w w w … <TEXT_END>
  <DT:k> <IMAGE_START> <REP:VQ> <ROW> v v v v v v v <ROW> … <IMAGE_END>
  <DT:k> <TIME_SERIES_START> <REP:EVENT> <STATIC> <VAR:Age> <VBIN:9> … <VAR:HR> <VBIN:5> <DT:k> <VAR:HR> <VBIN:6> … <TIME_SERIES_END>
<EOS>
```

Rules:

1. Segments are ordered by `t_start` with a stable sort (ties keep their input order).
2. Each span is `<X_START> <REP:R> payload… <X_END>`.
3. Every span except the first is preceded by `<DT:k>` for the gap to the previous span.
4. `<BOS>` and `<EOS>` have modality −1. All other tokens carry the modality of their span.

`protocol/serialize.py:validate` enforces rules 2–4, the side-channel shapes, the id range, and the continuous-payload count.

### 4.1 Columnar container (`TokenSequence`)

| Field | Shape / type | Meaning |
|---|---|---|
| `ids` | `(L,) int64` | global token ids |
| `mod` | `(L,) int8` | modality id, −1 for `<BOS>/<EOS>` |
| `t` | `(L,) float32` | minutes since the sample's reference time, −1 = n/a |
| `pos` | `(L,3) int32` | modality-specific position (§5), −1 = n/a |
| `conf` | `(L,) float32` or absent | optional confidence, NaN = n/a |
| `spans` | list of dict | `{modality, rep, start, end, meta}`; `ids[start:end]` is the payload; `meta` is what the decoder needs |
| `cont` | list of `(k,d) float16` | continuous payloads, one per continuous span, in `<CONT>`-slot order |
| header | JSON | `protocol_version`, `sample_id`, `labels`, `spans` |

Storage:

- one sequence → `.npz` (`TokenSequence.save/load`);
- a dataset split → one `.npz` **shard** (`protocol/storage.py`): arrays concatenated with offsets, a JSON header, no pickle;
- one sequence → human-readable **JSONL** (`protocol/tokens.py:write_jsonl`), one `MedicalToken` per line.

Loading any of these checks the protocol version.

### 4.2 Row view (`MedicalToken`)

| Field | Type | Notes |
|---|---|---|
| `token_id` | int | global id |
| `token` | str | readable form, e.g. `<IMAGE_VQ:17>`, `<VAR:HR>`, `<DT:62>` |
| `token_type` | `TokenType` | `sequence_boundary, modality_start, modality_end, representation, layout, special, continuous, time_delta, variable, value_bin, event, segment, subword, code` |
| `modality` | str or None | |
| `sample_id` | str | |
| `index` | int | position in the sequence |
| `timestamp_min` | float or None | |
| `position` | (int,int,int) or None | |
| `code` | int or None | local index within the token's range (codebook entry, bin, state) |
| `embedding` | list[float] or None | the vector of a `<CONT>` slot |
| `confidence` | float or None | reserved for model-predicted tokens; no codec in this release sets it |
| `metadata` | dict | on modality-start tokens: `{rep, span_meta}` |

`to_tokens` and `from_tokens` are exact inverses. Spans are re-derived from the boundary tokens, and span metadata travels on the start token, so a JSONL file is self-describing.

## 5. Positional information

| Modality | `pos` = (p0, p1, p2) | `t` |
|---|---|---|
| text | (sub-word index, −1, −1) | segment time |
| image | (row, col, −1) of the latent grid; `<ROW>` = (row, −1, −1) | segment time |
| video | (latent time step t′ from the segment start, row, col); `<CLIP>` = (t′, −1, −1). Segment tokens: (first clip, −1, −1) | start time of the latent step / segment |
| time-series | VQ / continuous: (window, group, −1); events: (grid step, variable index, −1); static block −1 | window / grid-step time |
| point cloud | (x, y, z) voxel coordinate | segment time |

`positional_encoding/unified.py` turns these side channels into an additive encoding:

- a modality embedding;
- a sinusoidal encoding of log-scaled absolute time;
- three per-modality axis tables (learned) or sinusoids (`kind="sinusoidal"`), so an image row and a video frame index do not share weights;
- learned "n/a" vectors wherever a channel is −1.

Positions live in side channels rather than as extra tokens, so sequences stay short.

## 6. Codecs and reversibility

Every codec implements `encode(Segment) -> TokenSpan`, `decode(TokenSpan) -> payload`, `raw_bits`, `payload_bits` and `alphabet_size` (`codecs/base.py`). `MultimodalCodec` does the same at sample level and batches the neural codecs.

| Codec | Span layout | Reversibility | What is lost |
|---|---|---|---|
| `ClinicalTextTokenizer` | sub-word ids | **exact** (byte-level BPE, no `<UNK>`) | nothing |
| `ImageVQCodec` | g × (`<ROW>` + g codes) | lossy | quantisation + decoder error |
| `ImageContinuousCodec` | g² `<CONT>` slots + fp16 vectors | lossy | decoder error (+ fp16 rounding) |
| `TSEventCodec`, mode `obs` | `<STATIC>` block, then per grid step `[<DT>] (<VAR> <VBIN>)*` | **exact w.r.t. the binned grid**; lossy w.r.t. raw values | within-step averaging (gridding), value binning |
| `TSEventCodec`, mode `change` | as `obs`, but only when a variable's bin changes | exact on observed cells w.r.t. the binned grid | also the observation times of repeated values |
| `TSVQCodec` | `<STATIC>` block, then per window `<PATCH>` + G codes (or `<MISSING>`) | lossy | gridding, quantisation, decoder error; static values binned |
| `TSContinuousCodec` | `<STATIC>` block + W·G `<CONT>` slots | lossy | gridding, decoder error; static values binned |
| `VideoVQCodec` | per clip `<CLIP>` + t′·h·w codes | lossy | spatial / temporal downsampling, quantisation |
| `VideoContinuousCodec` | t′·h·w `<CONT>` slots per clip | lossy | downsampling, decoder error |
| `VideoSegmentCodec` | per run of equal states `<SEG:k> <DT:duration>` | the **state sequence** is exact (boundaries from `pos`); frames very lossy (rendered from state prototypes) | all within-state appearance |
| `AnnotatedPhaseEventCodec` | per phase change `[<DT:gap>] <EV:…:phase:name>` | **exact** at the sampled-frame rate (frame index in `pos`) | nothing at that rate |
| `PointCloudTokenizer` | one code per occupied voxel, Morton order | lossy | point positions within a voxel (resampled from a Gaussian) |

The static block of the time-series codecs is identical across the three representations, so any difference between them comes from the temporal tokens.

## 7. Bit accounting (used for compression ratios)

- **Raw bits** (`Codec.raw_bits`):
  - image and video: `H·W·C·8`;
  - time-series: 56 bits per observation (16-bit time + 8-bit variable + 32-bit value) plus 32 bits per static value;
  - text: UTF-8 bytes × 8;
  - point cloud: `N·3·32`;
  - phase annotations: `T·⌈log2(#phases+1)⌉`.
- **Fixed-length token bits**: every discrete payload token costs `log2(alphabet)`, where `alphabet` is the number of ids the codec can emit. Continuous vectors cost 16 bits per value (fp16). `<CONT>` placeholders cost nothing.
- **Entropy-coded bits**: discrete tokens at the empirical entropy of their id distribution over the evaluated split. This is a lower bound for any per-token prefix code.

Span delimiters and `<DT>` tokens between spans are shared protocol overhead. They are counted in sequence length but not in payload bits.

## 8. Adding a modality or dataset

1. Choose (or reserve) an id range and add special tokens by appending to `SPECIAL_TOKENS`. Never reorder it. Bump `PROTOCOL_VERSION` if an existing id changes meaning.
2. Implement a `Codec` subclass: `encode`, `decode`, `alphabet_size`, `raw_bits`, and `encode_batch` for neural codecs.
3. Add a dataset adapter in `data/datasets.py` and a YAML config.
4. Add synthetic tests: exact reversibility where claimed, shapes, and serialization round-trip.
