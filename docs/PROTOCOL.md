# MedTok unified data protocol and lexicon specification

Protocol version **0.1**. The code in `src/medtok/protocol/` is the reference implementation. If this document and the code disagree, the code is correct.

## 1. Scope

The protocol maps five medical modalities into **one flat token-id space** and **one sequence format**, so that a single sequence model can read any combination of them:

| Modality | Payload (`Segment.data`) | Implemented codecs | Status |
|---|---|---|---|
| `text` | UTF-8 string | byte-level BPE (`bpe`) | full |
| `image` | `uint8 (28,28,3)` | VQ-VAE codes (`vq`), continuous latents (`cont`) | full (MedMNIST v2) |
| `timeseries` | `{"obs": (n,3) [min, var, value], "static": {...}}` | event tokens (`event`), patch VQ (`vq`), continuous (`cont`) | full (PhysioNet 2012) |
| `video` | Cholec80 annotations `{"phase": (T,), "tools": (T,7)}` or frames `(T,28,28,3)` | surgical event tokens (`event`), frame VQ (`vq`) | interface + synthetic demo |
| `pointcloud` | `float32 (N,3)` | voxel + k-means codebook (`vq`) | interface + synthetic demo |

## 2. Unified sample schema (`protocol/schema.py`)

```
Sample
  sample_id : str                      e.g. "physionet2012/132539", "bloodmnist/test/17"
  segments  : list[Segment]
  labels    : dict                     task labels, e.g. {"death": 0} or {"class": 3}
  split     : "train" | "val" | "test"
  meta      : dict

Segment
  modality  : one of text | image | timeseries | video | pointcloud
  data      : raw payload (table above)
  t_start   : float, minutes relative to the sample's reference time
  source    : dataset / file of origin
  meta      : dict
```

A codec turns a `Segment` into a `TokenSpan`. The serializer turns a whole `Sample` into a `TokenSequence`, which is the wire format:

| Field | Shape / type | Meaning |
|---|---|---|
| `ids` | `(L,) int64` | global token ids in `[0, 13056)` |
| `mod` | `(L,) int8` | modality index (order as in the table), `-1` for `<bos>`/`<eos>` |
| `t` | `(L,) float32` | absolute time in minutes since the reference time, `-1` if not applicable |
| `pos` | `(L,3) int32` | spatial / structural index, `-1` if not applicable (see §5) |
| `spans` | list of dict | `{modality, rep, start, end, meta}`; `start:end` indexes the payload tokens |
| `cont` | list of `(k,d) fp16` | continuous payloads, one per `cont` span, in order |
| header | JSON | protocol version, `sample_id`, labels, spans |

On disk a sequence is one `.npz` file (`TokenSequence.save/load`). A dataset is a directory of `.npz` files plus `manifest.jsonl`, which holds one `sample_to_manifest` record per sample (ids, labels, per-segment metadata and shapes, never raw data).

## 3. Sequence layout (`protocol/serialize.py`)

```
<bos>
  <txt> <rep:bpe> t t t ... </txt>
  <dt:k> <img> <rep:vq> <row> v v v v v v v <row> ... </img>
  <dt:k> <ts>  <rep:event> <static> <var:Age> <vbin:9> ... <var:HR> <vbin:5> <dt:k> <var:HR> ... </ts>
<eos>
```

* Segments are ordered by `t_start` (the sort is stable, so ties keep their original order).
* Every span is `<open> <rep:X> payload… <close>`. The `<rep:X>` marker records which representation produced the span, so one model can mix representation types.
* Every span after the first is preceded by a `<dt:k>` token that gives the gap since the previous span.
* `validate(seq)` checks this structure. `split_spans` and `deserialize` invert the layout.

## 4. Vocabulary (`protocol/vocab.py`)

The ranges are fixed by the protocol, not learned. A sequence produced by one checkpoint of the toolkit is therefore readable by a model trained on another, as long as the protocol version matches. A codebook smaller than its range leaves the tail of the range unused.

| Range | Ids | Size | Content |
|---|---|---|---|
| `special` | 0–255 | 256 | structural tokens (below), the rest reserved |
| `dt` | 256–511 | 256 | time-delta buckets `<dt:k>` |
| `var` | 512–767 | 256 | variable names `<var:NAME>` (41 used) |
| `vbin` | 768–1023 | 256 | value quantile bins `<vbin:k>` (16 used) |
| `event` | 1024–1279 | 256 | surgical events `<ev:...>` (21 used) |
| `text` | 1280–9471 | 8192 | BPE sub-words (884 learned) |
| `image_vq` | 9472–10495 | 1024 | image codebook (512 used) |
| `ts_vq` | 10496–11007 | 512 | time-series codebook (512 used) |
| `video_vq` | 11008–12031 | 1024 | video frame codebook |
| `pc_vq` | 12032–13055 | 1024 | point-cloud codebook (256 used) |

**Total vocabulary: 13,056 ids.**

### 4.1 Special tokens

Ids are positional; new tokens may only be appended.

| id | token | id | token | id | token |
|---|---|---|---|---|---|
| 0 | `<pad>` | 10 | `<ts>` | 20 | `<row>` (image row start) |
| 1 | `<bos>` | 11 | `</ts>` | 21 | `<frame>` (video frame start) |
| 2 | `<eos>` | 12 | `<vid>` | 22 | `<patch>` (time-series patch start) |
| 3 | `<unk>` | 13 | `</vid>` | 23 | `<missing>` |
| 4 | `<sep>` | 14 | `<pc>` | 24 | `<static>` (static attributes follow) |
| 5 | `<mask>` | 15 | `</pc>` | 25 | `<role:clin>` |
| 6 | `<txt>` | 16 | `<rep:bpe>` | 26 | `<role:label>` |
| 7 | `</txt>` | 17 | `<rep:vq>` | 27 | `<cont>` (continuous slot) |
| 8 | `<img>` | 18 | `<rep:event>` | | |
| 9 | `</img>` | 19 | `<rep:cont>` | | |

### 4.2 Time-delta tokens

There are 256 buckets for gaps, in minutes. The schedule is designed so that clinically common gaps are represented **exactly**:

| Bucket | Gap |
|---|---|
| 0 | same time (≤ 0.5 min) |
| 1–60 | 1 … 60 minutes, exact |
| 61–107 | 2 … 48 hours, exact |
| 108–255 | log-spaced from 48 h to 30 days (upper edge ≤ 1.9 % above the true gap) |

`<dt>` tokens let a model see gaps in the sequence itself. The exact time is always carried in the `t` side channel.

### 4.3 Enumerated entries (`artifacts/lexicon.json`)

* **Variables (41).** 4 static (`Age, Gender, Height, ICUType`) followed by 37 temporal PhysioNet-2012 variables (`Albumin … Weight`, full list in `data/physionet.py`).
* **Value bins (16).** Per-variable training-set quantiles. `artifacts/ts_binner.json` stores the edges and the bin medians used for decoding. Discrete variables (Gender, GCS, MechVent, ICUType) collapse to fewer bins.
* **Events (21).** `phase:<P>` for the 7 Cholec80 phases, and `on:<Tool>` / `off:<Tool>` for the 7 Cholec80 tools.

`Lexicon.id_to_str` renders any id in human-readable form. See `data/samples_demo/five_modality.tokens.txt` for an example.

## 5. Positional side channels

| Modality | `t` | `pos` |
|---|---|---|
| text | segment time | (token index, 0, 0) |
| image | segment time | (row, col, 0) on the 7×7 latent grid; `<row>` has (row, −1, 0) |
| timeseries / event | observation hour × 60 | (hour, variable index, −1); `<dt>` has (hour, −1, −1) |
| timeseries / vq, cont | patch start time | (patch, group, −1); `<patch>` has (patch, −1, −1) |
| video / event | exact event time (1 s resolution) | −1 |
| video / vq | frame time | (frame, row, col) |
| pointcloud | segment time | voxel (x, y, z) on the 8³ grid |

The reference model (`eval/downstream.py`) turns these into embeddings as follows: a sinusoidal encoding of `t` (in hours, with a learned vector for "n/a"), plus three learned index embeddings for `pos`, plus a modality embedding. This keeps sequences short, because no coordinate tokens are needed.

## 6. Codec specifications

| Codec | Tokens per sample | Reversibility | Notes |
|---|---|---|---|
| `TextBPECodec` | ≈ 1 per 4 characters | **exact** (byte-level) | trained on training-split notes only |
| `ImageVQCodec` | 56 (7 × (`<row>` + 7 codes)) | approximate (VQ-VAE decoder) | 512 codes, dim 32, cosine codebook |
| `ImageContCodec` | 49 `<cont>` slots + 49×32 fp16 payload | approximate | same encoder, before quantization |
| `TSEventCodec` | static block + 2 per observed (hour, var) cell + 1 `<dt>` per active hour | exact up to hourly averaging and bin median | variable length; missing data costs 0 tokens |
| `TSVQCodec` | static block + 12 × (`<patch>` + 4 codes) = static + 60 | approximate (decoder) | 4-h patches, product quantization G=4, 512 codes of dim 16 |
| `TSContCodec` | static block + 48 `<cont>` + 48×16 fp16 payload | approximate | same encoder |
| `SurgicalEventCodec` | 1 per phase/tool change (+ `<dt>`) | **exact** at 1 s | reads Cholec80 `videoXX-phase.txt` / `-tool.txt` |
| `VideoFrameVQCodec` | 50 per frame | approximate | wraps any frame VQ model |
| `PointCloudVoxelCodec` | 1 per occupied voxel (≤ 512) | approximate (Gaussian resampling) | Morton-ordered; `depth_to_points` for SCARED depth maps |

The static block in all time-series codecs is `<static>` followed by (`<var:k>` `<vbin:q>`) for each observed static attribute.

## 7. Adding a modality or codec

1. Subclass `codecs.base.Codec`. Set `modality`, `rep` and `reversible`, and implement `encode`, `decode`, `raw_bits` and `alphabet_size`.
2. Emit ids only inside the range reserved for your modality, plus special tokens. `Range.to_global` raises an error on overflow.
3. Fill `t` and `pos` following §5.
4. Register the codec in the codec dict passed to `serialize()`.

## 8. Data governance

* PhysioNet 2012 is open (ODC-By), but the course rules forbid uploading patient-level derived data. `data/raw`, `data/processed` and `data/samples` are therefore git-ignored and are regenerated from the scripts. Only synthetic samples (`data/samples_demo/`) are committed.
* MIMIC is **not** used. Clinical text is templated from admission-time structured fields (see `data/notes.py`), so it contains no free text and no outcome information.
* Cholec80 and SCARED require data requests, so only their file formats are supported, exercised on synthetic data.
