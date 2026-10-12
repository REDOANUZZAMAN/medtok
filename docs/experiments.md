# Experiments

## Research question

*Can heterogeneous medical modalities be converted into a unified discrete token representation while preserving useful information, reducing sequence length, and supporting downstream medical AI tasks?*

We compare three representation strategies on three fully implemented modalities, using identical architectures and training budgets within each modality:

| | 2D images (MedMNIST) | Time-series (PhysioNet 2012) | Surgical video (Cataract-101) |
|---|---|---|---|
| **A. Continuous projection** | autoencoder latents, 7×7×32, as `<CONT>` slots | window-AE latents, 12×4×16 | clip-AE latents, 2×4×4×32 per 8 s clip |
| **B. Vector quantisation** | same AE + VQ (K = 1,024) → 49 codes + 7 `<ROW>` | same AE + VQ (K = 512) → 12 × (`<PATCH>` + 4 codes) | same AE + VQ (K = 512) → `<CLIP>` + 32 codes per clip |
| **C. Event-based tokens** | not defined for a static image | measurement events `<VAR><VBIN>` (mode `obs`) and change events (mode `change`) | unsupervised appearance-state segments `<SEG:k><DT>` (K = 32 states) |

All three time-series representations carry the same discrete static block (age, gender, height, ICU type).

**Continuous projection is a continuous baseline, not a discrete tokenization.** Its `<CONT>` positions carry fp16 vectors; they are reported as *continuous vectors*, never counted as vocabulary tokens. Compression comparisons use the same raw input scale for every strategy and count fp16 precision for continuous vectors, `log2(alphabet)` bits per token index, and (in the "+overhead" ratio) the span delimiters and decoding metadata (`docs/token_protocol.md` §7).

**Scope (six-week plan, agreed with the supervisor).** Images and physiological time-series are the two fully implemented modalities, each with a downstream task (MedMNIST classification, PhysioNet 2012 mortality). Surgical video is adapted through Cataract-101 phase events; text is supported through synthetic templated notes; point clouds have an interface and tests. Video encoders and point-cloud training beyond this are extensions, not commitments. The unified protocol still defines fields and vocabulary ranges for all five modality categories.

## Protocol of each experiment

1. **Tokenizer training** (`python -m medical_tokenizer.train`).
   - Images: union of the three MedMNIST training splits, K = 1,024 codebook, 10 epochs, AdamW 2e-3, one-cycle schedule, batch 128, loss = MSE + VQ loss.
   - Time-series: 100 epochs, AdamW 1e-3, batch 64. Loss = MSE on observed cells in z-units + VQ loss. A random 20 % of observed cells is hidden from the encoder but still scored (a denoising objective, so codes cannot copy cells one to one).
   - Video: 8-frame clips at 1 fps, 64 px, up to 40 random clips per training video, 8 epochs, AdamW 1e-3, batch 32.
   - Event tokenizers learn nothing except quantile bins (time-series) or k-means states over the continuous video encoder's pooled clip latents (video), all fitted on the training split.
   - Every codebook, normaliser, quantile binner, k-means model and the text BPE is fitted on the **training split only**. Splits are fixed (seed 0): patient-level for PhysioNet, whole-video for Cataract-101 (adjacent frames of one surgery never cross splits), the official splits for MedMNIST.
2. **Tokenizer evaluation** (`python -m medical_tokenizer.evaluate`) on the **test split**:
   - reconstruction: images / frames — MSE, MAE, PSNR, SSIM in 0–255 pixel units; time-series — MAE / RMSE in per-variable z-units on observed grid cells only;
   - compression (§ bit accounting in `docs/token_protocol.md`);
   - sequence length;
   - codebook utilisation: active codes, utilisation, entropy, perplexity, dead %, top-10 % share.
3. **Downstream validation** (`python -m medical_tokenizer.downstream`). The same `UnifiedTokenTransformer` (d = 128, 2 pre-LN layers, 4 heads, 2.78 M parameters, of which 1.77 M are the 13,824 × 128 token embedding) is trained from scratch on the token shards:
   - AdamW 5e-4, one-cycle schedule, batch 32, up to 20 epochs;
   - early stopping on the validation metric (patience 5), best-validation checkpoint tested once;
   - class-weighted cross-entropy: each class is weighted by n / (K · count) from the training labels, so rare classes (e.g. DermaMNIST melanoma, ICU deaths ≈ 14 %) count as much as common ones;
   - binary tasks (PneumoniaMNIST, ICU mortality): the decision cut-off on P(positive) is the one that maximises balanced accuracy on the **validation** split, then applied unchanged to test. AUROC / AUPRC do not depend on it; accuracy, balanced accuracy and macro-F1 do. Multiclass tasks use argmax;
   - both rules were approved by the supervisor on 2026-10-12 and apply to every representation alike;
   - training seconds per epoch are recorded as a cost measure (self-attention cost grows with the square of sequence length);
   - **3 seeds**, reported as mean ± std.

   The model sees only protocol fields and never raw data.

   | Task | Tokens from | Label | Selection metric | Also reported |
   |---|---|---|---|---|
   | BloodMNIST cell type | image tokenizer | 8 classes | AUROC (one-vs-rest, macro) | **macro-F1**, accuracy |
   | PneumoniaMNIST | image tokenizer | binary | AUROC | **macro-F1**, accuracy |
   | DermaMNIST lesion type | image tokenizer | 7 classes | AUROC (one-vs-rest, macro) | **macro-F1**, accuracy |
   | In-hospital mortality | time-series tokenizer | binary (13.8 % positive) | AUROC | **AUPRC**, balanced accuracy |
   | Surgical phase (online) | video tokenizer | 10 phases, phase of the last frame of a 32 s window | macro-F1 | accuracy, balanced accuracy |
   | Multi-task | all three VQ tokenizers | all of the above in **one** model | mean of the selection metrics | — |

   Headline metrics: **AUROC and macro-F1 for images, AUROC and AUPRC for ICU mortality.** The two required downstream tasks are MedMNIST classification and PhysioNet 2012 mortality; surgical phase and multi-task are additional.

   Interpretation limits:
   - Surgical phase uses Cataract-101 (10 cataract phases), not Cholec80 (7 cholecystectomy phases). Its numbers are **not comparable with Cholec80 results**.
   - Annotated phase-event tokens (`AnnotatedPhaseEventCodec`) are generated from ground-truth labels. They show that annotations can be encoded and losslessly decoded; they say nothing about recognising phases from raw video. The phase-recognition task above uses only video tokens.
   - `ds_mortality_event_text` adds **synthetic** templated admission notes built from admission-time fields only; the outcome never enters the text. It is a protocol / layout test, not evidence about clinical text.

4. **Text acceptance.** The text modality is accepted on (i) lossless round-trip (decoded note == original), (ii) field preservation (every value, unit, time and negation cue survives the round-trip, `note_fields`), and (iii) protocol compatibility (text spans serialize to JSONL / NPZ and back together with other modalities; malformed input raises). These tests do **not** replace the downstream evaluation above and support no claim about real clinical report understanding, image-text alignment or clinical generalisation. Real image-text alignment would require authorised, genuinely paired reports.

5. **Ablations** (`configs/ablations.yaml`), each group at an equal, reduced budget with its own reference run:
   - image: codebook size K ∈ {64 … 1024}; token dimension ∈ {8, 32, 64}; patch size ∈ {2, 4, 7} (one patch-embedding architecture for all); codebook-collapse remedies switched off one at a time (dead-code restart, cosine lookup, EMA updates, all three = vanilla VQ-VAE);
   - time-series: K; window length / stride (incl. overlapping windows); grid sampling interval ∈ {30, 60, 120} min; missing-value filling (zero vs LOCF); vanilla VQ-VAE;
   - video: temporal sampling rate ∈ {0.5, 1, 2} fps; K.

## Reproducing everything

```bash
python -m medical_tokenizer.data.prepare --config configs/data.yaml
python -m medical_tokenizer.pipeline --config configs/experiment_comparison.yaml --skip-existing
python -m medical_tokenizer.pipeline --config configs/ablations.yaml --skip-existing
python -m medical_tokenizer.report          # -> reports/RESULTS.md, reports/tables/*.csv, reports/figures/*.png
```

Each run directory stores its resolved config, environment (library versions, git commit, dirty flag), per-epoch log, checkpoint, codebook, token shards and metrics. Results are deterministic up to the non-determinism of multi-threaded CPU kernels.
