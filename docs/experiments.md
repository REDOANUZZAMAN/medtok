# Experiments

## Research question

*Can heterogeneous medical modalities be converted into a unified discrete token representation while preserving useful information, reducing sequence length, and supporting downstream medical AI tasks?*

We compare three representation strategies on three fully implemented modalities, using identical architectures and training budgets within each modality:

| | 2D images (MedMNIST) | Time-series (PhysioNet 2012) | Surgical video (Cataract-101) |
|---|---|---|---|
| **A. Continuous projection** | autoencoder latents, 7×7×32, as `<CONT>` slots | window-AE latents, 12×4×16 | clip-AE latents, 2×4×4×32 per 8 s clip |
| **B. Vector quantisation** | same AE + VQ (K = 512) → 49 codes + 7 `<ROW>` | same AE + VQ (K = 512) → 12 × (`<PATCH>` + 4 codes) | same AE + VQ (K = 512) → `<CLIP>` + 32 codes per clip |
| **C. Event-based tokens** | not defined for a static image | measurement events `<VAR><VBIN>` (mode `obs`) and change events (mode `change`) | unsupervised appearance-state segments `<SEG:k><DT>` (K = 32 states) |

All three time-series representations carry the same discrete static block (age, gender, height, ICU type).

## Protocol of each experiment

1. **Tokenizer training** (`python -m medical_tokenizer.train`).
   - Images: union of the three MedMNIST training splits, 10 epochs, AdamW 2e-3, one-cycle schedule, batch 128, loss = MSE + VQ loss.
   - Time-series: 100 epochs, AdamW 1e-3, batch 64. Loss = MSE on observed cells in z-units + VQ loss. A random 20 % of observed cells is hidden from the encoder but still scored (a denoising objective, so codes cannot copy cells one to one).
   - Video: 8-frame clips at 1 fps, 64 px, up to 40 random clips per training video, 8 epochs, AdamW 1e-3, batch 32.
   - Event tokenizers learn nothing except quantile bins (time-series) or k-means states over the continuous video encoder's pooled clip latents (video), all fitted on the training split.
2. **Tokenizer evaluation** (`python -m medical_tokenizer.evaluate`) on the **test split**:
   - reconstruction: images / frames — MSE, MAE, PSNR, SSIM in 0–255 pixel units; time-series — MAE / RMSE in per-variable z-units on observed grid cells only;
   - compression (§ bit accounting in `docs/token_protocol.md`);
   - sequence length;
   - codebook utilisation: active codes, utilisation, entropy, perplexity, dead %, top-10 % share.
3. **Downstream validation** (`python -m medical_tokenizer.downstream`). The same `UnifiedTokenTransformer` (d = 128, 4 layers, 4 heads, ≈ 2.6 M parameters, mostly the 13,824 × 128 token embedding) is trained from scratch on the token shards:
   - AdamW 5e-4, one-cycle schedule, up to 30 epochs;
   - early stopping on the validation metric (patience 6), best-validation checkpoint tested once;
   - **3 seeds**, reported as mean ± std.

   The model sees only protocol fields and never raw data.

   | Task | Tokens from | Label | Selection metric | Also reported |
   |---|---|---|---|---|
   | BloodMNIST cell type | image tokenizer | 8 classes | accuracy | macro-F1, AUROC |
   | PneumoniaMNIST | image tokenizer | binary | AUROC | accuracy, macro-F1 |
   | DermaMNIST lesion type | image tokenizer | 7 classes | accuracy | macro-F1, AUROC |
   | In-hospital mortality | time-series tokenizer | binary (13.8 % positive) | AUROC | AUPRC, balanced accuracy |
   | Surgical phase (online) | video tokenizer | 10 phases, phase of the last frame of a 32 s window | macro-F1 | accuracy, balanced accuracy |
   | Multi-task | all three VQ tokenizers | all of the above in **one** model | mean of the selection metrics | — |

4. **Ablations** (`configs/ablations.yaml`), each group at an equal, reduced budget with its own reference run:
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
