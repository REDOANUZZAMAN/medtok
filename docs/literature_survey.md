# Literature survey, round 1 (2026-10-23)

Status: built from web-search summaries only. No paper has been read in full and no number has been checked,
so treat every claim as "to verify". Chinese-language literature is not covered yet (CNKI / Wanfang need a manual search).

## 1. Medical image tokenizers
| Work | What it does | Relation to MedTok |
|---|---|---|
| MedITok (arXiv 2505.19225) | One discrete VQ tokenizer (32,768 codes) for medical images, trained on 30M+ images; reconstruction + clinical semantics in two stages | Images only; fixed token count per image. Use as an image baseline |
| MedUAG (arXiv 2608.18937) | Unified understanding and generation for medical models; ViT + VAE + flow matching, not discrete VQ | Related, different design |

## 2. ECG tokenizers (the most crowded area)
| Work | Method |
|---|---|
| QINCo ECG tokenizer, ICLR 2025 (MIMIC-IV) | Adaptive residual quantization for ECG; linear probe on 6 cardiac conditions |
| Clinically faithful ECG reports, NeurIPS 2025 | QINCo tokenizer on 12-lead signals, report generation |
| HeartLLM (arXiv 2508.15338, AAAI) | Fixed-scale quantization (FSQ) tokens, LLM pretraining |
| AnyECG (arXiv 2411.17711) | Vector-quantized rhythm codebook |
| ECG2TOK (IJCAI 2025) | Self-distillation + online clustering for semantic tokens |
| ECG-Byte (arXiv 2412.14373) | Byte-pair-encoding style tokenizer |
| Beat-synchronous tokenization (arXiv 2608.30367), GeoRVQ (arXiv 2609.27018) | Newer ECG tokenizers, not yet read |

**Risk:** residual quantization for ECG on MIMIC-IV already exists (QINCo). "We use residual VQ on ECG" cannot be the contribution.

## 3. Multimodal EHR with tokens
| Work | Method |
|---|---|
| Autoregressive EHR Foundation Models with Multimodal Inputs (arXiv 2607.22264, ICML 2026) | Event-token stream (MEDS format, MIMIC-IV); ECG, chest X-ray and notes added through gated cross-attention with missing-modality masks. **Closest competitor to the protocol idea** |
| MedTok, ICML 2025 (arXiv 2502.04397) | Tokenizes medical codes (text + graph), not images or waveforms |
| MedPatch | Late fusion of EHR, CXR and notes; no shared token sequence |
| MedM2T (arXiv 2510.27321) | Time-aware EHR + ECG modelling with cross-attention |
| Symile / Symile-MIMIC | CXR + ECG + labs, representation learning (from the research plan, not re-searched) |

## 4. Adaptive token budget
| Work | Domain |
|---|---|
| FlexTok, ALIT, One-D-Piece, STAT, AdaTok, ElasticTok | Natural images: variable-length 1D tokenizers, tail dropping, learned allocation |
| GSToken, TokenSeg, CARVE | Medical 3D: lesion-aware or slice-wise token allocation for segmentation / understanding |

## 5. What the first round suggests
- No work found that puts chest X-ray, ECG and labs in one shared token budget and allocates it by clinical value.
  A quick search finding nothing is not proof; a deeper search is needed before claiming this.
- Safer claim: clinical-utility-based budget allocation across modalities, compared against fixed-rate and plain residual VQ.
- Baselines to run or cite: MedITok (images), QINCo (ECG), the ICML 2026 multimodal EHR model, FlexTok-style adaptive length.

## 6. Next
1. Read the 6 closest papers in full: QINCo ICLR + NeurIPS, ICML 2026 multimodal EHR, MedITok, GSToken, AdaTok.
2. Search again: "residual quantization adaptive length", "rate-distortion medical tokenizer", "ECG chest X-ray joint tokenization".
3. Chinese literature: manual CNKI / Wanfang search, plus Chinese groups on medical VQ tokenizers.
