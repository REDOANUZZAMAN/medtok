# Datasets

All numbers in this file come from `data/processed/data_stats.json`, which `python -m medical_tokenizer.data.prepare` writes from the files actually on disk. Re-run it to regenerate them.

## Data-handling rules followed

- Only **open** datasets are downloaded automatically: MedMNIST v2 and PhysioNet 2012 set-a.
- **Controlled or request-only** datasets (MIMIC-*, Cholec80, SCARED) are never downloaded by code. Their loaders only read a local copy that the user has obtained under the dataset's terms.
- Cataract-101 is openly downloadable but has a non-commercial licence. It is fetched manually (one `curl`, below) after reading the licence, and the code only reads the local zip.
- `data/raw/`, `data/processed/` and `experiments/` (checkpoints, token shards containing patient-level derived data) are git-ignored. Only aggregated metrics, tables and figures (`reports/`) are committed.
- No patient identifier is logged. Sample ids are the datasets' own anonymous record / video ids.

## Used datasets

### MedMNIST v2 — 2D medical images (primary image modality)

- Source: Zenodo record 10519652 (Yang et al., *Scientific Data* 2023). Licence CC BY 4.0.
- Subsets: **BloodMNIST** (8 classes), **PneumoniaMNIST** (2 classes, grayscale → replicated to RGB), **DermaMNIST** (7 classes), at 28 × 28.
- Official train / val / test splits are used unchanged.

| Subset | train | val | test |
|---|---|---|---|
| BloodMNIST | 11,959 | 1,712 | 3,421 |
| PneumoniaMNIST | 4,708 | 524 | 624 |
| DermaMNIST | 7,007 | 1,003 | 2,005 |

The image tokenizer is trained on the union of the three training splits (23,674 images).

### PhysioNet/CinC Challenge 2012, set-a — physiological time-series

- Source: physionet.org/content/challenge-2012/1.0.0. Licence ODC-By 1.0 (open access).
- 4,000 ICU stays. Each has up to 48 h of irregular measurements of 37 variables plus 4 static descriptors (age, gender, height, ICU type), and the in-hospital mortality label.
- Cleaning:
  - `-1` missing codes and physiologically impossible values are dropped (ranges in `data/physionet.py:VALID`);
  - static values are taken at 00:00 only.
- Split: patient-level, stratified by outcome, seed 0 — 70 / 15 / 15 % → 2,799 / 599 / 602 stays; mortality 13.8 / 13.9 / 14.0 %; about 430 observations per stay (median 420).
- Normalisation (z-score after `log1p` for heavy-tailed labs) and value bins (16 per-variable quantiles) are fitted on the **training split only**.

### Cataract-101 — surgical video (primary video modality, Cholec80 stand-in)

- Source: ftp.itec.aau.at/datasets/ovid/cat-101 (Schoeffmann et al., MMSys 2018). Licence **CC BY-NC 4.0**, research use only; cite the paper.
- 101 cataract surgeries, H.264, 720 × 540 at 25 fps, mean 8.3 min.
- Annotations are phase **start frames** for 10 quasi-standardised phases. A phase lasts until the next annotation. Frames before the first annotation are unannotated (label −1) and are excluded from phase recognition. Phase 2 ("viscous agent injection") legitimately occurs twice per surgery.
- Obtain manually:
  ```bash
  mkdir -p data/raw/cataract101 && curl -L -o data/raw/cataract101/cataract-101.zip \
      http://ftp.itec.aau.at/datasets/ovid/cat-101/downloads/cataract-101.zip     # 9.0 GB
  ```
- `prepare` samples every video at **2 fps**, centre-crops it to 540 × 540 and resizes it to **64 × 64** into `data/processed/cataract101_2fps_64/<video>.npz`. Experiments subsample this cache to 1 fps or 0.5 fps.
- Split: video-level (= patient-level), seed 0, 70 / 15 / 15 %.

**Why Cataract-101 instead of Cholec80:** Cholec80's public archive now returns HTTP 403, so it needs a CAMMA data request. Cataract-101 is the closest openly available surgical-video dataset with expert phase annotations. `data/video.py` includes Cholec80 parsers (`cholec80_phases`, `cholec80_tools`), and the same frame sampler and codecs apply unchanged once a local copy exists.

### Text — templated notes (interface modality)

No open dataset in scope contains clinical free text; MIMIC notes are credentialed-access. `data/notes.py` renders short notes from structured fields:

- ICU admission notes use admission-time fields only, so no outcome leakage;
- image-request notes give modality only, never the diagnosis;
- video notes give the procedure type only.

They exercise the text tokenizer and the multimodal layout. **No conclusion about clinical language is drawn from them.**

## Not used, interfaces only

| Dataset | Access | Status |
|---|---|---|
| Cholec80 | CAMMA data request | loaders for the phase / tool annotation files; `AnnotatedPhaseEventCodec` and the lexicon already include Cholec80 phases and tool on/off events |
| SCARED | grand-challenge account | `PointCloudTokenizer` + `depth_to_points` (depth map → point cloud); demonstrated on synthetic surfaces only |
| MIMIC-CXR / MIMIC notes | PhysioNet credentialing + DUA | not used; `ClinicalTextTokenizer.from_pretrained` accepts any sub-word tokenizer for future use |
