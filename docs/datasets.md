# Datasets

All numbers in this file come from `data/processed/data_stats.json`, which `python -m medical_tokenizer.data.prepare` writes from the files actually on disk. Re-run it to regenerate them.

## Data-handling rules followed

- Only **open** datasets are downloaded automatically: MedMNIST v2 and PhysioNet 2012 set-a.
- **Controlled or request-only** datasets (MIMIC-*, Cholec80, SCARED) are never downloaded by code. Their loaders only read a local copy that the user has obtained under the dataset's terms.
- **MIMIC-IV** (credentialed, PhysioNet DUA) is processed only on the project machine. Models, token shards and example reconstructions stay in `experiments/` (git-ignored). Only aggregate tables (`reports/mimic/`) are committed: no images, signals, report text or per-patient rows.
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
- Split: video-level (= patient-level), seed 0, 70 / 15 / 15 %. Whole surgeries are assigned to one split, so adjacent frames never cross train / val / test. Codebooks, the k-means segment states and any other learned preprocessing are fitted on the training videos only.
- Fixed and reported settings: frame sampling 1 fps for the tokenizers and downstream task (2 fps cache; 0.5 / 2 fps only in the sampling-rate ablation), clip length 8 frames = 8 s, no clip overlap, downstream window 4 clips = 32 s with stride 1 clip, frames 64 × 64. Event times are in seconds from the start of the video in the cache and in minutes in the protocol `t` field.
- Phase vocabulary (`CATARACT101_PHASES`, 10 classes, used as-is with no merging or remapping): the dataset's 10 phase ids map one-to-one to `<EV:cataract101:phase:…>` tokens; unannotated frames before the first annotation map to label −1 / `<EV:cataract101:phase:none>`.

| Split | videos | frames @ 2 fps | unannotated |
|---|---|---|---|
| train | 71 | 72,280 | 1,692 |
| val | 15 | 15,462 | 261 |
| test | 15 | 13,351 | 431 |

The phases are imbalanced. Phacoemulsification (phase 5) covers 31 % of annotated training frames, and capsule polishing (phase 7) 4 %. This is why phase recognition is selected and reported on macro-F1.

**Why Cataract-101 instead of Cholec80 (approved by the supervisor):** Cholec80's public archive now returns HTTP 403, so it needs a CAMMA data request. Cataract-101 is the closest openly available surgical-video dataset with expert phase annotations. `data/video.py` includes Cholec80 parsers (`cholec80_phases`, `cholec80_tools`), and the same frame sampler and codecs apply unchanged once a local copy exists.

This is a change of **surgical domain and label set**: cataract surgery (ophthalmology, operating microscope, 10 phases, ~8 min per case) instead of laparoscopic cholecystectomy (7 phases + 7 tools, ~38 min per case). Consequences:

- Results on Cataract-101 are **not comparable** with published Cholec80 results and are never presented as such.
- Event tokens generated from the ground-truth phase labels (`AnnotatedPhaseEventCodec`) demonstrate **annotation encoding** only (lossless phase-label → token → phase-label round-trip). They are not evidence of automatic phase recognition from raw video; the phase-recognition task uses video tokens only.

### Text — synthetic templated notes (interface modality)

No open dataset in scope contains clinical free text. (Real radiology reports are used only in the separate MIMIC-IV validation below.) `data/notes.py` renders short notes from structured fields. **All of this text is synthetic.**

- Every text segment carries the source tag `synthetic-template:<kind>@v<version>` (current version 2). The generation rules and the fields each kind reads (`TEMPLATE_RULES`) are written to `data/processed/data_stats.json` and to the interfaces run.
- Coverage: several wording variants per sentence, units (cm, kg, bpm, mmHg, °C), missing values ("not recorded"), negation ("Not mechanically ventilated in the first hour", "GCS not assessed", "No vital signs recorded"), and times relative to admission ("at 00:12 after admission").
- No label leakage: ICU admission notes read minutes 0–59 only and never the outcome; image-request notes give modality / body site only, never the class; video notes give the procedure type only, never the phases. Text is therefore never derived from the label it is used to predict.
- No fake pairing: samples are built within one dataset. Records from different datasets are never concatenated and presented as patient-matched multimodal data.
- Acceptance is limited to lossless round-trip, field preservation and protocol compatibility (see `docs/experiments.md`). **No conclusion about clinical language, image-text alignment or clinical generalisation is drawn from them.** A real alignment claim would need authorised, genuinely paired reports.

### MIMIC-IV linked multimodal subset — validation on real hospital data

Supervisor's demo copy (given 2026-10-17), placed at `data/raw/mimic_demo_5gb/mimic_demo_5gb/`, with the `p10-p12` and `p13-p15` archives unpacked into `cxr/files/` next to `p16-p19`. Loader: `data/mimic.py`. Used only by `python -m medical_tokenizer.mimic_validate` (see `docs/mimic_validation.md`).

- 600 patients, one chest X-ray study each (MIMIC-CXR-JPG, all views), its free-text radiology report, the 12-lead ECGs (MIMIC-IV-ECG, 500 Hz, 10 s) and the lab results (MIMIC-IV `labevents`) of the 7 days up to the radiograph. Nothing after the radiograph is included. Join key: `subject_id`.
- These are **genuinely linked** records of the same patient, unlike the within-dataset samples above.
- Reference time t = 0 is the radiograph minus 7 days, so all segments have t ≥ 0. Label: death within 365 days of the radiograph (`patients.dod`).
- Splits: patient-level 70/15/15, stratified by the label (seed 0), same rule as PhysioNet 2012.
- Lab variables: the 40 most frequent numeric tests on the training split, named `lab:<label>` and appended to the lexicon after the PhysioNet and ECG-lead names, so existing ids do not change.

## Not used, interfaces only

| Dataset | Access | Status |
|---|---|---|
| Cholec80 | CAMMA data request | loaders for the phase / tool annotation files; `AnnotatedPhaseEventCodec` and the lexicon already include Cholec80 phases and tool on/off events |
| SCARED | grand-challenge account | `PointCloudTokenizer` + `depth_to_points` (depth map → point cloud); demonstrated on synthetic surfaces only |
