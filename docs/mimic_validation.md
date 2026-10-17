# Validation on linked MIMIC-IV data

Supervisor, 2026-10-17: *"this week you will use the MIMIC-IV multimodal data that Rafi mentioned yesterday to test and validate your method."*

```bash
python -m medical_tokenizer.mimic_validate --config configs/mimic_validation.yaml      # full run (CPU, ~1 h)
python -m medical_tokenizer.mimic_validate --report experiments/mimic_validation/latest   # tables only
```

Results: [`reports/mimic/MIMIC_RESULTS.md`](../reports/mimic/MIMIC_RESULTS.md) (generated; never edit the numbers by hand).

## Why this validation matters

The main experiments use three separate open datasets: MedMNIST images, PhysioNet 2012 ICU charts and Cataract-101 video. Their text is synthetic, and samples never mix datasets. That leaves three questions open:

1. Do the tokenizers still work on **real clinical data from a different hospital source**?
2. Does the protocol hold when one patient has **several real modalities at once**, with real time gaps between them?
3. Can **real clinical text** go through the text tokenizer losslessly?

The MIMIC-IV subset answers all three. Each of its 600 patients has a chest X-ray study, the radiology report, 12-lead ECGs and lab results, all linked by `subject_id` (data details: `docs/datasets.md`).

## What is tested, and how

| part | what is run | what it shows |
|---|---|---|
| Chest X-ray | (a) the MedMNIST image VQ tokenizer used **unchanged** (zero-shot) at 28 px; (b) the same architecture and codebook size (K = 1024) trained on MIMIC training images at 28 px and 64 px; (c) the MIMIC 28 px tokenizer applied back to PneumoniaMNIST | whether a codebook learned on small public images transfers to real CXRs, and what training on in-domain data gains |
| ECG (new signal type) | the existing window VQ tokenizer (`TimeSeriesWindowAutoencoder` + `TSVQCodec`), 12 leads at 100 Hz, 0.2 s windows, 4 codes per window | the time-series tokenizer handles a dense high-rate waveform **without code changes**; errors in mV, PRDN and lead correlation |
| Labs | the existing event tokenizer (`TSEventCodec`) with 40 lab variables over 7 days, at 1 h, 6 h and 24 h grid steps | irregular hospital labs fit the event representation; the time-granularity trade-off (tokens vs merged results) |
| Reports | byte-level BPE trained on the synthetic notes vs BPE retrained on MIMIC training reports (1,024 and 4,096) | exact round-trip on real text; how much the synthetic-trained vocabulary costs on real language |
| Unified protocol | one sequence per patient: labs → ECG(s) → X-ray image(s) → report, saved to a shard, loaded, validated and decoded | the protocol works for real linked multimodal records; token budget per modality |
| Downstream | death within 1 year of the radiograph, from bag-of-tokens features, per modality and fused | the tokens carry clinically useful information, and modalities add up |

### Choices and their reasons

- **CPU only, on this machine.** The data is credentialed. The run is small (600 patients), so the CPU is enough.
- **Reference time = radiograph − 7 days.** The protocol uses t = −1 for "no time", so all real times must be ≥ 0.
- **Two time-series sources in one sample.** ECG (waveform VQ) and labs (events) are both the time-series modality but need different codecs. `RoutedCodec` (`codecs/base.py`) picks the codec from `Segment.source` and writes the source into the span metadata, so a sequence still decodes with no side information. This is a protocol extension found by using real data.
- **Long time gaps.** Labs span 7 days. Single `<DT>` buckets above 48 h are log-spaced, so they are not exact. In a first run they moved lab events to the wrong time step for every patient with a gap of more than 48 h between two lab results (column "patients with a gap > 48 h" in the report). The event codec now writes a gap over 48 h as a chain of exact ≤ 48 h `<DT>` tokens (`vocab.dt_chain`). Gaps up to 48 h, which covers every PhysioNet 2012 case, give exactly the same tokens as before, so earlier results are unchanged. Test: `test_ts_event_long_gaps_keep_exact_time`.
- **Images:** grayscale, centre square crop, resized, replicated to 3 channels so the MedMNIST tokenizer can read them. Raw bits count one channel.
- **ECG at 100 Hz** (block average of 5 samples): 1,000 steps × 12 leads. Errors are measured against the 100 Hz signal the tokenizer sees. Compression is reported against both the stored 500 Hz 16-bit record and a 100 Hz 16-bit signal.
- **Labs:** the 40 most frequent numeric tests on the training split, 16 quantile bins fitted on the training split, and the static block (age, sex) shared with PhysioNet.
- **Downstream: cross-validation instead of one test split.** With 600 patients and 90 deaths, a fixed test split has only 15 positives, so one AUROC number would be very noisy. Instead: 5 × repeated stratified 5-fold CV, logistic regression (L2, balanced class weights) on standardised log1p token counts. The model is deliberately simple. It tests whether the tokens carry the information, not how strong a model can be.
- **Leakage check:** labs and ECGs are from before the radiograph; the report is written at the radiograph. The label is a death date afterwards. No input contains information from after t = 7 days.

## Findings (numbers: `reports/mimic/MIMIC_RESULTS.md`)

1. **The protocol works on real linked data.** All 600 patient sequences pass structural validation. The shard save → load round-trip is identical, report text comes back exactly, and the lab observation pattern comes back exactly. Before the long-gap fix, the lab pattern was wrong for the patients with a gap of more than 48 h between lab results. Building this check is what found the bug.
2. **Images: the public-data codebook transfers only partly.** The MedMNIST tokenizer reads real CXRs zero-shot, but it uses only about half of its codebook and reconstructs clearly worse than the same design trained on MIMIC. The MIMIC-trained tokenizer uses almost the whole codebook and still works on PneumoniaMNIST. Conclusion: the architecture generalises; the codebook should be trained on in-domain data.
3. **ECG: works, but this is the weakest part.** The existing time-series tokenizer handles 12-lead ECG with no code changes and very high compression. Reconstruction keeps the overall waveform shape (moderate lead correlation) but loses detail (high PRDN). The RMSE mean is dominated by a few noisy recordings (large SD). Next step: more codes per window or a convolutional encoder.
4. **Labs: the event tokenizer fits 7-day hospital labs.** The observation pattern and re-encoding are 100 % exact at every grid step. A coarser grid (24 h) saves tokens but merges more results into one cell; 6 h merges few and is the chosen setting.
5. **Real text: the synthetic-template BPE is lossless but inefficient.** It needs more than twice as many tokens per report as a BPE retrained on real reports. Round-trip is exact for all three tokenizers.
6. **Downstream: the tokens carry information, and the modalities add up.** Every single modality is above chance for 1-year mortality, and fusing all four is better than any one. But age + sex alone is already strong, and "all tokens" is only about as good as demographics, so the gain on top of demographics is small and within the cross-validation spread. Report it as "tokens are informative and combinable", not as a mortality model.

## Limits (say these when presenting)

- 600 patients from one hospital system. This is a validation of the method, not a clinical model.
- The images are downsampled to 28 / 64 px, far below diagnostic resolution. The image metrics measure tokenizer fidelity, not diagnostic quality.
- Mortality is a convenience label derived from the linked tables. Results are exploratory, with wide confidence intervals.
- Data privacy: models trained on MIMIC (codebooks, BPE merges) stay local, because they are derived from credentialed data.
