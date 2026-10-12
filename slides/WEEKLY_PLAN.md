# Weekly supervisor updates

The work in this repo is released to the supervisor one slice per week. Each deck has its own builder script (`build_weekNN.py` → `weekNN_progress.pptx`).
Any result number on a slide must be copied from `reports/RESULTS.md` or `reports/tables/`. Never type one by hand.

Scope agreed with the supervisor (feedback of 2026-10-08 on the Week 1 pending decisions): a scoped, reproducible experimental prototype.
Images and physiological time-series are fully implemented; text is supported in the protocol (synthetic templated notes); surgical video is adapted through Cataract-101 phase events; point clouds get an interface and tests. At least two downstream tasks are kept (MedMNIST classification, PhysioNet 2012 mortality). Not all five modalities are promised as trained implementations.

| Week | Date | Focus | Deliverables | Source material |
|---|---|---|---|---|
| 1 | Oct 8 | Foundation and protocol | protocol v0.2, 13,824-token vocabulary, synthetic tests (done) | `docs/token_protocol.md`, `docs/datasets.md` |
| 2 | Oct 15 | Image tokenization | MedMNIST continuous projection vs 1,024-entry VQ codebook; reconstruction error, codebook utilisation, token counts | `experiments/image_*`, figures `recon_images`, `codebook_usage`, `tsne_image_latents` |
| 3 | Oct 22 | Time series | PhysioNet 2012; missing values and time encoding; continuous, VQ and event representations | `experiments/timeseries_*`, figure `recon_timeseries` |
| 4 | Oct 29 | Adaptation and integration | synthetic templated text (v2), Cataract-101 phase events, point-cloud interface; integration tests | `experiments/interfaces`, `experiments/video_*`, `tests/test_pipeline_integration.py` |
| 5 | Nov 5 | Tasks and ablations | image classification (AUROC, macro-F1) and ICU mortality (AUROC, AUPRC) on fixed splits and budgets; codebook-size and time-granularity ablations | `experiments/ds_*`, `experiments/abl_*`, figures `downstream`, `ablation_codebook_size` |
| 6 | Nov 12 | Reproducibility and delivery | compression, reconstruction, utilisation and task metrics summarised; code, specification, report, demo | `reports/RESULTS.md`, `docs/`, `scripts/run_all.sh`, notebook |

Each deck follows the same outline: recap of last week → work done → results / figures → issues and decisions → next week plan + questions.

## Conditions to keep visible on the slides

- **End of Week 2 checkpoint:** confirm the data are readable, the compute is enough, and both downstream tasks run. If behind schedule: reduce sample size with fixed splits, keep images, time-series and both tasks, and move advanced video encoding and point-cloud training to "extensions".
- Continuous projection is a continuous baseline; never count its vectors as vocabulary tokens. Compression comparisons count vector precision (fp16), token-index bit width and required metadata (`ratio (+overhead)`).
- Cataract-101 results are not comparable with Cholec80. Phase-event tokens from ground-truth labels show annotation encoding, not phase recognition.
- Templated text is synthetic; text is accepted on round-trip, field preservation and protocol compatibility only, and never replaces the downstream tasks.
- Cite Cataract-101 (Schoeffmann et al., MMSys 2018) and respect CC BY-NC 4.0.
