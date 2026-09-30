# data/

Git-ignored except this file.

```
data/raw/medmnist/            MedMNIST v2 npz files          (downloaded by `prepare`)
data/raw/physionet2012/       set-a + Outcomes-a.txt         (downloaded by `prepare`)
data/raw/cataract101/         cataract-101.zip               (download manually, see docs/datasets.md)
data/processed/assets/        lexicon.json, text_bpe.json
data/processed/cataract101_2fps_64/   sampled frame cache, one npz per video
data/processed/physionet2012_stays_s0.pkl   parsed-stay cache
data/processed/data_stats.json        dataset statistics computed from the files present
```

Never commit anything below `raw/` or `processed/`: it is patient-level (derived) data.
Controlled datasets (MIMIC, Cholec80, SCARED) must not be placed in the repository.
