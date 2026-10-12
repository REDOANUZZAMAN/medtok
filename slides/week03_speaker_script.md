# Week 3 progress report: speaker script

Deck: `week03_progress.pptx` (9 slides). Target length: about 8 minutes plus questions.

All numbers on the slides come from `reports/tables/` (`table_00.csv` for the recap, `table_01.csv`
for time series, `table_08.csv` and `table_11.csv` for the grid error and the missing-value runs).
The builder script reads them; nothing is typed by hand.

---

## Slide 1: Title (about 15 s)

Good morning, Professor. This is my third weekly report. This week is about the second fully
implemented modality: ICU time series from PhysioNet 2012.

## Slide 2: Recap (about 40 s)

Last week was images. On BloodMNIST, continuous reached a higher PSNR than VQ, but VQ gives real
tokens and a much higher compression ratio. With the three codebook tricks, almost all codes are
used, while vanilla VQ-VAE collapses.

The Week-2 checkpoint passed: the data is readable, the compute is enough with the GPU re-run, and
both downstream tasks run. So there is no scope reduction.

## Slide 3: Data, missing values, time (about 1 min 30 s)

Three parts.

**Data.** PhysioNet 2012 set A is open access. I use the first 48 hours of each ICU stay: 37
variables and 4 static values. The split is at patient level with a fixed seed. Normalisation and
bins are fitted on training stays only, so there is no leakage. The task is in-hospital mortality.

**Missing values.** This is the first hard part. Most variables are measured only a few times a
day. I put the data on an hourly grid, and I keep a mask that says which cells are really observed.
Empty cells get 0 for the encoder. I only score errors on observed cells, never on the filled ones.
During training I also hide some observed cells, so the codes cannot just copy the input.

**Time.** This is the second hard part. A `<DT>` token gives the coarse time gap between events. The
exact time in minutes is kept in a side channel on every token.

Moving the raw data onto the hourly grid already costs some error. I report this "gridding error"
separately, because all methods share it.

**Why a mask:** without it, the model cannot know if a value is real or filled.

## Slide 4: Three representations (about 1 min)

- **A, continuous:** a window autoencoder over the grid gives fp16 vectors. It is the baseline, and
  its vectors are not vocabulary tokens.
- **B, VQ:** the same autoencoder, but each latent vector becomes the nearest of 512 codebook
  entries. Each window becomes a `<PATCH>` token plus 4 codes.
- **C, event tokens:** each measurement becomes two tokens: the variable, like heart rate, and its
  value bin, from 16 quantile bins. The "changes only" variant writes a value only when its bin
  changes.

All three start with the same static block, use the same splits, and are scored the same way.

**Why event tokens:** they learn nothing except the bins. They are simple, and a doctor could read
them.

## Slide 5: Results table (about 1 min 15 s)

This table is the main result of the week.

- Continuous has a low error (second-lowest MAE and the lowest RMSE), but its positions are
  vectors, not tokens. The yellow box says this.
- VQ is the shortest discrete sequence and has the highest compression, but also the highest error.
- Event tokens have the lowest error. It is close to the gridding error itself. But the sequences
  are very long: several hundred positions on average.
- The "changes only" variant gives the same error with a shorter sequence.

The ratio with overhead also counts the block delimiters and metadata, as for images.

**Why the event error is so low:** the event codec stores every observed value as a bin. The only
error is the bin width.

## Slide 6: Figure (about 50 s)

This is one real test stay. Black dots are the real measurements. The lines are the decoded values.

The green event line passes through the dots. The blue continuous line follows the trend well. The
orange VQ line is smoother and misses some peaks, because only 4 codes must describe a whole window.

The GCS panel looks noisy, but that is only because the value is almost constant and the scale is
small.

## Slide 7: Missing values (about 1 min)

Here I compare two ways to fill empty cells for the VQ encoder.

Zero fill with the mask is my main setting. LOCF means "last observation carried forward": it
copies the last value. LOCF almost doubles the error, even though the mask is given in both cases.
It also uses the codebook less evenly: the perplexity is much lower. My explanation is that copied
values make the windows look busy, and codes are wasted on them.

The vanilla VQ-VAE row shows again that the codebook tricks matter: without them, some codes die.

So my decision is to keep zero fill with the mask. Window length and sampling interval are
ablations for Week 5.

**Why test this:** missing values are the main problem of ICU data. I need to show that my choice
is tested, not a guess.

## Slide 8: Issues and decisions (about 1 min)

1. **Event sequences are long:** the longest stay has more than a thousand positions. So I added the
   "changes only" variant.
2. **VQ loses detail:** VQ has the highest error. It is the most compact, so Week 5 checks if the
   task still works.
3. **Fair compression:** the same raw-bit scale for all methods, fp16 for continuous, and the
   overhead counted.
4. **Grid error is shared:** it is reported on its own, so no method is blamed for it.

## Slide 9: Next week and questions (about 45 s)

Next week: synthetic text, Cataract-101 video with phase events, the point-cloud interface, and
integration tests where all modalities go into one sequence and one model.

My questions:
1. Is zero fill with a mask acceptable, or should I also try interpolation?
2. Is the 48-hour window with an hourly grid fine?
3. Should the main time-series method in the thesis be event tokens or VQ?

Thank you.

---

## Possible questions and short answers

- **Why score only observed cells?** Filled cells are not real data. Scoring them would reward a
  model for guessing the fill value.
- **Why is VQ worse than continuous here?** Both use the same encoder, but VQ must pick one of 512
  codes per vector. With only 4 codes per window, peaks are lost.
- **Why 16 bins?** Quantile bins fitted on training data; 16 gives a small error and a small
  vocabulary range.
- **What is the gridding error?** The error from moving irregular measurements onto an hourly grid.
  Every method has it, so I report it separately.
- **Why do both event variants have the same error?** "Changes only" drops a value only when the bin
  did not change, so decoding gives the same bins on observed cells.
