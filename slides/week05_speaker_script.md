# Week 5 progress report: speaker script

Deck: `week05_progress.pptx` (10 slides). Target length: about 9 minutes plus questions.

All numbers on the slides come from `reports/tables/` (`table_03.csv` for downstream tasks,
`table_04.csv` and `table_08.csv` for codebook size, `table_09.csv` and `table_10.csv` for time
granularity). Downstream values are copied exactly as written in the CSV, like "0.983 ± 0.002".

---

## Slide 1: Title (about 15 s)

Good morning, Professor. This is my fifth weekly report. This week is the most important one for
the research question: do the tokens still work for real medical tasks? I also show two ablations.

## Slide 2: Recap (about 40 s)

Last week I added the adapted modalities. Synthetic text round-trips exactly and keeps all fields.
Cataract-101 video VQ uses almost all its codes. The point-cloud interface is tested on synthetic
surfaces, and one model can read image, ICU and video tokens.

The yellow box repeats two conditions: the text is synthetic, and Cataract-101 is not comparable
with Cholec80.

## Slide 3: Setup (about 1 min)

Every run uses the same model: a small Transformer trained from scratch on the token files. It only
sees protocol fields, never raw data. For each task, every representation gets the same model size
and the same training budget. I use early stopping on the validation set, and I test the best
checkpoint only once. Every result is the mean and standard deviation over 3 seeds.

The two main tasks are image classification, with AUROC and macro-F1, and ICU mortality, with AUROC
and AUPRC. Surgical phase and multi-task are extra.

**Why these metrics:** AUROC does not depend on a threshold. Macro-F1 and AUPRC show how the rare
classes do. Mortality is imbalanced: most patients survive, so accuracy alone would be misleading.

**Why the same budget:** otherwise a better score could come from more training, not from a better
representation.

## Slide 4: All downstream tasks (about 1 min)

This figure shows all five tasks. Blue is continuous, orange is VQ, green is event or segments.

For images, continuous is best on all three subsets. VQ is a bit lower, but it clearly learns the
task.

For mortality, event tokens reach about the same AUROC as continuous, but with real discrete tokens.
VQ is the weakest on mortality.

## Slide 5: Image classification (about 1 min)

This is the table for images. The VQ sequence is a bit longer, because of the row markers.

The gap between VQ and continuous is small in AUROC, and bigger in macro-F1. DermaMNIST macro-F1 is
low for both methods, because some lesion classes have very few examples.

**Why VQ is lower:** VQ lost some detail in reconstruction, as we saw in Week 2. Some of that detail
helps classification.

## Slide 6: ICU mortality and extra tasks (about 1 min 30 s)

On the left is mortality.
- Continuous and event tokens have about the same AUROC.
- But event tokens need much longer sequences, more than ten times longer.
- VQ is the weakest.
- The last row adds a synthetic admission note to the event tokens. The yellow box is important:
  this note is synthetic, built from admission fields only, never the outcome. It is a layout test,
  not a clinical text result.

On the right are the extra tasks. Surgical phase scores are low. It uses Cataract-101, which is not
comparable with Cholec80, so it stays an extra task. The multi-task table shows that one VQ model
trained on all three tasks together gets close to the single-task VQ scores.

**Why the multi-task result matters:** it shows that one shared vocabulary can really serve several
modalities in one model.

## Slide 7: Codebook-size ablation (about 1 min)

Here I change the codebook size K from 64 to 1,024.

The left plot shows the effective number of codes, the perplexity. It stays close to the dashed line,
which means all codes are used at every size. The right plot shows that PSNR goes up as K grows.

But a bigger codebook means more bits per token, so the compression ratio goes down. This is a
trade-off. For time series, the error also falls as K grows, and all codes are used.

These image runs use a short 5-epoch budget, so they are lower than the main run.

**Why this ablation:** codebook size is the main design choice of a VQ tokenizer.

## Slide 8: Time-granularity ablation (about 1 min)

Two settings for time-series VQ.

**Window length:** shorter windows give a lower error but more tokens. Overlapping 8-hour windows
recover part of the loss. I keep 4 hours as the balance point.

**Sampling interval:** a coarse 120-minute grid gives the tokenizer a lower error. But the grid
itself loses much more information: the gridding error is much higher. So both errors must be
reported together, and I keep 60 minutes.

**Why both errors:** if I only showed the tokenizer error, a coarse grid would look better, but it is
not.

## Slide 9: Issues and decisions (about 1 min)

1. **VQ is below continuous** on every image task and on mortality. I report this honestly: it is
   the price of a shared discrete vocabulary.
2. **Event tokens are long:** about the same AUROC as continuous on mortality, but much longer
   sequences, so attention costs more.
3. **Surgical phase is weak:** low macro-F1 with 64-pixel frames. It stays an extra task.
4. **Fair budgets:** the same model, splits, epochs, early stopping and seeds for every run.

## Slide 10: Next week and questions (about 45 s)

Next week is the last one: I summarise all metrics, check that one script reproduces every table and
figure, and finish the code, specification, report, demo and notebook.

My questions:
1. Is the VQ vs continuous gap acceptable as a finding?
2. Is the ablation set enough?
3. Should I recommend event tokens for ICU data?

Thank you.

---

## Possible questions and short answers

- **Why not a bigger Transformer?** Same small model for all runs keeps the comparison fair and
  fast. The goal is to compare representations, not to get the best possible score.
- **Why 3 seeds?** To show the spread. If two methods differ by less than their standard deviation,
  I do not call one better.
- **Why is DermaMNIST macro-F1 so low?** The classes are very imbalanced; rare lesion types are often
  missed, and macro-F1 weighs every class equally.
- **Does the synthetic note help mortality?** The AUROC is the same as event tokens without the note.
  It is only a layout test; I make no clinical claim.
- **Why is surgical phase so poor?** Frames are only 64 by 64 and the encoder is small. Phase
  recognition is hard; it is an extra task, not a promise.
