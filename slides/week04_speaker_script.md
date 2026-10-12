# Week 4 progress report: speaker script

Deck: `week04_progress.pptx` (9 slides). Target length: about 8 minutes plus questions.

All numbers on the slides come from `reports/tables/` (`table_01.csv` for the recap, `table_02.csv`
for video, `table_13.csv` for text and point clouds). The builder script reads them.

---

## Slide 1: Title (about 15 s)

Good morning, Professor. This is my fourth weekly report. This week is about adaptation and
integration: synthetic text, surgical video from Cataract-101, and point clouds, all in the same
protocol.

## Slide 2: Recap (about 40 s)

Last week was ICU time series. Event tokens had the lowest error but long sequences. The "changes
only" variant made them shorter with the same error. VQ was the shortest and most compact, but had
the highest error. For missing values I kept zero fill with a mask.

So images and time series are now the two fully implemented modalities. This week I add the three
"adapted" modalities.

## Slide 3: Scope of this week (about 1 min)

This table is important, because it says what I promise and what I do **not** claim.

- **Text:** synthetic templated notes. I promise lossless round-trip, field preservation and
  protocol compatibility. I do not claim clinical language understanding.
- **Video:** Cataract-101 with 10 phases. I promise VQ, continuous and segment tokens, and phase
  events. I do not compare with Cholec80 results.
- **Point clouds:** synthetic surfaces, interface and tests only. No trained model.

The yellow box is the citation: Cataract-101 is from Schoeffmann et al., MMSys 2018, with a CC
BY-NC 4.0 licence, so research use only.

**Why this slide:** it keeps the claims honest and matches the scope you approved.

## Slide 4: Synthetic text (about 1 min 15 s)

The notes are made from structured fields. They have different wordings, units, missing values,
negation like "not ventilated", and times. Every note carries a tag that says it is synthetic, with
the template version, v2. ICU notes read only the first hour and never the outcome, so there is no
label leakage. The BPE tokenizer is trained on the training split only.

The results: exact round-trip is 1.000, so every note decodes to exactly the same text. Field
preservation is also 1.000: every value, unit, time and negation survives.

The red box is a condition we agreed on: this text is synthetic. It is accepted only on round-trip,
field preservation and protocol compatibility, and it never replaces the downstream tasks.

**Why synthetic text:** there is no open clinical free text in scope. MIMIC needs credentialed
access.

## Slide 5: Cataract-101 video (about 1 min 30 s)

Three representations of the video:
- **A, continuous:** fp16 vectors, not vocabulary tokens.
- **B, VQ:** codes from a codebook; almost all codes are used.
- **C, segment events:** k-means states. A new token comes only when the scene changes, so the
  sequence is very short. But it reconstructs the frames worst.

The last row is different. It encodes the **ground-truth** phase labels as event tokens. It shows
that the protocol can store annotations and decode them exactly. It is **not** phase recognition
from video. The two red boxes say this, and they also say that Cataract-101 results are not
comparable with Cholec80, because it is another surgery with another label set.

Phase recognition from video tokens is a downstream task next week.

**Why Cataract-101:** Cholec80 needs a request to the CAMMA group; Cataract-101 is the closest open
surgical video dataset with phase labels.

## Slide 6: Point clouds (about 45 s)

The pipeline is: normalise the points, group them into voxels, compute features, and map them to
codebook ids. The table shows three voxel resolutions. Finer voxels give a lower Chamfer error, but
need more tokens and compress less. This is the same trade-off as patch size for images.

These are synthetic surfaces with non-learned features. It shows the interface works; it is not a
trained tokenizer. A `depth_to_points` function is ready for SCARED if access is granted.

## Slide 7: Integration (about 1 min)

On the left is one real test sequence: a synthetic note as sub-word tokens, followed by ICU event
tokens, in one sequence.

On the right are the four integration tests. They run the full pipeline for images, time series and
video: train, evaluate, encode, decode and downstream. The last test trains one multi-task model
that reads image, ICU and video tokens together.

The tests use small synthetic files that look exactly like the real datasets, so they need no
download. Every sequence passes the validator, bad input is rejected, and the file formats
round-trip without loss.

**Why this matters:** it proves the "unified" part of the protocol: different modalities really
share one format and one model.

## Slide 8: Issues and decisions (about 1 min)

1. **No open clinical text:** I use versioned synthetic templates, with limited claims.
2. **Cholec80 not accessible:** I use Cataract-101, marked as not comparable, cited, and used
   non-commercially.
3. **Labels vs recognition:** ground-truth events only show storage. Recognition is measured from
   video tokens only.
4. **Point clouds stay small:** synthetic only, kept as an extension.

## Slide 9: Next week and questions (about 45 s)

Next week: the downstream tasks and the ablations. Image classification with AUROC and macro-F1,
ICU mortality with AUROC and AUPRC, one Transformer, fixed splits, the same budget, and 3 seeds. The
ablations are codebook size and time granularity.

My questions:
1. Are the text acceptance criteria enough?
2. Can Cataract-101 phase recognition be an extra task, not a main result?
3. Should the point-cloud interface be in the main text or the appendix?

Thank you.

---

## Possible questions and short answers

- **Does 1.000 round-trip mean the text model is good?** No. It only means the tokenizer is lossless
  on these notes. It says nothing about clinical language.
- **Why is the phase-event ratio (+overhead) so much lower than the fixed ratio?** The label
  sequence is very short, so the fixed block delimiters and metadata are a big part of it.
- **Why are segment events so short?** They only write a token when the scene state changes, which
  is rare in a minute of video.
- **Could the text leak the label?** No. ICU notes read only the first hour and never the outcome;
  image notes never name the class.
- **What is Chamfer distance?** The average distance between the original points and the decoded
  points, here divided by the object size.
