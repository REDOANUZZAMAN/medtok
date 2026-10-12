# Week 6 progress report: speaker script

Deck: `week06_progress.pptx` (10 slides). Target length: about 9 minutes plus questions.

All numbers on the slides come from `reports/tables/` (`table_00.csv` to `table_03.csv`) and the
figures in `reports/figures/`. The builder script reads them; nothing is typed by hand.

---

## Slide 1: Title (about 15 s)

Good morning, Professor. This is my last weekly report of the six-week plan. This week I summarise
all results, make the work reproducible, and deliver the code, the specification, the report and
the demo.

## Slide 2: Recap (about 40 s)

Last week were the downstream tasks. On BloodMNIST, continuous had the best AUROC and VQ was a bit
lower. On mortality, continuous and event tokens were about the same, and VQ was lower. The
ablations showed that bigger codebooks reconstruct better, and that time granularity trades error
for sequence length.

## Slide 3: Compression and length (about 1 min)

The top row shows compression; the bottom row shows sequence length, for images, time series and
video.

- VQ gives the biggest storage saving for images and time series.
- Event tokens are readable and exact on the bins, but long.
- Video segments are the shortest sequences.
- Continuous saves little for images and ICU data, because it stores fp16 vectors.

The yellow box repeats the rule: continuous positions are fp16 vectors, not vocabulary tokens. The
tables also give the ratio with overhead, which counts delimiters and metadata.

**Why show length:** a Transformer's cost grows with the square of the sequence length, so length is
a real cost, not only storage.

## Slide 4: Reconstruction and codebook use (about 1 min)

Top: reconstruction for each modality. Continuous reconstructs best for images and video. Event
tokens reconstruct ICU data best.

Bottom left: codebook use for the three VQ tokenizers. Almost every code is used in all three, so
there is no codebook collapse. This comes from the three tricks: EMA updates, cosine lookup and
dead-code restarts.

## Slide 5: The two main modalities (about 1 min 30 s)

This is the summary table I will use in the thesis. It has the two fully implemented modalities,
images and ICU time series. For each method: sequence length, reconstruction, compression with
overhead, and the two task metrics.

How to read it:
- For images, continuous wins on reconstruction and on the task, and VQ wins clearly on
  compression.
- For ICU data, event tokens have the best reconstruction and about the same AUROC as continuous,
  but much longer sequences. VQ is the most compact, but has the lowest task scores.

Again, continuous "positions" are fp16 vectors, not tokens. Video and text are reported separately
because they are adapted modalities.

**Why only two modalities here:** they are the ones we agreed to fully implement, each with a
downstream task.

## Slide 6: Reproducibility (about 1 min)

One command, `scripts/run_all.sh`, runs everything in six steps: prepare the data, train and
evaluate the tokenizers, check the interfaces, run the downstream tasks, run the ablations, and
write the report.

What makes it repeatable: fixed seeds and splits; everything learned is fitted on training data
only; every run stores its config, library versions, git commit and logs; and the tests run without
any download.

And numbers are never typed. The report command writes all tables and figures, and these slides
read the same CSV files. The GPU re-run on Modal repeated all downstream runs on the same hardware.
Small differences can remain, because multi-threaded CPU code is not fully deterministic.

**Why this matters:** a supervisor or examiner can check every number back to its source.

## Slide 7: Deliverables (about 45 s)

Six deliverables:
1. The code: the Python package `medical_tokenizer`.
2. The specification: `docs/token_protocol.md`, with id ranges, special tokens, wire format and bit
   accounting.
3. The report: `reports/RESULTS.md`, with tables and figures generated from the experiments.
4. The demo: a browser app that shows data turned into tokens and back, for each modality.
5. The notebook, to explore real token sequences.
6. The documentation: architecture, datasets with licences and splits, and the experiment protocol.

## Slide 8: Limitations (about 1 min)

These are the limitations I will write in the thesis. They are the conditions we agreed on.
1. **Continuous is a baseline.** Its vectors are never counted as tokens, and the ratios include
   fp16 precision, index bits, delimiters and metadata.
2. **Cataract-101 is not Cholec80.** The results are not comparable, and ground-truth phase events
   are annotation encoding only.
3. **The text is synthetic.** It is accepted only on round-trip, field preservation and protocol
   compatibility, and it never replaces the downstream tasks.
4. **Small inputs and a small model.** This was chosen to compare many settings fairly, not to beat
   the state of the art.

At the bottom is the Cataract-101 citation and licence.

## Slide 9: Issues and decisions (about 1 min)

1. **No single winner.** VQ is the most compact, continuous has the best task scores, and event
   tokens reconstruct ICU data best. So the thesis presents the trade-offs, not one winner.
2. **Scope kept.** Images and time series are fully done with two tasks; the other three modalities
   are adapted, as agreed.
3. **Extensions:** better video encoders, point-cloud training, real paired clinical text, and
   Cholec80 if access is granted.
4. **Licences:** only open data; Cataract-101 is non-commercial; no patient-level data is in the
   repository.

## Slide 10: Next steps and questions (about 45 s)

After this week I will write the thesis chapters from the report and the specification, use these
decks as the outline for the defence, and freeze the code at a tagged commit.

My questions:
1. Is "trade-offs, no single winner" the right framing?
2. Are the deliverables complete?
3. Which results go in the main text and which in the appendix?
4. When can I send you the first full draft?

Thank you very much for your guidance over these six weeks.

---

## Possible questions and short answers

- **So which tokenizer should people use?** It depends on the goal. For storage and short sequences:
  VQ. For the best task score: continuous, but it is not a discrete vocabulary. For readable,
  accurate ICU data: event tokens.
- **Can someone else reproduce your results?** Yes: one script, fixed seeds and splits, and every run
  stores its config and git commit. Only open data is downloaded automatically; Cataract-101 needs
  one manual download because of its licence.
- **Why should we trust the numbers on the slides?** They are read by the builder scripts from the
  same CSV files the report writes. Nothing is typed by hand.
- **What is the main contribution?** One fixed vocabulary and wire format for several medical
  modalities, plus a fair, reproducible comparison of three tokenisation strategies on real tasks.
