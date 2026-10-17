# Week 7 progress report: speaker script

Deck: `week07_progress.pptx` (10 slides). Target length: about 9 minutes plus questions.

All numbers on the slides come from `reports/mimic/mimic_metrics.json`, which
`python -m medical_tokenizer.mimic_validate` writes. The builder script reads them; nothing is typed by hand.
The numbers below are the same ones as on the slides. If the run is repeated, read them from the slides.

---

## Slide 1: Title (about 15 s)

Good morning, Professor. This week I did what you asked: I used the MIMIC-IV multimodal data from
Rafi to test and validate my method.

## Slide 2: The task and the data (about 1 min)

Until now I tested on three separate public datasets, and my text was synthetic. MIMIC is
different: each patient has several real kinds of data at the same time.

The subset has 600 patients. Each one has a chest X-ray study, the real radiology report, the
ECGs and the lab tests of the 7 days before the X-ray. 90 patients died within one year.

I put each patient on one timeline. Time zero is seven days before the X-ray, and the X-ray and
report are at day seven. Nothing after the X-ray is used.

The data is credentialed, so it stays on our server. My slides and GitHub show only summary
numbers, never images or report text.

## Slide 3: Method (about 1 min)

Each kind of data goes through one of my existing tokenizers: labs become event tokens, the ECG
becomes window VQ tokens, the X-ray becomes image VQ tokens, and the report becomes sub-word
tokens. Then everything is put into one sequence per patient, in time order.

I only needed two small additions. First, ECG and labs are both "time series" but need different
tokenizers, so I added a router that picks the right one and writes the choice into the sequence.
Second, a fix for long time gaps, on the next slide. The tokenizer designs and the protocol format
did not change, and all 62 tests pass.

## Slide 4: Result 1, the protocol works on real data (about 1 min)

All 600 sequences pass the structural checks. Saving and loading gives exactly the same
sequences. The report text and the lab pattern come back exactly. A sequence has about 1,100
tokens on average. Images take the largest share, and the report is the shortest part.

This is the main claim of my thesis, that one format can hold many medical data types, now
shown on real linked patients.

## Slide 5: Result 2, real data found a bug (about 1 min)

My time-gap tokens were exact only up to 48 hours. PhysioNet stays are 48 hours long, so this
never showed up before. MIMIC labs cover 7 days, and about 7 % of patients have a gap of more than
48 hours between lab tests. For them the lab times came back in the wrong place.

The fix writes a long gap as a chain of exact gaps of up to 48 hours. Short gaps give the same tokens
as before, so my earlier results do not change. Now 100 % of patients come back exactly.
This shows why validation on real data is important.

## Slide 6: Result 3, chest X-rays (about 1 min)

First I used my old MedMNIST image tokenizer without changes. It works, but it uses only about
half of its codebook and the images are blurrier: 25.4 dB.

The same design trained on MIMIC X-rays uses almost all 1,024 codes and is clearly sharper:
29.2 dB at 28 pixels, 31.6 dB at 64 pixels. It still works on the old pneumonia images too.

So my conclusion: the tokenizer design generalises, but the codebook should be trained on
images similar to the target data.

## Slide 7: Result 4, ECG and labs (about 1 min)

ECG is a new signal type. My time-series tokenizer handled it with no code changes and
compresses it about 76 times. But this is my weakest result: the rough shape is kept (lead
correlation about 0.73), and fine detail is lost. Next I will give it more codes per window or a
convolutional encoder.

The lab tests fit my event tokenizer well: the pattern of which tests were done and when is exact at
every time step. A coarser grid saves tokens but merges more results, so I chose 6 hours.

## Slide 8: Result 5, real reports (about 45 s)

All text tokenizers are lossless on real reports. But the old one, trained on my synthetic
notes, needs more than twice as many tokens per report as one trained on real reports. This
supports your Week 4 feedback: synthetic text is fine for round-trip tests, but the vocabulary should
come from real notes.

## Slide 9: Result 6, are the tokens informative? (about 1 min)

To check that the tokens carry useful information, I predicted death within one year with a
simple model. Because there are only 90 deaths, I used repeated cross-validation instead of one
small test set.

Every modality alone is better than chance, and all four together are the best, at AUROC 0.78.
But age and sex alone already give 0.76. So I don't claim a mortality model. I only claim that the
tokens are informative and can be combined.

## Slide 10: Limits, next steps, questions (about 1 min)

Limits: 600 patients from one hospital, small images, and a simple label. The models trained on
MIMIC stay on our server.

Next: improve the ECG tokenizer, try one shared image codebook for MedMNIST and MIMIC, and write
the MIMIC chapter.

My questions: is this validation enough for the thesis? Should I keep the mortality task? And
should the MIMIC results go in the main text or the appendix?

Thank you.

---

## Likely questions and short answers

- **Why only 600 patients?** It's the demo subset I was given. It's enough to validate the method, not to build a clinical model.
- **Why is the ECG weak?** Each 0.2-second window gets only 4 tokens, which is too few for fine detail.
- **Why cross-validation?** A fixed test set would have only 15 deaths, so the score would be very noisy.
- **Is the data safe?** Yes. It is processed on our server, and only summary numbers are shared.
