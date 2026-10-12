# Week 1 progress report: speaker script

Deck: `week01_progress.pptx` (8 slides). Target length: about 7 minutes plus questions.

---

## Slide 1: Title (about 20 s)

Good morning, Professor. This is my first weekly progress report for my thesis, "Design of a
Medical Multimodal-Specific Tokenizer and a Unified Data Protocol". This week I worked on three
things: defining the problem, choosing the datasets, and designing the unified token protocol.
There are no experimental results yet. Those start next week.

## Slide 2: Problem and goal (about 1 min)

First, the problem. Medical data comes in many different forms: 2D images, ICU vital signs,
surgical video, clinical text and 3D scans. Usually each type gets its own model and its own input
format. So a single Transformer cannot read a mix of them, because they do not share a format.

My goal has three parts. First, one discrete token vocabulary that covers all the medical
modalities. Second, one sequence format, which I call the wire format, that also carries time and
position. Third, a fair comparison of different tokenisation strategies on real medical tasks.

I am comparing three strategies.
- **A, continuous projection:** the encoder's output vectors go straight into the model, with no
  discretisation. This is the baseline.
- **B, vector quantisation:** each encoder vector is replaced by the nearest entry in a learned
  codebook, so it becomes a discrete token.
- **C, event-based tokens:** measurements and annotations become symbolic tokens. For example,
  heart rate becomes a variable token plus a value-bin token.

## Slide 3: Six-week plan (about 40 s)

This is the six-week plan. Each week covers one part of the work.
- Week 1, this week: the problem, the datasets and the protocol.
- Week 2: the image tokenizer on MedMNIST, comparing VQ against continuous.
- Week 3: time-series tokenizers on PhysioNet 2012, comparing all three strategies.
- Week 4: surgical-video tokenizers on Cataract-101, plus simple interfaces for text and point clouds.
- Week 5: the downstream comparison, with one Transformer across all strategies and tasks.
- Week 6: ablation studies, final results and the thesis draft.

## Slide 4: Datasets (about 1 min 15 s)

I only used open-access datasets.
- For **2D images**: MedMNIST version 2, the Blood, Pneumonia and Derma subsets, at 28 by 28 pixels.
  That is 23,674 training images, with 8-, 2- and 7-class classification tasks.
- For **ICU time series**: PhysioNet 2012, set A. It has 37 variables plus 4 static ones, and 4,000
  ICU stays split into 2,799 for training, 599 for validation and 602 for testing. The task is
  predicting in-hospital mortality.
- For **surgical video**: Cataract-101. That is 101 surgeries, split 71, 15 and 15, sampled at
  2 frames per second and resized to 64 by 64. The task is recognising 10 surgical phases.
- **Clinical text** and **3D point clouds** are interface only: templated notes and synthetic
  surfaces, to show that the protocol can carry those modalities.

On the right, a note about Cholec80. I planned to use it, but its archive returns HTTP 403 because
it needs a request to the CAMMA group. Cataract-101 is the closest open surgical-video dataset with
phase labels. The Cholec80 loaders are already written, so I can switch if access is granted.

Data-handling rules: splits are done at the patient or video level with a fixed seed, so there is no
leakage. Normalisation and value bins are fitted on the training split only. And restricted datasets
like MIMIC and SCARED are never downloaded by the code.

## Slide 5: Unified token protocol (about 1 min 15 s)

This is the core design of the week: one vocabulary of 13,824 token ids.

The bar at the top shows the layout. The first part is six structural ranges of 256 ids each:
- special tokens, such as begin, end, and the modality start and end markers;
- time-gap tokens, for gaps from seconds up to 30 days;
- time-series variable and value-bin tokens;
- expert-annotated events, such as surgical phases;
- unsupervised segment, or clip-state, tokens.

After that come 8,192 sub-word pieces for text, and then four VQ codebooks of 1,024 entries each,
for image, time series, video and point cloud.

Three design choices:
1. The ranges are fixed and do not overlap. They are never learned, so any tokenizer can be swapped
   in without changing the rest of the system.
2. Events and segments are kept apart, so an annotated phase is never confused with a cluster id.
3. The exact time is kept in a side channel. The time-gap tokens are only a coarse view of it.

## Slide 6: Sequence format and real example (about 1 min)

This is the wire format. A sequence starts with BOS and ends with EOS. In between, each modality is
a block with a start marker, a token saying which representation is used, the content, and an end
marker. Between blocks there is a time-gap token. Images use row markers to keep the 2D layout.

Below is a real output from my code: a short note plus 12 ICU measurements, 40 tokens in total.
First the text block with sub-word tokens, then a time gap of zero, then the time-series block. You
can read each measurement as a pair, for example "VAR HR, VBIN 4", meaning heart rate in value bin 4.

Each token also carries side channels: the modality id, the exact time in minutes, a 3D position,
and an optional confidence. Data is stored in .npz shards for training and in human-readable JSONL
with one token per line. Both are lossless, and a validator checks every sequence.

## Slide 7: Work completed this week (about 45 s)

To summarise what is done this week:
- **Problem scope:** I defined the 5 modalities, the 3 strategies and the evaluation metrics, which
  are compression, reconstruction, codebook use, sequence length and downstream score.
- **Data pipeline:** download, cleaning and splitting for MedMNIST and PhysioNet 2012, and the
  Cataract-101 frame cache at 2 fps, 64 by 64.
- **Protocol specification, version 0.2:** vocabulary ranges, special tokens, time-gap buckets, the
  wire format, side channels and the storage formats.
- **Reference code:** a Python package called medical_tokenizer, with the protocol, serializer,
  validator and JSONL export.
- **Tests:** unit and end-to-end tests on synthetic data, so the format can be checked without
  downloading anything.

## Slide 8: Next week and questions (about 1 min)

Next week is the image tokenizer. I will train a patch autoencoder on the three MedMNIST subsets.
Version B adds vector quantisation with a 1,024-entry codebook. Version A uses the same encoder
with continuous outputs as the baseline. I will measure reconstruction error, codebook utilisation
and tokens per image, and I will show reconstructed images next to the originals.

I have three questions for you, Professor:
1. Is Cataract-101 acceptable instead of Cholec80, or should I file the CAMMA data request?
2. Is templated text acceptable for the text modality, or do you expect MIMIC credentialed access?
3. Is the six-week plan and its weekly scope acceptable?

Thank you. I am happy to take any questions.

---

## Possible questions and short answers

- **Why 1,024 codebook entries?** It is the starting size. Codebook size is one of the Week 6
  ablations, so I will test other sizes then.
- **Why 28 by 28 images? That is very small.** It keeps training fast enough to compare many
  strategies in six weeks. The protocol itself does not depend on image size.
- **Why fixed id ranges instead of a learned vocabulary?** So that any tokenizer can be swapped in
  without retraining the others, and every id can be traced back to its modality.
- **How do you stop data leakage?** Splits are at the patient or video level with a fixed seed, and
  normalisation and value bins are fitted on the training split only.
- **Do you have any results yet?** Not yet. This week was the foundation. The first results, image
  reconstruction and codebook use, come next week.
