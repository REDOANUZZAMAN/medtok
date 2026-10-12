# Week 2 progress report: speaker script

Deck: `week02_progress.pptx` (10 slides). Target length: about 9 minutes plus questions.

All numbers on the slides come from `reports/tables/table_00.csv` (images) and `table_03.csv`
(list of downstream runs). If someone asks "where is this number from?", the answer is: the
builder script reads it from that CSV file, and the CSV is written by the report command.

---

## Slide 1: Title (about 20 s)

Good morning, Professor. This is my second weekly report. This week I worked on the first fully
implemented modality: 2D medical images from MedMNIST. I compare two ways to turn an image into a
sequence: continuous projection and vector quantisation, or VQ. At the end I also show the
end-of-Week-2 checkpoint that we agreed on.

## Slide 2: Recap (about 45 s)

On the left is the six-week plan. Week 1 is done, and we are now in Week 2.

Last week I built the unified token protocol: one vocabulary and one sequence format. I also chose
the open datasets and wrote tests.

On the right is your feedback from October 8. We agreed that the thesis is a scoped, reproducible
prototype. Images and time series are fully implemented, and there are at least two downstream
tasks. Text uses synthetic notes, video uses Cataract-101 phase events, and point clouds only get an
interface and tests.

**Why this slide:** it shows that this week's work follows the scope you approved.

## Slide 3: Work done, two image tokenizers (about 1 min)

Both tokenizers use the same autoencoder. It takes a 28 by 28 image, makes a 7 by 7 grid of latent
vectors, and the decoder rebuilds the image from that grid.

- **A, continuous projection:** the 49 latent vectors go straight to the model as fp16 numbers.
  This is the baseline. Important: these vectors are **not** vocabulary tokens. I always report
  them as "continuous vectors", never as tokens.
- **B, VQ:** each latent vector is replaced by the id of the nearest entry in a codebook with 1,024
  entries. So one image becomes 56 real tokens: 49 codes plus 7 row markers.

To keep the codebook alive I use three tricks: EMA updates, cosine lookup and dead-code restarts.
I also train a vanilla VQ-VAE without these tricks, as a control.

**Why the same encoder:** if both methods share the encoder, the only difference is the
discretisation step. So the comparison is fair.

## Slide 4: Reconstruction (about 1 min)

On the left are real test images. For each subset, the first row is the original, the second is
VQ, and the third is continuous.

The table shows PSNR, SSIM and MAE for each subset. Continuous is better on every subset. That is
expected, because continuous keeps all the numbers, and VQ must choose one of 1,024 codes.

But look at the VQ images: the cell shape, the lung shape and the lesion colour are still clear.
MAE is in pixel units from 0 to 255.

**Why PSNR and SSIM:** PSNR measures the pixel error; SSIM measures if the structure looks the
same. Using both is standard for image reconstruction.

## Slide 5: Token counts and compression (about 1 min 15 s)

This table counts positions, vocabulary tokens and continuous vectors.

Continuous has 49 positions, but 0 vocabulary tokens, because its positions carry vectors. VQ has
56 positions, and all 56 are vocabulary tokens. The yellow box says this clearly, because it is one
of the conditions we agreed on.

The compression ratio is raw bits divided by token bits.
- **ratio (fixed):** each VQ index costs log2 of 1,024, so 10 bits. Each continuous value costs
  16 bits, because it is fp16.
- **ratio (+overhead):** this also counts the start, representation and end tokens of each block,
  and the metadata the decoder needs. So it is the more honest number, and it is a bit lower.

VQ compresses a lot. Continuous has a ratio below 1, so it is bigger than the raw image.

**Why count the overhead:** without it, the ratio would look better than reality. The supervisor
asked for a fair comparison.

## Slide 6: Codebook utilisation (about 1 min)

The plot shows how often each code is used, sorted from most to least used, on a log scale.

The grey line is the vanilla VQ-VAE. It drops to almost zero after about 74 codes. This is called
**codebook collapse**: most codes are never used. The orange line is my VQ with the three tricks:
almost all 1,024 codes are used.

In the table: "util" is active codes divided by 1,024, and "perplexity" is the effective number of
codes. Higher perplexity means the codes are used more evenly.

PneumoniaMNIST uses fewer codes. Chest X-rays are grey and very similar to each other, so they need
fewer different codes.

**Why this matters:** unused codes are wasted vocabulary. A collapsed codebook carries less
information per token.

## Slide 7: t-SNE (about 45 s)

This is a 2D map of the continuous latents for BloodMNIST test cells. Each dot is one cell, and the
colour is its true cell type.

The encoder never saw the labels. But platelets, on the right, and erythroblasts form their own
groups. So the latent space already holds medical information, and VQ quantises this space.

Some white-cell types overlap, because they look very similar at 28 by 28 pixels.

**Why only qualitative:** t-SNE is only a picture. The real test is the classification task in
Week 5.

## Slide 8: End-of-Week-2 checkpoint (about 1 min)

We agreed to check three things at the end of Week 2.
1. **Data readable:** yes. MedMNIST and PhysioNet load, clean and split with fixed seeds, and the
   Cataract-101 frame cache is built.
2. **Compute enough:** yes. I re-ran the training on a GPU on Modal. It took about 10 minutes
   instead of about 17 hours on the CPU. All downstream runs were repeated on the same hardware, so
   they are comparable.
3. **Both downstream tasks run:** yes. MedMNIST classification and PhysioNet mortality both train
   end to end on the token files, with 3 seeds each.

So my decision is: we are on schedule, and no reduction is needed. If I fall behind later, the
plan is to use smaller samples on the same fixed splits, and to move advanced video and point-cloud
training to "extensions".

**Why this slide:** it was a condition from your feedback. It shows the project is safe.

## Slide 9: Issues and decisions (about 1 min)

Four issues:
1. **Codebook collapse:** the vanilla VQ-VAE uses few codes. I keep the three tricks, and I keep the
   vanilla model as a control row so the effect is visible.
2. **Low use on PneumoniaMNIST:** only 563 active codes. The X-rays are simple. I still use one
   shared codebook for all three subsets, because one shared vocabulary is the point of the thesis.
3. **VQ loses detail:** this is the price of discrete tokens. Week 5 will show if it hurts the
   classification task.
4. **Continuous ratio below 1:** fp16 vectors use more bits than 8-bit pixels. That is fine: it is a
   baseline, not a compressor.

## Slide 10: Next week and questions (about 45 s)

Next week is time series on PhysioNet 2012. The two hard parts are missing values and irregular
time. I will compare continuous, VQ and event tokens, and report error, sequence length and
compression.

My questions:
1. Are PSNR, SSIM, MAE and codebook use enough for the image part?
2. Do you accept the checkpoint, with no scope reduction?
3. Is one shared codebook for all three subsets OK?

Thank you.

---

## Possible questions and short answers

- **Why is VQ worse than continuous?** VQ must pick one of 1,024 codes for each patch, so some detail
  is lost. Continuous keeps all the numbers. The gain of VQ is real discrete tokens and much better
  compression.
- **Why 1,024 codes?** It matches the image range in the vocabulary. In Week 5 I show a codebook-size
  ablation from 64 to 1,024.
- **Why is continuous not counted as tokens?** Because its positions hold vectors, not ids from the
  vocabulary. Counting them as tokens would make the comparison unfair.
- **What does "+overhead" add?** The start, representation and end tokens of each block, and the
  metadata the decoder needs, like grid shape.
- **Where does the 10 min vs 17 h come from?** From the Modal GPU re-run compared with the earlier
  full CPU run. It is a time measurement, not a result metric.
- **Is 28 by 28 too small?** It keeps training fast, so I can compare many settings in six weeks.
  The protocol does not depend on image size.
