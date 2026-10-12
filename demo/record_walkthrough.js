// Records a captioned walkthrough of the demo app -> demo/media/walkthrough.webm
//   node demo/record_walkthrough.js [http://localhost:8093]
// Needs playwright-core + a chromium build (paths below are the ones installed on the VPS).
const path = require('path');
const fs = require('fs');
const { chromium } = require(process.env.PW_CORE || '/usr/lib/node_modules/openclaw/node_modules/playwright-core');

const BASE = process.argv[2] || 'http://localhost:8093';
const OUT = path.join(__dirname, 'media');
const SAMPLE = process.env.SAMPLE_IMAGE || '/tmp/medtok_cell.png';
const W = 1280, H = 720;

const CAPTION_CSS = `
#wt-cap { position:fixed; left:50%; bottom:22px; transform:translateX(-50%); z-index:9999; max-width:900px; width:calc(100% - 340px);
  margin-left:128px; background:rgba(11,19,36,.94); color:#f1f5f9; font:500 19px/1.45 Inter,system-ui,sans-serif;
  padding:14px 22px 15px; border-radius:12px; box-shadow:0 10px 30px rgba(0,0,0,.3); border-left:5px solid #14b8a6; transition:opacity .3s; }
#wt-cap small { display:block; color:#5eead4; font-size:12.5px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:3px; }
#wt-ring { position:fixed; z-index:9998; border:3px solid #f59e0b; border-radius:10px; pointer-events:none; transition:all .35s;
  box-shadow:0 0 0 4000px rgba(15,23,42,.18); }
#wt-end { position:fixed; inset:0; z-index:10000; background:linear-gradient(135deg,#0b1324,#123b3a); color:#fff; display:flex;
  flex-direction:column; align-items:center; justify-content:center; font-family:Inter,system-ui,sans-serif; text-align:center; }
#wt-end h1 { font-size:46px; margin:0 0 10px; } #wt-end p { font-size:21px; color:#cbd5e1; margin:6px; }`;

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    executablePath: process.env.CHROME || '/root/.cache/ms-playwright/chromium-1148/chrome-linux/chrome', args: ['--no-sandbox'] });
  const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: OUT, size: { width: W, height: H } } });
  const page = await ctx.newPage();
  const wait = (s) => page.waitForTimeout(s * 1000);
  const setup = () => page.evaluate((css) => {
    if (document.getElementById('wt-style')) return;
    const st = document.createElement('style'); st.id = 'wt-style'; st.textContent = css; document.head.appendChild(st);
    const c = document.createElement('div'); c.id = 'wt-cap'; c.style.opacity = 0; document.body.appendChild(c);
  }, CAPTION_CSS);
  const say = async (step, text, secs = 5) => {
    await page.evaluate(([s, t]) => { const c = document.getElementById('wt-cap'); c.innerHTML = `<small>${s}</small>${t}`; c.style.opacity = 1; }, [step, text]);
    await wait(secs);
  };
  const ring = (sel) => page.evaluate((sel) => {
    let r = document.getElementById('wt-ring');
    const el = sel && document.querySelector(sel);
    if (!el) { if (r) r.remove(); return; }
    if (!r) { r = document.createElement('div'); r.id = 'wt-ring'; document.body.appendChild(r); }
    const b = el.getBoundingClientRect();
    Object.assign(r.style, { left: b.left - 6 + 'px', top: b.top - 6 + 'px', width: b.width + 12 + 'px', height: b.height + 12 + 'px' });
  }, sel);
  const scrollTo = async (sel, secs = 1.2) => {
    await page.evaluate((sel) => { const el = document.querySelector(sel); if (el) window.scrollTo({ top: el.getBoundingClientRect().top + scrollY - 100, behavior: 'smooth' }); }, sel);
    await wait(secs);
  };
  const tab = async (t, secs = 3) => { await ring(null); await page.click(`nav button[data-t="${t}"]`); await wait(secs); };

  await page.goto(BASE + '/#start');
  await wait(3); await setup();

  // 1. overview
  await say('Welcome', 'This is <b>MedTok</b>, the demo app of the thesis “Design of a Medical Multimodal-Specific Tokenizer and a Unified Data Protocol”.', 6);
  await say('The idea', 'AI models read <b>tokens</b>: small numbered pieces. MedTok turns pictures, ICU charts, surgery videos and notes into tokens from <b>one shared vocabulary</b>.', 7);
  await ring('.kpis'); await say('Key numbers', 'These numbers are read live from the data: the vocabulary size, the image codebook, and how many test samples the demo can show.', 6);
  await scrollTo('.pipe'); await ring('.pipe');
  await say('How it works', 'Four steps: medical data → tokenizer → one token sequence (saved as JSONL) → one AI model.', 6);
  await scrollTo('.tri'); await ring('.tri');
  await say('Three methods', 'The thesis compares three ways of making tokens: <b>continuous</b> (the baseline), a <b>VQ codebook</b>, and <b>events</b>.', 7);

  // 2. images
  await tab('image', 3.5); await setup();
  await say('Images', 'A 28×28 medical picture from the <b>test set</b>, which the models never saw during training.', 5);
  await ring('.grid7'); await say('Image → tokens', 'The picture is cut into a 7×7 grid. Each piece becomes the number of the closest entry in a codebook of 1,024 learned pieces.', 7);
  await ring('#imgview figure:nth-of-type(3)'); await say('Tokens → image', 'From the numbers alone the picture is rebuilt. Higher PSNR and SSIM mean it is closer to the original.', 6);
  await scrollTo('#imgview table'); await ring('#imgview table');
  await say('Smaller', '“Smaller by” compares the bits of the tokens with the bits of the pixels. The continuous version is often <i>bigger</i>, so it is only a baseline.', 7);
  await scrollTo('.pred'); await ring('.pred');
  await say('Classifier', 'A small Transformer reads <b>only the tokens</b>, never the pixels, and still names the cell type. Research demo, not a diagnosis.', 7);
  await ring(null); await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' })); await wait(1);
  await page.selectOption('#sub', 'pneumoniamnist'); await wait(3); await setup();
  await say('Chest X-ray', 'The same tokenizer and the same codebook also work for chest X-rays…', 4);
  await page.selectOption('#sub', 'dermamnist'); await wait(3); await setup();
  await say('Skin lesions', '…and for skin lesion photos. One tokenizer, one vocabulary.', 4);
  if (fs.existsSync(SAMPLE)) {
    await page.selectOption('#sub', 'bloodmnist'); await wait(3); await setup();
    await ring('label.act'); await say('Your own picture', 'You can also upload your own picture. It is cut to a square, shrunk to 28×28, tokenized, rebuilt and classified.', 5);
    await page.setInputFiles('#up', SAMPLE); await wait(4); await ring(null);
    await scrollTo('.pred', 1.5); await say('Upload result', 'Here is the uploaded picture as tokens, rebuilt, and classified.', 5);
  }

  // 3. ICU
  await tab('icu', 4); await setup();
  await say('ICU charts', 'One ICU patient’s first 48 hours from PhysioNet 2012: heart rate, blood pressure, temperature and more, measured at irregular times.', 7);
  await ring('.charts'); await say('Kept information', 'Black dots are the real measurements. The coloured lines are what comes back after tokenizing and decoding: events, VQ and continuous.', 8);
  await scrollTo('.card:nth-of-type(4)'); await ring('.card:nth-of-type(4) table');
  await say('Token counts', 'Events are long but readable and keep every time. VQ is short but blurs details. This trade-off is a main finding of the thesis.', 7);

  // 4. video
  await tab('video', 7); await setup();
  await say('Surgery video', 'Cataract surgery videos from Cataract-101. Each 8-second clip becomes 32 codebook tokens.', 6);
  await ring('.timeline'); await say('Phases as events', 'The coloured bar shows the 10 surgery phases, stored as event tokens from the doctors’ labels. Only about 30 tokens for a whole surgery.', 7);
  await say('Honest limits', 'These phase tokens come from the true labels: they show labels can be <i>stored</i> as tokens, not that the AI recognises phases. Not comparable with Cholec80.', 7);
  await scrollTo('.frames'); await ring(null);
  await say('Frames', 'Top row: the real frames. Below: frames rebuilt from VQ tokens and from continuous vectors.', 6);

  // 5. text
  await tab('text', 3); await setup();
  await say('Clinical text', 'Real doctor notes are private, so the notes are <b>synthetic</b>, made from templates. They are only used to test the protocol.', 7);
  await scrollTo('.card:nth-of-type(5)'); await ring('.card:nth-of-type(5)');
  await say('Round trip', 'The note is cut into sub-word tokens and decoded again. Exactly the same text and every value come back.', 6);

  // 6. sequence
  await tab('seq', 3); await setup();
  await say('One sequence', 'The main idea: a note (text) and an ICU chart (events) written <b>in one token sequence</b> with the same vocabulary.', 7);
  await ring('.card:nth-of-type(4) .toks'); await say('Markers', 'Special tokens like &lt;TEXT_START&gt; and &lt;TIME_SERIES_START&gt; show where each part begins and ends. Colours show the token type.', 7);
  await scrollTo('pre'); await ring('pre');
  await say('JSONL file', 'Saved as JSONL, the protocol’s file format. Every token keeps its id, type, modality, time and position, so it can be read back exactly.', 7);

  // 7. results
  await tab('results', 3); await setup();
  await say('Results', 'All results are generated by a script from the experiment runs and are never typed by hand.', 6);
  await page.evaluate(() => window.scrollTo({ top: 900, behavior: 'smooth' })); await wait(2);
  await page.evaluate(() => window.scrollTo({ top: 2100, behavior: 'smooth' })); await wait(2);
  await say('Downstream tasks', 'Images are scored with AUROC and macro-F1, ICU survival with AUROC and AUPRC. Between 0 and 1, higher is better.', 7);

  // 8. API
  await tab('api', 3); await setup();
  await say('For developers', 'Other programs can use the tokenizer through a JSON API, or install it as a Python package.', 6);
  if (fs.existsSync(SAMPLE)) {
    await page.setInputFiles('#tfile', SAMPLE); await wait(1);
    await ring('.try'); await page.click('#tgo2'); await wait(3);
    await ring('#tout'); await say('Live call', 'One API call: picture in, class probabilities out, as JSON.', 6);
  }
  await scrollTo('.card:nth-of-type(3)'); await ring(null);
  await say('Endpoints', 'Each endpoint has a ready-to-copy curl example.', 5);

  // end card
  await page.evaluate(() => {
    const e = document.createElement('div'); e.id = 'wt-end';
    e.innerHTML = '<h1>MedTok</h1><p>One token language for medical data</p><p style="font-size:16px;color:#5eead4;margin-top:22px">Images · ICU time-series · Surgical video · Clinical text · One protocol</p>';
    document.body.appendChild(e);
    document.getElementById('wt-cap').style.opacity = 0;
  });
  await wait(4);

  const video = page.video();
  await ctx.close();
  const src = await video.path();
  const dst = path.join(OUT, 'walkthrough.webm');
  fs.renameSync(src, dst);
  await browser.close();
  console.log('saved', dst);
})();
