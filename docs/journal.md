# Experiment journal

Raw notes for the write-up, in order, with what was tried, what failed and why. Times are UTC.
Figures referenced here live in `docs/journal/`.

The question: what does a model know about kanji when kanji is all it has seen, and what changes when
it starts from a model that has seen billions of images? Two arms, same data (the 6,413 characters with
KanjiVG strokes and KANJIDIC2 meanings), same two captions per character, same 10% caption dropout,
same evaluation (`kanji_diffusion.evaluate`).

## Arm 1: small latent diffusion model from scratch, laptop CPU (2026-10-21, 01:00–09:00)

- Setup: GlyphVAE 64×64×1 to 16×16×4 (1.6M params), frozen CLIP ViT-B/32, U-Net 17.4M params,
  v-prediction, cosine schedule, AdamW lr 4e-4 (warmup 300, cosine), batch 64, EMA 0.999.
- Autoencoder: 3,000 steps, 43 min, held-out L1 0.0055.
- Denoiser: 10,000 steps in 3 h 17 min, then resumed to 16,000 with a second cosine (~2 h).
- Training sheets: `journal/small/step000500.png` (dotted strokes), `step002000.png` (clean strokes,
  radicals in place: 魚 for fish, 雨 for rain, 金 for computer), `step010000.png`, `step016000.png`.
- Evaluation at 16k, guidance 3: dictionary kanji is the nearest training glyph in 18/48 samples
  (3/48 at 10k); no sample above the near-copy line (0.78, the 99th percentile of real-to-real
  similarity); 4,908 glyphs for 1,227 concepts, none above it either (max 0.73).
- Guidance sweep: fidelity 0% at w=0, 19% at w=1, 38–42% from w=1.5 to 8; variety falls monotonically
  with w. w=1.5 keeps the fidelity of w=3 with more variety.
- Weakness the author noticed in the picker page: characters look *typical*, the same radical in all
  four samples of a prompt (木 ×4 for Euler–Lagrange, 糸 ×4 for Hessian). Diagnosis: the model's only
  visual vocabulary is kanji components; CLIP B/32 gives abstract terms a vague embedding; guidance 3
  collapses variety. That motivated arm 2.

## Arm 2, attempt 1: Stable Diffusion 1.5 fine-tune on a RunPod A100 (2026-10-21, 08:32–10:30)

- Hardware: RunPod Secure Cloud, A100 SXM 80GB, US-MD-1, $1.79/h, pod `4m4t283g931qdo`.
  Details of the rental: the Bits2Bricks piece and `C:\Users\yagom\Desktop\runpod-diario.md`.
- Data: same characters rendered at 512 px (`prepare --size 512`), black on white, as RGB in [-1, 1].
- Setup: `finetune_sd.py`, full U-Net fine-tune from `stable-diffusion-v1-5`, frozen CLIP ViT-L/14 and
  VAE, epsilon prediction (the base model's), AdamW **lr 1e-5**, warmup 500, **cosine to zero**,
  batch 32, 6,000 steps (30 epochs), bf16 autocast, gradient checkpointing, EMA 0.999.
- Speed: 40-step smoke test measured 0.95 s/step (estimate had been 2.5–3.5); the full run averaged
  1.04 s/step, 104 min. GPU 100%, 33.5 GB of 80.
- Smoke test samples at 40 steps (`journal/sd15-attempt1/smoke-step000040.png`): still photographs
  (a bird, mountains, Elon Musk, pizza), already greyscale.
- 1,000 steps (`step001000.png`): kanji-like, thick marker strokes, dense compositions, grey vignetted
  backgrounds, one sample still a photo of a bird on a branch.
- 2,000–6,000 steps: style polished, **semantics never appear**. With the same seed, the sheets from
  3,000 on are almost identical to each other (`step003000.png` … `step006000.png`): the weights were
  barely moving.
- Side by side at 5,000 steps (`compare-small16k-vs-sd5k.png`): the small model gives 木 山 雨 鳥 and 魚
  where they belong; SD answers almost every word with the same kind of overloaded rare-looking
  character. Tree, mountain, rain look nothing like their kanji; no meaningful radicals (門, 食/欠).
- Evaluation (`evaluate`, 12 dictionary words × 4 samples): **fidelity 0/48 at every guidance tried
  (0, 1, 1.5, 2, 3)**, top-10 0/48; within-prompt similarity 0.50–0.51; no near-copies (mean nearest
  similarity 0.53). The sweep was stopped before 5 and 8, and the 1,227-concept picker generation was
  stopped at 128/1,227 once the diagnosis below showed the model ignores the text: neither could tell
  us anything.

### Why attempt 1 failed (analysis written before the full evaluation, to be checked against it)

1. **Learning rate too low for the task (most likely).** 1e-5 decayed on a cosine is the usual value
   for teaching SD a *style*; the run's mean lr was about 5e-6, against 4e-4 for the small model. The
   task here is not a style: the cross-attention has to unlearn that "tree" means a photo of a tree and
   learn that it means the component 木. The cheap part (black ink, dense glyph on white) explains
   almost all of the loss (0.014, dominated by white background) and is learned first; the
   meaning-to-component signal is a small part of the loss and, at this lr, barely moves the weights.
   Consistent with: style fixed early, semantics absent, sheets frozen from 3,000 steps on.
2. **Grey backgrounds: SD 1.x noise schedule.** The schedule never destroys the mean brightness of the
   image, so sampling starts from zero-mean (mid-grey) noise and the model never learned to drive the
   whole image to pure white (Lin et al., 2023, *Common Diffusion Noise Schedules and Sample Steps Are
   Flawed*). Standard fixes: offset noise, or zero terminal SNR with v-prediction (what the small model
   uses, which is why it has no vignette).
3. **Bug: EMA rounded to bf16 at every sample sheet.** `ema.to(dtype)` … `ema.float()` (lines 142–143
   of `finetune_sd.py`) cast the EMA weights to bf16 and back every 1,000 steps: about 0.4% relative
   error on every weight, six times. Adds noise to exactly the weights that are evaluated.
4. **Process: flying blind.** The smoke test measured speed, not learning. During the run only the
   sheets were watched, while a semantic metric existed (fidelity: does *tree* give 木). Measuring it
   every 1,000 steps would have shown at 2,000 that semantics were not rising. No resumable checkpoint
   (optimizer state) was saved either.

### Diagnosis: does the model read the caption? (10:35–10:45)

`kanji_diffusion/diagnose.py`: take 256 training glyphs, noise them at t = 200, 500, 800, and compare the
model's estimate of the clean latent with each glyph's own caption against the caption of a random other
glyph. The gain is how much lower the error is with the right caption.

- First version measured the error on the predicted *noise*. It gave 0.0% for SD fine-tuned **and for
  the original SD 1.5**, which certainly uses text: the measure was blind. In epsilon prediction, at high
  noise the noise is almost the input itself, so predicting it is trivial with or without the caption.
- Fixed version measures on the clean-latent estimate x̂₀, the same quantity for every model:

  | model | t=200 | t=500 | t=800 |
  |---|---|---|---|
  | small model (CPU, v-prediction) | 3.1% | 17.5% | **47.0%** |
  | SD 1.5, original | 0.0% | 0.0% | 0.5% |
  | SD 1.5, fine-tuned (attempt 1) | 0.0% | 0.0% | **0.0%** |

- The fine-tuned SD reconstructs kanji far better than the original (x̂₀ error at t=800: 0.087 against
  0.216) **without using the caption at all**. The fine-tune turned it into an unconditional kanji
  generator: it learned what a kanji looks like and learned that, for it, the caption predicts nothing.
- Weights (`/tmp/w.py` on the pod): relative change from the base model 1.2% overall, 0.7% in the
  cross-attention keys and values. Nothing broken (no NaN, no zeros): the weights barely moved.

**The mechanism, now with evidence.** SD 1.5 is trained to predict the noise. The steps where the
character's components are decided are the noisy ones, and there the epsilon loss is tiny (0.003 at
t=800 against 0.02 at t=200) and almost independent of the caption, so the gradient towards using the
text is almost nil. With a low learning rate the model takes the cheap route: style yes, semantics no.
The small model predicts the velocity, which at high noise is the clean image; there the caption is
worth 47% of the accuracy, and it learned the radicals by step 2,000.

Corrections to the earlier analysis: cause 1 (learning rate) stands but is the smaller half; the
objective (epsilon) is the main reason. Cause 2 (grey backgrounds) and cause 3 (EMA bug) stand.

### Attempt 2: train SD like the small model (launched 10:49)

- `--v-prediction`: velocity target, so the high-noise steps carry a clean-image loss.
- `--zero-snr`: schedule rescaled to zero terminal SNR, trailing timesteps, DDIM sampling with guidance
  rescale 0.7 (Lin et al., 2023). Fixes the grey vignette and fits v-prediction.
- `--lr 2e-5 --lr-xattn 1e-4 --constant`: five times more on the cross-attention, the layers that read
  the caption; no decay to zero.
- Starts from attempt 1's U-Net (`--init-unet runs/sd15/pipeline/unet`), which already has the style.
- EMA bug fixed (sheets sampled from a bf16 copy); resumable `state.pt` every 1,000 steps.
- Every 1,000 steps: caption gain at t=800, fidelity on 12 words × 2 samples, within-prompt similarity.
  Stop rule: if the caption gain is still ~0% at 2,000 steps, stop.
- 30-step smoke test passed after fixing two bugs it caught (a local variable shadowing the sampler
  function; the caption-gain figure being overwritten in the log line). Baseline at step 30: caption gain
  0.1%, fidelity 0/24.
- Repeat of the `pkill -f` trap from the Bits2Bricks piece: a remote `pgrep -f "run_post.sh"` matched its
  own SSH command and killed it (exit 255). Pattern with brackets (`[r]un_post`) or kill by pid.
- 1,000 steps (11:09, `journal/sd15-attempt2/step001000.png`): **backgrounds white**, strokes thin and
  uniform like the training renders (zero terminal SNR worked). **Semantics appear in the sheet**: water
  gives two characters with 氵, tree two with 木 on the left, fish two with 魚; nothing like it in the
  6,000 steps of attempt 1. Metrics lag the sheet: caption gain 0.5% (from 0.1% at step 30; the small
  model is at 47%), fidelity 0/24 (radicals right, not yet the exact dictionary character),
  within-prompt similarity 0.334. Decision: continue to the 2,000-step check.
- 2,000 steps (11:26, `journal/sd15-attempt2/step002000.png`): **caption gain 3.0%**, six times the
  1,000-step figure; the stop rule passes. Sheet: water 氵 in both samples, tree 木 in both, fish 魚 in
  both, bird one with 鳥 at the foot, fire one with 灬. Fidelity still 0/24: the right radical, not yet the
  exact character. Within-prompt similarity 0.371 (variety falling a little). Continue to 6,000.

### Research and hypotheses (11:30–11:45)

A literature and model-card search (sources in the session; key ones here):
- The January 2023 "fake kanji" model that hardmaru shared was trained by **enpitsu** (HF user `epts`):
  10,358 kanji at 512 px, captions = all meanings joined by commas (our long caption); demo samples at
  256 px, DPM++ 25 steps, **CFG 7.5**. Training hyperparameters are in a gated repo (`epts/kanji`).
- **Glyffuser** (yue-here.com/posts/glyffuser, from scratch, T5, Unihan definitions) reports our exact
  symptom: meanings switch on their radical in the usual position, but **the base character rarely
  appears, because far more training samples contain the radical than are the radical alone**; suggests
  frequency weighting; CFG 5–15.
- Text acts at high noise and is nearly ignored at low noise (eDiff-I, arXiv 2211.01324; TGATE,
  arXiv 2404.02747): consistent with the diagnosis above.
- Interventions with published evidence: logit-normal timestep sampling (SD3, arXiv 2403.03206);
  Min-SNR-γ weighting, γ=5, ~3.4× faster convergence (arXiv 2303.09556); cross-attention K/V only is
  enough for new concepts (Custom Diffusion, arXiv 2212.04488); training the text encoder at 1e-6 to
  2e-6 helps (HF DreamBooth study); epoch budgets of ~100 (Glyffuser) against our ~60.

Hypothesis test H2, "guidance too low" (`journal/sd15-attempt2/cfg_sweep_step2000.png`, rows w = 3, 7.5,
12 with rescale 0.7, and 7.5 without; columns water, tree, mountain, fish, rain, heart, woman, gold):
**rejected**. Higher guidance cleans the strokes but does not fix the character: water (氵), tree (木) and
fish (魚) are right at every scale, mountain (no 山), heart (no 心), woman (no 女) and gold (no 金) wrong at
every scale. The pattern is Glyffuser's: right where the meaning is a frequent radical, wrong for base
characters that are rare as recognisable components.

Proposed attempt 3 (after attempt 2 ends): logit-normal timesteps shifted to high noise + Min-SNR γ=5;
oversample simple (few-stroke) characters; high lr on attn2 to_k/to_v; last CLIP layers unfrozen at 2e-6;
~12,000 more steps with the usual stop rule. Estimated 3.5 h, ~$6.5.
- 3,000 steps (11:44, `journal/sd15-attempt2/step003000.png`): caption gain 4.8% (0.5, 3.0, 4.8:
  rising, decelerating). New hits: fire gives 火 in one sample, bird has 鳥 in both. Heart gives 中
  ("middle, inside"), a neighbouring meaning of *heart*, not 心. Fidelity 0/24; within-prompt similarity
  0.394 (variety still falling).

### Attempt 3 (approved 11:48, armed to start when attempt 2 ends)

Starts from attempt 2's final U-Net. Changes, each against the evidence above:
- `--t-logit-mean 0.5`: timesteps drawn logit-normally, median about t=620, so the high-noise steps where
  the components are decided are visited more (SD3's logit-normal sampling, shifted).
- Min-SNR weighting was planned and **dropped**: for v-prediction its weight is min(SNR, γ)/(SNR + 1),
  which tends to zero at the highest noise, exactly the steps the diagnosis says carry the caption. The
  plain v loss (the small model's) is kept.
- `--simple-boost 3 --simple-strokes 6`: characters with at most 6 strokes (base characters such as 山,
  心, 女) are drawn four times as often, against Glyffuser's radical-frequency bias.
- `--train-text-layers 4 --lr-text 2e-6`: the last four CLIP text layers and the final layer norm are
  unfrozen, so meanings can separate before the cross-attention.
- Same as attempt 2: v-prediction, zero terminal SNR, lr 1e-4 on attn2 and 2e-5 elsewhere, constant.
- 12,000 steps, checks every 1,000; a 30-step smoke test runs first in the chain and stops it if it fails.

### Red-team review and revision (12:00–12:25): attempt 3 disarmed

An independent review of the journal and code, plus two checks, changed the plan.

- **Caption gain is not comparable across the two models.** Same t, different noise: ᾱ at t=800 is
  0.093 for the small model's cosine schedule and 0.018 for SD with zero terminal SNR (the small model's
  t=800 corresponds to SD's t≈641). SD latents also carry texture no caption can predict. The 47% was
  never an attainable target for SD; attempt 1's conclusion stands (0.0% at every level).
- **"Radical yes, base no" is not radical frequency** (Glyffuser's explanation). Co-occurrence check on
  KanjiVG components (CPU): the predictor is the number of training captions linking the word to its
  component. Successes: tree 67 captions, water 46, fish 31, bird 17. Failures: heart 11, sun 11,
  woman 9, rain 8, moon 6, gold 4, even where the association is near perfect (woman: 89% of its captions
  belong to characters with 女). Exception: mountain (27, fails). Few examples per word–component pair,
  which SD needs more of than the small model because it must unlearn its photographic prior first.
- **Fidelity (exact top-1 by pixel IoU after 512 to 64) is unfair to SD and measures the opposite of the
  goal**: the author wants *more original* glyphs, and reproducing the dictionary character rewards the
  typical. 24 samples from one seed cannot separate 0/24 from 3/24.
- **Attempt 3 as armed was a bad experiment** and was disarmed at 12:19, before it started: four changes
  at once; the simple-character boost would have inflated fidelity by memorisation (8 of the 12 test
  words have ≤6 strokes); weight decay 1e-2 at lr 1e-4 would erode the cross-attention (~1.2% over 12k
  steps, more than attempt 1 moved it in total); the trainable text encoder had no EMA; `evaluate.py`
  samples without guidance rescale while training logs used it.
- Min-SNR: dropping it was right; the reasoning was incomplete (plain v loss already overweights low
  noise; a cap on the high-SNR side only would keep full weight at high noise). The logit-normal draw over
  the DDPM index is a heuristic, not SD3's distribution.
- Risks not watched: variety falling (within-prompt similarity 0.334 to 0.394); no copy check during
  training.

Revised plan, by value per GPU hour:
1. Evaluate attempt 2 on what the project asks: invented words, copies, variety, and a blind comparison
   against the small model on the 1,227-concept picker.
2. Small model with SD's CLIP ViT-L/14 and a larger U-Net (the planned third arm): minutes on the A100,
   separates "better text encoder" from "pretrained on photographs".
3. Hybrid: small-model samples re-rendered by SD attempt 2 through img2img at strength 0.4–0.6.
4. Only then, if SD earns it, a clean attempt 3 with one change (more epochs, or LoRA at 256 px with
   synonym caption augmentation for words with few examples), weight decay 0 on attn2, eval parity,
   a rank-based caption metric, a copy check, and numeric stop rules.

## Decision log from 12:30 (the author delegated the route: "go, justify every change")

Goal, fixed with the author's earlier words: invented characters that are original and meaningful,
judged blind by the author. Every decision below records what, why, what it costs and why this
justification and not another.

**D1. Attempt 2 is closed as a result, not continued** (12:32). Final: caption gain 7.2% (4.8, 6.0, 4.5,
7.2 at 3k–6k: noisy plateau), within-prompt similarity 0.442 (0.334 at 1k: variety collapsing). Why:
further full fine-tuning buys little meaning and costs the variety that is the only reason to use SD.
Cost: none. Why this justification: the two numbers that matter for the goal (meaning, originality) both
argue against continuing; fidelity is not used because it rewards copying the dictionary.

**D2. New primary automatic metric: semantic hit@20 with a chance control** (`evaluate.py`). For each
sample, the share of its 20 nearest training glyphs whose caption contains the prompt word; chance is the
same count against the other prompts' words. Why: it rewards the right family of components without
rewarding a copy, which matches the goal; exact top-1 stays as a secondary figure. Also added: median
target rank, a gibberish prompt set (made-up words: a model that reads meaning should not draw them as
convincingly), 8 samples per word instead of 4 (24 samples could not separate 0 from 3). Small model,
guidance 1.5: hit@20 0.065 against chance 0.005 (13×), exact 39/96, median rank 10, within-prompt 0.443.
A first run gave 0.000 for everything: the `\b` of the regex was destroyed by shell escaping; fixed with
the edit tool, the known positive (the small model) is what exposed it.

**D3. Evaluation parity**: `sample.py` now passes guidance rescale 0.7 to zero-terminal-SNR pipelines, as
their training sheets did. Why: the reviewer found the final evaluation would not match the logged curve.

**D4. Arm 3: the small model with SD's text encoder (CLIP ViT-L/14), nothing else changed** (12:41).
Why: it isolates the question "is the small model's typicality due to its weak text encoder?" from "is SD
better because it saw photographs?", which every SD run confounded. One deviation from arm 1, recorded:
16,000 steps on one cosine instead of 10,000 + 6,000 on two. Cost: ~25 min on the A100 (11 steps/s,
against 5 h on the laptop CPU). Why this arm before the component idea: it is the baseline the component
idea must beat, and it is the cheapest unanswered question.

**D5. Arm 4: arm 3 plus a third caption naming the parts** (12:43), e.g. 明 "bright, made of day (left) and
month (right)", from KanjiVG's first-level decomposition (6,029 of 6,413 characters), part names from
KANJIDIC first meanings, variant forms named by their original (氵 as water), KanjiVG's Japanese position
terms translated (tare: top-left, kamae: around, nyō: bottom-left), dictionary notes stripped ("radical",
"variant", "(no. 58)"). One change against arm 3. Why: the measured bottleneck is few examples per
word–component pair (gold: 4 captions), while each part name has tens to hundreds of examples (tree 377,
water 356, eat 41, gate 48, shellfish 81); a model that learns parts can be asked for a composition.
The caption keeps the meaning in front so the same model still answers plain words.

**D6. The designer: component designs for the 12 invented words** (`kanji_diffusion/designs.json`), two
per word, written by Claude as an inventor would (a semantic part, often a second part that narrows it),
using only part names present in the training captions. Why two: if two different designs give the
same glyph, the model ignores the design. Why in-vocabulary: a design the model never saw a word for
(food instead of eat) would test vocabulary, not composition.

**D7. Design metric** (`kanji_diffusion/design.py`): for each design, the share of a sample's 20 nearest
training glyphs whose component caption contains every part of that design, against the same share for
the other design of the same word. The other design is the control. Arm 3 (no component captions) is the
control for arm 4. Chained on the pod to run when both arms finish.

**D8. Correction: by the goal's own metrics, attempt 2 beats the small model** (12:51). Same evaluation,
8 samples per word:

| | small model | SD attempt 2 |
|---|---|---|
| semantic hit@20 (chance), guidance 1.5 | 0.065 (0.005), 13× | 0.076 (0.003), 25× |
| semantic hit@20 (chance), guidance 3 | | 0.080 (0.003) |
| within-prompt similarity, dictionary words (lower = more varied) | 0.443 | 0.405 |
| within-prompt similarity, invented words | 0.417 | 0.348 |
| near-copies | 0 | 0 |
| exact dictionary character | 39/96 | 0/96 (median rank 6,040) |
| caption gain at t=800 (its own schedule) | | 7.9% (attempt 1: 0.0%) |

SD attempt 2 draws components of the right family more often over chance and is more varied; it never
reproduces the dictionary character, which is the one thing the goal does not ask for. Calling attempt 2
a failure (D1, and the 12:20 analysis) leaned on exact fidelity, the metric already judged misaligned.
D1 stands as "stop fine-tuning it" (the variety trend), not as "it failed". The decision now belongs to
the blind comparison: attempt 2's 1,227-concept picker generation launched (guidance 3, as the small
model's picker), ~45 min, in parallel with arms 3 and 4.

**D9. Arm 3 result: a better text encoder makes the small model more faithful and more typical** (13:12).
Guidance 1.5 (guidance 3 in brackets): semantic hit@20 0.080 (0.085) against chance 0.005; exact
dictionary character 45/96 (52/96), median target rank 2 (1); within-prompt similarity 0.529 (0.601) on
dictionary words and 0.463 (0.539) on invented words, both worse than the small model's 0.443 / 0.417;
7/96 (12/96) samples of dictionary words above the near-copy line, which for these words are the
dictionary character itself. Reading: stronger conditioning narrows the outputs. The small model's
typicality was not caused by a weak text encoder; SD attempt 2 remains the most varied of the three
(0.405 / 0.348), which points at its visual pretraining as the source of variety. One-change arm, so the
attribution holds within this setup (one seed of training).
Practical: attempt 2's picker generation is slow while sharing the GPU (208/1,227 after 20 min).

**D10. Arm 4 composes from the design; the first design metric could not see it** (13:23). Arm 4's
evaluation on plain words barely moves against arm 3 (hit@20 0.073/0.078 against 0.080/0.085, exact
55–57/96, within-prompt 0.486/0.573 on dictionary words, 0.428/0.512 on invented words): the third caption
does not hurt plain prompts. The design test printed 3 of 24 for arm 4 and 1 of 24 for arm 3, but the
sheets (`journal/arm4-components/designs_arm4_g3.png`, control `designs_arm3_control_g3.png`) say the
opposite of a failure: asked for "pizza, made of eat (left) and pelt (right)", arm 4 draws 食 on the left
and 皮 on the right in all six samples; 門 around 糸 for "gate (around) and thread", 金 beside 鳥 for
"gold (left) and bird (right)", 雨 over 灬 for "rain (top) and fire (bottom)", 亻 for person, 扌 for hand,
車 for car. Arm 3, which never saw "made of" captions, draws unrelated glyphs for the same prompts.
Why the metric read zero: it required one training neighbour to contain every part of the design, and
for an invented combination such a neighbour almost never exists (that is what makes it invented).
Replaced with a per-part score (for each part, the share of the 20 nearest glyphs containing it, averaged),
plus a near-copy count and within-design variety, rerun at guidance 1.5 and 3.
Two weaknesses visible already, recorded before the numbers: (1) the six samples of one design are
nearly the same glyph (low variety, likely made worse by guidance 3); (2) several outputs may be real
characters (金+鳥 could be an existing rare kanji); the copy count will say. Some parts are not drawn:
"dish (top)" came out as 艹, "say (right)" as a generic right side. Complexity so far: one flag in
prepare, no model change; the decisive evidence was looking at the picture, which the logged metrics missed.

**D11. The per-part score confirms composition, with no copies; variety is the open weakness** (13:40).
12 words × 2 designs × 6 samples (216 glyphs per run):

| | part hit, own design | part hit, other design (control) | designs won | near-copies | within-design similarity |
|---|---|---|---|---|---|
| arm 4, guidance 1.5 | 0.349 | 0.158 | 17/24 | 0 | 0.597 |
| arm 4, guidance 3 | 0.349 | 0.160 | 16/24 | 0 | 0.671 |
| arm 3 (control), guidance 1.5 | 0.101 | 0.049 | 13/24 | 0 | 0.532 |
| arm 3 (control), guidance 3 | 0.092 | 0.042 | 12/24 | 0 | 0.624 |

Arm 4's own-design score is 3.5× the control's at the same guidance, and 2.2× its score for the other
design of the same word: the design, not the word, decides the glyph. Zero near-copies, so the designed
characters are new. Weakness: one design gives nearly the same glyph every time (0.60–0.67, against
0.43–0.51 for plain words). Decision: guidance 1.5 for arm 4 from now on (same composition, more variety),
and variety is delegated to the designer, not to sampling noise: different designs for the same word
give different, still meaningful glyphs, which is how a human inventor varies too. Why this and not
lowering guidance further: guidance trades meaning for noise, a design change trades one meaning for
another meaning. Cost: zero training.

**D12. The blind test becomes a three-way round robin** (13:41): small model (arm 1, plain word), SD
attempt 2 (plain word) and arm 4 with designs written by an LLM (one design per concept, four samples,
guidance 1.5). 300 concepts drawn at random from the 1,227, each shown as one pairing of two of the
three systems, sides shuffled; ~100 judgements per pairing, scored as win rates and a Bradley-Terry
ranking. Why three-way in one test: the author's time is the scarce resource, and one session answers
both questions (does pretraining help, does design help). Why 300: about 15 minutes of judging.

**D13. The designer: an LLM subagent wrote one design for each of the 300 blind-test concepts**
(`kanji_diffusion/designs300.json`), from 148 part names that each appear in 15 or more training
characters, positions restricted to the seven the captions use, steering away from compositions of
known kanji. Arm 4 drew four glyphs per design at guidance 1.5 (`journal/arm4-components/llm_designs_preview.png`).
Read by eye: most rows carry both parts (言+火 for Landauer's principle, 金+頁 for Mimiron's head,
車+牛 for the Lamborghini, 耳 for the iPod Shuffle, 目+黒 for Oakley); some keep only the radical
(Widowmaker: 女 instead of "husband"; Jensen Huang: the leather part is lost). Why an LLM and not a
lookup table: choosing an apt second part (a pun, an allegory) is the creative act the diffusion model
cannot do, and it is exactly what the author found missing ("typical" outputs).

**D14. Pod stopped while the author judges** (13:55 UTC). The blind test is live (300 concepts, three
systems, one pairing per concept; votes stored with the hidden assignment). Arms 3 and 4 copied to
`runs/smallL`, `runs/smallLC` and run on the laptop CPU; evaluations copied to `outputs/pod/`. Pod
4m4t283g931qdo stopped (status EXITED, read back): compute billing ends, the 50 GB volume at
/workspace survives at the stopped-disk rate, the container disk is wiped. Uptime 19,326 s at $1.79/h,
about $9.6 of compute. Why stop now: every next GPU step (more designs per concept, a larger arm 4,
hanzi data) depends on the verdict, and an idle A100 costs more per hour than any of those steps.
Risk accepted: on restart the GPU on that host may be taken; the volume can then be reached with a
zero-GPU start, and every result needed is already on the laptop.

**D15. The blind test was stopped after 7 judgements; the author found all three systems bad** ("me
parecen todos una mierda"). Votes read from the artifact's db: SD attempt 2 won 3 of its 4 pairings, the
small model 4 of 5, the design arm lost all 4 (to the small model three times, to SD once). Seven votes
decide nothing statistically, but two facts survive: no system clears the author's bar, and the arm every
metric of mine preferred (D10, D11) is the one he picked against every time. Third time a metric of mine
and the author's eye disagree (D8, D10). Reading: the metrics reward legibility and faithfulness to a
recipe, which is what makes a glyph look like a dictionary entry; his preference so far goes to the
stranger, more varied outputs. Decision: no further GPU work until the failure is named by the author
(what *bad* is: illegible, too typical, ugly strokes, low resolution), because optimising a fourth proxy
without it repeats the same mistake. Complexity of the decision: none; cost of skipping it: another
A100 session spent on the wrong target.

**D16. The author's diagnosis, and arm 5: reproduce enpitsu instead of inventing a fifth idea.** Asked
what was wrong, he marked three things: the glyphs look like dictionary kanji, the link to the word
cannot be seen, and the four samples are nearly identical. His reference is the fake kanji hardmaru
shared in January 2023 (*Skyscraper, Pikachu, Elon Musk, Deep Learning, YouTube, Gundam*), which he
wants with beautiful calligraphy and a pictographic quality one intuits before one can explain it.
Those images are enpitsu's: Stable Diffusion taught *漢字とその意味を1万字*, ten thousand kanji with their
meanings, and shown one chosen image per concept (sources:
https://x.com/enpitsu/status/1610587513059684353, https://x.com/hardmaru/status/1611237067589095425;
no training settings were published).
Three differences from my SD arm follow, each cheap to close:
- Vocabulary: 10,384 characters (every KANJIDIC2 entry with an English meaning, which is enpitsu's
  ten thousand) instead of the 6,413 KanjiVG also draws. Why it matters for the author's complaint: a
  model that knows 60% more meanings has more parts to borrow from when a word is new.
- Style: drawn from fonts instead of KanjiVG's uniform hairline strokes, Yuji Syuku (a brush face) for
  the 6,627 it covers and Shippori Mincho ExtraBold for the other 3,757, chosen over the regular mincho
  so the two weights match (`kanji_diffusion/prepare_font.py`). Two changes in one run break the
  one-variable rule on purpose: the goal of this arm is to reach the reference, not to attribute it;
  an ablation follows only if it works.
- Selection: enpitsu showed the best of many. Our blind test showed four random samples, which is a
  fair test of a sampler and an unfair one of the idea. The next judgement will show many candidates
  per concept and ask which ones the author would keep; the keep rate per system is the metric, and the
  kept ones are the gallery.
Training keeps attempt 2's recipe (v-prediction, zero terminal SNR, lr 2e-5 with 1e-4 on
cross-attention, constant), from base SD 1.5 rather than attempt 2's U-Net, since that one learned
the hairline style that this arm replaces; 8,000 steps instead of 6,000 for the same reason.
Infrastructure: pod 4m4t283g931qdo cannot start (its host's GPU is taken, the risk accepted in D14);
new pod cumyascnzdr67p, H100 SXM at $3.99/h (no A100 with CUDA 12.8+ in stock), about twice as fast,
so the run should cost about the same, near $6.
