# kanji-diffusion

A small latent diffusion model that invents kanji for English words. It learns from
the 6,413 characters that have both stroke data in [KanjiVG](https://kanjivg.tagaini.net)
and English meanings in [KANJIDIC2](https://www.edrdg.org/wiki/index.php/KANJIDIC_Project),
and then draws characters that never existed for prompts like *pizza*, *internet* or
*Elon Musk*. The idea comes from [@hardmaru](https://twitter.com/hardmaru)'s 2022 experiment,
which fine-tuned Stable Diffusion on the same data:

![@hardmaru's fake kanji](docs/hardmaru_matrix.PNG)

This version follows the same architecture as Stable Diffusion at a size a laptop CPU can
train overnight:

| Stage | Stable Diffusion 1.x | kanji-diffusion |
|---|---|---|
| Image | 512×512×3 | 64×64×1 (ink) |
| Autoencoder latent | 64×64×4 | 16×16×4 (`GlyphVAE`, 1.6M params) |
| Text encoder (frozen) | CLIP ViT-L/14 | CLIP ViT-B/32 |
| Denoiser | UNet, 860M params | UNet with cross-attention, 17M params |
| Objective | ε-prediction | v-prediction, cosine schedule |
| Sampler | PNDM / DDIM | DPM-Solver++, classifier-free guidance |

Everything runs on CUDA when it is available and falls back to the CPU otherwise.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; source .venv/bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
pip install -e .
```

## Run

```bash
# 1. Download KanjiVG and KANJIDIC2, render every character at 64x64 and embed the
#    meanings with CLIP (~5 minutes). Writes data/kanji64.pt and a preview sheet.
python -m kanji_diffusion.prepare

# 2. Train the autoencoder (~1 hour on an 8-core CPU, a few minutes on a GPU).
python -m kanji_diffusion.autoencoder --steps 3000 --batch 32

# 3. Train the denoiser. It saves the EMA weights and a sample sheet every 500 steps
#    under runs/small/, and continues from its last checkpoint if restarted.
python -m kanji_diffusion.train --steps 10000

# 4. Generate.
python -m kanji_diffusion.sample runs/small "pizza" "internet" "Elon Musk" -n 4 --out outputs/sheet.png
```

`--guidance` (default 3) trades variety for fidelity to the prompt; `--seed` changes the
draw. On a GPU, `--preset base` trains a wider denoiser.

## Layout

```
kanji_diffusion/
  prepare.py      data: KanjiVG strokes rendered without cairo, KANJIDIC2 meanings, CLIP embeddings
  text.py         the frozen CLIP text encoder
  autoencoder.py  GlyphVAE and its training loop
  model.py        the UNet and the noise schedule
  train.py        denoiser training with EMA and caption dropout
  sample.py       generation and labelled sample sheets
```

## Data licences

KanjiVG is © Ulrich Apel, CC BY-SA 3.0. KANJIDIC2 is © the Electronic Dictionary Research
and Development Group, CC BY-SA 4.0. Neither is redistributed here: `prepare` downloads them.
