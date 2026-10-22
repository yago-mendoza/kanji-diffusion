"""Sample one pipeline over a grid of resolution and guidance, one sheet per setting (rows: words, columns: seeds).

    python outputs/keep/sweep.py runs/arm5/pipeline outputs/keep/probe.txt outputs/sweep --sizes 256 --guidance 3 7.5

Built to test two levers read off the enpitsu demo (fixed at 256 px, guidance 7.5) without training anything:
a smaller latent leaves room for fewer strokes, and a higher guidance makes the caption weigh more.
"""
import argparse, time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("pipe", type=Path)
ap.add_argument("words", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--sizes", type=int, nargs="+", default=[256])
ap.add_argument("--guidance", type=float, nargs="+", default=[3.0, 7.5])
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--steps", type=int, default=25)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(args.pipe, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
pipe.set_progress_bar_config(disable=True)
extra = {"guidance_rescale": 0.7} if pipe.scheduler.config.get("rescale_betas_zero_snr") else {}
words = [w.strip() for w in args.words.read_text(encoding="utf-8").splitlines() if w.strip()]
args.out.mkdir(parents=True, exist_ok=True)
T = 128
for size in args.sizes:
    for g in args.guidance:
        t0 = time.time()
        sheet = Image.new("L", (110 + args.n * T, len(words) * T), 255)
        d = ImageDraw.Draw(sheet)
        for r, w in enumerate(words):
            gen = torch.Generator("cpu").manual_seed(100 + r)
            ims = pipe([w] * args.n, height=size, width=size, num_inference_steps=args.steps, guidance_scale=g,
                       generator=gen, **extra).images
            d.text((4, r * T + T // 2 - 6), w[:16], fill=0)
            for k, im in enumerate(ims):
                sheet.paste(im.convert("L").resize((T, T), Image.LANCZOS), (110 + k * T, r * T))
        path = args.out / f"{args.pipe.parent.name}_s{size}_g{g:g}.png"
        sheet.save(path)
        print(f"{path.name}  {time.time() - t0:.0f}s", flush=True)
