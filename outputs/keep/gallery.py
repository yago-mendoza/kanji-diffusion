"""Many samples per word from one pipeline, one PNG per sample, for picking the best by eye.

    python outputs/keep/gallery.py runs/arm9/pipeline runs/arm9/gallery outputs/keep/hardmaru_words.txt outputs/keep/extra_words.txt --n 24
"""
import argparse, re
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

ap = argparse.ArgumentParser()
ap.add_argument("pipe"); ap.add_argument("out", type=Path); ap.add_argument("words", nargs="+", type=Path)
ap.add_argument("--n", type=int, default=24); ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--size", type=int, default=256)
args = ap.parse_args()
p = StableDiffusionPipeline.from_pretrained(args.pipe, torch_dtype=torch.float16, safety_checker=None).to("cuda")
p.scheduler = DPMSolverMultistepScheduler.from_config(p.scheduler.config)
p.set_progress_bar_config(disable=True)
for f in args.words:
    for w in [x.strip() for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]:
        d = args.out / re.sub(r"[^a-z0-9]+", "-", w.lower()).strip("-"); d.mkdir(parents=True, exist_ok=True)
        ims = p([w] * args.n, num_inference_steps=25, guidance_scale=args.guidance, height=args.size, width=args.size,
                generator=[torch.Generator("cuda").manual_seed(s) for s in range(args.n)]).images
        for s, im in enumerate(ims):
            im.convert("L").save(d / f"{s:02d}.png")
        print(w, flush=True)
