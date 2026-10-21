"""Candidates for the keep test: many glyphs per concept from a fine-tuned SD run, kept at 256 px.

    python outputs/keep/gen_keep.py runs/arm5/pipeline outputs/keep/concepts.txt outputs/keep/data.json

Half the candidates at guidance 1.5, half at 3, so the author's keeps also say which guidance he prefers.
The blind test showed 64 px reductions (sample.generate brings SD output to 64 px to compare it with the
small model); here the glyph is judged as it is drawn.
"""
import base64, io, json, sys, time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

pipe_dir, concepts_path, out_path = map(Path, sys.argv[1:4])
N, SIZE, GUIDANCE = 12, 256, (1.5, 3.0)
concepts = [c.strip() for c in concepts_path.read_text(encoding="utf-8").splitlines() if c.strip()]
pipe = StableDiffusionPipeline.from_pretrained(pipe_dir, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
extra = {"guidance_rescale": 0.7} if pipe.scheduler.config.get("rescale_betas_zero_snr") else {}


def b64(im):
    buf = io.BytesIO()
    im.convert("L").resize((SIZE, SIZE), Image.LANCZOS).save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


rows, t0 = [], time.time()
for k, c in enumerate(concepts):
    cands = []
    for gi, g in enumerate(GUIDANCE):
        gen = torch.Generator("cuda").manual_seed(7000 + 10 * k + gi)
        ims = pipe([c] * (N // 2), num_inference_steps=30, guidance_scale=g, generator=gen, **extra).images
        cands += [{"g": g, "i": b64(im)} for im in ims]
    rows.append({"c": c, "cands": cands})
    print(f"{k + 1}/{len(concepts)}  {time.time() - t0:.0f}s", flush=True)
out_path.write_text(json.dumps({"n": N, "concepts": rows}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("done", out_path.stat().st_size // 1024, "KB")
