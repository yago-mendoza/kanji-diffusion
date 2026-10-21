"""Generate kanji for English prompts with a trained run.

    python -m kanji_diffusion.sample runs/small "Elon Musk" "pizza" "internet" -n 4
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw, ImageFont

from .autoencoder import GlyphVAE
from .model import build_scheduler
from .text import TextEncoder


@lru_cache(maxsize=2)
def _encoder(name: str) -> TextEncoder:
    return TextEncoder(name)


@torch.no_grad()
def generate(unet, vae, prompts: list[str], *, text_model: str, emb_null: torch.Tensor | None = None,
             n: int = 1, steps: int = 30, guidance: float = 3.0, seed: int = 0, device: str = "cpu") -> np.ndarray:
    """Returns uint8 images (len(prompts) * n, H, W), black ink on white."""
    enc = _encoder(text_model)
    cond = enc.encode([p for p in prompts for _ in range(n)]).to(device)
    null = (emb_null if emb_null is not None else enc.encode([""])[0]).to(device).expand_as(cond)

    sched = DPMSolverMultistepScheduler.from_config(build_scheduler().config)
    sched.set_timesteps(steps)
    g = torch.Generator().manual_seed(seed)
    c, size = unet.config.in_channels, unet.config.sample_size
    z = torch.randn((cond.shape[0], c, size, size), generator=g).to(device)
    for t in sched.timesteps:
        out = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([cond, null])).sample
        cv, uv = out.chunk(2)
        z = sched.step(uv + guidance * (cv - uv), t, z).prev_sample
    ink = vae.decode(z).sigmoid().squeeze(1).cpu().numpy()
    return ((1 - ink) * 255).round().astype(np.uint8)


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sample_grid(unet, vae, prompts: list[str], path: Path, *, n: int = 1, scale: int = 2, **kw) -> Path:
    """Labelled sheet: one column per prompt, n rows of samples."""
    imgs = generate(unet, vae, prompts, n=n, **kw)
    size = imgs.shape[-1] * scale
    label = 6 + 3 * scale
    sheet = Image.new("L", (len(prompts) * size, n * size + label), 255)
    draw = ImageDraw.Draw(sheet)
    font = _font(4 + 3 * scale)
    for i, p in enumerate(prompts):
        draw.text((i * size + 3, 2), p[:16], fill=0, font=font)
        for j in range(n):
            tile = Image.fromarray(imgs[i * n + j]).resize((size, size), Image.LANCZOS)
            sheet.paste(tile, (i * size, label + j * size))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def load_run(run: Path, device: str = "cpu"):
    cfg = json.loads((run / "config.json").read_text())
    unet = UNet2DConditionModel.from_pretrained(run / "unet").to(device).eval()
    return unet, GlyphVAE.load(run / "vae", device), cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("-n", type=int, default=4, help="samples per prompt")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("outputs/sheet.png"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, vae, cfg = load_run(args.run, device)
    path = sample_grid(unet, vae, args.prompts, args.out, n=args.n, scale=args.scale, text_model=cfg["text_model"],
                       steps=args.steps, guidance=args.guidance, seed=args.seed, device=device)
    print(path)


if __name__ == "__main__":
    main()
