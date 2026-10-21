"""Fine-tune a pretrained Stable Diffusion on the same kanji and captions as the small model (needs a GPU).

The second arm of the comparison: same 6,413 characters, same two captions per character, same 10% caption
dropout, but starting from a model that has already seen billions of images.

    python -m kanji_diffusion.prepare --size 512 --out data/kanji512.pt
    python -m kanji_diffusion.finetune_sd --data data/kanji512.pt --out runs/sd15 --steps 10000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .train import DEMO_PROMPTS, lr_at


def to_pixels(images: torch.Tensor) -> torch.Tensor:
    """uint8 black-on-white (N,H,W) to the RGB [-1, 1] range Stable Diffusion's autoencoder expects."""
    x = images.float().div(127.5).sub(1).unsqueeze(1)
    return x.expand(-1, 3, -1, -1)


@torch.no_grad()
def sample_sheet(unet, vae, text_encoder, tokenizer, prompts, path: Path, n: int = 2, steps: int = 25,
                 guidance: float = 3.0, size: int = 512, seed: int = 0):
    from PIL import Image
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                   scheduler=DPMSolverMultistepScheduler.from_config(SCHED_CONFIG),
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator(unet.device).manual_seed(seed)
    imgs = pipe([p for p in prompts for _ in range(n)], num_inference_steps=steps, guidance_scale=guidance,
                height=size, width=size, generator=g).images
    tile = 128
    sheet = Image.new("L", (len(prompts) * tile, n * tile), 255)
    for i, im in enumerate(imgs):
        sheet.paste(im.convert("L").resize((tile, tile), Image.LANCZOS), ((i // n) * tile, (i % n) * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


SCHED_CONFIG = None


def main() -> None:
    global SCHED_CONFIG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--data", type=Path, default=Path("data/kanji512.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/sd15"))
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cond-drop", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "fine-tuning Stable Diffusion needs a GPU"
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device, dtype).requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device, dtype).requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(device)
    unet.enable_gradient_checkpointing()
    sched = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")
    SCHED_CONFIG = sched.config
    pred = sched.config.prediction_type

    data = torch.load(args.data)
    chars, images = data["chars"], data["images"]
    full = data["captions"]
    first = [c.split(", ")[0] for c in full]
    size = images.shape[-1]
    print(f"{len(chars)} kanji at {size}px, base {args.model}, prediction {pred}", flush=True)

    # Encode every glyph once: the autoencoder is frozen, so its latents never change.
    with torch.no_grad():
        latents = torch.cat([
            vae.encode(to_pixels(images[i:i + 64]).to(device, dtype)).latent_dist.sample() * vae.config.scaling_factor
            for i in range(0, len(images), 64)
        ]).float()

    def embed(texts):
        ids = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                        return_tensors="pt").input_ids.to(device)
        return text_encoder(ids)[0].float()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arm.json").write_text(json.dumps({"base": args.model, "data": str(args.data), "steps": args.steps,
                                                    "batch": args.batch, "lr": args.lr, "cond_drop": args.cond_drop}, indent=2))
    ema = copy.deepcopy(unet).requires_grad_(False)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    alphas = sched.alphas_cumprod.to(device)
    unet.train()
    t0, running = time.time(), None
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(chars), (args.batch,))
        z0 = latents[idx.to(device)]
        texts = [full[i] if torch.rand(()) < 0.5 else first[i] for i in idx.tolist()]
        texts = ["" if torch.rand(()) < args.cond_drop else t for t in texts]
        cond = embed(texts)
        t = torch.randint(0, sched.config.num_train_timesteps, (args.batch,), device=device)
        noise = torch.randn_like(z0)
        a = alphas[t].sqrt().view(-1, 1, 1, 1)
        s = (1 - alphas[t]).sqrt().view(-1, 1, 1, 1)
        zt = a * z0 + s * noise
        target = noise if pred == "epsilon" else a * noise - s * z0
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        with torch.autocast("cuda", dtype=dtype):
            out = unet(zt, t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(out.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(p, 1 - min(args.ema, (1 + step) / (10 + step)))
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 25 == 0:
            eta = (time.time() - t0) / step * (args.steps - step)
            print(f"step {step:6d}  loss {running:.4f}  lr {opt.param_groups[0]['lr']:.2e}  eta {eta / 60:5.0f}min", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            ema.eval()
            sample_sheet(ema.to(dtype), vae, text_encoder, tokenizer, DEMO_PROMPTS, args.out / f"samples/step{step:06d}.png", size=size)
            ema.float()
            print(f"saved samples at step {step}", flush=True)

    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=ema.to(torch.float16),
                                   scheduler=DPMSolverMultistepScheduler.from_config(SCHED_CONFIG),
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.to(torch.float16).save_pretrained(args.out / "pipeline")
    print(f"saved pipeline to {args.out / 'pipeline'} in {math.ceil((time.time() - t0) / 60)} min", flush=True)


if __name__ == "__main__":
    main()
