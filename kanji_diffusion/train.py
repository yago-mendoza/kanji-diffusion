"""Train the text-conditioned latent denoiser.

    python -m kanji_diffusion.train --steps 10000
    python -m kanji_diffusion.train --steps 20000   # same --out: continues where it stopped
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

from .autoencoder import GlyphVAE, ink
from .model import build_scheduler, build_unet
from .sample import sample_grid

DEMO_PROMPTS = ["water", "fire", "tree", "mountain", "sun", "moon", "fish", "bird",
                "rain", "heart", "computer", "robot", "pizza", "internet", "Elon Musk", "democracy"]


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base * step / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/kanji64.pt"))
    ap.add_argument("--vae", type=Path, default=Path("runs/vae"))
    ap.add_argument("--out", type=Path, default=Path("runs/small"))
    ap.add_argument("--preset", default="small")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--cond-drop", type=float, default=0.1, help="caption dropout for classifier-free guidance")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    data = torch.load(args.data)
    vae = GlyphVAE.load(args.vae, device)
    with torch.no_grad():
        x = ink(data["images"]).to(device)
        latents = torch.cat([vae.encode(x[i:i + 512]) for i in range(0, len(x), 512)])
    emb = data["emb"].to(device)
    emb_null = data["emb_null"].to(device)
    n, text_dim = latents.shape[0], emb.shape[-1]

    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "state.pt"
    unet = build_unet(latents.shape[-1], latents.shape[1], text_dim, args.preset).to(device)
    ema = copy.deepcopy(unet).requires_grad_(False)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    if state_path.exists():
        st = torch.load(state_path, map_location=device)
        unet.load_state_dict(st["unet"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        step = st["step"]
        print(f"resumed at step {step}")
    vae.save(args.out / "vae")  # a run carries the autoencoder it was trained on
    (args.out / "config.json").write_text(json.dumps({
        "preset": args.preset, "text_model": data["text_model"], "data": str(args.data),
    }, indent=2))

    sched = build_scheduler()
    alphas = sched.alphas_cumprod.to(device)
    print(f"{n} kanji, latents {tuple(latents.shape[1:])}, "
          f"{sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M params, device {device}", flush=True)

    t0, start, running = time.time(), step, None
    unet.train()
    while step < args.steps:
        idx = torch.randint(0, n, (args.batch,), device=device)
        z0 = latents[idx]
        variant = torch.randint(0, emb.shape[1], (args.batch,), device=device)
        cond = emb[idx, variant].float()
        cond[torch.rand(args.batch, device=device) < args.cond_drop] = emb_null.float()

        t = torch.randint(0, sched.config.num_train_timesteps, (args.batch,), device=device)
        noise = torch.randn_like(z0)
        a = alphas[t].sqrt().view(-1, 1, 1, 1)
        s = (1 - alphas[t]).sqrt().view(-1, 1, 1, 1)
        zt = a * z0 + s * noise
        v = a * noise - s * z0

        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, args.warmup, args.steps)
        loss = F.mse_loss(unet(zt, t, encoder_hidden_states=cond).sample, v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        step += 1

        decay = min(args.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(p, 1 - decay)

        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 25 == 0:
            eta = (time.time() - t0) / (step - start) * (args.steps - step)
            print(f"step {step:6d}  loss {running:.4f}  lr {opt.param_groups[0]['lr']:.2e}  "
                  f"elapsed {time.time() - t0:6.0f}s  eta {eta / 60:5.0f}min", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step}, state_path)
            ema.save_pretrained(args.out / "unet")
            sample_grid(ema.eval(), vae, DEMO_PROMPTS, args.out / f"samples/step{step:06d}.png", n=2,
                        text_model=data["text_model"], emb_null=emb_null.float(), steps=25, device=device)
            print(f"saved checkpoint at step {step}", flush=True)


if __name__ == "__main__":
    main()
