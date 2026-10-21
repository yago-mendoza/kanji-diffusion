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
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline,
                       UNet2DConditionModel)
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
                                   scheduler=sampler(),
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    g = torch.Generator(unet.device).manual_seed(seed)
    imgs = pipe([p for p in prompts for _ in range(n)], num_inference_steps=steps, guidance_scale=guidance, **guidance_kw(),
                height=size, width=size, generator=g).images
    tile = 128
    sheet = Image.new("L", (len(prompts) * tile, n * tile), 255)
    for i, im in enumerate(imgs):
        sheet.paste(im.convert("L").resize((tile, tile), Image.LANCZOS), ((i // n) * tile, (i % n) * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


SCHED_CONFIG = None
ZERO_SNR = False


def sampler():
    """Sampling scheduler matching training. With zero terminal SNR, DDIM with trailing timesteps (Lin et al., 2023)."""
    if ZERO_SNR:
        return DDIMScheduler.from_config(SCHED_CONFIG, timestep_spacing="trailing", rescale_betas_zero_snr=True)
    return DPMSolverMultistepScheduler.from_config(SCHED_CONFIG)


def guidance_kw():
    return {"guidance_rescale": 0.7} if ZERO_SNR else {}


def make_fidelity(path: Path):
    """Fidelity check during training: 12 dictionary words, 2 samples each, scored as in evaluate.py."""
    from PIL import Image
    from .evaluate import KNOWN, masks, similarity
    data = torch.load(path)
    real, chars = data["images"].numpy(), data["chars"]
    real_m = masks(real)
    words = list(KNOWN)

    @torch.no_grad()
    def run(unet, vae, text_encoder, tokenizer, size):
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                       scheduler=sampler(),
                                       safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        g = torch.Generator(unet.device).manual_seed(0)
        imgs = pipe([w for w in words for _ in range(2)], num_inference_steps=25, guidance_scale=3.0, **guidance_kw(),
                    height=size, width=size, generator=g).images
        gen = __import__("numpy").stack([__import__("numpy").asarray(i.convert("L").resize((64, 64), Image.LANCZOS)) for i in imgs])
        sim = similarity(masks(gen), real_m)
        top1 = sum(int((sim[k] > sim[k, chars.index(KNOWN[words[k // 2]])]).sum() == 0) for k in range(len(gen)))
        own = similarity(masks(gen), masks(gen))
        var = float(sum(own[2 * i, 2 * i + 1] for i in range(len(words))) / len(words))
        return top1, var
    return run


def main() -> None:
    global SCHED_CONFIG, ZERO_SNR
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
    ap.add_argument("--init-unet", type=Path, help="start from this U-Net (e.g. a previous run's pipeline/unet)")
    ap.add_argument("--lr-xattn", type=float, help="learning rate for the cross-attention layers (default: --lr)")
    ap.add_argument("--constant", action="store_true", help="constant learning rate after warmup instead of a cosine")
    ap.add_argument("--offset-noise", type=float, default=0.0, help="per-channel noise offset, lets backgrounds reach white")
    ap.add_argument("--v-prediction", action="store_true", help="train on the velocity instead of the noise")
    ap.add_argument("--zero-snr", action="store_true", help="rescale the schedule to zero terminal SNR (fixes grey backgrounds)")
    ap.add_argument("--eval-data", type=Path, default=Path("data/kanji64.pt"), help="64 px set for the fidelity check")
    ap.add_argument("--t-logit-mean", type=float, help="sample timesteps logit-normally with this mean (>0 favours high noise)")
    ap.add_argument("--simple-boost", type=float, default=0.0,
                    help="extra sampling weight for characters with few strokes (base characters, rare as components)")
    ap.add_argument("--simple-strokes", type=int, default=6)
    ap.add_argument("--kanjivg", type=Path, default=Path("data/raw/kanjivg.xml.gz"), help="stroke data, for --simple-boost")
    ap.add_argument("--train-text-layers", type=int, default=0, help="unfreeze the last N CLIP text layers")
    ap.add_argument("--lr-text", type=float, default=2e-6)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "fine-tuning Stable Diffusion needs a GPU"
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(device).requires_grad_(False)
    text_params = []
    if args.train_text_layers:
        for m in list(text_encoder.text_model.encoder.layers[-args.train_text_layers:]) + [text_encoder.text_model.final_layer_norm]:
            m.requires_grad_(True)
        text_params = [p for p in text_encoder.parameters() if p.requires_grad]
    else:
        text_encoder.to(dtype)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(device, dtype).requires_grad_(False)
    unet = (UNet2DConditionModel.from_pretrained(args.init_unet) if args.init_unet
            else UNet2DConditionModel.from_pretrained(args.model, subfolder="unet")).float().to(device)
    unet.enable_gradient_checkpointing()
    sched = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler",
                                          **({"prediction_type": "v_prediction"} if args.v_prediction else {}),
                                          **({"rescale_betas_zero_snr": True, "timestep_spacing": "trailing"} if args.zero_snr else {}))
    ZERO_SNR = args.zero_snr
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
        with torch.autocast("cuda", dtype=dtype):
            return text_encoder(ids)[0].float()

    # Sampling weights: base characters (few strokes) are rare as recognisable components, so the model learns
    # their radicals before the characters themselves; --simple-boost shows them more often.
    weights = torch.ones(len(chars))
    if args.simple_boost:
        from .prepare import read_strokes
        strokes = read_strokes(args.kanjivg)
        n_strokes = torch.tensor([len(strokes.get(c, [])) for c in chars])
        weights += args.simple_boost * (n_strokes <= args.simple_strokes).float()
        print(f"{int((n_strokes <= args.simple_strokes).sum())} characters with <= {args.simple_strokes} strokes, "
              f"weight {1 + args.simple_boost}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arm.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    ema = copy.deepcopy(unet).requires_grad_(False)
    # The cross-attention (attn2) is what reads the caption: it can get its own, higher learning rate.
    xattn = [p for n, p in unet.named_parameters() if ".attn2." in n]
    rest = [p for n, p in unet.named_parameters() if ".attn2." not in n]
    groups = [{"params": rest, "base": args.lr}, {"params": xattn, "base": args.lr_xattn or args.lr}]
    if text_params:
        groups.append({"params": text_params, "base": args.lr_text})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-2)
    alphas = sched.alphas_cumprod.to(device)
    start = 1
    state_path = args.out / "state.pt"
    if state_path.exists():
        st = torch.load(state_path, map_location=device)
        unet.load_state_dict(st["unet"]); ema.load_state_dict(st["ema"]); opt.load_state_dict(st["opt"])
        if "text" in st:
            text_encoder.load_state_dict(st["text"])
        start = st["step"] + 1
        print(f"resumed at step {st['step']}", flush=True)
    fidelity = make_fidelity(args.eval_data) if args.eval_data.exists() else None
    # Caption gain: how much the right caption improves the clean-latent estimate at high noise, against a
    # shuffled caption (as in diagnose.py). 0% means the model ignores the text.
    gk = torch.Generator().manual_seed(1)
    probe = torch.randperm(len(chars), generator=gk)[:64]
    probe_wrong = probe[torch.randperm(64, generator=gk)]
    probe_noise = torch.randn((64,) + tuple(latents.shape[1:]), generator=gk).to(device)

    @torch.no_grad()
    def caption_gain(model, t_val=800):
        z0p = latents[probe.to(device)]
        a_, s_ = alphas[t_val].sqrt(), (1 - alphas[t_val]).sqrt()
        zt_ = (a_ * z0p + s_ * probe_noise).to(dtype)
        tt = torch.full((64,), t_val, device=device, dtype=torch.long)
        errs = []
        for ids in (probe, probe_wrong):
            e = embed([first[i] for i in ids.tolist()]).to(dtype)
            out = model(zt_, tt, encoder_hidden_states=e).sample.float()
            x0 = (zt_.float() - s_ * out) / a_ if pred == "epsilon" else a_ * zt_.float() - s_ * out
            errs.append(F.mse_loss(x0, z0p).item())
        return (errs[1] - errs[0]) / errs[1]
    unet.train()
    t0, running = time.time(), None
    for step in range(start, args.steps + 1):
        idx = torch.multinomial(weights, args.batch, replacement=True)
        z0 = latents[idx.to(device)]
        texts = [full[i] if torch.rand(()) < 0.5 else first[i] for i in idx.tolist()]
        texts = ["" if torch.rand(()) < args.cond_drop else t for t in texts]
        cond = embed(texts)
        T = sched.config.num_train_timesteps
        if args.t_logit_mean is not None:
            t = (torch.sigmoid(torch.randn(args.batch, device=device) + args.t_logit_mean) * (T - 1)).round().long()
        else:
            t = torch.randint(0, T, (args.batch,), device=device)
        noise = torch.randn_like(z0)
        if args.offset_noise:
            noise = noise + args.offset_noise * torch.randn(z0.shape[0], z0.shape[1], 1, 1, device=device)
        a = alphas[t].sqrt().view(-1, 1, 1, 1)
        s = (1 - alphas[t]).sqrt().view(-1, 1, 1, 1)
        zt = a * z0 + s * noise
        target = noise if pred == "epsilon" else a * noise - s * z0
        for g in opt.param_groups:
            g["lr"] = (g["base"] * min(1.0, step / args.warmup) if args.constant
                       else lr_at(step, g["base"], args.warmup, args.steps))
        with torch.autocast("cuda", dtype=dtype):
            out = unet(zt, t, encoder_hidden_states=cond).sample
        loss = F.mse_loss(out.float(), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(unet.parameters()) + text_params, 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), unet.parameters()):
                pe.lerp_(p, 1 - min(args.ema, (1 + step) / (10 + step)))
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if step % 25 == 0:
            eta = (time.time() - t0) / (step - start + 1) * (args.steps - step)
            print(f"step {step:6d}  loss {running:.4f}  lr {opt.param_groups[0]['lr']:.2e}  eta {eta / 60:5.0f}min", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            # Sample from a bf16 copy: casting the EMA itself rounded its weights at every sheet (attempt 1's bug).
            sampling_unet = copy.deepcopy(ema).to(dtype).eval()
            sampling_text = copy.deepcopy(text_encoder).to(dtype).eval()
            sample_sheet(sampling_unet, vae, sampling_text, tokenizer, DEMO_PROMPTS, args.out / f"samples/step{step:06d}.png", size=size)
            line = f"  caption gain {100 * caption_gain(sampling_unet):.1f}%"
            if fidelity:
                f1, var = fidelity(sampling_unet, vae, sampling_text, tokenizer, size)
                line += f"  fidelity {f1}/24  within-prompt {var:.3f}"
            del sampling_unet, sampling_text
            torch.save({"unet": unet.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                        **({"text": text_encoder.state_dict()} if text_params else {})}, state_path)
            print(f"saved samples and state at step {step}{line}", flush=True)

    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder.to(torch.float16), tokenizer=tokenizer, unet=ema.to(torch.float16),
                                   scheduler=sampler(),
                                   safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.to(torch.float16).save_pretrained(args.out / "pipeline")
    print(f"saved pipeline to {args.out / 'pipeline'} in {math.ceil((time.time() - t0) / 60)} min", flush=True)


if __name__ == "__main__":
    main()
