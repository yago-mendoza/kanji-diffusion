"""Does the model use the caption? Error of the clean-latent estimate with each glyph's own caption against a shuffled one.

    python -m kanji_diffusion.diagnose runs/small            # small model, 64 px data
    python -m kanji_diffusion.diagnose runs/sd15 --data data/kanji512.pt

At each noise level, the same noisy latents are denoised three times: with the right caption, with the
caption of another character, and with the empty caption. If the model reads the meaning, the right
caption gives a clearly lower error than a wrong one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/kanji64.pt"))
    ap.add_argument("-n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator().manual_seed(args.seed)
    data = torch.load(args.data)
    idx = torch.randperm(len(data["chars"]), generator=g)[:args.n]
    wrong = idx[torch.randperm(args.n, generator=g)]
    caps = [c.split(", ")[0] for c in data["captions"]]

    if (args.run / "pipeline" / "model_index.json").exists():
        from diffusers import StableDiffusionPipeline
        from .finetune_sd import to_pixels
        pipe = StableDiffusionPipeline.from_pretrained(args.run / "pipeline", torch_dtype=torch.float16).to(device)
        unet, vae, tok, te = pipe.unet, pipe.vae, pipe.tokenizer, pipe.text_encoder
        alphas = pipe.scheduler.alphas_cumprod.to(device)
        pred = pipe.scheduler.config.prediction_type

        def embed(texts):
            ids = tok(texts, padding="max_length", max_length=tok.model_max_length, truncation=True, return_tensors="pt").input_ids
            return te(ids.to(device))[0]
        z0 = torch.cat([vae.encode(to_pixels(data["images"][idx[i:i + 32]]).to(device, torch.float16)).latent_dist.mean
                        for i in range(0, args.n, 32)]) * vae.config.scaling_factor
        dtype = torch.float16
    else:
        from .model import build_scheduler
        from .sample import load_run, _encoder
        from .autoencoder import ink
        unet, vae, cfg = load_run(args.run, device)
        enc = _encoder(cfg["text_model"])
        alphas = build_scheduler().alphas_cumprod.to(device)
        pred = "v_prediction"

        def embed(texts):
            return enc.encode(texts).to(device)
        z0 = vae.encode(ink(data["images"][idx]).to(device))
        dtype = torch.float32

    right_e = embed([caps[i] for i in idx.tolist()]).to(dtype)
    wrong_e = embed([caps[i] for i in wrong.tolist()]).to(dtype)
    null_e = embed([""] * args.n).to(dtype)
    z0 = z0.to(dtype)
    report = {}
    for t_val in (200, 500, 800):
        t = torch.full((args.n,), t_val, device=device, dtype=torch.long)
        noise = torch.randn(z0.shape, generator=g).to(device, dtype)
        a, s = alphas[t_val].sqrt().to(dtype), (1 - alphas[t_val]).sqrt().to(dtype)
        zt = a * z0 + s * noise
        target = noise if pred == "epsilon" else a * noise - s * z0
        err = {}
        for name, e in (("right", right_e), ("wrong", wrong_e), ("empty", null_e)):
            out = torch.cat([unet(zt[i:i + 32], t[i:i + 32], encoder_hidden_states=e[i:i + 32]).sample for i in range(0, args.n, 32)]).float()
            # Compared on the clean-latent estimate, the same quantity whatever the model predicts: an error on
            # the noise is nearly blind to the caption at high noise, where predicting the noise is trivial.
            x0 = (zt.float() - s.float() * out) / a.float() if pred == "epsilon" else a.float() * zt.float() - s.float() * out
            err[name] = F.mse_loss(x0, z0.float()).item()
        gain = (err["wrong"] - err["right"]) / err["wrong"]
        report[t_val] = {**{k: round(v, 5) for k, v in err.items()}, "right_vs_wrong_gain": round(gain, 4)}
        print(f"t={t_val}: error right {err['right']:.5f}  wrong {err['wrong']:.5f}  empty {err['empty']:.5f}"
              f"  -> the right caption lowers the error by {100 * gain:.1f}% against a wrong one", flush=True)
    (args.run / "diagnose.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
