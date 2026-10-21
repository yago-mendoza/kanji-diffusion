"""A light variational autoencoder: 64x64 glyphs to a 16x16x4 latent, as Stable Diffusion
does at 512x512 to 64x64x4. It is cheap enough to train on a CPU, and diffusion then runs
on 16 times fewer positions.

    python -m kanji_diffusion.autoencoder --steps 6000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class Res(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class GlyphVAE(nn.Module):
    def __init__(self, channels=(32, 64, 128), latent: int = 4):
        super().__init__()
        c1, c2, c3 = channels
        self.config = {"channels": list(channels), "latent": latent}
        self.encoder = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), Res(c1),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), Res(c2),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), Res(c3), Res(c3),
            nn.GroupNorm(8, c3), nn.SiLU(), nn.Conv2d(c3, 2 * latent, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent, c3, 3, padding=1), Res(c3), Res(c3),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c3, c2, 3, padding=1), Res(c2),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c1, 3, padding=1), Res(c1),
            nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 1, 3, padding=1),
        )
        # Latents are divided by this before diffusion so they have unit variance.
        self.register_buffer("scale", torch.tensor(1.0))

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        z = mean + torch.randn_like(mean) * (0.5 * logvar.clamp(-30, 20)).exp() if sample else mean
        return z / self.scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Logits of ink; sigmoid gives ink in [0, 1]."""
        return self.decoder(z * self.scale)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "vae.pt")
        (path / "vae.json").write_text(json.dumps(self.config))

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "GlyphVAE":
        cfg = json.loads((path / "vae.json").read_text())
        vae = cls(tuple(cfg["channels"]), cfg["latent"])
        vae.load_state_dict(torch.load(path / "vae.pt", map_location=device))
        return vae.to(device).eval()


def ink(images_uint8: torch.Tensor) -> torch.Tensor:
    """uint8 black-on-white (N, H, W) to float ink in [0, 1], shape (N, 1, H, W)."""
    return (1 - images_uint8.float() / 255).unsqueeze(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/kanji64.pt"))
    ap.add_argument("--out", type=Path, default=Path("runs/vae"))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--kl", type=float, default=1e-5)
    args = ap.parse_args()

    torch.manual_seed(0)
    x_all = ink(torch.load(args.data)["images"])
    n = len(x_all)
    held = torch.arange(n)[::50]  # 2% held out to check reconstructions honestly
    train_idx = torch.tensor(sorted(set(range(n)) - set(held.tolist())))

    vae = GlyphVAE()
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps, pct_start=0.05)
    print(f"{sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M params")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x = x_all[train_idx[torch.randint(0, len(train_idx), (args.batch,))]]
        h = vae.encoder(x)
        mean, logvar = h.chunk(2, dim=1)
        logvar = logvar.clamp(-30, 20)
        z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
        logits = vae.decoder(z)
        rec = F.binary_cross_entropy_with_logits(logits, x) + F.l1_loss(logits.sigmoid(), x)
        kl = 0.5 * (mean.pow(2) + logvar.exp() - 1 - logvar).mean()
        loss = rec + args.kl * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 100 == 0:
            with torch.no_grad():
                hx = x_all[held]
                err = F.l1_loss(vae.decode(vae.encode(hx)).sigmoid(), hx).item()
            print(f"step {step:5d}  rec {rec.item():.4f}  kl {kl.item():.2f}  held-out L1 {err:.4f}  {time.time() - t0:6.0f}s", flush=True)

    vae.eval()
    with torch.no_grad():
        z = torch.cat([vae.encode(x_all[i:i + 512]) for i in range(0, n, 512)])
        vae.scale.fill_(z.std().item())
        hx = x_all[held[:32]]
        rec = vae.decode(vae.encode(hx)).sigmoid()
    vae.save(args.out)
    pair = torch.cat([hx, rec], dim=2).squeeze(1)  # originals over reconstructions
    sheet = np.concatenate(list(((1 - pair.numpy()) * 255).astype(np.uint8)), axis=1)
    Image.fromarray(sheet).save(args.out / "reconstructions.png")
    print(f"latent std {vae.scale.item():.3f}; saved to {args.out}")


if __name__ == "__main__":
    main()
