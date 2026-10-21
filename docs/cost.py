"""Seconds per training step for the same U-Net on 64x64x1 pixels and on the 16x16x4 latent.

    python docs/cost.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kanji_diffusion.model import build_unet  # noqa: E402

torch.manual_seed(0)
for name, size, ch in (("latent 16x16x4", 16, 4), ("pixels 64x64x1", 64, 1)):
    unet = build_unet(size, ch, 512, "small")
    opt = torch.optim.AdamW(unet.parameters(), lr=1e-4)
    x = torch.randn(64, ch, size, size)
    cond = torch.randn(64, 32, 512)
    t = torch.randint(0, 1000, (64,))
    times = []
    for i in range(4):
        t0 = time.time()
        loss = F.mse_loss(unet(x, t, encoder_hidden_states=cond).sample, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i:  # the first step warms up allocators
            times.append(time.time() - t0)
    print(f"{name}: {sum(times) / len(times):.2f} s per step of batch 64", flush=True)
