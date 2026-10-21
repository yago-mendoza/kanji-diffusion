"""Explanatory figures drawn from a trained run: every glyph, latent and curve comes from the model itself.

    python docs/paper_figures.py runs/small pipeline autoencoder interpolation trajectory guidance
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kanji_diffusion.autoencoder import ink  # noqa: E402
from kanji_diffusion.model import build_scheduler  # noqa: E402
from kanji_diffusion.sample import _encoder, load_run  # noqa: E402

OUT = ROOT / "docs"
BG, FG, DIM, FAINT = "#0c0c0d", "#ece9e2", "#8c8a84", "#3a3936"
ACC, ACC_BG = "#d9773f", "#2a1a10"        # trained parts
FROZ, FROZ_BG = "#8fa6c4", "#141a22"       # frozen parts
MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": MONO, "text.color": FG, "axes.edgecolor": FAINT, "savefig.facecolor": BG})

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/small"
unet, vae, cfg = load_run(RUN)
DATA = torch.load(ROOT / "data/kanji64.pt")
CHARS = DATA["chars"]


def glyph(ch: str) -> torch.Tensor:
    return ink(DATA["images"][CHARS.index(ch)][None])  # (1,1,64,64) ink in [0,1]


def to_img(inkmap: np.ndarray) -> np.ndarray:
    """Ink in [0,1] to an RGB array: light strokes on the dark ground."""
    bg, fg = np.array([12, 12, 13]) / 255, np.array([236, 233, 226]) / 255
    return bg + inkmap[..., None] * (fg - bg)


def decode(z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return vae.decode(z).sigmoid()[:, 0].numpy()


def latent_mosaic(z: torch.Tensor) -> np.ndarray:
    """4x16x16 latent as a 2x2 mosaic of its channels, each scaled to its own range, grey."""
    z = z.numpy()
    tiles = []
    for c in range(z.shape[0]):
        a = z[c]
        a = (a - a.min()) / (a.max() - a.min() + 1e-6)
        tiles.append(np.pad(a, 1, constant_values=0))
    top, bot = np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)
    m = np.concatenate([top, bot], 0)
    return np.stack([m * 0.85 + 0.05] * 3, -1)


def canvas(w: float, h: float):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    return fig, ax


def box(ax, x, y, w, h, title, sub=None, kind="trained", size=11):
    edge, fill = {"trained": (ACC, ACC_BG), "frozen": (FROZ, FROZ_BG), "plain": (FAINT, BG)}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", lw=1.4, ec=edge, fc=fill))
    ty = y + h / 2 - (8 if sub else 0)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=size, color=FG, weight="bold")
    if sub:
        ax.text(x + w / 2, ty + 19, sub, ha="center", va="center", fontsize=size - 3, color=DIM)


def arrow(ax, p, q, color=DIM, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color,
                                 linestyle=(0, (4, 3)) if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))


def picture(ax, arr, x, y, s, label=None, frame=True):
    ax.imshow(arr, extent=(x, x + s, y + s, y), interpolation="lanczos" if arr.shape[0] > 20 else "nearest", zorder=2)
    if frame:
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="square,pad=0", lw=0.8, ec=FAINT, fc="none", zorder=3))
    if label:
        ax.text(x + s / 2, y + s + 14, label, ha="center", va="center", fontsize=8, color=DIM)


def label(ax, x, y, text, size=9, color=DIM, **kw):
    ax.text(x, y, text, fontsize=size, color=color, va="center", **kw)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", facecolor=BG)
    plt.close(fig)
    print(OUT / f"{name}.png")


def text_embed(prompts):
    return _encoder(cfg["text_model"]).encode(prompts)


@torch.no_grad()
def sample_with_trace(prompts, steps=30, guidance=3.0, seed=0, keep=()):
    """DPM-Solver++ sampling that also returns the model's clean-glyph estimate at chosen steps."""
    enc = _encoder(cfg["text_model"])
    cond = enc.encode(prompts)
    null = enc.encode([""])[0].expand_as(cond)
    sched = DPMSolverMultistepScheduler.from_config(build_scheduler().config)
    sched.set_timesteps(steps)
    ac = sched.alphas_cumprod
    z = torch.randn((len(prompts), 4, 16, 16), generator=torch.Generator().manual_seed(seed))
    trace = {}
    for k, t in enumerate(sched.timesteps):
        out = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([cond, null])).sample
        cv, uv = out.chunk(2)
        v = uv + guidance * (cv - uv)
        if k + 1 in keep:
            a = ac[t]
            trace[k + 1] = decode(a.sqrt() * z - (1 - a).sqrt() * v)
        z = sched.step(v, t, z).prev_sample
    return decode(z), trace


# ---------------------------------------------------------------- figures

def pipeline():
    """Training (top) and sampling (bottom) paths through the three networks."""
    W, H = 1500, 640
    fig, ax = canvas(W, H)
    s = 92
    x0 = glyph("魚")
    with torch.no_grad():
        z0 = vae.encode(x0)[0]
    g = torch.Generator().manual_seed(3)
    eps = torch.randn(z0.shape, generator=g)
    ac = build_scheduler().alphas_cumprod
    zt = ac[500].sqrt() * z0 + (1 - ac[500]).sqrt() * eps

    label(ax, 24, 28, "TRAINING", 10, FG, weight="bold")
    label(ax, 118, 28, "one step: noise a known kanji, learn to undo it", 9)
    y = 78
    picture(ax, to_img(x0[0, 0].numpy()), 30, y, s, "x  64×64×1")
    box(ax, 160, y + 6, 150, 80, "Encoder E", "1.6M · trained first")
    picture(ax, latent_mosaic(z0), 350, y, s, "z₀  16×16×4")
    ax.text(470, y + s / 2, "+", fontsize=20, color=DIM, ha="center", va="center")
    picture(ax, latent_mosaic(eps), 500, y, s, "noise ε")
    ax.text(615, y + s / 2, "=", fontsize=20, color=DIM, ha="center", va="center")
    picture(ax, latent_mosaic(zt), 640, y, s, "zₜ  (t = 500)")
    box(ax, 790, y - 4, 190, 100, "U-Net", "17.4M · trained second", size=12)
    box(ax, 1030, y + 6, 190, 80, "predicted v̂", "velocity", kind="plain")
    box(ax, 1270, y + 6, 200, 80, "loss ‖v̂ − v‖²", "v = √ᾱ·ε − √(1−ᾱ)·z₀", kind="plain", size=10)
    for a, b in [((30 + s, y + s / 2), (160, y + s / 2)), ((310, y + s / 2), (350, y + s / 2)),
                 ((640 + s, y + s / 2), (790, y + s / 2)), ((980, y + s / 2), (1030, y + s / 2)),
                 ((1220, y + s / 2), (1270, y + s / 2))]:
        arrow(ax, a, b)

    # text branch, shared by both paths
    ty = 300
    y2 = 460
    box(ax, 560, ty, 210, 74, "CLIP text encoder", "ViT-B/32 · frozen", kind="frozen")
    ax.text(470, ty + 37, "“fish”", fontsize=13, color=FG, ha="center", va="center")
    arrow(ax, (505, ty + 37), (560, ty + 37), FROZ)
    label(ax, 788, ty + 22, "32 tokens × 512", 9, FROZ)
    label(ax, 788, ty + 44, "read by cross-attention", 9, FROZ)
    arrow(ax, (770, ty + 30), (885, y + 96), FROZ, rad=-0.25)
    arrow(ax, (560, ty + 60), (300, y2 - 6), FROZ, rad=0.2)
    label(ax, 1000, ty + 33, "10% of captions dropped in training,", 8)
    label(ax, 1000, ty + 51, "so the same U-Net also denoises unconditioned", 8)

    label(ax, 24, 410, "SAMPLING", 10, FG, weight="bold")
    label(ax, 118, 410, "from pure noise", 9)
    gen, trace = sample_with_trace(["fish"], keep=(1, 6, 30), seed=2)
    z = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(2))
    picture(ax, latent_mosaic(z[0]), 30, y2, s, "z_T  noise")
    box(ax, 160, y2 - 4, 210, 100, "U-Net × 30", "DPM-Solver++ · guidance 3", size=12)
    label(ax, 160, y2 + 132, "each step: v̂ = v̂(∅) + 3·(v̂(fish) − v̂(∅))", 8)
    for i, (k, lab) in enumerate([(1, "estimate at step 1"), (6, "step 6"), (30, "step 30")]):
        picture(ax, to_img(trace[k][0]), 420 + i * 125, y2, s, lab)
    box(ax, 820, y2 + 6, 150, 80, "Decoder D", "trained with E")
    picture(ax, to_img(gen[0]), 1010, y2, s, "new glyph  64×64")
    arrow(ax, (30 + s, y2 + s / 2), (160, y2 + s / 2))
    arrow(ax, (370, y2 + s / 2), (420, y2 + s / 2))
    arrow(ax, (420 + 2 * 125 + s, y2 + s / 2), (820, y2 + s / 2))
    arrow(ax, (970, y2 + s / 2), (1010, y2 + s / 2))
    label(ax, 1135, y2 + 22, "the estimates are the clean glyph", 8)
    label(ax, 1135, y2 + 40, "the U-Net expects at that step,", 8)
    label(ax, 1135, y2 + 58, "decoded only to be shown", 8)

    ax.add_patch(FancyBboxPatch((1290, 590), 16, 12, boxstyle="round,pad=0,rounding_size=3", lw=1.2, ec=ACC, fc=ACC_BG))
    label(ax, 1312, 596, "trained", 8)
    ax.add_patch(FancyBboxPatch((1380, 590), 16, 12, boxstyle="round,pad=0,rounding_size=3", lw=1.2, ec=FROZ, fc=FROZ_BG))
    label(ax, 1402, 596, "frozen", 8)
    save(fig, "fig-pipeline")


def autoencoder():
    """The autoencoder's shapes, then real glyphs through it."""
    W, H = 1500, 700
    fig, ax = canvas(W, H)
    stages = [("64×64×1", 64, 1, "plain"), ("64×64×32", 64, 32, "trained"), ("32×32×64", 32, 64, "trained"),
              ("16×16×128", 16, 128, "trained"), ("16×16×4", 16, 4, "latent"), ("16×16×128", 16, 128, "trained"),
              ("32×32×64", 32, 64, "trained"), ("64×64×32", 64, 32, "trained"), ("64×64×1", 64, 1, "plain")]
    x, base = 40, 250
    centers = []
    for i, (name, sp, ch, kind) in enumerate(stages):
        h = sp * 2.6
        w = 10 + np.log2(ch + 1) * 9
        col = {"plain": FAINT, "trained": ACC, "latent": FG}[kind]
        fill = {"plain": BG, "trained": ACC_BG, "latent": "#2b2a27"}[kind]
        ax.add_patch(FancyBboxPatch((x, base - h / 2), w, h, boxstyle="square,pad=0", lw=1.2, ec=col, fc=fill))
        ax.text(x + w / 2, base + 100, name, ha="center", fontsize=8, color=DIM)
        centers.append((x, x + w))
        x += w + 118
    for (a0, a1), (b0, b1) in zip(centers, centers[1:]):
        arrow(ax, (a1 + 4, base), (b0 - 4, base))
    mid = centers[4]
    ax.text((mid[0] + mid[1]) / 2, base - 70, "latent z", ha="center", fontsize=10, color=FG, weight="bold")
    ax.text((centers[0][0] + centers[3][1]) / 2, 74, "encoder: 3 resolutions, residual convolutions, stride-2 downsampling",
            ha="center", fontsize=9, color=ACC)
    ax.text((centers[5][0] + centers[8][1]) / 2, 74, "decoder: the mirror, nearest-neighbour upsampling",
            ha="center", fontsize=9, color=ACC)
    ax.text((mid[0] + mid[1]) / 2, base + 124, "4,096 pixels into 1,024 numbers", ha="center", fontsize=9, color=FG)

    # real glyphs
    chars = "魚雨鳥電"
    y = 430
    label(ax, 40, y - 26, "glyph", 9)
    label(ax, 190, y - 26, "its four latent channels", 9)
    label(ax, 530, y - 26, "decoded", 9)
    label(ax, 690, y - 26, "|error|", 9)
    s = 110
    for i, ch in enumerate(chars[:2]):
        yy = y + i * (s + 30)
        x0 = glyph(ch)
        with torch.no_grad():
            z = vae.encode(x0)
        rec = decode(z)[0]
        picture(ax, to_img(x0[0, 0].numpy()), 40, yy, s)
        zz = z[0].numpy()
        for c in range(4):
            a = zz[c]
            a = (a - a.min()) / (a.max() - a.min() + 1e-6)
            picture(ax, np.stack([a * 0.85 + 0.05] * 3, -1), 190 + (c % 2) * 57, yy + (c // 2) * 57, 53)
        picture(ax, to_img(rec), 530, yy, s)
        err = np.abs(rec - x0[0, 0].numpy())
        picture(ax, to_img(np.clip(err * 4, 0, 1)), 690, yy, s)
    notes = [
        "Training asks one thing: rebuild the glyph",
        "from z. Nothing tells the network what a",
        "stroke or a radical is; the 16×16 grid keeps",
        "where ink is, and the 4 channels at each cell",
        "keep what that patch of ink looks like.",
        "",
        "Loss: binary cross-entropy + L1 on the ink,",
        "plus a KL term of weight 1e-5 that keeps z",
        "near a unit Gaussian without blurring it.",
        "",
        "Held-out mean error: 0.0055 per pixel",
        "(error maps shown ×4).",
    ]
    for i, t in enumerate(notes):
        label(ax, 870, y + 6 + i * 21, t, 9, FG if i < 5 else DIM)
    save(fig, "fig-autoencoder")


def interpolation():
    """Mixing two kanji in pixels and in the latent: the latent mix stays a drawing."""
    a, b = "木", "米"
    xa, xb = glyph(a), glyph(b)
    with torch.no_grad():
        za, zb = vae.encode(xa), vae.encode(xb)
    ts = np.linspace(0, 1, 7)
    pix = [(1 - t) * xa[0, 0].numpy() + t * xb[0, 0].numpy() for t in ts]
    lat = [decode((1 - t) * za + t * zb)[0] for t in ts]
    W, H = 1500, 420
    fig, ax = canvas(W, H)
    s, gap, x0 = 140, 22, 230
    for r, (name, row) in enumerate([("average of pixels", pix), ("average of latents,\nthen decoded", lat)]):
        y = 40 + r * (s + 50)
        ax.text(30, y + s / 2, name, fontsize=10, color=FG, va="center")
        for i, im in enumerate(row):
            picture(ax, to_img(np.clip(im, 0, 1)), x0 + i * (s + gap), y, s, f"{ts[i]:.2f}" if r == 1 else None)
    save(fig, "fig-interpolation")


def trajectory():
    prompts = ["fish", "pizza", "internet", "algorithm"]
    keep = (1, 2, 4, 7, 11, 16, 22, 30)
    _, tr = sample_with_trace(prompts, keep=keep, seed=11)
    W = 230 + len(keep) * 150
    H = 60 + len(prompts) * 150
    fig, ax = canvas(W, H)
    s = 128
    for j, k in enumerate(keep):
        ax.text(230 + j * 150 + s / 2, 28, f"step {k}", ha="center", fontsize=9, color=DIM)
    for i, p in enumerate(prompts):
        y = 50 + i * 150
        ax.text(30, y + s / 2, p, fontsize=11, color=FG, va="center")
        for j, k in enumerate(keep):
            picture(ax, to_img(tr[k][i]), 230 + j * 150, y, s)
    save(fig, "fig-trajectory")


def guidance():
    prompts = ["rain", "pizza", "Elon Musk"]
    scales = [0, 1, 2, 3, 5, 8]
    W, H = 230 + len(scales) * 150, 60 + len(prompts) * 150
    fig, ax = canvas(W, H)
    s = 128
    for j, g in enumerate(scales):
        ax.text(230 + j * 150 + s / 2, 28, f"guidance {g}", ha="center", fontsize=9, color=DIM)
        imgs, _ = sample_with_trace(prompts, guidance=float(g), seed=5)
        for i, p in enumerate(prompts):
            picture(ax, to_img(imgs[i]), 230 + j * 150, 50 + i * 150, s)
    for i, p in enumerate(prompts):
        ax.text(30, 50 + i * 150 + s / 2, p, fontsize=11, color=FG, va="center")
    save(fig, "fig-guidance")


if __name__ == "__main__":
    for name in sys.argv[2:]:
        globals()[name]()
