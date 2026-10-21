"""Paper-style figures (white ground, vector shapes, mathtext) drawn from a trained run.

    python docs/paper_figures.py runs/small overview architecture chain reconstruction trajectory \
        guidance_sweep memorization loss
Each figure is written to docs/paper/<name>.png (and .pdf).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kanji_diffusion.autoencoder import ink  # noqa: E402
from kanji_diffusion.model import build_scheduler  # noqa: E402
from kanji_diffusion.sample import _encoder, load_run  # noqa: E402

OUT = ROOT / "docs" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

# Fills follow the convention of the latent diffusion paper: pixel space pink, latent space green,
# conditioning grey, networks blue, attention orange.
PIX, PIX_E = "#f6dede", "#b45f5f"
LAT, LAT_E = "#e2eedb", "#5f8f52"
LAT_IN = "#f1f7ee"
COND, COND_E = "#efefef", "#8a8a8a"
NET, NET_E = "#d3e2f4", "#3f6fa8"
ATT, ATT_E = "#fbe2c2", "#c4721b"
BAR = "#3e6b3a"
INK = "#1a1a1a"
GREY = "#7a7a7a"
SERIES = ["#2f6db5", "#c46a1b", "#1f8a70"]  # validated: lightness band, chroma, CVD, contrast on white

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "cm",
    "font.size": 9, "axes.linewidth": 0.7, "axes.edgecolor": "#333333", "xtick.color": "#333333",
    "ytick.color": "#333333", "axes.labelcolor": INK, "text.color": INK, "savefig.facecolor": "white",
})

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/small"
unet, vae, cfg = load_run(RUN)
DATA = torch.load(ROOT / "data/kanji64.pt")
CHARS = DATA["chars"]


def glyph(ch: str) -> torch.Tensor:
    return ink(DATA["images"][CHARS.index(ch)][None])


def paper(inkmap: np.ndarray) -> np.ndarray:
    """Ink in [0,1] to black on white."""
    return 1 - np.clip(inkmap, 0, 1)


def decode(z: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return vae.decode(z).sigmoid()[:, 0].numpy()


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(OUT / f"{name}.png")


def board(w, h):
    """A figure whose data coordinates are millimetres, origin bottom-left."""
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def panel(ax, x, y, w, h, fc, ec, title=None, title_at="top", r=2.2, lw=0.9, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    if title:
        ty = y + h - 2.6 if title_at == "top" else y + 2.4
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=size, weight="bold")


def trapezoid(ax, x, y, w, h, narrow_right=True, label=None, size=11, fc=NET, ec=NET_E):
    s = h * 0.22
    pts = [(x, y), (x + w, y + s), (x + w, y + h - s), (x, y + h)] if narrow_right else \
          [(x, y + s), (x + w, y), (x + w, y + h), (x, y + h - s)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=0.9))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)


def latent_bar(ax, x, y, h, label=None, w=1.6):
    ax.add_patch(Rectangle((x, y), w, h, fc=BAR, ec=BAR, lw=0.5))
    if label:
        ax.text(x + w / 2, y - 1.6, label, ha="center", va="top", fontsize=9)


def arrow(ax, p, q, color=INK, lw=0.9, ls="-", rad=0.0, head=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=head, lw=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def thumb(ax, img, x, y, s, border=True):
    ax.imshow(paper(img), cmap="gray", vmin=0, vmax=1, extent=(x, x + s, y, y + s), interpolation="lanczos", zorder=3)
    if border:
        ax.add_patch(Rectangle((x, y), s, s, fc="none", ec="#555555", lw=0.6, zorder=4))


def qkv(ax, x, y, w=5.6, h=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=ATT, ec=ATT_E, lw=0.8, zorder=3))
    ax.text(x + w / 2, y + h * 0.68, "$Q$", ha="center", va="center", fontsize=8, zorder=4)
    ax.text(x + w / 2, y + h * 0.3, "$K\\,V$", ha="center", va="center", fontsize=8, zorder=4)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#888888"); s.set_linewidth(0.5)


@torch.no_grad()
def sample_trace(prompts, steps=30, guidance=3.0, seed=0, keep=()):
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


# ------------------------------------------------------------------ 1. overview

def overview():
    """The model in the layout of Rombach et al. (2022), Fig. 3, with this project's shapes."""
    W, H = 178, 70
    fig, ax = board(W, H)
    x0 = glyph("魚")
    gen, _ = sample_trace(["fish"], seed=2)

    panel(ax, 2, 10, 24, 56, PIX, PIX_E, "Pixel space", "bottom")
    panel(ax, 28, 10, 112, 56, LAT, LAT_E, "Latent space")
    panel(ax, 142, 10, 34, 56, COND, COND_E, "Conditioning")

    thumb(ax, x0[0, 0].numpy(), 4.5, 47, 9)
    ax.text(9, 44.5, "$x$", ha="center", va="top", fontsize=10)
    trapezoid(ax, 16, 45, 8, 13, narrow_right=True, label="$\\mathcal{E}$")
    thumb(ax, gen[0], 4.5, 20, 9)
    ax.text(9, 17.5, "$\\tilde{x}$", ha="center", va="top", fontsize=10)
    trapezoid(ax, 16, 18, 8, 13, narrow_right=False, label="$\\mathcal{D}$")
    arrow(ax, (13.6, 51.5), (16, 51.5))
    arrow(ax, (16, 24.5), (13.6, 24.5))

    latent_bar(ax, 31, 45, 13, "$z_0$")
    arrow(ax, (24, 51.5), (31, 51.5))
    ax.add_patch(FancyBboxPatch((38, 55.6), 90, 4.2, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=LAT_E, lw=0.8))
    ax.text(83, 57.7, "Diffusion process  $q(z_t\\mid z_0)=\\mathcal{N}(\\sqrt{\\bar\\alpha_t}\\,z_0,\\,(1-\\bar\\alpha_t)\\,I)$",
            ha="center", va="center", fontsize=7.6)
    arrow(ax, (32.6, 57.7), (38, 57.7))
    latent_bar(ax, 133, 45, 13, None)
    ax.text(135.4, 51.5, "$z_T$", ha="left", va="center", fontsize=9)
    arrow(ax, (128, 57.7), (133.8, 57.7))

    panel(ax, 38, 14, 90, 37, LAT_IN, LAT_E, None, r=1.6, lw=0.7)
    ax.text(83, 47.6, "Denoising U-Net  $\\hat v_\\theta(z_t,\\,t,\\,\\tau_\\theta(y))$", ha="center", va="center", fontsize=8.5, weight="bold")
    latent_bar(ax, 131, 22, 13, None)
    ax.text(131.8, 20.2, "$z_t$", ha="center", va="top", fontsize=9)
    arrow(ax, (133.8, 45), (133.8, 35.4), lw=0.8)
    trapezoid(ax, 106, 19, 20, 22, narrow_right=False, label="")
    trapezoid(ax, 46, 19, 20, 22, narrow_right=True, label="")
    ax.text(116, 30, "down", ha="center", va="center", fontsize=7, color=NET_E)
    ax.text(56, 30, "up", ha="center", va="center", fontsize=7, color=NET_E)
    arrow(ax, (131, 28.5), (126, 28.5))
    for xq in (94, 84, 74):
        qkv(ax, xq, 21.5)
    arrow(ax, (106, 26), (99.6, 25.2))
    arrow(ax, (94, 25.2), (89.6, 25.2))
    arrow(ax, (84, 25.2), (79.6, 25.2))
    arrow(ax, (74, 25.2), (66, 26))
    for xs, xe, y in ((122, 50, 39.0), (114, 58, 36.6)):
        arrow(ax, (xs, y), (xe, y), color=GREY, ls=(0, (3, 2)), lw=0.7, head=5)
    latent_bar(ax, 40.3, 22, 13, None)
    arrow(ax, (46, 28.5), (41.9, 28.5))
    ax.text(41.1, 20.2, "$z_{t-1}$", ha="center", va="top", fontsize=9)
    ax.plot([41.1, 41.1, 131.8], [35.2, 43.2, 43.2], color=INK, lw=0.7)
    arrow(ax, (131.8, 43.2), (131.8, 35.4), lw=0.7, head=5)
    ax.text(44, 41.4, "$\\times\\,T$: the output becomes the next input (30 solver steps at sampling)", ha="left",
            va="center", fontsize=6.6, color=GREY)
    arrow(ax, (40.3, 22), (33.4, 22), lw=0.8)
    latent_bar(ax, 31.8, 16, 12, None)
    ax.text(32.6, 14.4, "$z_0$", ha="center", va="top", fontsize=9)
    arrow(ax, (31.8, 24.5), (24, 24.5))

    ax.add_patch(FancyBboxPatch((148, 50), 22, 7, boxstyle="round,pad=0,rounding_size=1", fc="white", ec=COND_E, lw=0.8))
    ax.text(159, 53.5, "$y$ = “fish”", ha="center", va="center", fontsize=8.5)
    trapezoid(ax, 152, 28, 14, 16, narrow_right=False, label="$\\tau_\\theta$", fc="#e4e4e4", ec=COND_E)
    ax.text(159, 25.5, "CLIP ViT-B/32\n(frozen)", ha="center", va="top", fontsize=6.8, color=GREY, linespacing=1.1)
    arrow(ax, (159, 50), (159, 44))
    ax.plot([152, 146, 146], [33, 33, 17.5], color=ATT_E, lw=0.8)
    ax.plot([146, 76.8], [17.5, 17.5], color=ATT_E, lw=0.8)
    for xq in (96.8, 86.8, 76.8):
        arrow(ax, (xq, 17.5), (xq, 21.5), color=ATT_E, lw=0.8, head=5)
    ax.text(112, 15.6, "$\\tau_\\theta(y)\\in\\mathbb{R}^{32\\times 512}$", ha="center", va="center", fontsize=7.5, color=ATT_E)

    ax.text(9, 37.6, "$64\\times64\\times1$", ha="center", fontsize=6.6, color=GREY)
    ax.text(31.8, 37.6, "$16\\times16\\times4$", ha="center", fontsize=6.6, color=GREY)

    ly = 4.5
    qkv(ax, 19, ly - 3.2, w=4.6, h=6.2)
    ax.text(25, ly - 0.2, "cross-attention", va="center", fontsize=7.5)
    ax.plot([52, 60], [ly, ly], color=GREY, lw=0.7, ls=(0, (3, 2)))
    ax.text(61.5, ly, "skip connection", va="center", fontsize=7.5)
    trapezoid(ax, 88, ly - 2.2, 4, 4.4, label="")
    ax.text(93.5, ly, "trained", va="center", fontsize=7.5)
    trapezoid(ax, 108, ly - 2.2, 4, 4.4, label="", fc="#e4e4e4", ec=COND_E)
    ax.text(113.5, ly, "frozen", va="center", fontsize=7.5)
    ax.add_patch(Rectangle((128, ly - 2.2), 1.4, 4.4, fc=BAR, ec=BAR))
    ax.text(131, ly, "latent tensor", va="center", fontsize=7.5)
    save(fig, "overview")


# ------------------------------------------------------------------ 2. architecture

def _bars(ax, x, y_mid, specs, gap=2.2, unit=1.1, wscale=0.045):
    """Ronneberger-style feature-map bars. specs: (channels, spatial, kind). Returns bar extents."""
    out = []
    for ch, sp, kind in specs:
        h = sp * unit
        w = max(0.7, ch * wscale)
        fc = {"res": "#6f9fd8", "att": "#f0a958", "io": "white", "lat": BAR}[kind]
        ec = {"res": "#2f5f99", "att": ATT_E, "io": "#333333", "lat": BAR}[kind]
        ax.add_patch(Rectangle((x, y_mid - h / 2), w, h, fc=fc, ec=ec, lw=0.6))
        ax.text(x + w / 2, y_mid + h / 2 + 0.8, f"{ch}", ha="center", va="bottom", fontsize=6.3, color="#444444")
        out.append((x, x + w, y_mid - h / 2, y_mid + h / 2))
        x += w + gap
    return out


def _link(ax, a, b, color="#1d3f8f", y=None):
    ya = y if y is not None else (a[2] + a[3]) / 2
    arrow(ax, (a[1] + 0.2, ya), (b[0] - 0.2, ya), color=color, lw=0.8, head=5)


def architecture():
    W, H = 200, 150
    fig, ax = board(W, H)

    ax.text(2, H - 3, "(a)  Autoencoder  $\\mathcal{E}$, $\\mathcal{D}$  (1.6M parameters)", fontsize=9, weight="bold", va="top")
    ym = H - 26
    enc = [(1, 64, "io"), (32, 64, "res"), (32, 64, "res"), (64, 32, "res"), (64, 32, "res"),
           (128, 16, "res"), (128, 16, "res"), (128, 16, "res"), (8, 16, "res")]
    dec = [(4, 16, "lat"), (128, 16, "res"), (128, 16, "res"), (128, 16, "res"), (64, 32, "res"), (64, 32, "res"),
           (32, 64, "res"), (32, 64, "res"), (1, 64, "io")]
    unit = 0.38
    eb = _bars(ax, 8, ym, enc, gap=4.6, unit=unit, wscale=0.05)
    db = _bars(ax, eb[-1][1] + 8, ym, dec, gap=4.6, unit=unit, wscale=0.05)
    seq = eb + db
    for i, (a, b) in enumerate(zip(seq, seq[1:])):
        kind = "down" if i in (2, 4) else "up" if i in (12, 14) else "sample" if i == 8 else "conv"
        color = {"down": "#a3201d", "up": "#2c8a2c", "sample": BAR, "conv": "#1d3f8f"}[kind]
        _link(ax, a, b, color=color, y=ym)
    for b, s in ((seq[0], 64), (seq[3], 32), (seq[5], 16), (seq[-4], 32), (seq[-2], 64)):
        ax.text(b[0] - 0.2, b[2] - 0.6, f"${s}^2$", ha="left", va="top", fontsize=6.2, color=GREY)
    ax.text((seq[8][0] + seq[9][1]) / 2, ym - 9, "$\\mu,\\,\\log\\sigma^2$, then $z=\\mu+\\sigma\\,\\epsilon$", ha="center",
            va="top", fontsize=7.2)
    ax.text((eb[0][0] + eb[-1][1]) / 2, ym + 16, "encoder $\\mathcal{E}$", ha="center", fontsize=8)
    ax.text((db[0][0] + db[-1][1]) / 2, ym + 16, "decoder $\\mathcal{D}$", ha="center", fontsize=8)
    ax.text((db[0][0] + db[-1][1]) / 2, ym - 21, "loss: BCE + L1 on ink, $+\\,10^{-5}\\,\\mathrm{KL}(q(z\\mid x)\\,\\|\\,\\mathcal{N}(0,I))$",
            ha="center", fontsize=7.2, color=GREY)

    top = H - 52
    ax.text(2, top, "(b)  Denoising U-Net  $\\hat v_\\theta$  (17.4M parameters)", fontsize=9, weight="bold", va="top")
    u = 1.35
    y1, y2, y3 = top - 18, top - 46, top - 68
    ws = 0.024
    l1 = _bars(ax, 12, y1, [(4, 16, "lat"), (64, 16, "res"), (64, 16, "res"), (64, 16, "res")], gap=2.4, unit=u, wscale=ws)
    l2 = _bars(ax, 30, y2, [(128, 8, "res"), (128, 8, "att"), (128, 8, "res"), (128, 8, "att")], gap=1.9, unit=u, wscale=ws)
    l3 = _bars(ax, 48, y3, [(192, 4, "res"), (192, 4, "att"), (192, 4, "res"), (192, 4, "att")], gap=1.7, unit=u, wscale=ws)
    mid = _bars(ax, l3[-1][1] + 4, y3, [(192, 4, "res"), (192, 4, "att"), (192, 4, "res")], gap=1.7, unit=u, wscale=ws)
    r3 = _bars(ax, mid[-1][1] + 4, y3, [(192, 4, "res"), (192, 4, "att"), (192, 4, "res"), (192, 4, "att"),
                                       (192, 4, "res"), (192, 4, "att")], gap=1.7, unit=u, wscale=ws)
    r2 = _bars(ax, r3[-1][1] + 5, y2, [(128, 8, "res"), (128, 8, "att"), (128, 8, "res"), (128, 8, "att"),
                                      (128, 8, "res"), (128, 8, "att")], gap=1.9, unit=u, wscale=ws)
    r1 = _bars(ax, r2[-1][1] + 5, y1, [(64, 16, "res"), (64, 16, "res"), (64, 16, "res"), (4, 16, "lat")], gap=2.4, unit=u, wscale=ws)
    for row in (l1, l2, l3, mid, r3, r2, r1):
        for a, b in zip(row, row[1:]):
            _link(ax, a, b)
    _link(ax, l3[-1], mid[0])
    _link(ax, mid[-1], r3[0])
    for a, b in ((l1[-1], l2[0]), (l2[-1], l3[0])):
        arrow(ax, ((a[0] + a[1]) / 2, a[2] - 0.3), ((b[0] + b[1]) / 2 - 2.5, b[3] + 1.6), color="#a3201d", lw=0.9, head=6)
    for a, b in ((r3[-1], r2[0]), (r2[-1], r1[0])):
        arrow(ax, ((a[0] + a[1]) / 2, a[3] + 0.3), ((b[0] + b[1]) / 2 - 1, b[2] - 0.6), color="#2c8a2c", lw=0.9, head=6)
    for a, b, y in ((l1[-1], r1[0], y1 + 16 * u / 2 + 3.6), (l2[-1], r2[0], y2 + 8 * u / 2 + 3.6), (l3[-1], r3[0], y3 + 4 * u / 2 + 3.6)):
        ax.plot([a[1] - 0.5, a[1] - 0.5], [a[3], y], color="#9a9a9a", lw=0.8)
        ax.plot([a[1] - 0.5, b[0] + 0.5], [y, y], color="#9a9a9a", lw=0.8)
        arrow(ax, (b[0] + 0.5, y), (b[0] + 0.5, b[3] + 0.1), color="#9a9a9a", lw=0.8, head=5)
    for row, s in ((l1, 16), (l2, 8), (l3, 4), (r1, 16), (r2, 8), (r3, 4)):
        ax.text(row[0][0] - 0.3, row[0][2] - 0.6, f"${s}^2$", ha="left", va="top", fontsize=6.2, color=GREY)
    ax.text(l1[0][0] - 2, y1, "$z_t$", ha="right", va="center", fontsize=10)
    ax.text(r1[-1][1] + 2, y1, "$\\hat v$", ha="left", va="center", fontsize=10)
    ax.text((mid[0][0] + mid[-1][1]) / 2, y3 - 8, "middle block", ha="center", fontsize=7, color=GREY)

    lx, ly = 12, y3 - 16
    items = [("#1d3f8f", "ResBlock: 2 × (GroupNorm, SiLU, conv 3×3), residual"),
             (ATT_E, "transformer: self-attn., cross-attn. to $\\tau_\\theta(y)$, MLP; 8 heads"),
             ("#a3201d", "downsample: conv 3×3, stride 2"),
             ("#2c8a2c", "upsample: nearest ×2, conv 3×3"),
             ("#9a9a9a", "skip: feature maps concatenated"),
             (None, "timestep $t$: sinusoidal embedding, MLP to 256, added in every ResBlock")]
    for i, (c, t) in enumerate(items):
        yy = ly - (i % 3) * 4.4
        xx = lx + (i // 3) * 100
        if c is None:
            pass
        elif c == ATT_E:
            ax.add_patch(Rectangle((xx, yy - 1.6), 1.6, 3.2, fc="#f0a958", ec=ATT_E, lw=0.6))
        else:
            arrow(ax, (xx - 1, yy), (xx + 3, yy), color=c, lw=0.9, head=5)
        ax.text(xx + 5, yy, t, va="center", fontsize=6.8)
    save(fig, "architecture")


# ------------------------------------------------------------------ 3. Markov chain

def chain():
    """Ho et al. (2020), Fig. 2, redrawn in latent space with real latents decoded for display."""
    W, H = 178, 52
    fig, ax = board(W, H)
    x0 = glyph("魚")
    with torch.no_grad():
        z0 = vae.encode(x0)
    eps = torch.randn(z0.shape, generator=torch.Generator().manual_seed(4))
    ac = build_scheduler().alphas_cumprod
    zt = lambda t: ac[t].sqrt() * z0 + (1 - ac[t]).sqrt() * eps  # noqa: E731
    nodes = [(12, "$z_T$", 999, True), (62, "$z_t$", 600, True), (100, "$z_{t-1}$", 560, True), (156, "$z_0$", 0, False)]
    y = 38
    for x, lab, t, filled in nodes:
        ax.add_patch(plt.Circle((x, y), 6.2, fc="#c9c9c9" if filled else "white", ec=INK, lw=0.9))
        ax.text(x, y, lab, ha="center", va="center", fontsize=12)
    for x, _, t, _ in nodes:
        img = decode(zt(t))[0] if t else x0[0, 0].numpy()
        thumb(ax, img, x + 4, 8, 15)
    for a, b in ((18.6, 30), (44, 55.6), (106.6, 118), (132, 149.6)):
        arrow(ax, (a, y), (b, y))
    ax.text(37, y, "$\\cdots$", ha="center", va="center", fontsize=12)
    ax.text(125, y, "$\\cdots$", ha="center", va="center", fontsize=12)
    arrow(ax, (68.6, y + 0.4), (93.6, y + 0.4))
    ax.text(81, y + 4.2, "$p_\\theta(z_{t-1}\\mid z_t,\\,y)$", ha="center", fontsize=9)
    arrow(ax, (94, y - 4.5), (68.5, y - 4.5), ls=(0, (3, 2)), rad=-0.35)
    ax.text(81, y - 12.5, "$q(z_t\\mid z_{t-1})$", ha="center", fontsize=9)
    ax.text(2, 3.0, "Nodes live in the 16×16×4 latent space; images are $\\mathcal{D}(z_t)$, decoded only to be seen "
            "($t = 999,\\,600,\\,560,\\,0$).", fontsize=6.8, color=GREY)
    save(fig, "chain")


# ------------------------------------------------------------------ 4. reconstruction and interpolation

def reconstruction():
    chars = "魚雨鳥電龍鬱"
    fig, axes = plt.subplots(3, len(chars), figsize=(5.6, 3.0), gridspec_kw=dict(hspace=0.06, wspace=0.06))
    for j, ch in enumerate(chars):
        x = glyph(ch)
        with torch.no_grad():
            z = vae.encode(x)
        rec = decode(z)[0]
        for i, im in enumerate((x[0, 0].numpy(), rec, np.abs(rec - x[0, 0].numpy()) * 4)):
            axes[i, j].imshow(paper(im), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            clean(axes[i, j])
    for i, lab in enumerate(("$x$", "$\\mathcal{D}(\\mathcal{E}(x))$", "$4\\,|x-\\tilde x|$")):
        axes[i, 0].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9, labelpad=6)
    save(fig, "reconstruction")

    a, b = "木", "米"
    xa, xb = glyph(a), glyph(b)
    with torch.no_grad():
        za, zb = vae.encode(xa), vae.encode(xb)
    ts = np.linspace(0, 1, 7)
    fig, axes = plt.subplots(2, len(ts), figsize=(5.6, 1.85), gridspec_kw=dict(hspace=0.08, wspace=0.06))
    for j, t in enumerate(ts):
        rows = ((1 - t) * xa[0, 0].numpy() + t * xb[0, 0].numpy(), decode((1 - t) * za + t * zb)[0])
        for i, im in enumerate(rows):
            axes[i, j].imshow(paper(im), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            clean(axes[i, j])
        axes[1, j].set_xlabel(f"$\\lambda={t:.2f}$", fontsize=7.5)
    axes[0, 0].set_ylabel("pixels", rotation=0, ha="right", va="center", fontsize=8.5, labelpad=6)
    axes[1, 0].set_ylabel("latents", rotation=0, ha="right", va="center", fontsize=8.5, labelpad=6)
    save(fig, "interpolation")


# ------------------------------------------------------------------ 5. trajectory

def trajectory():
    prompts = ["fish", "pizza", "internet", "algorithm"]
    keep = (1, 2, 4, 7, 11, 16, 22, 30)
    _, tr = sample_trace(prompts, keep=keep, seed=11)
    fig, axes = plt.subplots(len(prompts), len(keep), figsize=(5.6, 2.95), gridspec_kw=dict(hspace=0.06, wspace=0.06))
    for i, p in enumerate(prompts):
        for j, k in enumerate(keep):
            axes[i, j].imshow(paper(tr[k][i]), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
            clean(axes[i, j])
            if i == 0:
                axes[i, j].set_title(f"step {k}", fontsize=7.5, pad=3)
        axes[i, 0].set_ylabel(f"“{p}”", rotation=0, ha="right", va="center", fontsize=8.5, labelpad=6)
    save(fig, "trajectory")


# ------------------------------------------------------------------ 6. guidance sweep

def _spines(ax):
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def guidance_sweep():
    rows = []
    for d in sorted((ROOT / "outputs" / "sweep").glob("g*"), key=lambda p: float(p.name[1:])):
        r = json.loads((d / "report.json").read_text(encoding="utf-8"))
        rows.append((float(d.name[1:]), r["known"]["target_top1"] / r["known"]["samples"],
                     1 - r["novel"]["within_prompt"], r["novel"]["mean_similarity"]))
    g, fid, var, nov = map(np.array, zip(*rows))
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.3), gridspec_kw=dict(wspace=0.62))
    ax = axes[0]
    ax.plot(var, fid, color=SERIES[0], lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8)
    offs = {0.0: (-24, 4), 1.0: (5, 0), 1.5: (4, 4), 2.0: (-10, 7), 3.0: (-6, -12), 5.0: (2, 5), 8.0: (-6, 8)}
    for gi, x, y in zip(g, var, fid):
        ax.annotate(f"$w={gi:g}$", (x, y), textcoords="offset points", xytext=offs.get(gi, (5, 3)), fontsize=7,
                    color="#333333")
    ax.set_xlabel("variety: 1 − similarity within a prompt")
    ax.set_ylabel("fidelity: dictionary kanji nearest")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _spines(ax)
    ax = axes[1]
    ax.plot(g, nov, color=SERIES[1], lw=1.6, marker="o", ms=4.5, mec="white", mew=0.8)
    ax.axhline(0.783, color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.text(g.max(), 0.783 + 0.006, "99th pct., real vs. real", ha="right", va="bottom", fontsize=6.8, color=GREY)
    ax.axhline(0.574, color=GREY, lw=0.8, ls=(0, (1, 2)))
    ax.text(0, 0.574 + 0.006, "median, real vs. real", ha="left", va="bottom", fontsize=6.8, color=GREY)
    ax.set_xlabel("guidance scale $w$")
    ax.set_ylabel("invented words: similarity\nto nearest training glyph")
    ax.set_ylim(0.4, 0.82)
    _spines(ax)
    save(fig, "guidance_sweep")
    (OUT / "guidance_sweep.json").write_text(json.dumps([dict(guidance=a, fidelity=b, variety=c, novelty=d)
                                                         for a, b, c, d in rows], indent=1))


# ------------------------------------------------------------------ 7. memorization

def memorization():
    """Somepalli et al. (2023) style: nearest-training-glyph similarity for real and generated glyphs."""
    from kanji_diffusion.evaluate import masks, similarity
    real = DATA["images"].numpy()
    rm = masks(real)
    idx = torch.randperm(len(real), generator=torch.Generator().manual_seed(0))[:2000]
    s = similarity(rm[idx], rm)
    s[torch.arange(len(idx)), idx] = -1
    base = s.max(1).values.numpy()
    gen = np.load(ROOT / "outputs/picker/glyphs.npy")
    gm = masks(gen)
    best = np.concatenate([similarity(gm[i:i + 512], rm).max(1).values.numpy() for i in range(0, len(gen), 512)])
    p99 = np.percentile(base, 99)
    fig, ax = plt.subplots(figsize=(5.4, 2.3))
    bins = np.linspace(0.2, 1.0, 61)
    for data, c, lab in ((base, SERIES[0], "real kanji vs. nearest other real kanji (2,000)"),
                         (best, SERIES[1], f"generated vs. nearest training kanji ({len(best):,} glyphs, 1,227 prompts)")):
        ax.hist(data, bins=bins, density=True, histtype="stepfilled", color=c, alpha=0.22, lw=0)
        ax.hist(data, bins=bins, density=True, histtype="step", color=c, lw=1.3, label=lab)
    ax.axvline(p99, color=INK, lw=0.8, ls=(0, (3, 2)))
    above = (best > p99).mean()
    ax.text(p99 + 0.006, ax.get_ylim()[1] * 0.62,
            f"99th pct. of real: {p99:.2f}\nnone of {len(best):,} generated above\n(highest {best.max():.2f})",
            fontsize=6.8, va="top")
    ax.set_xlabel("similarity to nearest training glyph (ink IoU, strokes thickened 1 px)")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    save(fig, "memorization")
    (OUT / "memorization.json").write_text(json.dumps(dict(p99=float(p99), median_real=float(np.median(base)),
                                                           median_gen=float(np.median(best)), above_p99=float(above),
                                                           n_gen=int(len(best)), max_gen=float(best.max())), indent=1))


# ------------------------------------------------------------------ 8. loss

def loss():
    pat = re.compile(r"step\s+(\d+)\s+loss\s+([\d.]+)")
    pts = {}
    for log in ("runs/train.log", "runs/train2.log"):
        for line in (ROOT / log).read_text(encoding="utf-8", errors="ignore").splitlines():
            m = pat.search(line)
            if m:
                pts[int(m.group(1))] = float(m.group(2))
    st, ls = map(np.array, zip(*sorted(pts.items())))
    fig, ax = plt.subplots(figsize=(5.4, 2.1))
    ax.plot(st, ls, color=SERIES[0], lw=1.4)
    ax.axvline(10000, color=GREY, lw=0.8, ls=(0, (3, 2)))
    ax.text(10150, ls.max() * 0.95, "resumed with a\nsecond cosine to 16k", fontsize=6.8, va="top", color="#333333")
    ax.set_xlabel("training step (batch 64)")
    ax.set_ylabel("$\\|\\hat v-v\\|^2$ (running mean)")
    ax.set_yscale("log")
    ax.grid(True, which="both", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "loss")


# ------------------------------------------------------------------ 9. stages: what trains, what is frozen, where each loss goes

FROZ, FROZ_E = "#e6e6e6", "#8a8a8a"
GRAD = "#c0392b"


def box(ax, x, y, w, h, text, fc, ec, size=8.5, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (1.6 if sub else 0), text, ha="center", va="center", fontsize=size, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, ha="center", va="center", fontsize=6.4, color="#555555", zorder=3)


def line(ax, pts, color=INK, lw=0.9, ls="-", head=True):
    """An orthogonal polyline through pts, with an arrowhead on its last segment."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    if head:
        arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, head=6)


def stages():
    W, H = 182, 138
    fig, ax = board(W, H)
    s = 10
    x0 = glyph("魚")
    with torch.no_grad():
        z0 = vae.encode(x0)
    rec = decode(z0)[0]
    gen, _ = sample_trace(["fish"], seed=2)
    noise = torch.randn((1, 4, 16, 16), generator=torch.Generator().manual_seed(2))

    def mosaic(z):
        z = z[0].numpy()
        t = [(c - c.min()) / (c.max() - c.min() + 1e-6) for c in z]
        m = np.concatenate([np.concatenate(t[:2], 1), np.concatenate(t[2:], 1)], 0)
        return 1 - m  # shown so that more ink reads darker

    def band(y, h, title, sub):
        ax.add_patch(Rectangle((1, y), W - 2, h, fc="#fafafa", ec="#cfcfcf", lw=0.7, zorder=0))
        ax.text(3, y + h - 2.6, title, fontsize=8.6, weight="bold", va="center")
        ax.text(3, y + h - 6.4, sub, fontsize=6.6, color="#555555", va="center")

    # ---- stage 1
    y = 104
    band(y - 6, 38, "Stage 1. Autoencoder", "3,000 steps, 43 min on the CPU")
    cy = y + 8
    thumb(ax, x0[0, 0].numpy(), 8, cy - s / 2, s)
    ax.text(13, cy - s / 2 - 2, "$x$", ha="center", va="top", fontsize=9)
    trapezoid(ax, 26, cy - 7, 12, 14, narrow_right=True, label="$\\mathcal{E}$")
    latent_bar(ax, 48, cy - 6, 12, None)
    ax.text(48.8, cy - 8, "$z$", ha="center", va="top", fontsize=9)
    trapezoid(ax, 58, cy - 7, 12, 14, narrow_right=False, label="$\\mathcal{D}$")
    thumb(ax, rec, 80, cy - s / 2, s)
    ax.text(85, cy - s / 2 - 2, "$\\tilde x$", ha="center", va="top", fontsize=9)
    box(ax, 104, cy - 6, 46, 12, "$\\mathcal{L}_1=\\mathrm{BCE}(\\tilde x,x)+|\\tilde x-x|+10^{-5}\\,\\mathrm{KL}$", "white", INK, size=7.6)
    line(ax, [(18, cy), (26, cy)])
    line(ax, [(38, cy), (48, cy)])
    line(ax, [(49.6, cy), (58, cy)])
    line(ax, [(70, cy), (80, cy)])
    line(ax, [(90, cy), (104, cy)])
    line(ax, [(13, cy + s / 2), (13, cy + 13), (127, cy + 13), (127, cy + 6)])
    ax.text(70, cy + 14.2, "target: the same glyph", ha="center", fontsize=6.4, color="#555555")
    line(ax, [(127, cy - 6), (127, cy - 11), (64, cy - 11), (64, cy - 7)], color=GRAD, ls=(0, (3, 2)))
    line(ax, [(64, cy - 11), (32, cy - 11), (32, cy - 7)], color=GRAD, ls=(0, (3, 2)))
    ax.text(96, cy - 13.4, "$\\nabla\\mathcal{L}_1$ updates $\\mathcal{E}$ and $\\mathcal{D}$", ha="center", va="center",
            fontsize=7, color=GRAD)

    # ---- stage 2
    y = 62
    band(y - 6, 38, "Stage 2. Denoiser", "16,000 steps, about 5 h on the CPU; $\\mathcal{E}$ and CLIP no longer change")
    cy = y + 8
    thumb(ax, x0[0, 0].numpy(), 8, cy - s / 2, s)
    trapezoid(ax, 22, cy - 6, 10, 12, narrow_right=True, label="$\\mathcal{E}$", fc=FROZ, ec=FROZ_E, size=10)
    latent_bar(ax, 37, cy - 6, 12, None)
    ax.text(37.8, cy - 8, "$z_0$", ha="center", va="top", fontsize=9)
    ax.add_patch(plt.Circle((48, cy), 2.6, fc="white", ec=INK, lw=0.9, zorder=2))
    ax.text(48, cy, "+", ha="center", va="center", fontsize=10, zorder=3)
    thumb(ax, mosaic(noise), 44.5, cy - 14.5, 7, border=True)
    ax.text(53, cy - 11, "$\\epsilon\\sim\\mathcal{N}(0,I)$, random $t$", ha="left", va="center", fontsize=6.6)
    latent_bar(ax, 57, cy - 6, 12, None)
    ax.text(57.8, cy + 7.4, "$z_t$", ha="center", va="bottom", fontsize=9)
    box(ax, 68, cy - 7, 30, 14, "U-Net $\\hat v_\\theta$", NET, NET_E, size=9, sub="cross-attention to $\\tau_\\theta(y)$")
    box(ax, 104, cy - 5, 12, 10, "$\\hat v$", "white", INK, size=9)
    box(ax, 124, cy - 6, 26, 12, "$\\mathcal{L}_2=\\|\\hat v-v\\|^2$", "white", INK, size=8)
    line(ax, [(18, cy), (22, cy)])
    line(ax, [(32, cy), (37, cy)])
    line(ax, [(38.6, cy), (45.4, cy)])
    line(ax, [(48, cy - 7.5), (48, cy - 2.6)])
    line(ax, [(50.6, cy), (57, cy)])
    line(ax, [(58.6, cy), (68, cy)])
    line(ax, [(98, cy), (104, cy)])
    line(ax, [(116, cy), (124, cy)])
    line(ax, [(37.8, cy + 6), (37.8, cy + 12), (137, cy + 12), (137, cy + 6)])
    ax.text(88, cy + 13.4, "target $v=\\sqrt{\\bar\\alpha_t}\\,\\epsilon-\\sqrt{1-\\bar\\alpha_t}\\,z_0$", ha="center", fontsize=6.8,
            color="#555555")
    line(ax, [(137, cy - 6), (137, cy - 11), (83, cy - 11), (83, cy - 7)], color=GRAD, ls=(0, (3, 2)))
    ax.text(112, cy - 13.2, "$\\nabla\\mathcal{L}_2$ updates the U-Net only", ha="center", va="center", fontsize=7, color=GRAD)

    # ---- conditioning, shared by stage 2 and sampling
    ky = 46
    ax.add_patch(FancyBboxPatch((112, ky - 6), 64, 12, boxstyle="round,pad=0,rounding_size=1.2", fc=FROZ, ec=FROZ_E, lw=0.9, zorder=2))
    ax.text(144, ky + 1.8, "CLIP text encoder $\\tau_\\theta$ (frozen)", ha="center", va="center", fontsize=8, zorder=3)
    ax.text(144, ky - 2.6, "$y$ = “fish”  to  $\\tau_\\theta(y)\\in\\mathbb{R}^{32\\times512}$, computed once", ha="center",
            va="center", fontsize=6.4, color="#555555", zorder=3)
    line(ax, [(112, ky), (90, ky), (90, cy - 7)], color=ATT_E)
    line(ax, [(90, ky), (90, 25)], color=ATT_E)

    # ---- sampling
    y = 4
    band(y, 34, "Sampling", "no loss, nothing is updated; about 0.3 s per glyph in a batch")
    cy = 18
    thumb(ax, mosaic(noise), 8, cy - s / 2, s)
    ax.text(13, cy - s / 2 - 2, "$z_T\\sim\\mathcal{N}(0,I)$", ha="center", va="top", fontsize=8)
    box(ax, 68, cy - 7, 30, 14, "U-Net $\\hat v_\\theta$", NET, NET_E, size=9, sub="once with $y$, once with $\\varnothing$")
    line(ax, [(18, cy), (68, cy)])
    ax.text(43, cy + 2, "$z_t$", ha="center", va="bottom", fontsize=9)
    box(ax, 104, cy - 6, 30, 12, "$\\hat v=\\hat v_\\varnothing+w\\,(\\hat v_y-\\hat v_\\varnothing)$", "white", INK, size=7.4,
        sub="guidance, $w=3$")
    line(ax, [(98, cy), (104, cy)])
    line(ax, [(119, cy - 6), (119, cy - 12), (30, cy - 12), (30, cy)])
    ax.text(74.5, cy - 10.4, "DPM-Solver++ step: $z_t$ to $z_{t-1}$, repeated 30 times", ha="center", fontsize=6.8,
            color="#555555")
    line(ax, [(134, cy), (140, cy)])
    latent_bar(ax, 140, cy - 6, 12, None)
    ax.text(140.8, cy - 8, "$z_0$", ha="center", va="top", fontsize=9)
    trapezoid(ax, 146, cy - 6, 10, 12, narrow_right=False, label="$\\mathcal{D}$", fc=FROZ, ec=FROZ_E, size=10)
    line(ax, [(141.6, cy), (146, cy)])
    thumb(ax, gen[0], 163, cy - s / 2, s)
    line(ax, [(156, cy), (163, cy)])
    ax.text(168, cy - s / 2 - 2, "new glyph", ha="center", va="top", fontsize=7)

    # legend
    lx, ly = 112, 132
    ax.add_patch(Rectangle((lx, ly - 1.6), 4, 3.2, fc=NET, ec=NET_E, lw=0.8))
    ax.text(lx + 5.5, ly, "trained in this stage", va="center", fontsize=6.6)
    ax.add_patch(Rectangle((lx + 36, ly - 1.6), 4, 3.2, fc=FROZ, ec=FROZ_E, lw=0.8))
    ax.text(lx + 41.5, ly, "frozen", va="center", fontsize=6.6)
    ax.plot([lx + 54, lx + 60], [ly, ly], color=GRAD, lw=0.9, ls=(0, (3, 2)))
    ax.text(lx + 61.5, ly, "gradient", va="center", fontsize=6.6)
    save(fig, "stages")


if __name__ == "__main__":
    for name in sys.argv[2:]:
        globals()[name]()
