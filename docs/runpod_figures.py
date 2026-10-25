"""Figures for the Bits2Bricks piece on renting the A100 (infraphysics article 6402857).

    .venv/Scripts/python.exe docs/runpod_figures.py [out_dir]

Every number below is copied from the rental, not computed from a model:
- the catalog read at 08:31 UTC on 2026-10-21 (Secure Cloud, `list-gpu-types`), as noted in the
  rental diary and in the article's sizing section;
- the memory measured on the running job (33.5 GB) and the bytes per parameter of the article's sheet;
- the step times: the estimate made beforehand (2.5 to 3.5 s), the 40-step smoke test (0.95 s) and the
  average of the full 6,000-step run (1.04 s, 104 min), from docs/journal.md, arm 2 attempt 1;
- the sample sheets of attempt 1 at step 40 and step 6,000 (docs/journal/sd15-attempt1/).
Default out_dir is the article's media folder in the infraphysics repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "infraphysics/media/articles/6402857/figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, GREY, LIGHT = "#1a1a1a", "#7a7a7a", "#d9d9d9"
GREEN, RED, BLUE = "#5f8f52", "#b03a2e", "#2f6db5"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.labelcolor": INK, "text.color": INK, "savefig.facecolor": "white",
})

PRICE = 1.79  # A100 SXM 80GB, $ per hour of compute
NEED = 33.5   # GB measured on the running job, batch 32 at 512 px

# name, memory GB, $ per hour, stock at 08:31 UTC
CATALOG = [
    ("RTX 4090", 24, 0.89, "high"),
    ("A100 PCIe", 80, 1.79, "none"),
    ("A100 SXM", 80, 1.79, "medium"),
    ("RTX PRO 6000", 96, 2.49, "medium"),
    ("H100 SXM", 80, 3.99, "medium"),
]


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color="#eeeeee", lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(path)


def sizing():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 3.0), gridspec_kw={"width_ratios": [1.25, 1.6], "wspace": 0.6})

    # Left: what the 33.5 GB is made of, against the three card sizes.
    p = 0.86  # billion parameters
    parts = [("weights", 4 * p, "#9bb98f"), ("gradients", 4 * p, "#7fa673"), ("AdamW moments", 8 * p, GREEN),
             ("EMA copy", 4 * p, "#4a7440"), ("activations and the rest", None, LIGHT)]
    y = 0
    for (label, gb, col), label_y in zip(parts, (2, 9, 16, 29, 36)):  # label rows kept clear of the 24 GB line
        gb = NEED - y if gb is None else gb
        a.bar(0, gb, bottom=y, width=0.5, color=col, edgecolor="white", lw=0.6)
        a.plot([0.25, 0.4], [y + gb / 2, label_y], color=GREY, lw=0.5)
        a.text(0.43, label_y, f"{label}  {gb:.1f}", va="center", fontsize=7.2)
        y += gb
    for cap, lab in ((24, "RTX 4090, 24 GB"), (40, "A100 40GB"), (80, "A100 or H100, 80 GB")):
        col = RED if cap < NEED else GREY
        a.axhline(cap, color=col, lw=0.8, ls=(0, (3, 2)))
        a.text(2.2, cap + 1.3, lab, fontsize=7, color=col, ha="right")
    a.set_xlim(-0.3, 2.2)
    a.set_ylim(0, 86)
    a.set_xticks([])
    a.set_ylabel("GPU memory (GB)")
    a.set_title(f"the job: {NEED} GB at batch 32", fontsize=9, loc="left")
    for s in ("top", "right", "bottom"):
        a.spines[s].set_visible(False)

    # Right: price per hour, greyed where the card is too small or out of stock.
    names = [f"{n}  ({m} GB)" for n, m, _, _ in CATALOG]
    for i, (n, m, usd, stock) in enumerate(CATALOG):
        fits, avail = m >= NEED, stock != "none"
        col = GREEN if (fits and avail) else LIGHT
        hatch = None if avail else "////"
        b.barh(i, usd, color=col, edgecolor=GREY if not avail else "white", hatch=hatch, lw=0.6, height=0.6)
        note = ("too small" if not fits else "") + ("no stock" if not avail else f"stock {stock}" if fits else "")
        b.text(usd + 0.06, i, f"${usd:.2f}   {note}", va="center", fontsize=7.2,
               color=INK if (fits and avail) else GREY)
    b.set_yticks(range(len(CATALOG)), names, fontsize=7.6)
    b.set_xlim(0, 5.3)
    b.set_xlabel("$ per hour, Secure Cloud")
    b.set_title("the catalog at 08:31 UTC", fontsize=9, loc="left")
    clean(b)
    save(fig, "sizing")


def plan():
    steps = 6000
    rows = [  # label, s per step low, high, colour
        ("estimate beforehand\n(published throughput)", 2.5, 3.5, LIGHT),
        ("40-step test\n(measured)", 0.95, 0.95, BLUE),
        ("the full run\n(measured average)", 1.04, 1.04, GREEN),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 2.0))
    for i, (lab, lo, hi, col) in enumerate(rows):
        h_lo, h_hi = steps * lo / 3600, steps * hi / 3600
        ax.barh(i, h_hi, color=col, height=0.55, edgecolor="white")
        if hi > lo:
            ax.barh(i, h_lo, color="#bdbdbd", height=0.55, edgecolor="white")
            txt = f"{lo} to {hi} s/step:  {h_lo:.1f} to {h_hi:.1f} h,  ${h_lo * PRICE:.1f} to {h_hi * PRICE:.1f}"
        else:
            txt = f"{lo} s/step:  {h_hi * 60:.0f} min,  ${h_hi * PRICE:.2f}"
        ax.text(h_hi + 0.08, i, txt, va="center", fontsize=7.4)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 10)
    ax.set_xlabel("hours of A100 for 6,000 steps at batch 32 (cost at $1.79 per hour)")
    clean(ax)
    save(fig, "plan")


def samples():
    src = ROOT / "docs/journal/sd15-attempt1"
    rows = [("step 40", src / "smoke-step000040.png"), ("step 6,000", src / "step006000.png")]
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 2.35), gridspec_kw={"hspace": 0.32})
    for ax, (lab, path) in zip(axes, rows):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_axis_off()
        ax.set_title(lab, fontsize=8.5, loc="left", pad=3)
    save(fig, "samples")


if __name__ == "__main__":
    sizing()
    plan()
    samples()
