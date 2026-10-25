"""Caption gain of every arm trained with Stable Diffusion's own objective, against passes per caption.

    python caption-gain-passes.py        (writes caption-gain-passes.png beside this file)

Data, nothing interpolated:
  - arms 9, 9b, 10, 10b, 11: the 'caption gain' lines of the training logs in
    stable_diffusion_kanji_generator/runs/<arm>/<arm>.log
  - arm 8 and the 256-glyph overfitting test: docs/journal.md, D21 and D20 (their logs stayed on the pod)
  - arm 7: docs/journal.md, D19, which records only the range (-0.4% to 0.1% at every save, steps 1,000 to 5,000)
Passes per caption = step x batch 32 / characters in the set (10,358 full set, 1,937 Jōyō, 256 test).
Arm 2 is left out: it trained on another noise schedule, so its gain is not on the same scale.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path(r"C:/Users/yagom/dev/stable_diffusion_kanji_generator/runs")
HERE = Path(__file__).resolve().parent
BATCH, FULL, JOYO, TEST = 32, 10358, 1937, 256
INK, GREY, LIGHT, HAIR, BAND = "#121212", "#69655f", "#a9a39a", "#d6d1c8", "#efe9df"


def from_log(arm):
    text = (RUNS / arm / f"{arm}.log").read_text(encoding="utf-8", errors="replace")
    return [(int(s), float(g)) for s, g in re.findall(r"at step (\d+)\s+caption gain (-?[\d.]+)%", text)]


def passes(series, n):
    return [s * BATCH / n for s, _ in series], [g for _, g in series]


arm8 = list(zip(range(1000, 8001, 1000), [0.2, 1.8, 5.9, 11.2, 16.3, 22.8, 30.3, 36.8]))  # journal D21
overfit = [(500, 3.4), (1000, 22.0), (1500, 44.0)]  # journal D20

plt.rcParams.update({"font.family": "Georgia", "font.size": 15, "axes.edgecolor": GREY, "xtick.color": GREY,
                     "ytick.color": GREY, "axes.labelcolor": GREY, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(10, 6.4))
fig.subplots_adjust(top=.9, left=.09, right=.86, bottom=.11)
ax.grid(axis="y", color=HAIR, lw=.8)
ax.set_axisbelow(True)

# arm 7: only its range is recorded, so it is drawn as a band, not a line
ax.fill_between([1000 * BATCH / FULL, 5000 * BATCH / FULL], -0.4, 0.1, color=INK, lw=0)

lines = [  # series, characters, colour, width, style, marker, label
    (from_log("arm9"), JOYO, INK, 2.4, "-", "o", "arm 9, 256 px"),
    (from_log("arm9b"), JOYO, INK, 1.4, (0, (4, 3)), None, "arm 9b, its rerun"),
    (arm8, JOYO, GREY, 2.2, "-", "o", "arm 8, 512 px"),
    (overfit, TEST, GREY, 2.2, "-", "s", "256-glyph test, 512 px"),
    (from_log("arm11"), FULL, INK, 2.4, "-", "D", "arm 11, full set"),
    (from_log("arm10b"), JOYO, LIGHT, 2, "-", None, "arm 10b, LoRA 5e-4"),
    (from_log("arm10"), JOYO, LIGHT, 2, (0, (1, 2)), None, "arm 10, LoRA 1e-4"),
]
ends = []
for series, n, colour, lw, ls, marker, label in lines:
    x, y = passes(series, n)
    ax.plot(x, y, color=colour, lw=lw, ls=ls, marker=marker, ms=7, mec="white", mew=1.6, clip_on=False, zorder=3)
    ends.append((x[-1], y[-1], label, colour))

nudge = {"arm 9b, its rerun": -16, "arm 8, 512 px": -5, "arm 11, full set": 2, "arm 10b, LoRA 5e-4": 5,
         "arm 10, LoRA 1e-4": -2, "256-glyph test, 512 px": 3}
for x, y, label, colour in ends:
    ax.annotate(label, (x, y), xytext=(8, nudge.get(label, 0)), textcoords="offset points",
                va="center", fontsize=13, color=INK if colour == INK else GREY, style="italic", annotation_clip=False)
ax.annotate("arm 7, every save\nbetween \u22120.4% and 0.1%", (9, 0.2), xytext=(3, 46), textcoords="data", ha="left",
            fontsize=13, color=GREY, style="italic", arrowprops=dict(arrowstyle="-", color=GREY, lw=.8))

ax.set_xlim(0, 270)
ax.set_ylim(-2, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("times each training character had been seen (passes per caption)")
ax.set_title("How much the caption helps the drawing (caption gain)", loc="left", fontsize=16, color=INK, pad=14)
fig.savefig(HERE / "caption-gain-passes.png", dpi=110, facecolor="white")
print("ok")
