"""Figures for the README and the write-up: light ink on a dark ground, labelled in mono.

    python docs/figures.py progress            # training sheets under runs/small/samples
    python docs/figures.py data                # real training glyphs with their meanings
    python docs/figures.py sheet runs/small    # fresh samples for known and invented prompts
    python docs/figures.py pairs runs/small    # samples over their nearest training glyph
    python docs/figures.py cover runs/small    # an unlabelled field of invented characters
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs"
BG, INK, LABEL, DIM = (12, 12, 13), (236, 233, 226), (150, 148, 142), (88, 87, 84)


def font(size: int, cjk: bool = False):
    for name in (("YuGothM.ttc", "msgothic.ttc", "NotoSansCJK-Regular.ttc") if cjk else ("consola.ttf", "DejaVuSansMono.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def tile(a: np.ndarray, size: int, ink=INK) -> Image.Image:
    """uint8 black-on-white glyph to a coloured-ink-on-dark tile."""
    t = 1 - np.asarray(Image.fromarray(a).resize((size, size), Image.LANCZOS), dtype=np.float32) / 255
    rgb = np.array(BG, np.float32) + t[..., None] * (np.array(ink, np.float32) - np.array(BG, np.float32))
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8))


def grid(cells, cols: list[str], rows: list[str] | None, path: Path, size: int = 128, gap: int = 10,
         under: list[list[str]] | None = None, inks=None) -> Path:
    """cells[r][c] uint8 glyphs. Column labels on top, row labels on the left, optional captions under each."""
    pad, head = 28, 30
    left = 0 if rows is None else 118
    cap = 0 if under is None else 26
    w = pad * 2 + left + len(cols) * size + (len(cols) - 1) * gap
    h = pad * 2 + head + len(cells) * (size + cap) + (len(cells) - 1) * gap
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    f, fc = font(15), font(15, cjk=True)
    for c, name in enumerate(cols):
        x = pad + left + c * (size + gap)
        fs = 15
        while fs > 9 and d.textlength(name, font=font(fs)) > size + gap - 4:
            fs -= 1
        d.text((x + size / 2, pad + 8), name, fill=LABEL, font=font(fs), anchor="mm")
    for r, row in enumerate(cells):
        y = pad + head + r * (size + cap + gap)
        if rows is not None:
            d.text((pad, y + size / 2), rows[r], fill=LABEL, font=f, anchor="lm")
        for c, a in enumerate(row):
            x = pad + left + c * (size + gap)
            im.paste(tile(a, size, inks[r][c] if inks else INK), (x, y))
            if under is not None:
                d.text((x + size / 2, y + size + 13), under[r][c], fill=DIM, font=fc, anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)
    print(path)
    return path


def progress() -> None:
    """Crop the training sheets (16 prompts x 2 seeds, 128 px tiles under a 12 px label) into steps x prompts."""
    from kanji_diffusion.train import DEMO_PROMPTS
    prompts = ["tree", "fish", "bird", "rain", "computer", "pizza", "internet", "Elon Musk"]
    steps = [500, 1000, 2000, 4000, 10000, 16000]
    cells = []
    for s in steps:
        sheet = np.asarray(Image.open(ROOT / f"runs/small/samples/step{s:06d}.png").convert("L"))
        row = []
        for p in prompts:
            i = DEMO_PROMPTS.index(p)
            row.append(sheet[12:12 + 128, i * 128:(i + 1) * 128])
        cells.append(row)
    grid(cells, prompts, [f"step {s:,}" for s in steps], OUT / "progress.png", size=112)


def data() -> None:
    d = torch.load(ROOT / "data/kanji64.pt")
    chars, caps, imgs = d["chars"], d["captions"], d["images"].numpy()
    pick = "水火木山日月魚鳥雨心女金門食車電"
    cells, under = [], []
    for row in (pick[:8], pick[8:]):
        cells.append([imgs[chars.index(c)] for c in row])
        under.append([caps[chars.index(c)].split(",")[0][:14] for c in row])
    grid(cells, [""] * 8, None, OUT / "data.png", size=104, under=under)


def sheet(run: Path) -> None:
    from kanji_diffusion.sample import generate, load_run
    unet, vae, cfg = load_run(run)
    sets = {
        "dictionary": ["tree", "mountain", "woman", "fish", "bird", "rain", "gold", "sun"],
        "invented": ["pizza", "internet", "computer", "Elon Musk", "smartphone", "coffee", "algorithm", "cryptocurrency"],
    }
    for name, prompts in sets.items():
        g = generate(unet, vae, prompts, n=3, text_model=cfg["text_model"], seed=1)
        cells = [[g[i * 3 + j] for i in range(len(prompts))] for j in range(3)]
        grid(cells, prompts, None, OUT / f"{name}.png", size=120)


def pairs(run: Path, n: int = 4) -> None:
    """Each sample above the most similar of the 6,413 training glyphs (dimmed), as the evaluation measures it."""
    from kanji_diffusion.evaluate import KNOWN, NOVEL, masks, similarity
    from kanji_diffusion.sample import generate, load_run
    unet, vae, cfg = load_run(run)
    d = torch.load(ROOT / "data/kanji64.pt")
    real, chars = d["images"].numpy(), d["chars"]
    real_m = masks(real)
    for name, prompts in (("known", list(KNOWN)), ("novel", NOVEL)):
        g = generate(unet, vae, prompts, n=n, text_model=cfg["text_model"])
        best, idx = similarity(masks(g), real_m).max(dim=1)
        cells, under, inks = [], [], []
        for j in range(n):
            ks = [i * n + j for i in range(len(prompts))]
            cells += [[g[k] for k in ks], [real[idx[k]] for k in ks]]
            under += [[""] * len(ks), [f"{chars[idx[k]]} {best[k]:.2f}" for k in ks]]
            inks += [[INK] * len(ks), [DIM] * len(ks)]
        grid(cells, prompts, None, OUT / f"{name}.png", size=96, under=under, inks=inks)


COVER_WORDS = ["pizza", "internet", "computer", "robot", "Elon Musk", "democracy", "airplane", "smartphone",
               "coffee", "algorithm", "cryptocurrency", "climate change", "skyscraper", "YouTube", "Pikachu",
               "deep learning", "singularity", "spaceship", "electric guitar", "vaccine", "satellite", "hamburger",
               "television", "submarine", "nuclear power", "video game", "chocolate", "bicycle", "black hole",
               "password", "email", "software", "football", "museum", "hospital", "volcano", "tornado", "dinosaur",
               "telescope", "microscope", "battery", "neural network", "subway", "earthquake", "astronaut",
               "sushi", "jazz", "photograph", "printer", "rocket", "diffusion model"]


def cover(run: Path, cols: int = 16, rows: int = 7, size: int = 100) -> None:
    """16:7 field of invented characters for the article hero: no labels, ink varying a little in strength."""
    from kanji_diffusion.sample import generate, load_run
    unet, vae, cfg = load_run(run)
    words = (COVER_WORDS * 3)[:cols * rows]
    g = generate(unet, vae, words, n=1, text_model=cfg["text_model"], seed=7)
    rng = np.random.default_rng(3)
    order = rng.permutation(len(g))
    gap = size // 5
    im = Image.new("RGB", (cols * (size + gap) + gap, rows * (size + gap) + gap), BG)
    for k, i in enumerate(order[:cols * rows]):
        r, c = divmod(k, cols)
        strength = rng.uniform(0.35, 1.0)
        ink = tuple(int(BG[t] + strength * (INK[t] - BG[t])) for t in range(3))
        im.paste(tile(g[i], size, ink), (gap + c * (size + gap), gap + r * (size + gap)))
    im.save(OUT / "cover.png")
    print(OUT / "cover.png")


if __name__ == "__main__":
    what = sys.argv[1]
    run = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    {"progress": progress, "data": data, "sheet": lambda: sheet(run), "pairs": lambda: pairs(run),
     "cover": lambda: cover(run)}[what]()
