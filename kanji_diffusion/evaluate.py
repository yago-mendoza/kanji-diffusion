"""Check what the model learned: does a dictionary meaning bring back its kanji, and is a
made-up prompt answered with a new glyph or a copy of the nearest training character?

    python -m kanji_diffusion.evaluate runs/small --out outputs/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .sample import _font, generate, load_run

# Prompt and the kanji a dictionary gives for it.
KNOWN = {"water": "水", "fire": "火", "tree": "木", "mountain": "山", "sun": "日", "moon": "月", "fish": "魚",
         "bird": "鳥", "rain": "雨", "heart": "心", "woman": "女", "gold": "金"}
NOVEL = ["pizza", "internet", "computer", "robot", "Elon Musk", "democracy", "airplane", "smartphone",
         "coffee", "algorithm", "cryptocurrency", "climate change"]


def ink(a: np.ndarray) -> np.ndarray:
    return 1 - a.astype(np.float32) / 255


def masks(a: np.ndarray, grow: int = 1) -> torch.Tensor:
    """Thresholded ink, thickened by `grow` pixels so a stroke drawn one pixel off still overlaps."""
    m = torch.from_numpy(ink(a) > 0.5).float().unsqueeze(1)
    return F.max_pool2d(m, 2 * grow + 1, 1, grow).flatten(1)


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = a @ b.T
    return inter / (a.sum(1, keepdim=True) + b.sum(1) - inter).clamp(min=1)


def baseline(real: np.ndarray, k: int = 1000) -> np.ndarray:
    """How close a real kanji gets to its nearest *other* kanji: the bar a novel glyph is held to."""
    idx = torch.randperm(len(real), generator=torch.Generator().manual_seed(0))[:k]
    sim = similarity(masks(real[idx.numpy()]), masks(real))
    sim[torch.arange(k), idx] = -1
    return sim.max(1).values.numpy()


def pair_sheet(prompts, gen, real, idx, iou, chars, n, path: Path, scale: int = 2):
    """Per prompt: n samples, each over its nearest training kanji and their similarity."""
    s = gen.shape[-1] * scale
    label = 16
    w, h = len(prompts) * s, label + n * (2 * s + 14)
    sheet = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(sheet)
    font = _font(10)
    for i, p in enumerate(prompts):
        d.text((i * s + 3, 2), p[:16], fill=0, font=font)
        for j in range(n):
            k = i * n + j
            y = label + j * (2 * s + 14)
            sheet.paste(Image.fromarray(gen[k]).resize((s, s), Image.LANCZOS), (i * s, y))
            sheet.paste(Image.fromarray(real[idx[k]]).resize((s, s), Image.LANCZOS).point(lambda v: 140 + v * 115 // 255),
                        (i * s, y + s))
            d.text((i * s + 3, y + 2 * s), f"{chars[idx[k]]} {iou[k]:.2f}", fill=0, font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/kanji64.pt"))
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/eval"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, vae, cfg = load_run(args.run, device)
    data = torch.load(args.data)
    real, chars, captions = data["images"].numpy(), data["chars"], data["captions"]
    base = baseline(real)
    copy_at = float(np.percentile(base, 99))
    report = {"baseline_nearest_other": {"p10": round(float(np.percentile(base, 10)), 3),
                                         "median": round(float(np.median(base)), 3),
                                         "p90": round(float(np.percentile(base, 90)), 3),
                                         "near_copy_threshold_p99": round(copy_at, 3)}}
    print(f"distinct real kanji: median nearest similarity {np.median(base):.3f}, 99th percentile {copy_at:.3f}")
    real_m = masks(real)
    for name, prompts in (("known", list(KNOWN)), ("novel", NOVEL)):
        gen = generate(unet, vae, prompts, n=args.n, text_model=cfg["text_model"], guidance=args.guidance,
                       device=device)
        sim = similarity(masks(gen), real_m)
        best, idx = sim.max(dim=1)
        idx, iou = idx.numpy(), best.numpy()
        pair_sheet(prompts, gen, real, idx, iou, chars, args.n, args.out / f"{name}.png")
        rows = []
        for i, p in enumerate(prompts):
            ks = range(i * args.n, (i + 1) * args.n)
            row = {"prompt": p, "nearest": [chars[idx[k]] for k in ks],
                   "nearest_meaning": [captions[idx[k]] for k in ks],
                   "similarity": [round(float(iou[k]), 3) for k in ks]}
            if name == "known":
                # Rank of the dictionary kanji among all 6,413 by similarity to each sample (1 = nearest).
                t = chars.index(KNOWN[p])
                row["target"] = KNOWN[p]
                row["target_rank"] = [int((sim[k] > sim[k, t]).sum()) + 1 for k in ks]
            rows.append(row)
        # Variety: mean similarity between the samples of one prompt (1 = the same glyph every time).
        own = similarity(masks(gen), masks(gen))
        pairs = [float(own[i * args.n + a, i * args.n + b]) for i in range(len(prompts))
                 for a in range(args.n) for b in range(a + 1, args.n)]
        report[name] = {"mean_similarity": round(float(iou.mean()), 3), "within_prompt": round(float(np.mean(pairs)), 3),
                        "near_copies": int((iou > copy_at).sum()), "samples": len(iou), "prompts": rows}
        line = (f"{name}: mean nearest similarity {iou.mean():.3f}, {(iou > copy_at).sum()}/{len(iou)} above {copy_at:.3f}, "
                f"within-prompt similarity {np.mean(pairs):.3f}")
        if name == "known":
            ranks = np.array([r for row in rows for r in row["target_rank"]])
            report[name]["target_top1"] = int((ranks == 1).sum())
            report[name]["target_top10"] = int((ranks <= 10).sum())
            line += f", dictionary kanji nearest in {(ranks == 1).sum()}/{len(ranks)}, top 10 in {(ranks <= 10).sum()}"
        print(line)
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
