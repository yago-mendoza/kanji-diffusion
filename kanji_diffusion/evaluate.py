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
from PIL import Image, ImageDraw

from .sample import _font, generate, load_run

KNOWN = ["water", "fire", "tree", "mountain", "sun", "moon", "fish", "bird", "rain", "heart", "woman", "gold"]
NOVEL = ["pizza", "internet", "computer", "robot", "Elon Musk", "democracy", "airplane", "smartphone",
         "coffee", "algorithm", "cryptocurrency", "climate change"]


def ink(a: np.ndarray) -> np.ndarray:
    return 1 - a.astype(np.float32) / 255


def nearest(gen: np.ndarray, real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index of the closest training glyph by ink IoU, and that IoU, for every generated one."""
    g = torch.from_numpy(ink(gen).reshape(len(gen), -1) > 0.5).float()
    r = torch.from_numpy(ink(real).reshape(len(real), -1) > 0.5).float()
    inter = g @ r.T
    union = g.sum(1, keepdim=True) + r.sum(1) - inter
    iou = inter / union.clamp(min=1)
    best, idx = iou.max(dim=1)
    return idx.numpy(), best.numpy()


def pair_sheet(prompts, gen, real, idx, iou, chars, n, path: Path, scale: int = 2):
    """Per prompt: n samples, each over its nearest training kanji and the IoU between them."""
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

    unet, vae, cfg = load_run(args.run)
    data = torch.load(args.data)
    real, chars, captions = data["images"].numpy(), data["chars"], data["captions"]
    report = {}
    for name, prompts in (("known", KNOWN), ("novel", NOVEL)):
        gen = generate(unet, vae, prompts, n=args.n, text_model=cfg["text_model"], guidance=args.guidance)
        idx, iou = nearest(gen, real)
        pair_sheet(prompts, gen, real, idx, iou, chars, args.n, args.out / f"{name}.png")
        rows = []
        for i, p in enumerate(prompts):
            ks = range(i * args.n, (i + 1) * args.n)
            rows.append({"prompt": p, "nearest": [chars[idx[k]] for k in ks],
                         "nearest_meaning": [captions[idx[k]] for k in ks],
                         "iou": [round(float(iou[k]), 3) for k in ks]})
        report[name] = {"mean_iou": round(float(iou.mean()), 3),
                        "copies_above_0.8": int((iou > 0.8).sum()), "samples": len(iou), "prompts": rows}
        print(f"{name}: mean nearest IoU {iou.mean():.3f}, {(iou > 0.8).sum()}/{len(iou)} near-copies (IoU > 0.8)")
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
