"""Draw invented words from component designs, and measure whether the design decides the glyph.

    python -m kanji_diffusion.design runs/smallLC --data data/kanji64_LC.pt --out outputs/design

For each word in designs.json: n samples of the plain word, of design A ("pizza, made of eat (left) and pelt
(right)") and of design B. Design hit: the share of a sample's 20 nearest training glyphs whose component
caption contains every part of a design. A design that decides the glyph gives its own samples a higher hit
for its own parts than for the other design's parts; the other design is the control.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .evaluate import masks, similarity
from .sample import _font, generate, load_run

HERE = Path(__file__).parent


def parts_of(design: str) -> list[str]:
    return [re.sub(r"\s*\(.*\)$", "", p).strip() for p in design.split(" and ")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/kanji64_LC.pt"))
    ap.add_argument("--designs", type=Path, default=HERE / "designs.json")
    ap.add_argument("-n", type=int, default=6)
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--out", type=Path, default=Path("outputs/design"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet, vae, cfg = load_run(args.run, device)
    data = torch.load(args.data)
    real, comp = data["images"].numpy(), [c.lower() for c in data["component_captions"]]
    real_m = masks(real)
    designs = {k: v for k, v in json.loads(args.designs.read_text(encoding="utf-8")).items() if not k.startswith("_")}

    def has_parts(parts):
        return np.array([all(re.search(r"\b" + re.escape(p.lower()) + r"\b", c) for p in parts) for c in comp])

    rows, report, cells = [], {}, []
    for word, ((da, why_a), (db, why_b)) in designs.items():
        prompts = [word, f"{word}, made of {da}", f"{word}, made of {db}"]
        g = generate(unet, vae, prompts, n=args.n, text_model=cfg["text_model"], guidance=args.guidance, device=device)
        top = torch.topk(similarity(masks(g), real_m), 20, dim=1).indices.numpy()
        pa, pb = has_parts(parts_of(da)), has_parts(parts_of(db))
        hit = lambda k0, mask: float(np.mean([mask[top[k]].mean() for k in range(k0 * args.n, (k0 + 1) * args.n)]))
        r = {"A": da, "B": db, "A_hit_A": hit(1, pa), "A_hit_B": hit(1, pb), "B_hit_B": hit(2, pb), "B_hit_A": hit(2, pa),
             "plain_hit_A": hit(0, pa), "plain_hit_B": hit(0, pb), "base_rate_A": float(pa.mean()), "base_rate_B": float(pb.mean())}
        report[word] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        print(f"{word:15s} A[{da}] own {r['A_hit_A']:.3f} vs {r['A_hit_B']:.3f} | "
              f"B[{db}] own {r['B_hit_B']:.3f} vs {r['B_hit_A']:.3f}", flush=True)
        cells.append((word, da, db, g))
    wins = sum((v["A_hit_A"] > v["A_hit_B"]) + (v["B_hit_B"] > v["B_hit_A"]) for v in report.values())
    print(f"design decides the glyph in {wins} of {2 * len(report)} designs (own parts above the other design's)")
    report["_summary"] = {"wins": int(wins), "of": 2 * len(designs)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "design_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # Sheet: per word, three rows of n samples (plain, design A, design B), labelled.
    s, lab, gap = 96, 320, 6
    W, H = lab + args.n * (s + gap), len(cells) * (3 * (s + gap) + 16)
    sheet = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(sheet)
    f = _font(13)
    for i, (word, da, db, g) in enumerate(cells):
        y0 = i * (3 * (s + gap) + 16)
        for r, label in enumerate([word, "A: " + da, "B: " + db]):
            y = y0 + r * (s + gap)
            d.text((6, y + s // 2), label[:44], fill=0, font=f, anchor="lm")
            for j in range(args.n):
                sheet.paste(Image.fromarray(g[r * args.n + j]).resize((s, s), Image.LANCZOS), (lab + j * (s + gap), y))
    sheet.save(args.out / "designs.png")


if __name__ == "__main__":
    main()
