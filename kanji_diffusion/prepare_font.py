"""Render every KANJIDIC2 character with an English meaning from fonts, for the enpitsu reproduction (arm 5).

    python -m kanji_diffusion.prepare_font --fonts fonts/YujiSyuku-Regular.ttf fonts/ShipporiMincho-Regular.ttf \
        --size 512 --out data/kanji512_font.pt

enpitsu's fake kanji came from Stable Diffusion trained on about 10,000 characters with their meanings;
KANJIDIC2 has 10,384 with English meanings, against the 6,413 KanjiVG also draws. Each character is drawn
with the first font that has it: the brush font (Yuji Syuku, 6,627 of them) and a mincho for the rest.
Same fields as prepare.py's output that finetune_sd.py reads (chars, captions, images).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .prepare import KANJIDIC_URL, download, read_meanings

FILL = 0.82  # the glyph's longer side as a share of the frame, close to KanjiVG's margins


def render(char: str, font: ImageFont.FreeTypeFont, size: int, supersample: int = 2) -> np.ndarray:
    big = size * supersample
    img = Image.new("L", (big, big), 255)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), char, font=font)
    s = FILL * big / max(x1 - x0, y1 - y0)
    # Draw at the font's nominal size, then scale the ink box to the frame and centre it.
    glyph = Image.new("L", (x1 - x0 + 4, y1 - y0 + 4), 255)
    ImageDraw.Draw(glyph).text((2 - x0, 2 - y0), char, font=font, fill=0)
    w, h = max(1, round(glyph.width * s)), max(1, round(glyph.height * s))
    img.paste(glyph.resize((w, h), Image.LANCZOS), ((big - w) // 2, (big - h) // 2))
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonts", type=Path, nargs="+", required=True, help="in order of preference")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/kanji512_font.pt"))
    args = ap.parse_args()

    meanings = read_meanings(download(KANJIDIC_URL, args.raw / "kanjidic2.xml.gz"))
    cmaps = [TTFont(f).getBestCmap() for f in args.fonts]
    fonts = [ImageFont.truetype(str(f), 256) for f in args.fonts]
    chars, which = [], []
    for c in sorted(meanings):
        k = next((i for i, cm in enumerate(cmaps) if ord(c) in cm), None)
        if k is not None:
            chars.append(c)
            which.append(k)
    for i, f in enumerate(args.fonts):
        print(f"{f.name}: {which.count(i)} characters")

    images = np.stack([render(c, fonts[k], args.size) for c, k in zip(chars, which)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"chars": chars, "captions": [", ".join(meanings[c]) for c in chars],
                "images": torch.from_numpy(images), "font": [args.fonts[k].name for k in which], "size": args.size},
               args.out)
    step = max(1, len(chars) // 128)
    tiles = [Image.fromarray(images[i]).resize((64, 64), Image.LANCZOS) for i in range(0, 128 * step, step)]
    sheet = Image.new("L", (16 * 64, 8 * 64), 255)
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 16) * 64, (i // 16) * 64))
    sheet.save(args.out.with_suffix(".preview.png"))
    print(f"saved {args.out} ({images.shape}) and preview")


if __name__ == "__main__":
    main()
