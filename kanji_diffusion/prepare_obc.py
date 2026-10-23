"""Oracle bone characters (HUST-OBC, deciphered) mixed with enpitsu's kanji, for arm 12.

    python -m kanji_diffusion.prepare_obc --kanji data/kanji256_epts.pt --out data/obc_mix256.pt

Oracle bone script is drawn: a fish looks like a fish. The kanji set taught the model to read a word through the
dictionary and answer with a printed part; this set should teach it to answer with a drawing. Only the three clean
sources are kept (H, the HWOBC expert redrawings; X, a scanned dictionary of hand-copies; Y, a website of traced
glyphs): G and L are 40 to 80 px crops that would blur at 256. Each class is captioned with the English meanings of
its modern character, KANJIDIC2 first (the dictionary of the kanji set) and Unihan kDefinition otherwise; classes
with neither are dropped. Every oracle caption carries the prefix "oracle bone script: " so one model can be asked
for either style. HUST-OBC is CC BY-NC 4.0.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from .prepare import KANJIDIC_URL, download, read_meanings

HUST_URL = "https://www.modelscope.cn/datasets/wpj2003/HUST-OBC/resolve/master/HUST-OBC.zip"
UNIHAN_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
PREFIX = "oracle bone script: "
SOURCES = "HXY"


def unihan_definitions(raw: Path) -> dict[str, str]:
    z = zipfile.ZipFile(download(UNIHAN_URL, raw / "Unihan.zip"))
    out = {}
    for line in z.read("Unihan_Readings.txt").decode("utf-8").splitlines():
        if "\tkDefinition\t" in line:
            cp, _, d = line.split("\t")
            out[chr(int(cp[2:], 16))] = d
    return out


def unihan_caption(d: str) -> str:
    """'(same as 廩) a granary, to supply (foodstuff); to stockpile' becomes 'granary, supply, stockpile'."""
    d = re.sub(r"\([^)]*\)", "", d)
    senses = [re.sub(r"^(to|a|an|the)\s+", "", s.strip()) for s in re.split(r"[;,]", d)]
    senses = [s for s in senses if s and not re.search(r"[㐀-鿿\U00020000-\U0002ffff]", s)]
    return ", ".join(dict.fromkeys(senses[:4]))


def glyph(path: Path, size: int) -> np.ndarray:
    """Black ink on white, cropped to the ink, centred in a square with the margin of the kanji set."""
    im = Image.open(path).convert("L")
    a = np.asarray(im)
    if np.median(a) < 128:          # a few scans are white on black
        im = ImageOps.invert(im); a = np.asarray(im)
    ys, xs = np.where(a < 160)
    if len(xs) == 0:
        return None
    im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(im.size) / 0.84)
    canvas = Image.new("L", (side, side), 255)
    canvas.paste(im, ((side - im.size[0]) // 2, (side - im.size[1]) // 2))
    return np.asarray(canvas.resize((size, size), Image.LANCZOS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kanji", type=Path, default=Path("data/kanji256_epts.pt"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/obc_mix256.pt"))
    ap.add_argument("--oracle-share", type=float, default=0.7, help="share of each batch drawn from the oracle set")
    args = ap.parse_args()
    root = args.raw / "obc/HUST-OBC/deciphered"
    if not root.exists():
        zipfile.ZipFile(download(HUST_URL, args.raw / "obc/HUST-OBC.zip")).extractall(args.raw / "obc")
    kanji = torch.load(args.kanji)
    size = kanji["size"]
    kd = read_meanings(download(KANJIDIC_URL, args.raw / "kanjidic2.xml.gz"))
    uh = unihan_definitions(args.raw)
    ids = json.loads((root / "ID_to_chinese.json").read_text(encoding="utf-8"))
    chars, captions, images, dropped = [], [], [], set()
    for dp, _, files in sorted(os.walk(root)):
        cid = os.path.basename(dp)
        files = sorted(f for f in files if f[0] in SOURCES and not f.endswith(".json"))
        if not files:
            continue
        c = ids.get(cid, "?")
        cap = ", ".join(kd[c]) if c in kd else unihan_caption(uh[c]) if c in uh else ""
        if not cap:
            dropped.add(cid); continue
        for f in files:
            g = glyph(Path(dp) / f, size)
            if g is not None:
                chars.append(c); captions.append(cap); images.append(g)
    n_o, n_k = len(images), len(kanji["chars"])
    print(f"oracle: {n_o} glyphs, {len(set(chars))} characters, {len(dropped)} classes without an English meaning dropped")
    # Each oracle character weighs the same whatever its number of drawings, and the oracle set as a whole gets
    # --oracle-share of the batches.
    per_char = Counter(chars)
    w_o = torch.tensor([1.0 / per_char[c] for c in chars])
    w_o *= args.oracle_share / w_o.sum()
    w_k = torch.full((n_k,), (1 - args.oracle_share) / n_k)
    out = {"chars": list(kanji["chars"]) + chars, "captions": list(kanji["captions"]) + captions,
           "images": torch.cat([kanji["images"], torch.from_numpy(np.stack(images))]), "size": size,
           "prefix": [""] * n_k + [PREFIX] * n_o, "weights": torch.cat([w_k, w_o])}
    torch.save(out, args.out)
    pick = np.random.default_rng(0).choice(n_o, 32, replace=False)
    Image.fromarray(np.concatenate([np.concatenate([images[i] for i in pick[r * 8:(r + 1) * 8]], 1) for r in range(4)], 0)).resize(
        (1024, 512)).save(args.out.with_suffix(".png"))
    print(f"{n_k} kanji + {n_o} oracle glyphs at {size}px, oracle share {args.oracle_share}: {args.out}")


if __name__ == "__main__":
    main()
