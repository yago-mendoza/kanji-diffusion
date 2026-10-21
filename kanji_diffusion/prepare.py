"""Build the training set: KanjiVG strokes rendered to grayscale bitmaps, paired with
KANJIDIC2 English meanings, plus the frozen CLIP text embeddings of those meanings.

    python -m kanji_diffusion.prepare --size 64
"""

from __future__ import annotations

import argparse
import re
import gzip
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from svg.path import parse_path

from .text import TextEncoder

KANJIDIC_URL = "http://www.edrdg.org/kanjidic/kanjidic2.xml.gz"
KANJIVG_URL = "https://github.com/KanjiVG/kanjivg/releases/download/r20260714/kanjivg-20260714.xml.gz"
KVG_NS = "{http://kanjivg.tagaini.net}"
VIEWBOX = 109.0


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def read_meanings(path: Path) -> dict[str, list[str]]:
    """English meanings per character (KANJIDIC2 meanings without m_lang are English)."""
    root = ET.parse(gzip.open(path)).getroot()
    out: dict[str, list[str]] = {}
    for ch in root.iter("character"):
        meanings = [m.text for m in ch.iter("meaning") if "m_lang" not in m.attrib and m.text]
        if meanings:
            out[ch.findtext("literal")] = meanings
    return out


POSITIONS = {"left": "left", "right": "right", "top": "top", "bottom": "bottom",
             "tare": "top-left", "kamae": "around", "nyo": "bottom-left", "nyoc": "bottom-left"}


def _clean(meaning: str) -> str:
    """'pig's head radical variant (no. 58)' to 'pig's head': the text model reads words, not dictionary notes."""
    m = re.sub(r"\(no\.?\s*\d+\)", "", meaning)
    m = re.sub(r"\b(radical|variant|kanji)\b", "", m)
    return re.sub(r"\s+", " ", m).strip(" ,-") or meaning


def read_components(path: Path, meanings: dict[str, list[str]]) -> dict[str, str]:
    """A caption of each character's first-level parts and where they sit, e.g. 明 -> 'day (left) and month (right)'.

    Parts come from KanjiVG's decomposition; a part written in a variant form (氵) is named by the character it
    stands for (水), which KanjiVG records as kvg:original.
    """
    root = ET.parse(gzip.open(path)).getroot()

    def parts(g):
        out = []
        for c in g.findall("g"):
            el = c.get(KVG_NS + "element")
            if el:
                name = meanings.get(c.get(KVG_NS + "original") or el) or meanings.get(el)
                if name:
                    out.append((_clean(name[0]), POSITIONS.get(c.get(KVG_NS + "position") or "")))
            else:
                out += parts(c)
        return out

    out: dict[str, str] = {}
    for kanji in root.iter("kanji"):
        if "-" in kanji.get("id", ""):
            continue
        g = kanji.find("g")
        if g is None or not g.get(KVG_NS + "element"):
            continue
        ps = parts(g)
        if len(ps) >= 2:
            out[g.get(KVG_NS + "element")] = " and ".join(f"{m} ({p})" if p else m for m, p in ps)
    return out


def read_strokes(path: Path) -> dict[str, list[str]]:
    """SVG path data of every stroke, in drawing order, per character."""
    root = ET.parse(gzip.open(path)).getroot()
    out: dict[str, list[str]] = {}
    for kanji in root.iter("kanji"):
        kid = kanji.get("id", "")
        # Variant drawings carry a suffix (kvg:kanji_05b57-Kaisho); keep the standard one.
        if "-" in kid:
            continue
        g = kanji.find("g")
        if g is None:
            continue
        char = g.get(KVG_NS + "element") or chr(int(kid.split("_")[-1], 16))
        paths = [p.get("d") for p in kanji.iter("path") if p.get("d")]
        if paths:
            out[char] = paths
    return out


def render(paths: list[str], size: int, stroke: float = 3.4, supersample: int = 4) -> np.ndarray:
    """Black strokes on white, antialiased by drawing large and downsampling."""
    big = size * supersample
    scale = big / VIEWBOX
    img = Image.new("L", (big, big), 255)
    draw = ImageDraw.Draw(img)
    width = max(1, round(stroke * scale))
    for d in paths:
        # Sample each Bézier segment at fixed steps: path.length() integrates numerically
        # and made rendering the whole set take over half an hour.
        pts = []
        for seg in parse_path(d):
            if type(seg).__name__ == "Move":
                continue
            start = 1 if pts else 0  # segments share endpoints
            pts.extend(seg.point(i / 16) for i in range(start, 17))
        if not pts:
            continue
        xy = [(p.real * scale, p.imag * scale) for p in pts]
        draw.line(xy, fill=0, width=width, joint="curve")
        r = width / 2
        for x, y in (xy[0], xy[-1]):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=0)
    return np.asarray(img.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/kanji64.pt"))
    ap.add_argument("--text-model", default=TextEncoder.DEFAULT)
    ap.add_argument("--components", action="store_true",
                    help="add a third caption naming the character's parts, e.g. 'bright, made of day (left) and month (right)'")
    args = ap.parse_args()

    meanings = read_meanings(download(KANJIDIC_URL, args.raw / "kanjidic2.xml.gz"))
    strokes = read_strokes(download(KANJIVG_URL, args.raw / "kanjivg.xml.gz"))
    chars = sorted(c for c in strokes if c in meanings)
    print(f"{len(chars)} characters with both strokes and English meanings")

    with ProcessPoolExecutor() as pool:
        images = np.stack(list(pool.map(partial(render, size=args.size), [strokes[c] for c in chars], chunksize=64)))
    print(f"rendered {len(images)} images", flush=True)

    # Two captions per character: all meanings, and the first one alone, so short
    # prompts at sampling time are in distribution.
    full = [", ".join(meanings[c]) for c in chars]
    first = [meanings[c][0] for c in chars]
    enc = TextEncoder(args.text_model)
    emb_full = enc.encode(full)
    emb_first = enc.encode(first)
    emb_null = enc.encode([""])[0]
    variants = [emb_full, emb_first]
    comp_caps = None
    if args.components:
        comps = read_components(args.raw / "kanjivg.xml.gz", meanings)
        comp_caps = [f"{first[i]}, made of {comps[c]}" if c in comps else first[i] for i, c in enumerate(chars)]
        print(f"{sum(c in comps for c in chars)} characters get a component caption, e.g. {comp_caps[chars.index('明')]}")
        variants.append(enc.encode(comp_caps))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "chars": chars,
            "captions": full,
            "images": torch.from_numpy(images),
            "emb": torch.stack(variants, dim=1).half(),  # (N, 2 or 3, T, D)
            "component_captions": comp_caps,
            "emb_null": emb_null.half(),
            "text_model": args.text_model,
            "size": args.size,
        },
        args.out,
    )
    preview = Image.fromarray(np.concatenate([np.concatenate(images[i * 16:(i + 1) * 16], 1) for i in range(8)], 0))
    preview.save(args.out.with_suffix(".preview.png"))
    print(f"saved {args.out} ({images.shape}) and preview")


if __name__ == "__main__":
    main()
