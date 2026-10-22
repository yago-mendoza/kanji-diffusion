"""enpitsu's own training set (Hugging Face epts/kanji-full) in this repo's format, for the faithful reproduction (arm 7).

    python -m kanji_diffusion.prepare_epts --size 512 --out data/kanji512_epts.pt

10,358 Mincho glyphs at 512 px, black on white, captioned with the raw KANJIDIC2 meanings ("Asia, rank next,
come after, -ous"). The set carries no characters, only images and captions; each caption is matched back to its
character through KANJIDIC2 so later analysis can name what the model drew. Unmatched captions keep "?".
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from .prepare import KANJIDIC_URL, download, read_meanings

PARQUET = ("https://huggingface.co/datasets/epts/kanji-full/resolve/main/data/"
           "train-00000-of-00001-7a6d03bbb08bdf94.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/kanji512_epts.pt"))
    args = ap.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    src = args.raw / "epts-kanji-full.parquet"
    if not src.exists():
        urllib.request.urlretrieve(PARQUET, src)
    table = pq.read_table(src).to_pydict()
    captions = table["text"]
    images = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("L").resize((args.size, args.size), Image.LANCZOS))
                       for im in table["image"]])
    by_caption = {", ".join(m): c for c, m in read_meanings(download(KANJIDIC_URL, args.raw / "kanjidic2.xml.gz")).items()}
    chars = [by_caption.get(c, "?") for c in captions]
    print(f"{len(captions)} glyphs at {args.size}px, {sum(c != '?' for c in chars)} matched to a character")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"chars": chars, "captions": captions, "images": torch.from_numpy(images), "size": args.size}, args.out)
    Image.fromarray(np.concatenate([np.concatenate(list(images[i * 8:(i + 1) * 8]), 1) for i in range(4)], 0)).resize(
        (1024, 512)).save(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
