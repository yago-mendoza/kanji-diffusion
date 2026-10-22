"""The kanji whose dictionary meaning sits nearest a word in SD's own text space (CLIP ViT-L/14, pooled).

    python outputs/keep/nearest.py runs/arm5/pipeline data/kanji64_font.pt outputs/keep/ref_words.txt [--k 6]

A test of the mechanism behind enpitsu's fake kanji with no diffusion at all: if a model learns caption to glyph,
a new word should land near the glyphs of its nearest captions (王, "king, rule, magnate", for Elon Musk).
"""
import argparse, sys
from pathlib import Path

import torch
from transformers import CLIPTextModel, CLIPTokenizer

sys.stdout.reconfigure(encoding="utf-8")
ap = argparse.ArgumentParser()
ap.add_argument("pipe", type=Path)
ap.add_argument("data", type=Path)
ap.add_argument("words", type=Path)
ap.add_argument("--k", type=int, default=6)
args = ap.parse_args()

d = torch.load(args.data)
chars, caps = d["chars"], d["captions"]
tok = CLIPTokenizer.from_pretrained(args.pipe, subfolder="tokenizer")
te = CLIPTextModel.from_pretrained(args.pipe, subfolder="text_encoder").float().eval()


@torch.no_grad()
def emb(texts):
    out = []
    for i in range(0, len(texts), 256):
        ids = tok(texts[i:i + 256], padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
        out.append(torch.nn.functional.normalize(te(ids).pooler_output, dim=-1))
    return torch.cat(out)


# Each meaning on its own: a caption with many senses ("Asia, rank next, come after") should match on any one.
senses, owner = [], []
for j, c in enumerate(caps):
    for s in c.split(", "):
        senses.append(s); owner.append(j)
E = emb(senses)
owner = torch.tensor(owner)
words = [w.strip() for w in args.words.read_text(encoding="utf-8").splitlines() if w.strip()]
for w, s in zip(words, emb(words) @ E.T):
    best = torch.full((len(chars),), -1.0).scatter_reduce(0, owner, s, "amax")
    v, i = best.topk(args.k)
    print(f"{w:22s}", "  ".join(f"{chars[j]} {caps[j].split(', ')[0][:14]}" for j in i.tolist()))
