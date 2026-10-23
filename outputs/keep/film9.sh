#!/bin/bash
# Arm 9 again, filmed: the probe words drawn every 25 steps to 10,000, so the turn from photograph to invented
# character to dictionary kanji can be shown as continuous frames. Runs after arm 12 on the same pod.
cd /workspace/kanji-diffusion
until grep -qE "GALLERY_READY|Traceback" /workspace/arm12.log; do sleep 30; done
git pull -q && git log --oneline -1
python - <<'PY'
import re, gzip, torch
joyo = set()
for ch in gzip.open("data/raw/kanjidic2.xml.gz", "rt", encoding="utf-8").read().split("<character>")[1:]:
    lit = re.search(r"<literal>(.+?)</literal>", ch).group(1); gr = re.search(r"<grade>(\d+)</grade>", ch)
    if gr and int(gr.group(1)) <= 8: joyo.add(lit)
d = torch.load("data/kanji256_epts.pt"); ix = [i for i, c in enumerate(d["chars"]) if c in joyo]
print(len(ix), "joyo glyphs at 256")
torch.save({"chars": [d["chars"][i] for i in ix], "captions": [d["captions"][i] for i in ix], "images": d["images"][ix], "size": d["size"]}, "data/epts_joyo256.pt")
PY
python -m kanji_diffusion.finetune_sd --data data/epts_joyo256.pt --eval-data /none --out runs/film9 \
  --steps 10000 --lr 1e-5 --lr-xattn 1e-4 --constant --save-every 2000 \
  --probe outputs/keep/film_words.txt --probe-guidance 7.5 --probe-every 25
rm -f runs/film9/state.pt
tar cf /workspace/film9.tar -C runs/film9 probe && echo FILM_DONE
