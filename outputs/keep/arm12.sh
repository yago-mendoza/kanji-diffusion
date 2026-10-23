#!/bin/bash
# Arm 12: oracle bone script (HUST-OBC, clean sources, 70% of each batch) mixed with enpitsu's kanji (30%), oracle
# captions prefixed "oracle bone script: ". Two starts with the same data, run one after the other:
#   12a from arm 11's U-Net at step 24,000 (already reads English through the dictionary),
#   12b from Stable Diffusion 1.5 (keeps more of what things look like, arm 6).
# Arm 9's recipe otherwise (256 px, cross-attention 1e-4, the rest 1e-5). U-Net and probes every 1,000 steps.
# Usage: bash arm12.sh STEPS   (the arm 11 U-Net must be in /workspace/init/arm11_24k)
# /workspace is a network volume: everything here outlives the pod (the first run lost its results with the pod's disk).
STEPS=${1:-8000}
cd /workspace
[ -d kanji-diffusion ] || git clone -q https://github.com/yago-mendoza/kanji-diffusion.git
cd kanji-diffusion && git pull -q && git log --oneline -1
pip install -q --break-system-packages -e . pyarrow 2>&1 | tail -1
[ -f data/kanji256_epts.pt ] || python -m kanji_diffusion.prepare_epts --size 256 --out data/kanji256_epts.pt
[ -f data/obc_mix256.pt ] || python -m kanji_diffusion.prepare_obc --kanji data/kanji256_epts.pt --out data/obc_mix256.pt
rm -rf data/raw/obc data/raw/epts-kanji-full.parquet   # 50 GB disk: keep room for 16 U-Nets
df -h /workspace | tail -1
COMMON="--data data/obc_mix256.pt --eval-data /none --steps $STEPS --lr 1e-5 --lr-xattn 1e-4 --constant --save-every 1000
  --save-unets --probe outputs/keep/obc_probe.txt --probe-guidance 7.5"
until [ -f /workspace/init/arm11_24k/.done ]; do sleep 5; done   # the U-Net upload runs alongside the data prep
python -m kanji_diffusion.finetune_sd $COMMON --out runs/arm12a --init-unet /workspace/init/arm11_24k
rm -f runs/arm12a/state.pt; echo ARM12A_DONE
python -m kanji_diffusion.finetune_sd $COMMON --out runs/arm12b
rm -f runs/arm12b/state.pt; echo ARM12B_DONE
# The comparison gallery, drawn here on the GPU before the pod stops: 30 words, 12 samples, both starts, four points each.
for r in arm12a arm12b; do for s in 002000 004000 006000 008000; do
  python outputs/keep/gallery_hf.py runs/$r/unets/step$s runs/gallery/$r-$s outputs/keep/obc_gallery.txt --n 12
done; done
tar cf /workspace/gallery.tar -C runs gallery && echo GALLERY_READY
