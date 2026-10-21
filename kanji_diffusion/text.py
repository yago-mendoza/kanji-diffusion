"""Frozen CLIP text encoder shared by data preparation and sampling."""

from __future__ import annotations

import torch
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder:
    DEFAULT = "openai/clip-vit-base-patch32"
    MAX_TOKENS = 32  # KANJIDIC meaning lists rarely run longer

    def __init__(self, name: str = DEFAULT, device: str = "cpu"):
        self.tokenizer = CLIPTokenizer.from_pretrained(name)
        self.model = CLIPTextModel.from_pretrained(name).to(device).eval()
        self.device = device
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts: list[str], batch: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), batch):
            tok = self.tokenizer(
                texts[i:i + batch],
                padding="max_length",
                truncation=True,
                max_length=self.MAX_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            out.append(self.model(**tok).last_hidden_state.float().cpu())
        return torch.cat(out)
