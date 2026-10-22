"""LoRA on the U-Net's attention, written out so it needs nothing beyond torch.

Each query, key, value and output projection of every attention layer (self and cross) keeps its pretrained
weight frozen and gains a low-rank update B·A beside it. Only A and B train, so Stable Diffusion's own weights,
and what they know about how things look, stay as they were.
"""
import copy

import torch
from torch import nn



class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        w = base.weight
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=w.device, dtype=w.dtype)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=w.device, dtype=w.dtype)
        nn.init.normal_(self.lora_down.weight, std=1 / rank)
        nn.init.zeros_(self.lora_up.weight)  # starts as the identity: step 0 is plain Stable Diffusion
        self.scale = alpha / rank

    def forward(self, x, *args, **kwargs):
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale


def _swap(model: nn.Module, fn):
    """Apply fn to the four projections of every transformer attention (attn1 reads the image, attn2 the caption)."""
    for name, attn in list(model.named_modules()):
        if not name.endswith(("attn1", "attn2")):
            continue
        for leaf in ("to_q", "to_k", "to_v"):
            setattr(attn, leaf, fn(getattr(attn, leaf)))
        attn.to_out[0] = fn(attn.to_out[0])


def add_lora(unet: nn.Module, rank: int, alpha: float) -> list[nn.Parameter]:
    """Freeze the U-Net and wrap its attention projections; returns the parameters to train."""
    unet.requires_grad_(False)
    _swap(unet, lambda m: LoRALinear(m, rank, alpha) if isinstance(m, nn.Linear) else m)
    return [p for n, p in unet.named_parameters() if "lora_" in n]


def fused(unet: nn.Module) -> nn.Module:
    """A copy with every update folded into its weight: a plain U-Net that save_pretrained and pipelines accept."""
    m = copy.deepcopy(unet)

    def fold(layer):
        if not isinstance(layer, LoRALinear):
            return layer
        delta = (layer.lora_up.weight.float() @ layer.lora_down.weight.float()) * layer.scale
        layer.base.weight.data += delta.to(layer.base.weight.dtype)
        return layer.base

    _swap(m, fold)
    return m


def lora_state(unet: nn.Module) -> dict:
    return {n: p.detach().to("cpu", torch.float16) for n, p in unet.named_parameters() if "lora_" in n}
