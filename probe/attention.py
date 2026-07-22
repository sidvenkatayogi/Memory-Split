"""Q1 — attention dependency-tracing + head ablation (NEW-PROBES 1.4).

- ``ablate_heads`` / ``head_ablation_effect``: zero specific attention heads and
  measure the causal effect on any scorer (which heads implement dependency
  lookup). Analogue of ``interp.ablate_mlp_neurons`` for heads.
- ``attention_weights``: recover per-head attention matrices for one layer
  (the model uses fused SDPA which hides them, so we recompute
  softmax(QKᵀ/√d) with RoPE) — inspect whether heads attend query→needed-operands.

Forward hooks only; no edit to ``train/model.py``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager

import torch

from train.model import _apply_rope


@contextmanager
def ablate_heads(model, heads):
    """Zero the output of specific attention heads for the context's duration.

    ``heads``: iterable of ``(layer, head)``. Head ``h`` occupies columns
    ``[h*head_dim:(h+1)*head_dim]`` of the attention output fed to ``attn.wo``;
    a forward_pre_hook on ``wo`` zeros them (clone avoids autograd hazards).
    """
    hd = model.cfg.head_dim
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, head in heads:
        by_layer[layer].append(head)
    handles = []

    def mk(hs):
        cols = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in sorted(set(hs))])

        def pre_hook(_m, inp):
            x = inp[0].clone()
            x[..., cols] = 0.0
            return (x,) + tuple(inp[1:])
        return pre_hook

    try:
        for layer, hs in by_layer.items():
            handles.append(model.blocks[layer].attn.wo.register_forward_pre_hook(mk(hs)))
        yield
    finally:
        for h in handles:
            h.remove()


def head_ablation_effect(model, heads, scorer) -> dict:
    """``scorer(model)->float`` with vs without the heads ablated (baseline−ablated)."""
    baseline = float(scorer(model))
    with ablate_heads(model, heads):
        ablated = float(scorer(model))
    return {"baseline": baseline, "ablated": ablated, "delta": baseline - ablated}


def attention_weights(model, tok, prompt, layer, device) -> torch.Tensor:
    """Per-head causal attention matrices ``[n_head, T, T]`` for one layer.

    Recomputes ``softmax(QKᵀ/√d)`` (with RoPE) from the layer's normed input,
    since the model's fused SDPA does not expose weights. Rows sum to 1 over the
    causal prefix. Read: does head h put mass from the query token onto its
    needed-operand tokens?
    """
    ids = tok.encode(prompt) or [tok.EOT]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    blk = model.blocks[layer]
    cap: dict = {}

    def pre_hook(_m, inp):
        cap["x"] = inp[0].detach()

    h = blk.attn.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            model.forward(x)
    finally:
        h.remove()

    xin = cap["x"]  # [1, T, d]
    B, T, _ = xin.shape
    nh, hd = model.cfg.n_head, model.cfg.head_dim
    attn = blk.attn
    q = attn.wq(xin).view(B, T, nh, hd).transpose(1, 2)  # [1,H,T,hd]
    k = attn.wk(xin).view(B, T, nh, hd).transpose(1, 2)
    cos, sin = model.rope_cos[:T].to(xin.device), model.rope_sin[:T].to(xin.device)
    q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # [1,H,T,T]
    mask = torch.triu(torch.ones(T, T, device=xin.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1)[0]  # [H,T,T]
