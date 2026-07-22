"""Q2 — where an external fact value enters the residual stream (NEW-PROBES 2.5).

Proxy for the split arm's inference-time value injection: contrast the residual
when the fact value IS present in context vs ABSENT, per layer, and see where the
representation diverges — i.e. the layer(s) at which the retrieved value is
incorporated. Complements NR-5 (which localizes stored-fact neurons in dense): this
shows how the split arm instead *splices* the value from the store.

Reuses ``mechanism.capture_last_token``. Forward passes only.
"""

from __future__ import annotations

import torch

from corpusgen.bios import RELATION_PHRASES
from evals.mechanism import capture_last_token


def residual_diff_by_layer(model, tok, ctx_with, ctx_without, device, batch_size=16) -> dict:
    """Per-layer mean L2 residual difference between paired contexts (last token).

    ``ctx_with[i]`` and ``ctx_without[i]`` differ only by the presence of the fact
    value. Returns {layer: mean ||h_with − h_without||}. A rise at layer ℓ marks
    where the value's presence propagates into the stream.
    """
    assert len(ctx_with) == len(ctx_without)
    rw = capture_last_token(model, tok, ctx_with, device, batch_size)["resid"]   # [N,L,D]
    ro = capture_last_token(model, tok, ctx_without, device, batch_size)["resid"]
    diff = (rw - ro).norm(dim=-1)  # [N, L]
    return {l: float(diff[:, l].mean()) for l in range(diff.size(1))}


def value_injection_profile(model, tok, records, device, attrs=None, batch_size=16) -> dict:
    """Where fact values enter the stream, over a set of records.

    Builds paired contexts "{name}'s {rel} is {value}" (with) vs "{name}'s {rel} is"
    (without) for each (entity, attr) and returns
    {profile: {layer: mean L2 diff}, n, peak_layer}. The peak layer is a candidate
    'splice point' for how externalized values are incorporated.
    """
    attrs = attrs or tuple(RELATION_PHRASES)
    ctx_with, ctx_without = [], []
    for rec in records:
        for a in attrs:
            base = f"{rec.name}'s {RELATION_PHRASES[a]} is"
            ctx_without.append(base)
            ctx_with.append(base + f" {rec.attrs[a]}")
    if not ctx_with:
        return {"profile": {}, "n": 0, "peak_layer": None}
    profile = residual_diff_by_layer(model, tok, ctx_with, ctx_without, device, batch_size)
    peak = max(profile, key=profile.get) if profile else None
    return {"profile": profile, "n": len(ctx_with), "peak_layer": peak}
