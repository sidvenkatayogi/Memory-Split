"""Q1 (GATED) — step-localization via activation patching (NEW-PROBES 1.5).

Which layer/position causally carries the reasoning result. Patch a layer's
residual from a *clean* (correct-operand) run into a *corrupted* (wrong-operand)
run and measure the recovery of the gold answer logit — using a CONTINUOUS metric
(Δ log p(gold)) so it works below the argmax threshold.

⚠️ GATED: informative only once the model reasons above chance (at chance there is
no clean/corrupt behavioral gap to trace). Thin wrapper over
``interp.activation_patch_logits``; runs today (tested on toy), low-signal until P0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from evals.interp import activation_patch_logits


def step_patch_effect(model, tok, clean_prompt, corrupt_prompt, layer,
                      gold_answer, device, positions=None) -> dict:
    """Δ log p(gold first token) from patching ``layer`` clean→corrupt.

    Returns {clean_logp, corrupt_baseline_logp, patched_logp, recovery} where
    recovery = patched − corrupt_baseline (how much patching that layer restores
    the gold answer). ``clean_prompt`` should induce the right answer,
    ``corrupt_prompt`` a wrong one (e.g. one operand changed).
    """
    gid = (tok.encode(" " + gold_answer) or [tok.EOT])[0]
    out = activation_patch_logits(model, tok, clean_prompt, corrupt_prompt,
                                  layer, positions=positions, device=device)
    clean_lp = F.log_softmax(out["clean"].float(), dim=-1)[gid].item()
    patched_lp = F.log_softmax(out["patched"].float(), dim=-1)[gid].item()
    # corrupt baseline = clean forward of the corrupt prompt (no patch)
    ids = torch.tensor([tok.encode(corrupt_prompt) or [tok.EOT]], device=device)
    with torch.no_grad():
        logits, _ = model.forward(ids)
    corrupt_lp = F.log_softmax(logits[0, -1].float(), dim=-1)[gid].item()
    return {
        "clean_logp": clean_lp,
        "corrupt_baseline_logp": corrupt_lp,
        "patched_logp": patched_lp,
        "recovery": patched_lp - corrupt_lp,
        "layer": layer,
    }
