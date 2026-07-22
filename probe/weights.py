"""Q3/Q2 — weight-level fact storage & causal separation.

- ``fact_weight_attribution`` (NEW-PROBES 3.1): first-order attribution
  |grad ⊙ W| of a fact's recall NLL onto each layer's MLP weights → *which
  parameters* store the fact. Dense localizes; split should be diffuse/absent.
- ``double_dissociation`` (2.3): ablate the dense arm's top memorization neurons
  and measure recall vs a reasoning scorer — the fact-side half of the
  double-dissociation (reasoning-side ablation is gated on reasoning > chance).
- ``participation_ratio`` / ``fact_superposition`` (3.2): effective dimensionality
  of fact activations → how superposed/crowded storage is (sweep across dose N).
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from evals.interp import ablate_mlp_neurons


def fact_weight_attribution(model, tok, prompt: str, answer: str, device,
                            keys=("mlp.w1", "mlp.w2", "mlp.w3")) -> dict:
    """Per-layer |grad ⊙ W| attribution of NLL(answer | prompt) onto MLP weights.

    Teacher-forces ``prompt``+``answer``, backprops the answer NLL, and returns
    {layer: attribution_sum} over the named weight matrices. High, concentrated
    attribution = the weights holding that fact. Compare dense (localized) vs split
    (diffuse/near-zero). Enables grad (no no_grad).
    """
    ctx = tok.encode(prompt) or [tok.EOT]
    ans = tok.encode(" " + answer)
    if not ans:
        return {}
    ids = torch.tensor([ctx + ans], dtype=torch.long, device=device)
    model.zero_grad(set_to_none=True)
    logits, _ = model.forward(ids)
    logp = F.log_softmax(logits.float(), dim=-1)
    # predict answer token j from position (len(ctx)-1 + j)
    c = len(ctx)
    pos = torch.arange(c - 1, c - 1 + len(ans), device=device)
    tgt = torch.tensor(ans, device=device)
    nll = -logp[0, pos, tgt].sum()
    nll.backward()

    per_layer: dict[int, float] = {}
    for name, p in model.named_parameters():
        m = re.match(r"blocks\.(\d+)\.(.+)\.weight$", name)
        if not m or p.grad is None:
            continue
        sub = m.group(2)
        if any(sub.endswith(k) for k in keys):
            layer = int(m.group(1))
            attr = (p.grad * p.detach()).abs().sum().item()
            per_layer[layer] = per_layer.get(layer, 0.0) + attr
    model.zero_grad(set_to_none=True)
    return per_layer


def double_dissociation(model, fact_neurons, recall_scorer, reasoning_scorer) -> dict:
    """Ablate ``fact_neurons`` (from ``mechanism.top_neurons``) and measure both
    scorers. Fact-side of the dissociation: recall should drop sharply while
    reasoning is ~unchanged (facts and reasoning use different units).

    ``recall_scorer``/``reasoning_scorer``: callables ``model->float``. Returns
    baselines, ablated values, and deltas (baseline−ablated) for each.
    """
    neurons = [(l, n) for l, n, *_ in fact_neurons]
    rec0, rea0 = float(recall_scorer(model)), float(reasoning_scorer(model))
    with ablate_mlp_neurons(model, neurons):
        rec1, rea1 = float(recall_scorer(model)), float(reasoning_scorer(model))
    return {
        "recall": {"baseline": rec0, "ablated": rec1, "delta": rec0 - rec1},
        "reasoning": {"baseline": rea0, "ablated": rea1, "delta": rea0 - rea1},
        "dissociation": (rec0 - rec1) - (rea0 - rea1),
    }


def participation_ratio(acts: torch.Tensor) -> float:
    """Effective dimensionality of ``[N, D]`` activations:
    PR = (Σλ)² / Σλ² over covariance eigenvalues. In [1, D]; low ⇒ activity packed
    into few directions (more superposition). [MT]
    """
    X = acts.float()
    X = X - X.mean(0, keepdim=True)
    cov = (X.T @ X) / max(1, X.size(0) - 1)
    lam = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1, s2 = lam.sum(), (lam * lam).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def fact_superposition(acts: torch.Tensor) -> dict:
    """Superposition summary for fact activations ``[N, D]``: participation ratio
    and its fraction of full dim. Sweep across dose N: lower PR at higher N ⇒
    facts increasingly share directions (crowding). """
    D = acts.size(1)
    pr = participation_ratio(acts)
    return {"participation_ratio": pr, "dim": D, "pr_fraction": pr / D if D else 0.0}
