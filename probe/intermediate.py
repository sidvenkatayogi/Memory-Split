"""Q1 — intermediate-quantity probe (NEW-PROBES 1.1). iGSM-specific.

Beyond the final answer (NR-7), probe whether the model computes the *intermediate*
results along the reasoning chain. For each "The number of … is (a op b) mod 23 =
{val}." step in the gold CoT, build a context truncated **before** the value
(value not yet in context ⇒ non-copy) and probe that value (0-22) from each layer's
hidden state. Above-chance decodability = genuine step-by-step computation (the
finest 'hidden reasoning' signal, Physics 2.1).

Reuses ``mechanism.{capture_last_token,fit_linear_probe,probe_accuracy}``.
"""

from __future__ import annotations

import math
import re

import torch

from evals.mechanism import capture_last_token, fit_linear_probe, probe_accuracy

_STEP = re.compile(r"= (\d+)\.")


def _intermediate_contexts(item):
    """[(context_ending_before_value, value_int)] for each CoT compute step.

    Context = prompt + solution up to and including the '=' of a compute step
    (value withheld). Excludes the final step whose value == the answer only if
    it coincides with the 'Answer:' copy — kept here since the value is still
    withheld at the '=' slot (non-copy). Requires meta['solution'].
    """
    sol = item.meta.get("solution")
    if not sol:
        return []
    out = []
    for m in _STEP.finditer(sol):
        # truncate the solution right after '=' (before the space+value)
        cut = m.start() + 1  # include '='
        ctx = item.prompt + sol[:cut]
        out.append((ctx, int(m.group(1))))
    return out


def intermediate_value_probe(
    model, tok, items, device, batch_size: int = 16, test_frac: float = 0.3, seed: int = 0,
) -> dict | None:
    """Per-layer linear-probe accuracy for intermediate step values (mod 23).

    Pools all compute-step slots across items, fits a held-out probe per layer for
    the value class, and reports ``acc_by_layer`` vs ``chance=1/n_classes``,
    ``best_layer/acc``, ``decodable``. Non-copy (value withheld at the probe slot).
    Returns None if too few slots.
    """
    contexts, values = [], []
    for it in items:
        for ctx, val in _intermediate_contexts(it):
            contexts.append(ctx)
            values.append(val)
    if len(contexts) < 8:
        return None
    classes = sorted(set(values))
    if len(classes) < 2:
        return None
    cls_idx = {c: i for i, c in enumerate(classes)}
    labels = torch.tensor([cls_idx[v] for v in values])
    n_classes = len(classes)

    resid = capture_last_token(model, tok, contexts, device, batch_size)["resid"]  # [N,L,D]
    N, L, _ = resid.shape
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_test = max(1, int(N * test_frac))
    te = torch.zeros(N, dtype=torch.bool); te[perm[:n_test]] = True; tr = ~te
    if int(tr.sum()) < 2 or int(te.sum()) < 1:
        return None

    acc_by_layer = {}
    for layer in range(L):
        X = resid[:, layer, :]
        probe = fit_linear_probe(X[tr], labels[tr], n_classes, seed=seed)
        acc_by_layer[layer] = probe_accuracy(probe, X[te], labels[te])
    best = max(acc_by_layer, key=acc_by_layer.get)
    chance = 1.0 / n_classes
    se = math.sqrt(max(acc_by_layer[best] * (1 - acc_by_layer[best]), 1e-9) / int(te.sum()))
    return {
        "acc_by_layer": acc_by_layer, "chance": chance, "n_classes": n_classes,
        "best_layer": int(best), "best_acc": float(acc_by_layer[best]),
        "n_slots": N, "n_test": int(te.sum()),
        "decodable": bool(acc_by_layer[best] - chance > max(0.05, 2 * se)),
    }
