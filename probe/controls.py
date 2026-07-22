"""Positive / sanity controls for the probes.

A probe that reports "nothing there" is only trustworthy if the *same probe* can
recover something that IS there. These controls make a format / regime / capture
mismatch **fail loudly** instead of masquerading as a negative result.

- ``machinery_control``: probe for the relation word, which is literally present in
  the prompt "{name}'s {relation} is". A working capture+probe pipeline must
  recover it regardless of what the model learned; if it can't, the pipeline (not
  the model) is broken, and every null result is untrustworthy.
- ``fact_recovery_control``: on the DENSE arm, the attribute value should be
  linearly decodable if the model actually stored the fact and the prompt format
  matches training. Low bar (just beat chance). Use it before believing a
  "split shows nothing" contrast — if dense also shows nothing, suspect the setup.

Both reuse the same capture+probe path the real probes use, so they exercise the
exact code that would otherwise fail silently.
"""

from __future__ import annotations

import torch

from corpusgen.bios import RELATION_PHRASES, VALUE_POOLS
from evals.mechanism import capture_last_token, fit_linear_probe, probe_accuracy


def _probe_last_layer(model, tok, contexts, labels, n_classes, device,
                      seed: int = 0, test_frac: float = 0.3):
    resid = capture_last_token(model, tok, contexts, device)["resid"]  # [N, L, D]
    X = resid[:, -1, :]
    n = X.size(0)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(n * test_frac))
    te = torch.zeros(n, dtype=torch.bool); te[perm[:n_test]] = True; tr = ~te
    if int(tr.sum()) < 2 or int(te.sum()) < 1:
        return None, 1.0 / n_classes
    probe = fit_linear_probe(X[tr], labels[tr], n_classes, seed=seed)
    return probe_accuracy(probe, X[te], labels[te]), 1.0 / n_classes


def machinery_control(model, tok, records, device, per_attr: int = 16) -> dict:
    """Can the probe recover the RELATION that is verbatim in the prompt?

    Arm/training-independent: if this fails, the capture/probe/format is broken —
    NOT evidence about the model. Returns {kind, acc, chance, passed}.
    """
    attrs = list(RELATION_PHRASES)
    contexts, labels = [], []
    for i, rec in enumerate(records):
        a = attrs[i % len(attrs)]
        contexts.append(f"{rec.name}'s {RELATION_PHRASES[a]} is")
        labels.append(attrs.index(a))
        if len(contexts) >= per_attr * len(attrs):
            break
    acc, chance = _probe_last_layer(model, tok, contexts, torch.tensor(labels),
                                    len(attrs), device)
    return {"kind": "in_context_relation", "acc": acc, "chance": chance,
            "passed": bool(acc is not None and acc > max(0.5, 2 * chance))}


def fact_recovery_control(model, tok, records, device, attr: str = "major") -> dict:
    """DENSE-arm check: is the attribute value linearly decodable at all?

    A low bar (beat chance). If the dense model can't clear it, either it didn't
    learn the facts or the prompt format is off — either way, treat a
    "split stores nothing" contrast with caution. Returns {kind, acc, chance, passed}.
    """
    pool = VALUE_POOLS[attr]
    idx = {v: i for i, v in enumerate(pool)}
    contexts = [f"{r.name}'s {RELATION_PHRASES[attr]} is" for r in records]
    labels = torch.tensor([idx[r.attrs[attr]] for r in records])
    acc, chance = _probe_last_layer(model, tok, contexts, labels, len(pool), device)
    return {"kind": f"dense_fact_recovery[{attr}]", "acc": acc, "chance": chance,
            "passed": bool(acc is not None and acc > max(0.05, 2 * chance))}


def warn_if_failed(control: dict) -> bool:
    """Print a loud warning when a control fails; return True if it FAILED."""
    if not control.get("passed"):
        a = control.get("acc")
        a_str = "n/a" if a is None else f"{a:.3f}"
        print(f"⚠️  POSITIVE CONTROL FAILED [{control['kind']}]: recovered acc="
              f"{a_str} vs chance={control['chance']:.3f}. The probe pipeline "
              f"(capture/format), not necessarily the model, may be at fault — "
              f"treat null/absent results as UNTRUSTWORTHY until this passes.")
        return True
    return False
