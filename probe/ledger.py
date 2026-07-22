"""Q3 — fact-information ledger + capacity scaling (pure aggregation).

- ``fact_info_ledger`` (NEW-PROBES 3.4): reconcile recall-bits, probe-bits (NR-5),
  and MC-recognition (NR-4) into one per-arm accounting; the recall→probe gap is
  the L11 extractability confound made explicit.
- ``capacity_scaling`` (3.5): measured fact-bits vs params (160M→1B dense) → does
  storage track ~2 bits/param on our stack (anchors the memorization wall).
"""

from __future__ import annotations

import math


def fact_info_ledger(
    recall_bits: float | None = None,
    probe_bits: float | None = None,
    mc_recall_acc: float | None = None,
    n_entities: int | None = None,
) -> dict:
    """One reconciled fact-storage statement for an arm.

    ``extractability_gap = probe_bits - recall_bits`` (>0 ⇒ facts stored but not
    generatively extractable — L11). ``bits_per_entity`` normalizes by N when given.
    All fields optional; only what's provided is reported.
    """
    out: dict = {}
    if recall_bits is not None:
        out["recall_bits"] = float(recall_bits)
    if probe_bits is not None:
        out["probe_bits"] = float(probe_bits)
    if mc_recall_acc is not None:
        out["mc_recall_acc"] = float(mc_recall_acc)
    if recall_bits is not None and probe_bits is not None:
        out["extractability_gap"] = float(probe_bits - recall_bits)
    if n_entities:
        for k in ("recall_bits", "probe_bits"):
            if k in out:
                out[f"{k}_per_entity"] = out[k] / n_entities
    return out


def capacity_scaling(points: list[dict]) -> dict:
    """Fact-bits vs params across runs (dense) → bits-per-param per point + a fit.

    ``points``: [{name, params, bits}]. Returns per-point ``bits_per_param`` and a
    log-log slope (bits ∝ params^slope; ~1.0 ⇒ linear/2-bit-per-param regime). The
    2-bit/param law predicts bits_per_param ≈ const across scale.
    """
    pts = [p for p in points if p.get("params") and p.get("bits") is not None]
    per_point = [
        {"name": p.get("name"), "params": p["params"], "bits": p["bits"],
         "bits_per_param": p["bits"] / p["params"]}
        for p in pts
    ]
    slope = None
    if len(pts) >= 2:
        xs = [math.log(p["params"]) for p in pts]
        ys = [math.log(p["bits"]) for p in pts if p["bits"] > 0]
        if len(ys) == len(xs) and len(xs) >= 2:
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            var = sum((x - mx) ** 2 for x in xs)
            slope = cov / var if var > 0 else None
    return {"per_point": per_point, "loglog_slope": slope}
