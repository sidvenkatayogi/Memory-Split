"""Q2/Q3 — representational & weight geometry (pure, no model needed at call time).

- ``linear_cka`` / ``cross_arm_cka`` (NEW-PROBES 2.1): where dense and split diverge.
- ``effective_rank`` / ``weight_spectral`` (3.3): whether removing memorization
  lowers the rank/norm dense spends on facts.
- ``topk_neuron_overlap`` (2.4): are fact-neurons and reasoning-neurons disjoint;
  are dense's fact-neurons repurposed in split.
- ``subspace_overlap`` (2.2): overlap of a fact-probe vs reasoning-probe subspace.

All inputs are plain tensors/arrays (activations from ``mechanism.capture_last_token``
or probe weight matrices), so these are deterministic and unit-testable offline.
"""

from __future__ import annotations

import torch


def _center(X: torch.Tensor) -> torch.Tensor:
    return X - X.mean(dim=0, keepdim=True)


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two [N, D] activation matrices (same N rows/inputs).

    CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F), columns centered.
    In [0, 1]; 1 iff representations are equal up to orthogonal transform + scale;
    invariant to isotropic scaling and orthonormal rotation. [MT]
    """
    Xc, Yc = _center(X.float()), _center(Y.float())
    hsic_xy = (Yc.T @ Xc).pow(2).sum()
    hsic_xx = (Xc.T @ Xc).norm()
    hsic_yy = (Yc.T @ Yc).norm()
    denom = hsic_xx * hsic_yy
    return float(hsic_xy / denom) if denom > 0 else 0.0


def cross_arm_cka(acts_a: torch.Tensor, acts_b: torch.Tensor) -> dict[int, float]:
    """Per-layer CKA between two arms' ``[N, L, D]`` activations on shared inputs.

    Low CKA at a layer ⇒ the arms represent those inputs differently there — the
    layers where 'freed capacity' manifests. Report alongside per-layer weight
    deltas for a fuller picture.
    """
    assert acts_a.shape == acts_b.shape, "arms must share [N, L, D]"
    return {l: linear_cka(acts_a[:, l, :], acts_b[:, l, :]) for l in range(acts_a.size(1))}


def effective_rank(W: torch.Tensor) -> float:
    """Effective rank = exp(entropy of the normalized singular-value distribution).

    1 for a rank-1 matrix; approaches min(shape) for an isotropic one. A scalar
    proxy for 'how much of the matrix's capacity is used'. [MT]
    """
    s = torch.linalg.svdvals(W.float())
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return float(torch.exp(entropy))


def weight_spectral(state_dict: dict, keys=("mlp.w1", "mlp.w2", "mlp.w3")) -> dict:
    """Per-layer effective-rank + Frobenius norm for the named weight matrices.

    Feed two arms' state_dicts and diff to test whether the split arm's MLP
    weights are leaner (lower rank/norm) than dense's fact-laden ones (3.3).
    Returns {layer: {key: {eff_rank, fro_norm}}}.
    """
    import re
    out: dict[int, dict] = {}
    for name, W in state_dict.items():
        m = re.match(r"blocks\.(\d+)\.(.+)\.weight$", name)
        if not m or W.dim() != 2:
            continue
        layer, sub = int(m.group(1)), m.group(2)
        if any(sub.endswith(k) for k in keys):
            out.setdefault(layer, {})[sub] = {
                "eff_rank": effective_rank(W), "fro_norm": float(W.float().norm())
            }
    return out


def topk_neuron_overlap(sel_a: torch.Tensor, sel_b: torch.Tensor, k: int) -> dict:
    """Jaccard overlap of the top-k |selectivity| neurons of two maps ([L, H]).

    Use for: fact-selectivity vs reasoning-selectivity (disjoint? 2.4), or dense
    fact-neurons vs their split selectivity (repurposed?). Returns
    {overlap, n_a, n_b, n_shared}.
    """
    def topset(sel):
        flat = sel.reshape(-1).abs()
        idx = torch.topk(flat, min(k, flat.numel())).indices.tolist()
        return set(idx)
    A, B = topset(sel_a), topset(sel_b)
    inter = len(A & B)
    union = len(A | B)
    return {"overlap": inter / union if union else 0.0,
            "n_a": len(A), "n_b": len(B), "n_shared": inter}


def subspace_overlap(A: torch.Tensor, B: torch.Tensor) -> float:
    """Normalized overlap of the column spaces of A [d, ra] and B [d, rb].

    Orthonormalize each, then mean squared cosine of principal angles
    = ||Qa^T Qb||_F^2 / min(ra, rb) ∈ [0, 1]. 1 ⇒ nested/equal subspaces, 0 ⇒
    orthogonal. Use with fact-probe vs reasoning-probe weight matrices (2.2). [MT]
    """
    Qa, _ = torch.linalg.qr(A.float())
    Qb, _ = torch.linalg.qr(B.float())
    r = min(Qa.size(1), Qb.size(1))
    if r == 0:
        return 0.0
    return float((Qa.T @ Qb).pow(2).sum() / r)
