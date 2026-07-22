#!/usr/bin/env python
"""Rigorous split-vs-dense comparison on DEDUCTION (the learnable reasoning task) = the H1 test.

Uses the battery's per-item results (paired: same 1500 eval items for both arms), so we can run
a paired McNemar test + a bootstrap CI on the accuracy difference, per fact load. Deduction is
above its 0.50 yes/no floor, so unlike iGSM this actually tests "does the split model reason
better?" (H1). Reports per-depth too.

IMPORTANT caveat (printed): this quantifies EVAL-ITEM sampling variance. The variance that
matters for "split reasons better" is across TRAINING SEEDS, and we have one seed per arm — so
even a significant per-item gap could be a single-run fluke. Multi-seed is required to claim a
real direction.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

D = Path("outputs/_frozen_data/ded_results")
LOADS = ["n50k", "n200k", "n800k"]


def load(load_, arm):
    rows = {}
    for line in open(D / f"{load_}_{arm}.jsonl"):
        r = json.loads(line)
        rows[r["qid"]] = (bool(r["correct"]), r["meta"].get("depth"))
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial p at 0.5
    cdf = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)


def boot_ci(pairs, B=20000, seed=0):
    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    idx = range(n)
    # precompute per-item split-dense
    sd = [int(s) - int(d) for (d, s) in pairs]
    for _ in range(B):
        acc = 0
        for _ in range(n):
            acc += sd[rng.randrange(n)]
        diffs.append(acc / n)
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def main():
    print("Split-vs-dense on DEDUCTION (paired, per fact load). '+' = split better.\n")
    out = {}
    for L in LOADS:
        dz, sz = load(L, "dense"), load(L, "split")
        qids = [q for q in dz if q in sz]
        d = [dz[q][0] for q in qids]
        s = [sz[q][0] for q in qids]
        n = len(qids)
        acc_d, acc_s = sum(d) / n, sum(s) / n
        b = sum(1 for i in range(n) if d[i] and not s[i])   # dense right, split wrong
        c = sum(1 for i in range(n) if s[i] and not d[i])   # split right, dense wrong
        p = mcnemar_exact_p(b, c)
        pairs = list(zip(d, s))
        lo, hi = boot_ci(pairs)
        wd, ws = wilson(sum(d), n), wilson(sum(s), n)
        out[L] = {"n": n, "dense_acc": round(acc_d, 4), "split_acc": round(acc_s, 4),
                  "dense_CI95": [round(x, 4) for x in wd], "split_CI95": [round(x, 4) for x in ws],
                  "split_minus_dense": round(acc_s - acc_d, 4),
                  "diff_CI95": [round(lo, 4), round(hi, 4)],
                  "mcnemar_b_dense>split": b, "mcnemar_c_split>dense": c,
                  "mcnemar_p_twosided": round(p, 5),
                  "significant_0.05": p < 0.05}
        print(f"== {L} (n={n}) ==")
        print(f"  dense {acc_d:.3f} [{wd[0]:.3f},{wd[1]:.3f}]  split {acc_s:.3f} [{ws[0]:.3f},{ws[1]:.3f}]")
        print(f"  split-dense = {acc_s-acc_d:+.3f}  95% CI [{lo:+.3f},{hi:+.3f}]  McNemar p={p:.4f}"
              f"  {'SIGNIF' if p<0.05 else 'ns'}")
        # per depth
        for depth in (1, 2):
            qd = [q for q in qids if dz[q][1] == depth]
            if len(qd) < 20:
                continue
            ad = sum(dz[q][0] for q in qd) / len(qd)
            as_ = sum(sz[q][0] for q in qd) / len(qd)
            print(f"    depth {depth} (n={len(qd)}): dense {ad:.3f}  split {as_:.3f}  (split-dense {as_-ad:+.3f})")
        print()
    Path("outputs/mechinterp/h1_deduction").mkdir(parents=True, exist_ok=True)
    (Path("outputs/mechinterp/h1_deduction") / "deduction_h1.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/mechinterp/h1_deduction/deduction_h1.json")


if __name__ == "__main__":
    main()
