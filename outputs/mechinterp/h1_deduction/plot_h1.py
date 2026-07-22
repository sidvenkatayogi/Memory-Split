#!/usr/bin/env python
"""Forest-style plot of split-minus-dense deduction accuracy per load, with 95% CIs."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "deduction_h1.json").read_text())
LOADS = ["n50k", "n200k", "n800k"]
y = range(len(LOADS))
diffs = [d[L]["split_minus_dense"] for L in LOADS]
los = [d[L]["diff_CI95"][0] for L in LOADS]
his = [d[L]["diff_CI95"][1] for L in LOADS]

fig, ax = plt.subplots(figsize=(8.6, 4.0))
for i, L in enumerate(LOADS):
    err = [[diffs[i] - los[i]], [his[i] - diffs[i]]]
    color = "#4c72b0" if diffs[i] >= 0 else "#c44e52"
    ax.errorbar(diffs[i], i, xerr=err, fmt="o", color=color, capsize=5, ms=8)
    sig = "significant" if d[L]["significant_0.05"] else "n.s."
    ax.text(his[i] + 0.006, i, f"{diffs[i]:+.3f}  ({sig}, McNemar p={d[L]['mcnemar_p_twosided']})",
            va="center", fontsize=9)
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_yticks(list(y)); ax.set_yticklabels(["50k people", "200k people", "800k people"])
ax.set_xlabel("split accuracy − dense accuracy on deduction  (→ split better)")
ax.set_xlim(-0.11, 0.11)
ax.set_title("Does the split model reason better? (deduction, the learnable task)\n"
             "No consistent advantage: null at 50k, split +3pp at 200k, split −5.5pp at 800k.\n"
             "Bars = 95% CI over eval items — but these are SINGLE-SEED runs, so a few-pp gap\n"
             "is within training-seed noise (needs multiple seeds to trust any direction).",
             fontsize=9.5)
fig.tight_layout()
fig.savefig(HERE / "deduction_h1.png", dpi=130)
print("wrote", HERE / "deduction_h1.png")
