#!/usr/bin/env python
"""Extractability figure (SEEN / trained people) with self-contained labels."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOADS = ["n50k", "n200k", "n800k"]
XLAB = ["50k", "200k", "800k"]


def load(l, a):
    return json.loads((HERE / f"{l}_{a}_seen.json").read_text())


x = np.arange(len(LOADS))
w = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

# Panel 1: generative recall (can the model PRODUCE the value?)
dg = [load(l, "dense")["generative_recall"]["overall"] for l in LOADS]
sg = [load(l, "split")["generative_recall"]["overall"] for l in LOADS]
ax1.bar(x - w / 2, dg, w, label="dense (closed-book)", color="#c44e52")
ax1.bar(x + w / 2, sg, w, label="split (store OFF)", color="#4c72b0")
ax1.set_xticks(x); ax1.set_xticklabels(XLAB)
ax1.set_xlabel("fact load = number of people in training")
ax1.set_ylabel("generative recall (produces the exact value)")
ax1.set_ylim(0, 0.75)
ax1.set_title("Can the model PRODUCE the fact?\n(dense memorizes only at 50k; split stores nothing)")
ax1.legend(frameon=False, fontsize=9)
for i, (d, s) in enumerate(zip(dg, sg)):
    ax1.text(i - w / 2, d + 0.012, f"{d:.2f}", ha="center", fontsize=8)
    ax1.text(i + w / 2, s + 0.012, f"{s:.2f}", ha="center", fontsize=8)

# Panel 2: MC recognition (can it RECOGNIZE the value vs 3 distractors?)
dm = [load(l, "dense")["mc_recognition"]["overall"] for l in LOADS]
sm = [load(l, "split")["mc_recognition"]["overall"] for l in LOADS]
ax2.bar(x - w / 2, dm, w, label="dense", color="#c44e52")
ax2.bar(x + w / 2, sm, w, label="split", color="#4c72b0")
ax2.axhline(0.25, ls="--", color="gray", lw=1)
ax2.text(2.4, 0.27, "chance = 1/4", fontsize=8, color="gray", ha="right")
ax2.set_xticks(x); ax2.set_xticklabels(XLAB)
ax2.set_xlabel("fact load = number of people in training")
ax2.set_ylabel("MC recognition accuracy (4-way)")
ax2.set_ylim(0, 1.0)
ax2.set_title("Can it RECOGNIZE the value (vs 3 distractors)?\n"
              "dense@50k = 0.90 (facts are there); elsewhere ≈ chance")
ax2.legend(frameon=False, fontsize=9)
for i, (d, s) in enumerate(zip(dm, sm)):
    ax2.text(i - w / 2, d + 0.02, f"{d:.2f}", ha="center", fontsize=8)
    ax2.text(i + w / 2, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)

fig.suptitle("NR-4 extractability — measured on TRAINED people (frozen) · split & dense · "
             "final ckpt ≈3.2B tok · seed 0", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE / "extractability_overview.png", dpi=130)
print("wrote", HERE / "extractability_overview.png")
