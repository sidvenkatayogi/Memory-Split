#!/usr/bin/env python
"""Superposition figure with self-contained labels (trained-people = authoritative)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOADS = ["n50k", "n200k", "n800k"]
XLAB = ["50k", "200k", "800k"]


def pr(load, arm, tag):
    return json.loads((HERE / f"{load}{tag}.json").read_text())[arm]["participation_ratio"]


x = np.arange(len(LOADS)); w = 0.38
fig, ax = plt.subplots(figsize=(8.2, 5.0))
dense = [pr(l, "dense", "_seen") for l in LOADS]
split = [pr(l, "split", "_seen") for l in LOADS]
b1 = ax.bar(x - w/2, dense, w, label="dense (memorizes facts)", color="#c44e52")
b2 = ax.bar(x + w/2, split, w, label="split (looks facts up)", color="#4c72b0")
# overlay strangers as faint markers to show agreement
for i, l in enumerate(LOADS):
    ax.plot([i - w/2], [pr(l, "dense", "_unseen")], "k_", ms=16, mew=2)
    ax.plot([i + w/2], [pr(l, "split", "_unseen")], "k_", ms=16, mew=2,
            label="strangers (unseen), for comparison" if i == 0 else None)
ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(XLAB)
ax.set_xlabel("fact load = number of distinct people in training")
ax.set_ylabel("participation ratio (last-layer fact activations)\n"
              "HIGHER = roomier · LOWER = more crowded")
ax.set_title("Superposition probe: are fact representations crowded?\n"
             "Dense uses far fewer directions than split, and collapses once it can't memorize\n"
             "(split arm, trained people; '—' marks = strangers, which agree · final ckpt ≈3.2B tok · seed 0)",
             fontsize=10.5)
ax.set_ylim(0, 9)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "superposition.png", dpi=130)
print("wrote", HERE / "superposition.png")
