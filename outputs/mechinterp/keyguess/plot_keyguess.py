#!/usr/bin/env python
"""Plot key-gen generalization: held-out vs seen key accuracy + decomposition."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOADS = ["n50k", "n200k", "n800k"]
XLAB = ["50k", "200k", "800k"]


def load(l):
    return json.loads((HERE / f"{l}_keyguess.json").read_text())


D = {l: load(l) for l in LOADS}
x = np.arange(len(LOADS))
w = 0.38
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# Panel 1: held-out vs seen KEY accuracy (the generalization question)
held = [D[l]["heldout"]["key_accuracy"] for l in LOADS]
seen = [D[l]["seen"]["key_accuracy"] for l in LOADS]
ax1.bar(x - w / 2, held, w, label="unseen people (sample A)", color="#4c72b0")
ax1.bar(x + w / 2, seen, w, label="unseen people (sample B)", color="#55a868")
ax1.set_ylim(0.9, 1.005)
ax1.set_xticks(x); ax1.set_xticklabels(XLAB)
ax1.set_xlabel("fact load = number of people in training")
ax1.set_ylabel("exact organizer-key accuracy  (1.0 = perfect)")
ax1.set_title("Correct lookup address for people NEVER seen in training\n"
              "(two unseen samples agree ⇒ a general procedure)")
ax1.legend(frameon=False, fontsize=9, loc="lower center")
for i, (h, s) in enumerate(zip(held, seen)):
    ax1.text(i - w / 2, h + 0.002, f"{h:.3f}", ha="center", fontsize=8)
    ax1.text(i + w / 2, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)

# Panel 2: held-out decomposition (key / name-half / relation-half / answer)
metrics = [("key_accuracy", "exact key"), ("name_half_accuracy", "name-half (copy)"),
           ("relation_half_accuracy", "relation-half"), ("answer_accuracy", "answer")]
w2 = 0.2
colors = ["#4c72b0", "#dd8452", "#c44e52", "#8172b3"]
for k, (m, lab) in enumerate(metrics):
    vals = [D[l]["heldout"][m] for l in LOADS]
    ax2.bar(x + (k - 1.5) * w2, vals, w2, label=lab, color=colors[k])
ax2.axhline(1 / 6, ls="--", color="gray", lw=1)
ax2.text(2.35, 1 / 6 + 0.01, "relation chance (1/6)", fontsize=7.5, color="gray", ha="right")
ax2.set_ylim(0, 1.08)
ax2.set_xticks(x); ax2.set_xticklabels(XLAB)
ax2.set_xlabel("fact load = number of people in training")
ax2.set_ylabel("accuracy on unseen people  (1.0 = perfect)")
ax2.set_title("The two sub-skills of an address\n"
              "name-half = copy person · relation-half = pick attribute (6-way)")
ax2.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")

fig.suptitle("Key-generation on UNSEEN (in-distribution synthetic) people — split arm · "
             "final ckpt ≈3.2B tokens · seed 0 · n=1,200 probes/bar", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(HERE / "keyguess_overview.png", dpi=130)
print("wrote", HERE / "keyguess_overview.png")
