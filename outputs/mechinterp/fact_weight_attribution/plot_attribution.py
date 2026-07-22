#!/usr/bin/env python
"""Fact-weight-attribution figures with self-contained labels (trained people).

Regenerates the per-load per-layer curves from the SEEN (trained-people) runs — the
authoritative population — where dense and split are NOT distinguishable (the eye-catching
cross-arm gap only appears on strangers, and is a wrongness-gradient artifact). Also writes
an across-load summary of the one signal that survives (within-dense, higher where it
memorized).
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOADS = ["n50k", "n200k", "n800k"]


def layers(load, arm):
    d = json.loads((HERE / f"{load}_seen.json").read_text())[arm]
    xs = sorted(int(k) for k in d)
    return xs, [d[str(i)] for i in xs]


# per-load per-layer dense vs split (trained people)
for load in LOADS:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    xd, yd = layers(load, "dense"); xs, ys = layers(load, "split")
    ax.plot(xd, yd, "-o", color="#c44e52", label="dense (memorizes)")
    ax.plot(xs, ys, "-o", color="#4c72b0", label="split (looks up; not directly comparable)")
    ax.set_xlabel("transformer layer   (0 = input side  →  11 = output side)")
    ax.set_ylabel("weight-sensitivity for producing the fact\n|grad × weight| summed over the layer's MLP")
    memo = "dense memorized here (62% recall)" if load == "n50k" else "dense did NOT memorize here"
    ax.set_title(f"Fact-weight attribution · {load[1:]} people · measured on TRAINED people ({memo})\n"
                 "dense ≈ split → no clean cross-arm localization; magnitude confounded by prediction error\n"
                 "(20 'major' facts · final ckpt ≈3.2B tok · seed 0)", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / f"attribution_{load}.png", dpi=130)
    plt.close(fig)
    print("wrote", HERE / f"attribution_{load}.png")

# across-load: the surviving within-dense signal (total dense attribution)
fig, ax = plt.subplots(figsize=(7.4, 4.6))
tot = [sum(layers(l, "dense")[1]) for l in LOADS]
sp = [sum(layers(l, "split")[1]) for l in LOADS]
x = np.arange(len(LOADS)); w = 0.38
ax.bar(x - w/2, tot, w, color="#c44e52", label="dense")
ax.bar(x + w/2, sp, w, color="#4c72b0", label="split")
ax.set_xticks(x); ax.set_xticklabels(["50k", "200k", "800k"])
ax.set_xlabel("fact load = number of distinct people in training")
ax.set_ylabel("total weight-sensitivity across all layers (|grad × weight|)")
ax.set_title("The one signal that survives on trained people: WITHIN the dense arm,\n"
             "weight-sensitivity is far higher at the load where it actually memorized (50k)\n"
             "(dense ≈ split at each load — do NOT read a cross-arm gap · seed 0)", fontsize=9.5)
for i, v in enumerate(tot):
    ax.text(i - w/2, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "attribution_across_load.png", dpi=130)
print("wrote", HERE / "attribution_across_load.png")
