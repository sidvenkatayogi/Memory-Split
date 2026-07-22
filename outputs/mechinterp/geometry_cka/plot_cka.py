#!/usr/bin/env python
"""Cross-arm CKA-by-layer figure with self-contained labels (trained people)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LOADS = [("n50k", "50k people", "#55a868"),
         ("n200k", "200k people", "#dd8452"),
         ("n800k", "800k people", "#c44e52")]

fig, ax = plt.subplots(figsize=(9.2, 5.4))
for load, lab, col in LOADS:
    d = json.loads((HERE / f"{load}_seen.json").read_text())
    cka = {int(k): v for k, v in d["cka_by_layer"].items()}
    xs = sorted(cka)
    ax.plot(xs, [cka[i] for i in xs], "-o", color=col, label=lab)
ax.set_ylim(0, 1.0)
ax.set_xlabel("transformer layer   (0 = input side  →  11 = output side)")
ax.set_ylabel("cross-arm representation similarity (linear CKA)\n"
              "1 = dense & split represent the same inputs identically · 0 = unrelated")
ax.set_title("Geometry probe: WHERE do the dense and split models differ inside?\n"
             "Same prompts fed to both arms; low similarity = they compute differently there\n"
             "(measured on trained people · final ckpt ≈3.2B tok · seed 0)", fontsize=10.5)
ax.legend(title="fact load", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "cka_by_layer.png", dpi=130)
print("wrote", HERE / "cka_by_layer.png")
