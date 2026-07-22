#!/usr/bin/env python
"""Value-injection profile with self-contained, appropriately-hedged labels."""
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
    prof = {int(k): v for k, v in d["profile"].items()}
    xs = sorted(prof)
    ax.plot(xs, [prof[i] for i in xs], "-o", color=col, label=lab)
ax.set_xlabel("transformer layer   (0 = input side  →  11 = output side)")
ax.set_ylabel("effect of the looked-up value on internal activity\n"
              "(L2 distance: with-value vs without-value · larger = bigger effect)")
ax.set_title("Value-injection probe (split model): where does the looked-up value get used?\n"
             "Effect grows across depth — but note later layers have larger activations,\n"
             "so part of this rise is expected geometry, not proof of 'used only at the end'\n"
             "(split arm · final ckpt ≈3.2B tok · seed 0)", fontsize=9.5)
ax.legend(title="fact load", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "injection_profile.png", dpi=130)
print("wrote", HERE / "injection_profile.png")
