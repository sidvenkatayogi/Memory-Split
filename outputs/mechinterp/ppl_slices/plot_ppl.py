#!/usr/bin/env python
"""Plot ppl-slices summary: fact-value NLL and reasoning bpb, dense vs split."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
LOADS = ["n50k", "n200k", "n800k"]
XLAB = ["50k entities", "200k entities", "800k entities"]


def load(load_, arm):
    return json.loads((HERE / f"{load_}_{arm}.json").read_text())


fact = {arm: [load(l, arm)["fact_value_nll"]["nll_per_token"] for l in LOADS]
        for arm in ("dense", "split")}
reas = {arm: [load(l, arm)["slices"]["reasoning"]["bpb"] for l in LOADS]
        for arm in ("dense", "split")}

x = np.arange(len(LOADS))
w = 0.36
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# ---- fact-value NLL (the on-hypothesis panel) ----
ax1.bar(x - w / 2, fact["dense"], w, label="dense (closed-book)", color="#c44e52")
ax1.bar(x + w / 2, fact["split"], w, label="split (store OFF)", color="#4c72b0")
ax1.axhline(np.log(2) * 0, color="k", lw=0)  # keep baseline visible
ax1.set_xticks(x); ax1.set_xticklabels(XLAB)
ax1.set_ylabel("fact-value NLL  (nats / value-token, lower = better)")
ax1.set_title("Factual slice: predicting attribute VALUES\n"
              "for FRESH held-out entities from weights alone")
ax1.legend(frameon=False, fontsize=9)
for i, (d, s) in enumerate(zip(fact["dense"], fact["split"])):
    ax1.text(i - w / 2, d + 0.15, f"{d:.2f}", ha="center", fontsize=8)
    ax1.text(i + w / 2, s + 0.15, f"{s:.2f}", ha="center", fontsize=8)

# ---- reasoning bpb (the no-regression control panel) ----
ax2.bar(x - w / 2, reas["dense"], w, label="dense", color="#c44e52")
ax2.bar(x + w / 2, reas["split"], w, label="split", color="#4c72b0")
ax2.set_xticks(x); ax2.set_xticklabels(XLAB)
ax2.set_ylabel("reasoning bpb  (bits / byte, lower = better)")
ax2.set_ylim(0.17, 0.19)
ax2.set_title("Knowledge-free slice (iGSM + deduction):\n"
              "identical text scored on both arms")
ax2.legend(frameon=False, fontsize=9)
for i, (d, s) in enumerate(zip(reas["dense"], reas["split"])):
    ax2.text(i - w / 2, d + 0.0004, f"{d:.4f}", ha="center", fontsize=7.5)
    ax2.text(i + w / 2, s + 0.0004, f"{s:.4f}", ha="center", fontsize=7.5)

fig.suptitle("NR-6 ppl-slices — final checkpoint (step 6100 ≈ 3.2B tokens), seed 0",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(HERE / "ppl_slices_overview.png", dpi=130)
print("wrote", HERE / "ppl_slices_overview.png")
