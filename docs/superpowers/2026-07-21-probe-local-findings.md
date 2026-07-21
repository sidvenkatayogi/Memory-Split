# Local probe findings — what the surviving Mac artifacts show

2026-07-21. Probes from the `probe/` toolkit (parallel workstream) run
against the artifacts remaining on this machine after the cluster
cleanup. Driver: `scripts/probe_local.py`; raw numbers:
`outputs/probe-local/probe_local_results.json`.

## Artifacts probed

| Artifact | What it is | Probes it supports |
|---|---|---|
| `outputs/pulled/step0001520.pt` | dense 160M, gate round 2 (n200k dose, 0.8B tokens) | Q3 storage + Q1 attention, single-arm |
| `data/smoke/runs/smoke_{dense,split}/ckpt.pt` | toy 4Lx256 PAIR, gate-0 smoke | Q2 pair contrasts (demonstration scale) |
| `outputs/pulled/{igsm,deduction}.jsonl` | held-out reasoning sets (gate vintage) | prompts/scorers |
| corpus generators (deterministic seeds) | regenerate the 200k entities + fact probes exactly | fact prompts for attribution/selectivity |

Scale caveats apply throughout: the pair contrast is toy-scale (4 layers,
300 steps); the 160M results are single-arm (its split twin's checkpoint
no longer exists locally; the rebuilt sweep will supply new pairs). One
seed everywhere; all findings directional. Same-init pairing is what
makes unit-identity comparisons meaningful: both arms start from
byte-identical weights, so "which units became fact-selective" is a
trained difference, not init noise.

## Finding 1 (toy pair, Q2): fact-selective units are nearly disjoint

Top-40 fact-selective MLP units per arm (Cohen's d, fact vs reasoning
prompts): only 3/40 shared (Jaccard 0.04) despite identical
initialization. The arms built their fact-context machinery in different
units. Read together with the split arm's behavioral profile (0 measured
fact-bits, 0% store-off recall), the natural interpretation: dense's
fact units store values; split's fact units do something else (plausibly
lookup-context detection). Unit-level CLS division of labor.

## Finding 2 (toy pair, Q2): representations diverge globally

Cross-arm linear CKA on shared prompts, per layer:

- fact prompts: 0.44 / 0.36 / 0.38 / 0.39 (layers 0-3)
- reasoning prompts: 0.47 / 0.47 / 0.44 / 0.30

Same-init models trained on near-identical data would sit far higher.
The one-bit intervention (loss-mask fact values) reorganized the whole
computation, not a local patch: the split arm is a different solution,
not dense-minus-facts. Necessary (not sufficient) for the "freed
capacity is reused" claim; where it is reused needs the 160M pair.

## Finding 3 (toy pair, Q3): dense spends more weight rank

Effective rank of each layer's MLP down-projection (dense minus split):
+0.91 / +1.57 / +0.76 / +2.38 (layers 0-3). Arbitrary fact mappings are
incompressible, so memorization should consume rank; the dense arm pays
that cost at every layer, most at the top. Weight-level counterpart of
the bits-in-weights accounting. Caveat: small deltas at toy scale, and
corpus token-statistics differences (lookup wrappers) could contribute.

## Finding 4 (dense 160M, single-arm): storage geography and two instructive nulls

Full battery completed 2026-07-21 15:15 (CPU, ~10 min). Context for all
four: this checkpoint memorized almost nothing (0.7% closed-book recall
at the gate budget), so it probes the *undertrained* end of the storage
story.

**4a. Training compresses; the top layer most.** Effective rank of every
MLP down-projection fell 43-63 below its random-init value (largest drop:
final layer, -63). Fact-heavy training should RESIST this compression
(arbitrary mappings are incompressible) — the dense-vs-split rank gap at
matched steps, per dose, is the quantity to watch on the sweep pairs.

**4b. Fact attribution is U-shaped.** Gradient-times-weight attribution
of fact-recall NLL concentrates at the network's two ends: layer 0
(12.3%) and layers 9-11 (11-13% each), with a valley through layers 2-4.
The few facts this model does hold live embedding-adjacent and
readout-adjacent, not in the mid-layer MLPs the LLM knowledge-editing
literature emphasizes — plausibly a signature of the undertrained regime.

**4c. Fact contexts collapse to ~2 effective dimensions.** Participation
ratio of final-layer residuals: 2.3 for fact prompts vs 8.4 for reasoning
prompts (of 768). Fact prompts produce near-identical internal states:
the model treats "X's major is" generically instead of per-entity — the
activation-level face of "facts not stored" and the baseline point for
the superposition-vs-dose curve (prediction: PR rises as dense actually
memorizes, then compresses under crowding).

**4d. Ablation dissociation — an instructive floor-effect null.**
Ablating the top-60 fact-selective units left recall NLL essentially
unchanged (+0.003 nats) while *deduction* NLL worsened by 0.80 nats. No
recall damage is expected when there is no recall to damage (floor
effect); the deduction damage says those "fact-selective" units carry
load-bearing general computation at this stage. The dissociation test
needs a dense checkpoint that actually memorized — i.e. the full-budget
sweep arms.

**4e. Methodological catch (attention probe):** per-layer attention-band
ablation *improved* teacher-forced NLL on yes/no deduction answers
(layer 0 by a implausible 8.6 nats). This is a scorer artifact, not a
discovery: breaking the model collapses it toward generic high-frequency
tokens, which can lower NLL on short binary answers. The attention probe
needs a margin-based scorer (yes-vs-no logit difference) before its
numbers mean anything; flagged for the cluster runs.

Raw numbers for all five: `outputs/probe-local/probe_local_results.json`.

## What the full-scale versions will test (on the rebuilt sweep pairs)

- Double dissociation, causal: ablate dense's top fact units; recall
  should collapse with deduction untouched.
- Attribution contrast: fact-NLL gradients concentrated (dense) vs
  diffuse/near-zero (split).
- Superposition vs dose: dense fact activations should pack into fewer
  effective dimensions as N rises across n50k -> n200k -> n800k; split
  flat. The mechanistic mirror of the dose-response headline.
- CKA localization at 160M: divergence concentrated in the layers where
  capacity is freed.

These are the mediation layer of the experiment: the battery says
whether the split wins; the probes say why (dense spends units, rank,
and dimensions on storage; split spends the identical budget elsewhere).

## Ops notes

- `probe/` is the parallel workstream's export bundle (untracked here by
  design); its four reused modules were restored into `evals/`
  (mechanism, interp, reasoning_probe, continuous — pinned copies,
  attributed).
- Probes at 160M run fine on CPU (~20 min for the full single-arm
  battery); avoid MPS for anything touching `torch.linalg` (missing ops,
  and hard-to-debug silent exits under the fallback).
