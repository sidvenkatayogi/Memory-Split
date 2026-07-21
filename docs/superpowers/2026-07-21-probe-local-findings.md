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

## Finding 4 (dense 160M): PENDING — battery running

Weight-spectral-vs-init, fact weight attribution (which layers store
facts), fact-activation superposition (participation ratio), fact-unit
ablation (recall vs deduction dissociation), and per-layer attention-band
ablation are running on CPU at the time of writing (Apple-GPU backend
lacks the SVD op and silently killed two attempts; CPU is reliable).
This section is updated when the run lands.

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
