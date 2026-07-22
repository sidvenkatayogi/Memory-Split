# Are the dense model's weight matrices "fuller" than the split model's?

> **Entity population: N/A — weights only.** This probe reads the weight matrices
> directly with **no forward pass on any entity**, so the seen/unseen generator
> drift ([`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md)) does **not** affect
> it. **Result stands as-is.**

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **dense** twin memorizes facts, the **split** twin looks them up.*

**What this probe asks, in plain terms.** If the dense twin stuffs thousands of facts into
its weights, maybe you can *see* that in the weights themselves — its weight matrices might
look "fuller" or more used-up than the split twin's leaner ones. This probe inspects the
weight matrices **directly** (it never runs the model on any text).

**How to read the number.** For each feed-forward matrix we compute its **effective rank** —
roughly, **how many independent patterns the matrix actually uses**, out of the most it
*could* use (here, out of 768). Higher = more of the matrix's capacity is in play.

**Why it matters — significance & nuance.** One concrete version of "facts eat capacity"
predicts the memorizing dense twin should use **more** of its weight matrices than the
offloading split twin, so a dense > split gap would be parameter-level evidence for that
story. *Nuance:* effective rank is a **single, coarse, whole-matrix summary**, so a *null*
(no difference) does **not** mean "memorization is free" — the effect could be real but too
small or too localized for this blunt measure to catch. Read a null as "**this instrument
can't see it**," and pair it with the activation-level crowding probe (`superposition/`),
which *can*.

**Which models / checkpoint / people.** Both twins at all three loads (50k/200k/800k), at
the finished checkpoint `snapshots/step0006100.pt`. **Weights only — no people involved**,
so the seen/unseen question doesn't apply here. One seed ⇒ a direction, not a proof.

## Why this probe, and what it supports
**Motivation.** One concrete version of "facts consume capacity" predicts that a model
which memorizes many facts should use *more* of its weight matrices — a higher
effective rank — than a model that offloads them. This probe tests that prediction at
the **parameter level**. Paired with the activation-level crowding result
(`superposition/`), it localizes *where* capacity is and isn't spent, sharpening the
mechanistic story behind H1/H3: a null here plus a positive there says the difference
lives in *how the weights are used on fact prompts*, not in the raw weight matrices.
**Method/rigor note.** Effective rank = `exp(H)`, where H is the entropy of the
normalized singular-value distribution of a weight matrix — the *effective number of
independent directions* the matrix uses (a rank-1 matrix → 1; a perfectly isotropic
matrix → min(rows, cols)). Computed on the feed-forward (MLP) weight matrices; it reads
the weights directly, with no forward pass.

## Result — no difference (a clean null)
The effective rank is **essentially identical** between the two models at every load:

| load | dense | split |
|---|---|---|
| n50k  | 689 | 686 |
| n200k | 687 | 686 |
| n800k | 686 | 686 |

## What it means
At this model size, **this weight-level measurement does not distinguish the two
models.** Memorizing facts does not visibly "fill up" the dense model's weight matrices
relative to the split model's. So the "leaner weights" version of the capacity story is
**not** supported by this particular measurement — a clean, reportable negative result.

Worth noting: the crowding *does* show up when we look at the model's **activity**
(see the `superposition` probe: the dense model packs facts into fewer directions) —
just not in this coarse whole-matrix weight measure. In other words, the difference is
visible in *how the model uses its weights on fact prompts*, not in the raw size/shape
of the weight matrices.

## Deeper interpretation & rigor
Why this null is *expected* and still worth reporting: each feed-forward weight matrix
(e.g. 768×2048) is overwhelmingly devoted to the general language computation that
**both** models share; any fact-specific storage is a small perturbation on top of that
bulk. Effective rank is a single whole-matrix summary (≈686 of 768 — near full rank in
both arms), so it is far too coarse to detect a small, possibly low-rank fact-storage
component. The honest conclusion is therefore **not** "memorization uses no capacity,"
but "**this particular measurement cannot see it.**"

That is exactly why it's valuable *next to* the activation-level result: the capacity
effect shows up in **how the weights are used on fact prompts** — activations collapse
into few directions as load grows (see `superposition/`) — rather than in the bulk
geometry of the weight matrices themselves. In other words, the difference is in the
*computation on fact inputs*, not in the raw shape of the parameters. A sharper
weight-level test would target the specific subspaces implicated by the fact-neuron
localization, instead of scoring the whole matrix.

Rigor limits: coarse whole-matrix scalar; single seed; no targeted fact-subspace
analysis.

## Caveats (fine print)
- **One seed ⇒ a direction.** Effective rank is a single coarse number for a whole
  matrix; a real fact-storage difference could hide in a small part of the matrix
  without changing this overall number. At this scale the matrices are dominated by the
  shared language ability common to both models.

## Provenance
Analysis code added on top of the frozen experiment (to be noted in the write-up); no
training run was changed.
