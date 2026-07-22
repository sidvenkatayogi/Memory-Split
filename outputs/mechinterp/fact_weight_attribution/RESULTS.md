# Which layers hold a specific fact?

> **Entity population (labelled per the standing convention):** this probe was run on
> **both** the people the model was trained on ("seen") and a set of strangers it never
> saw ("unseen"). That distinction matters *a lot* here — more than for any other probe —
> because the way this probe scores a fact is partly driven by whether the model gets the
> fact right, and it only ever gets *trained* people right. So the **trained-people
> ("seen") numbers are the ones to trust**, and comparing them against the stranger
> numbers is what reveals the confound. Background on why both populations exist:
> [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md).

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **dense** twin memorizes facts, the **split** twin looks them up.*

**What this probe asks, in plain terms.** If the dense twin memorized "Kai's major is
Communications," **which layers' weights are responsible for producing that value?** The
hope was to point at specific weights and say "the fact lives here."

**How we measure it — and the built-in trap.** We ask the model to produce a specific value
and measure how **sensitive** that output is to each layer's weights (a gradient-based
"how much does this layer matter" score). A big, concentrated score at a layer would suggest
the fact is stored there. **The trap:** this score is large not only when a fact is *stored*
in a layer, but also whenever the model is simply **wrong** about the value — so the raw
number mixes "where it's stored" with "how badly it's failing." That confound turns out to
dominate, which is why the seen-vs-stranger split (next section) is decisive here.

**Why it matters — significance & nuance.** In principle this gives **per-fact, per-layer**
support for **H3** (a memorized fact sits in specific dense weights; the split twin has no
such store). *In practice*, because of the trap, the eye-catching dense-vs-split result was
an artifact — so **the only defensible reading is within the dense twin, across fact loads**
(more fact-weight-sensitivity at the load where it actually memorized). Treat this as a
**weak, confounded hint about localization, never as a dense-vs-split comparison.**

**Which models / checkpoint / people.** Both twins at all three loads (50k/200k/800k), at
the finished checkpoint `snapshots/step0006100.pt`, on 20 sample `major` facts. Run on both
the people the model **trained on** (the numbers to trust) and **strangers** (which expose
the trap). One seed; a sensitivity heuristic, **not** a causal ablation.

## Why this probe, and what it supports
**Motivation.** If the dense model memorizes a fact, that fact should be attributable to
specific weights; if the split model offloads it, no such localized in-weight store
should exist. In principle this gives **per-fact, per-layer** support for H3 (facts sit
in dense weights, but not in split weights) and connects the abstract "bits in weights"
idea to concrete parameters — i.e. *where* a memorized fact physically lives.

**Method/rigor note — and the built-in confound.** We teacher-force "prompt + true
value," backpropagate the value's negative log-likelihood, and sum `|gradient × weight|`
over each layer's MLP matrices. This is a **first-order (Taylor) sensitivity** heuristic,
not a causal ablation. The crucial subtlety: the **gradient is large precisely when the
model is *wrong*** about the value (a confidently-correct prediction has near-zero
gradient). So `|grad × weight|` mixes two things together: (a) *is there fact-machinery
in these weights?* and (b) *is the model currently failing to produce this value?* A big
number can come from either. This is exactly why the seen-vs-unseen split below is
decisive rather than cosmetic.

## Why the seen/unseen distinction is the whole story here
Feed the probe a **stranger** and the model is guaranteed to be wrong about that person's
attribute (it never saw them), so factor (b) — the "you're wrong" gradient — is maxed out.
Feed it a **trained** person and, at the load where it memorized, the model is usually
*right*, so that gradient is small. Because of this, the stranger numbers are inflated
wherever the model also happens to have fact-machinery, and the trained-people numbers
are the honest measure of "how much fact-structure is in these weights." When the two
disagree, believe the trained-people run.

## Result (figures `attribution_{n50k,n200k,n800k}.png` show the trained-people run; `attribution_across_load.png` summarizes the within-dense trend)
Summed `|grad × weight|` over 20 `major` facts, and the layer where it peaks, for both
populations. Higher = more weight-sensitivity for producing the value:

| load | dense, trained (seen) | dense, strangers (unseen) | split, trained | split, strangers |
|---|---|---|---|---|
| n50k  | **18,700** (peak layer 9) | 56,500 (peak layer 9) | 21,700 | 20,700 |
| n200k | 1,600 (peak layer 11) | 1,500 (peak layer 11) | 21,300 | 20,500 |
| n800k | 1,300 (peak layer 11) | 1,400 (peak layer 0) | 19,400 | 18,700 |

## What it means
- **There is no clean dense-vs-split localization contrast.** The stranger run reads a
  large dense-n50k value (56,500, far above split), but that number is inflated by the
  "you're wrong about this stranger" gradient. On the **trained** people (whom the n50k
  dense model actually recalls), the same measurement is **18,700 — essentially the same
  as the split arm's 21,700**. Measured on the people the model actually stored, the dense
  arm's per-fact weight-sensitivity is **not distinguishable from the split arm's**, so
  **do not cite a "dense localizes, split diffuse" result from this probe.**
- **What *does* survive is a within-dense pattern across loads, and it points the right
  way.** Staying inside the dense arm and looking across loads on the trained people,
  weight-sensitivity is far higher at **n50k (18,700)** than at **n200k (1,600)** or
  **n800k (1,300)** — roughly a **12× gap** — and the same ordering holds (even more
  strongly) on strangers. n50k is the one load where the dense model genuinely memorized
  (62% closed-book recall), so "more weight-structure implicated where the model actually
  stored facts" is the sensible reading. Note this ordering is *not* explained by the
  wrongness confound — if anything that confound would push n200k (where dense is more
  often wrong) *higher*, not lower — which is why the within-dense pattern is more
  trustworthy than the cross-arm one.
- **The peak layer is consistently layer 9** for dense-n50k under both populations, so
  "if there is fact-structure, it sits mid-to-late in the network" is a stable
  observation even though the magnitude is confounded.
- **The split arm is flat (≈20k) at every load and population**, consistent with it never
  developing value-specific weight structure — but remember its absolute value is not
  comparable to dense's (different training target for value tokens), which is the second
  reason the cross-arm comparison is off-limits.

## Deeper interpretation & rigor
A cross-probe observation worth flagging: this probe says the dense fact-structure (at
50k people) is most attributable to **layer 9**, whereas the mechanism-localization probe
finds the strongest fact-*selective activations* in the **early layers** (layer 0). That
is not a contradiction — the two measure different things: localization asks *where
fact-selective activity is*, while attribution asks *which weights most change the produced
value*. A plausible picture is that early layers *detect/route* the (name, attribute)
query while later layers *emit* the stored value; but reconciling the two is a job for a
causal follow-up (ablate candidate layers, see where recall actually breaks), not for this
correlational, magnitude-confounded probe.

Rigor limits: first-order `|grad × weight|` attribution can **under-count** storage that
is spread across many weights, its absolute scale is arbitrary, and — as emphasized above —
its magnitude is entangled with prediction error, so only *within-arm, within-population*
comparisons of the *layer shape* and *across-load ordering* are meaningful. Based on 20
facts of a single attribute (`major`), one seed, local run, on both populations.

## Caveats (fine print)
- **One seed ⇒ a direction.**
- **Do not compare dense vs split totals** — different training target for value tokens,
  *and* the magnitude is confounded by prediction error.
- **Trust the trained-people ("seen") numbers**; the stranger numbers are inflated wherever
  the model is wrong, which for facts is everywhere it didn't memorize.
- This is a sensitivity heuristic, not a causal test; confirming "where a fact lives" needs
  turning those weights off and checking recall drops.
- The per-layer PNGs are the trained-people run (dense ≈ split); the stranger run has the
  same peak-at-layer-9 shape but an inflated n50k magnitude (the confound described above).

## Provenance
Analysis code added on top of the frozen experiment (to be noted in the write-up); no
training run was changed. Each load was measured on the frozen trained people
(`outputs/_frozen_data/<load>_recall.jsonl`, saved `<load>_seen.json`) and on a regenerated
stranger set (`<load>_unseen.json`).
