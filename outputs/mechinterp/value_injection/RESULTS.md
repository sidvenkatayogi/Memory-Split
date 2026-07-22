# Where does a looked-up fact value enter the split model?

> **Entity population: UNSEEN (regenerated) — but this probe is entity-agnostic.**
> The value is *forced* into the residual stream by the harness, so where it enters
> does not depend on whether the entity was trained (repo generator drift — see
> [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md)). **Why it matters here:**
> it mostly **doesn't** — the injection-point conclusion is robust to seen/unseen.

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **split** twin doesn't know fact values — it emits a lookup query and an
external table hands the value back to it as text.*

**What this probe asks, in plain terms.** Because the split twin gets fact values from
**outside** (they appear as tokens in its context after a lookup), the natural mechanistic
question is: **where inside the network does that external value actually get absorbed and
begin to steer what the model does** — early layers, or late? (This only makes sense for
the **split** twin; the dense twin has facts baked into its weights, so there's no external
value to inject.)

**How we measure it.** For each fact we run the model on two inputs that differ by exactly
one thing — the value word **present** vs **absent** ("`Kai's major is Communications`" vs
"`Kai's major is`") — and at **each layer** measure how far apart the two runs' internal
states are. A bigger gap at a layer = that's where the value's presence is changing the
computation.

**Why it matters — significance & nuance.** We already know the split twin stores **no**
facts in its weights (the store ON-vs-OFF recall result). This is the flip side: confirming
it behaves as a **"use-a-fact-from-context" machine** and locating the **splice point**
where retrieved knowledge meets the reasoning model — the mechanistic complement to **H3**,
and a practically useful thing to know for the broader fast/slow-memory design (an external
store feeding a small reasoner). *Nuance / caveat:* in a transformer, later layers naturally
carry larger-magnitude activations, so a raw layer-by-layer gap **tends to grow with depth
for almost any change** — meaning "the effect peaks at the last layer" is only partial
evidence and needs a norm-normalized follow-up (see the rigor caveat below). Correlational,
not a causal test of the retrieval pathway.

**Which model / checkpoint / people.** The **split** twin only, at all three loads
(50k/200k/800k), finished checkpoint `snapshots/step0006100.pt`. One seed ⇒ a direction.
The value is supplied by us, so this is entity-agnostic (trained-vs-stranger doesn't change
it).

## Result (see `injection_profile.png`)
At all three loads the effect of the value **grows layer by layer and is largest at the
final layer** (the curves rise monotonically and peak at the last layer); the peak
magnitude is similar across loads (≈267 / 292 / 300).

## What it means (grounded, with an important rigor caveat)
- The plain reading: the injected value is **incorporated progressively and matters most
  near the output**, rather than being folded into the early/middle "knowledge" layers.
  That fits a **use-from-context (copy/attend-and-use) mechanism**: the split model
  carries the value forward and brings it to bear close to where it must emit the
  answer — which is exactly how a model *should* behave if the fact is not in its
  weights but supplied externally. Contrast with the dense model, whose fact is present
  in the weights from the early/middle layers and so needs no late external value.
- **Rigor caveat you must keep in mind (why the "peak at the last layer" is only
  partial evidence):** in a transformer the residual stream **accumulates** across
  depth, so later layers simply have **larger-norm activations**. That means an L2
  difference between two conditions will *tend* to grow with depth **even for a generic
  change**, not just for this value. So part of the monotonic rise is expected from
  residual-stream geometry, and the raw peak-at-the-end should **not** be over-read as
  "the value is only used at the very end." The robust claims are: (a) the value
  **measurably changes** the representation, and (b) its effect is **present and
  building across layers** — consistent with use-from-context. Pinning down the true
  "splice layer" needs a **norm-normalized** version (divide each layer's difference by
  that layer's typical activation norm, or use a relative/cosine measure), which is the
  right follow-up.

## Caveats (fine print)
- **One seed ⇒ a direction, not a proof.** This is **correlational** (a with-value vs
  without-value contrast), not a causal test of the retrieval pathway.
- We used "value shown in the sentence" as a **stand-in** for the true look-up, where
  the value instead arrives *after* the model emits `<db_retrieve>` and is force-decoded.
  A stronger version would inject the actual retrieved tokens mid-generation.
- **Not normalized for residual-stream growth** (see the rigor caveat above) — interpret
  the *shape/that-it-happens*, not the raw magnitudes, until the normalized version is run.
- Measured on a sample of people/attributes, locally.

## Provenance
Analysis code added on top of the frozen experiment (to be noted in the write-up); it
did not change any training run.
