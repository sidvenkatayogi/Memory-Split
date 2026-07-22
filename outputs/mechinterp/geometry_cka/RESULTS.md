# Where do the two models differ *inside*? (layer-by-layer representation similarity)

> **Entity population:** this comparison is run on **both** populations — fact prompts
> about people the models were **trained on** ("seen") and about **strangers** they never
> saw ("unseen"). Both are reported together below and — with one instructive exception at
> n50k — they agree. See [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md) for which
> population is authoritative and why we load the frozen trained people.

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **dense** twin memorizes facts, the **split** twin looks them up.*

**What this probe asks, in plain terms.** The two twins are identical except for how they
handle facts, so somewhere inside they must compute differently. This probe asks ***where*
in the network they diverge** — do they think alike early and split apart later, or the
reverse? We feed **both** twins the **same** prompts and, **layer by layer**, measure how
similarly they represent those inputs.

**How to read the number.** The similarity measure is **linear CKA**, a standard score
from **0 to 1**: **1 = the two twins arrange the same inputs identically** at that layer;
**0 = completely differently.** It compares the *geometry* of the internal activations
(which inputs are treated as similar), ignoring rotations and rescalings, and it does
**not** look at the weights. "Most-different layer" = where the twins represent the same
prompts most unlike each other.

**Why it matters — significance & nuance.** The split twin should have "spare" capacity
(it isn't memorizing values). If it spends that capacity differently, this probe
**localizes where** — which is where any *freed-capacity* effect behind **H1** would have
to live, and where fact off-loading (**H3**) actually changes the computation. *Nuance /
big caveat:* this is **exploratory and correlational** — it tells you *where to look*, not
what those layers *do*; and we have **no baseline** for how much two same-architecture
models differ from a different random seed alone, so read the **pattern**, not the absolute
similarity values.

**Which models / checkpoint / people.** Both twins at all three loads (50k/200k/800k), at
the finished checkpoint `snapshots/step0006100.pt`. One seed ⇒ a direction, not a proof.
Measured on both trained people and strangers (see next section); the prompts are identical
plain text with no lookup tokens, so any difference reflects what each twin *learned*, not
different inputs.

## Result (see `cka_by_layer.png`)
Similarity ranges 0–1 (1 = the two models represent inputs identically at that layer).
For each load we show the earliest layer, the layer where the two models are **most
different** (lowest similarity), and the final layer — measured once on trained people
and once on strangers:

| load | people | first layer | most-different layer (similarity) | last layer |
|---|---|---|---|---|
| n50k  | trained (seen)     | 0.30 | layer 11 (**0.27**) | 0.27 |
| n50k  | strangers (unseen) | 0.28 | layer 0 (0.28)      | 0.29 |
| n200k | trained (seen)     | 0.17 | layer 0 (0.17)      | 0.59 |
| n200k | strangers (unseen) | 0.14 | layer 0 (0.14)      | 0.59 |
| n800k | trained (seen)     | 0.33 | layer 0 (0.33)      | 0.76 |
| n800k | strangers (unseen) | 0.40 | layer 1 (0.40)      | 0.77 |

## What it means (grounded, but exploratory)
- **The two models are genuinely different where facts are processed, at every load,
  no matter whose facts we probe with.** Cross-arm similarity stays low (roughly
  0.15–0.40 at its lowest) in both the trained-people and stranger columns, and the
  two columns track each other closely. So the divergence is a real property of the two
  learned networks, not an artifact of which people we happened to test.
- **They differ most in the earliest layers and agree most in the middle.** One reading
  is mechanistic: early layers turn a name+attribute into a "recall this / look this up"
  representation — the computation the two arms handle differently — while middle layers
  do generic language processing shared by both. **But there is a more mundane
  alternative we cannot rule out:** the earliest layers include each model's
  **separately-trained embeddings and low-level features**, and *any* two independently
  trained models differ there — so low early-layer similarity may partly reflect
  "different random low-level solutions," not fact-specific computation. Both readings
  likely contribute; this probe can't separate them.
- **At the last layer the two models become more alike as fact load grows**
  (≈0.28 → 0.59 → 0.76). The interpretation: at 50k people the dense model still
  succeeds at memorization, so its output-stage representations are shaped by "produce
  the memorized value," which differs from the split model's "produce a lookup / use the
  returned value"; but at 200k/800k the dense model can no longer memorize (its recall
  collapses — see the mechanism/superposition write-ups), so its output-stage
  computation defaults toward something closer to the split model's.
- **A tempting-but-wrong sharpening (corrected by a follow-up test below).** It's natural to
  guess the n50k last-layer divergence is dense "emitting the memorized value" on the specific
  facts it recalled. We tested that directly (Test A below) and it is **not** the case — within
  n50k, cross-arm CKA is identical on facts dense recalled vs facts it failed. So the
  divergence at n50k is a **global** property of the low-load dense model's fact pathway, **not**
  a per-fact "am I reciting this one right now" effect. (The follow-up tests are the honest
  read; treat the layer-by-layer curves above as descriptive.)

## Follow-up tests — does our own data support "convergence = dense stops doing fact-special work"?
We ran two extra checks with the *same two models* (no 0.8B, no extra training) to interrogate
the "they converge because dense stops storing" story. See `<load>_cka_tests.json`.

**A generic within-model baseline (resolves the "no baseline" worry).** Cross-arm last-layer
CKA on **knowledge-free reasoning text** (which neither model stores) is **≈0.95 at every
load** — i.e. when neither arm is doing anything fact-special, the two arms are nearly
identical. So the fact-prompt CKA numbers below are genuinely about *fact* processing, not a
global model difference.

**Test B — fact-prompt CKA vs that generic baseline, per load:**

| load | cross-arm CKA on **fact** prompts | cross-arm CKA on **reasoning** text (generic baseline) |
|---|---|---|
| n50k  | 0.36 | 0.95 |
| n200k | 0.59 | 0.94 |
| n800k | 0.76 | 0.96 |

Reading: on fact prompts the arms start very different (0.36) and **converge toward the generic
baseline as load grows (0.36 → 0.59 → 0.76)** — i.e. at high load dense processes fact prompts
much more like split does. **But even at n800k it has not fully reached the generic ceiling
(0.76 < 0.95),** so some fact-arm-specific structure remains; the convergence is real but
*partial*.

**Test A — within n50k, does the divergence live on the facts dense actually memorized?** We
split the fact prompts into ones dense **recalled** vs **failed** and computed cross-arm CKA on
each:

| subset (n50k) | cross-arm last-layer CKA |
|---|---|
| facts dense **recalled** | 0.379 |
| facts dense **failed** | 0.380 |

**Identical.** So the arm-divergence is **not** concentrated on successfully-memorized facts —
it's uniform across all fact prompts. This rules out the simple "output stage emits the
memorized value → diverges only there" mechanism, and says the load-level effect is a broader
reconfiguration of dense's fact pathway. (Only n50k qualifies for this test — at n200k/n800k
dense recalls almost nothing, so there's no "recalled" subset to form.)

**What this means for the "not enough parameters" question.** Two things: (1) the reason dense
fails at high load is **too few exposures, not too few parameters** — the build report shows
each person is seen **196 / 49 / 12** times at n50k/n200k/n800k (fixed 736M-token bio budget),
and 160M params carry ≈320 Mbit of memorization headroom versus only ≈42 Mbit demanded by
800k people, so capacity is *not* the binding constraint at these loads. (2) The convergence is
consistent with "high-load dense stops doing fact-special work and drifts toward split's generic
computation" (Test B direction) — but it is a **global** drift, not a per-recalled-fact effect
(Test A), and it's **incomplete** even at n800k (0.76 vs 0.95). A clean way to settle the
parameter question specifically would be a bigger model at the same fact load (does dense then
re-memorize and re-diverge?) — that needs a run we don't have locally.

## Caveats (fine print)
- **Baseline (partly addressed).** Ideally we'd compare against a same-architecture,
  different-seed pair (dense-seed-0 vs dense-seed-1) to know how much CKA two of *our* models
  share by default. We don't have a second seed, but the **reasoning-text cross-arm CKA (≈0.95)**
  now serves as a within-models "generic, nothing-fact-special" ceiling — so the fact-prompt
  gap is real and fact-related. The remaining unquantified piece is pure seed-to-seed variation.
  Absent a second seed, read the *pattern* (early divergence; fact-prompt CKA rising toward
  the 0.95 generic ceiling with load) rather than the exact similarity values, and don't
  attribute the whole gap to fact-offloading.
- **The exact "most-different layer" is noisy.** On strangers the minimum sits at
  layer 0–1; on trained people at n50k it moves to the last layer. That movement is
  itself the interesting signal (above), but it means you should not cite a single
  "the divergence layer is L0" — lean on the shape of the whole curve.
- **One seed ⇒ a direction, not a proof.** This is a **correlational** description of
  representations, not a causal test of what the layers do.
- CKA can be **dominated by a few high-variance directions**, so treat the exact
  per-layer values as approximate.
- Measured on a sample of structurally-similar recall prompts; a broader prompt mix
  could shift the numbers. All three loads were run locally, on both populations.

## Provenance
Analysis code added on top of the frozen experiment (to be noted in the write-up); it
did not change any training run. Each load was measured twice — trained people from the
frozen `outputs/_frozen_data/<load>_recall.jsonl` and a regenerated stranger set — saved
as `<load>_seen.json` and `<load>_unseen.json`.
