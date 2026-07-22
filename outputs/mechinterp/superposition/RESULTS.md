# "Crowding" probe — are facts crammed together in the dense model?

> **Entity population:** measured on **both** the people the model was trained on
> ("seen") and a fresh set of strangers it never saw ("unseen"); both are reported
> together below because, for this question, they tell the same story. Which population
> is authoritative for which question — and why we load the frozen trained people rather
> than regenerating them — is in [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md).

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **dense** twin memorizes facts, the **split** twin looks them up.*

**What this probe asks, in plain terms.** When a model reads a fact prompt like
"`Kai's major is`", its internal activity is just a list of numbers. We ask: **how many
independent "directions" does the model spread that activity across?** Packing many facts
into a few shared directions makes them interfere — that's **crowding**. Spreading them
over many directions means there's room to spare.

**How to read the number.** The measure is the **participation ratio** — roughly, the
**effective number of directions** the fact activity occupies. **Higher = roomier; lower =
more crowded.** It's a property of the *activity*, not of the weights.

**Why it matters — significance & nuance.** A central reason to off-load facts is that
memorizing lots of arbitrary facts should **crowd** a small model's representations and
eat capacity it could otherwise use for reasoning (the premise behind **H1**). So the
on-hypothesis prediction is: **crowding rises with fact load in the dense twin, but not in
the split twin.** *Important nuance:* a *low* number is ambiguous — it can mean "many facts
crammed together" (crowding) **or** "nothing stored, so the activity collapsed to a
generic blur" (the model giving up). This probe alone can't fully separate the two; the
results below use the ordering across loads to argue between them.

**Which models / checkpoint / people.** Both twins at all three loads (50k/200k/800k), at
the finished checkpoint `snapshots/step0006100.pt`. **One seed ⇒ a direction, not a
proof.** We measured on both the people the model **trained on** and **strangers** it never
saw — the next section explains why, and why it doesn't change the answer. Correlational
(a property of the activity), not a causal test.

## Why this probe, and what it supports
**Motivation.** The whole project rests on a premise from the interpretability and
capacity literature: when a small model must store many arbitrary facts, those facts
compete for the same limited representation space and get packed together
("superposition" / "facts crowd out capacity"). This probe measures that crowding
**directly**, so it tests the *mechanism* behind the primary hypothesis — that removing
the fact-memorization burden frees representational capacity (H1) — and complements the
"facts leave the weights" result (H3). If crowding rises with fact load in the dense
model but not the split model, that is the internal signature of the effect the study
is built around.

**Method/rigor note.** The number is the **participation ratio** of the activations:
`(Σλ)² / Σλ²` over the eigenvalues λ of the last-layer activation covariance across
fact prompts. It equals the *effective number of dimensions* the activations occupy
(1 = all variance in a single direction; D = variance spread evenly over all D
dimensions). Lower ⇒ variance concentrated in fewer directions ⇒ more crowding. It is
a property of the **activations**, not the weights.

## Seen people vs unseen people — why we look at both, and why it doesn't change the answer
It is fair to ask whether "how crowded are the fact representations" depends on
*whose* facts we probe with. If crowding were really about **stored** facts jostling
each other, you might expect it to show up only when the model is recalling people it
actually memorized (seen), and to look different for strangers (unseen), whom it has
nothing stored about. So we measure both. The two populations give **essentially the
same numbers** (table below). That agreement is itself a finding: the low-dimensional
geometry we see is mostly a property of *how the model handles a fact-shaped prompt*
("someone's attribute is …"), not of whether it happens to know that particular someone.
It also means the crowding geometry is not an artifact of *which* people we probe with.

## Result (see `superposition.png`)
Participation ratio (higher = the model spreads fact activity over more independent
directions = *less* crowded; lower = *more* crowded). Shown for people the model was
trained on ("seen") and strangers ("unseen"):

| load | dense, seen | dense, unseen | split, seen | split, unseen |
|---|---|---|---|---|
| n50k  | 6.7 | 5.9 | 7.8 | 8.0 |
| n200k | **1.8** | **1.8** | 6.2 | 6.4 |
| n800k | 2.6 | 2.6 | 6.1 | 6.2 |

## What it means
Two things are solid, and one common-sense reading needs a caveat:
- **Solid, and it holds whichever people we probe with:** the **dense** model always
  uses **fewer** directions than the split model (dense < split at every load, seen or
  unseen), and there is a **large drop for dense between 50k and 200k people**
  (from about 6–7 down to about 1.8). The split model stays roughly steady (≈6–8)
  across all loads — consistent with it having offloaded the facts, so adding more
  people doesn't squeeze its representations. The seen and unseen columns move together,
  so none of this is an artifact of which people we chose.
- **Needs a caveat — it is *not* a clean "more people ⇒ always more crowded" trend.**
  Dense goes ≈6 → 1.8 → 2.6, i.e. 800k is *less* crowded than 200k, the opposite of
  what pure crowding would predict. So the reliable statement is "dense is far more
  crowded than split, with a big change once memorization fails (50k→200k)," **not** a
  smooth monotonic worsening. The 200k-vs-800k difference is small and, like the recall
  wobble, is probably near-floor noise at one seed.

**Two interpretations of the dense collapse — and the trained-people numbers help choose
between them.** There are two readings:
1. *Crowding / superposition:* with more facts than it can cleanly store, the dense
   model packs them into fewer shared directions (interference) — the mechanism the
   project hypothesizes, which would *explain* the recall collapse.
2. *Give-up / generic collapse:* once the dense model **fails** to memorize (200k/800k),
   it may simply have nothing fact-specific to represent, so its activations on these
   prompts flatten into a low-variance, generic "I don't know" state — which also lowers
   the participation ratio, but means something quite different (absence of stored
   content, rather than lots of content crammed together).

Reading the trained-people ("seen") column across loads tips the balance toward the
**give-up** interpretation. The dense model is at its **least** crowded exactly at
**n50k (≈6.7)** — which is the one and only load where it genuinely memorized its facts
(62% closed-book recall on the trained people). It is at its **most** crowded at
**n200k (1.8)**, the load where memorization **failed** (recall near zero). If crowding
were caused *by* successfully packing in many memorized facts, we would expect the
opposite ordering — the memorizing model (n50k) should look the most crammed, not the
roomiest. Instead, low dimensionality lines up with *failure* to store, which is what
you would expect if the activations are collapsing into a generic no-information state.
So the honest summary is: **the dense model's fact representations lose dimensionality
exactly when its memory fails, while the split model's never do** — and that loss looks
more like "nothing left to represent" than like "too much crammed in."

## Deeper interpretation & rigor
Context for the raw numbers: the activation space here is **768-dimensional**, yet the
effective number of directions used is only ≈2–8 — very low, for both seen and unseen
people. Two things cause that: (1) these prompts are structurally near-identical
("X's Y is"), so their activations naturally lie in a low-dimensional subspace, and
(2) on top of that, the **dense** model compresses *further* as facts pile up. So do
**not** read the absolute participation ratio as "the model only has N dimensions"; the
meaningful quantities are the **dense-vs-split gap at each load** and the **downward
trend for dense as load grows** — both of which reproduce across the seen and unseen
populations.

The concept this connects to is **superposition**: a network can represent more distinct
features than it has dimensions by letting them share directions — but at the cost of
**interference** between features. A dense participation ratio falling toward ≈2 at 200k
people means the fact activations are collapsing toward a near-one-dimensional manifold.
As discussed above, the trained-people ordering suggests this collapse coincides with
the model *giving up* on storage rather than with heavy interference from successful
storage; either way, the split model, having offloaded the facts, shows no such collapse.
This is the internal, representational counterpart to the external recall result.

Rigor limits: measured at the **last layer only** (a layer sweep would show where
crowding is worst); it summarizes *variance geometry*, not literally a count of facts;
it is single-seed; and it is correlational (a property of the activity, not a causal
test). Because the seen and stranger populations agree, the geometry is not an artifact
of which people we probe with.

## Caveats (fine print)
- **One seed ⇒ a direction, not a proof.**
- This is a **correlation** (a property of the activity), not a test of cause.
- The absolute numbers are low partly because these prompts are all similar in shape;
  the meaningful things are the **dense-vs-split gap** and the **trend as people
  increase**, not the exact values.
- Run locally on a sample of people, on both the trained and untrained populations.

## Provenance
Analysis code added on top of the frozen experiment (to be noted in the write-up); no
training run was changed. Numbers come from two runs per model — one on the frozen
trained people (`outputs/_frozen_data/<load>_recall.jsonl`) and one on a regenerated
stranger set — stored as `<load>_seen.json` and `<load>_unseen.json`.
