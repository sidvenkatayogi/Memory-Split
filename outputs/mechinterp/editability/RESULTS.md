# Editability: update / add / delete a split-model's facts by editing the store (no retraining)

> **Entity population: SEEN (frozen trained people).** We edit facts the model normally
> serves and check its answers follow; run on the frozen trained people via `evals/frozen.py`.
> See [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md).

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **split** twin doesn't store fact values in its weights — it looks them up
in an external table.*

**What this probe asks, in plain terms.** If the split model's facts really live in an
external table rather than in its trained weights, then **the three ways you'd edit a
database should just work, with no retraining:**
- **update** — overwrite a fact's value → the model returns the new value;
- **add** — insert a fact for a brand-new person → the model can now answer about them;
- **delete** — remove a fact → the model can no longer produce it (instant unlearning).
For each we also check **locality**: the edit must not disturb anyone else's facts.

**How to read the numbers.**
- **edit-success rate** — after changing a fact in the table, does the model's answer switch
  to the new value? (1.0 = always.)
- **locality rate** — after that edit, do *unrelated* facts still return their original
  values? (1.0 = the edit never leaked to anything else.)
Both are ordinary inference — **the model's weights are frozen throughout.**

**Why it matters — significance & nuance.** This is a **causal, behavioral** demonstration of
**H3** that needs no internal probing: a fact you can rewrite by editing an external table
(with zero gradient steps) was, by definition, *not stored in the weights*. It's also the
practically valuable property the whole design is chasing — **updatable, correctable, and
unlearnable knowledge** with no training. *Nuance:* this is a property of the **split arm
only**; the dense arm has no table to edit (its facts are baked into weights and can only be
changed by retraining), so there is no dense-vs-split number — the result is "the split
architecture behaves as designed," not a contrast.

**Which model / checkpoint / people.** The **split** twin at all three loads (50k/200k/800k),
final checkpoint `snapshots/step0006100.pt`, 100 edits + 100 separate locality checks per
load, on the frozen trained people. One seed.

## Background: what the "organizer" is, and why editing it works

**The organizer (a.k.a. "the store" or "lookup table")** is a plain dictionary that maps a
**key** — `"{person}, {relation}"` — to a **value**. For example:

```
"Kai Bramble, employer"    -> "Cascade Logistics"
"Kai Bramble, birth_city"  -> "Willowford"
...one row per (person, attribute) the model was trained on...
```

It is **not part of the neural network.** It's an ordinary external table that sits next to
the model at inference time. (See `organizer/store.py`.)

**Why the split model has no fact values in its weights.** During training, whenever a fact
value would have appeared in the text, it was **masked out of the loss** and replaced by a
"look-it-up" call. So the split model was *never rewarded for knowing the value* — only for
learning **when to ask and how to phrase the question**. That's why, with the table
unplugged, its recall is 0% (mechanism/ppl-slices probes): the value was never in the weights
to begin with.

**What actually happens in one inference (step by step).** When we ask the split model
"`Kai Bramble's employer is`", generation goes like this:
1. The model emits a special token `<db_start>`, then writes the **query** `Kai Bramble,
   employer`, then emits `<db_retrieve>`. (This is the "address" the `keyguess` probe scores.)
2. The decoding harness (see `evals/generate.py`) **intercepts** that `<db_retrieve>`, takes
   the query text, and **looks it up in the organizer**.
3. Whatever value the organizer returns is **fed back into the model's context as tokens**
   ("force-decoded"), and generation continues from there.

So the value the model "says" is **literally whatever the table returns for that key** — it is
read live from the table on every inference, not recalled from memory.

**Why editing the table changes the answer (it's not "retroactive" — it's live lookup).**
Because step 2 reads the value from the table *every time*, if we change the table entry
`"Kai Bramble, employer" -> "Delta Robotics"` and ask the same question again, step 2 now
returns "Delta Robotics", step 3 feeds that back, and the model emits "Delta Robotics." No
weight changed; the model is simply reading the new value from the (now edited) table. This is
different from the **dense** model, where "Kai's employer" is encoded in the weights — the only
way to change *that* answer is to retrain.

**Worked example of the actual test:**
- Pick a trained person + attribute, e.g. `("Kai Bramble", employer)`, whose true value is
  "Cascade Logistics".
- **Edit:** overwrite the organizer entry to a different valid value, e.g. "Delta Robotics".
- **Edit-success:** ask "`Kai Bramble's employer is`" with the edited table attached — does the
  model now answer **"Delta Robotics"**? (It does, 99–100% of the time.)
- **Locality:** ask a *different* person's fact (that we did **not** edit) with the same edited
  table — do they still get their **original** correct value? (They do, 100% of the time —
  because each key is independent, editing one row can't touch another.)

## Result
All three store operations succeed at every fact load, and every edit stays **local** (it
never disturbs other people's facts). Both columns are fractions (1.0 = perfect):

| operation | "success" means | success (n50k / n200k / n800k) | locality (n50k / n200k / n800k) |
|---|---|---|---|
| **update** | model returns the overwritten value | 1.00 / 1.00 / 0.99 | 1.00 / 1.00 / 1.00 |
| **add** | model answers about a brand-new person whose fact is **only** in the store | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| **delete** | model **no longer** produces the removed value (unlearning) | 1.00 / 1.00 / 1.00 | 1.00 / 0.99 / 1.00 |

**Delete is *clean* unlearning, not a confident wrong answer.** For a deleted fact the model
still emits the correct lookup query, but the store now returns nothing, so **every one of
those lookups misses** — e.g. at n50k: **870 lookups, 0 hits, 870 misses**. It fails by
*coming up empty*, and the true value is surfaced **0%** of the time. (After the miss the model
may continue with filler; what's guaranteed is that the deleted value is no longer produced —
if a hard "I don't know" refusal were required, that's a separate behavior to add.)

## Interpretation
**The split model's knowledge is fully external and behaves like an editable database —
update, add, and delete all work with zero weight updates.** Rewriting a row flips the answer
(update), inserting a row makes a brand-new person answerable (add), and removing a row makes
the fact unproducible (delete/unlearn) — each ≈99–100%, and none of them disturbs anyone
else's facts (≈99–100% locality). This is direct, outside-in proof of H3: the knowledge the
split model serves is the *table's*, not the *weights'*. It's the behavioral complement to two
earlier findings: (a) the split weights store ≈0 fact values (mechanism / ppl-slices), and
(b) the retrieved value is what actually drives the model's output (value-injection). It holds
regardless of fact load — expected, since these are properties of the *lookup path*, not of
how many facts exist.

**Why add and delete follow from the same live-lookup mechanism.**
- **Add** works because the model reads values from the table at inference — but there's a
  prerequisite it must satisfy first: it has to **emit the correct key** for the new person.
  It can, for **in-distribution** (synthetic) newcomers — that's exactly the `keyguess` result
  (≈100% correct addresses for never-trained synthetic people). So "add a person, model
  answers about them" is really "the model addresses them + the store returns the added value."
- **Delete** works because a removed key makes the lookup **miss**, and the split model has
  **nothing in its weights to fall back on** (store-OFF recall = 0%). The miss statistics make
  this concrete and clean: the model *does* form the right query, it just retrieves emptiness
  (0 hits / all misses), so the true value is never produced. This is instant unlearning by
  deletion — impossible for the dense model without retraining.

The reason edit-success isn't a surprising "the model changed its mind" is the mechanism
above: the model reads the value from the table on every inference, so **the table is the
single source of truth** for values. Editing the table is editing the model's knowledge.

Contrast with the dense arm (not run here because it's impossible by construction): a dense
model's fact is in its weights, so the only way to change it is to retrain — exactly the
rigidity the memory-split design removes. This is why `dense_editable` is reported as `false`.

## Guardrail: what happens when the store has no answer (no hallucination)
Editing/deleting raises an obvious safety question: **when the store can't answer, does the
model make something up?** We tested this directly (querying facts that are *not* in the store,
so every lookup misses) and the answer is a clean **no**.

**On a store miss, the split model fabricates a plausible fact value 0% of the time** (across
n50k / n200k / n800k, ≈450 miss queries). It writes the correct query, the store returns
nothing, and — instead of inventing a value — it emits **generic filler**, e.g.:
```
"Bila Hola Eastberg's birth city is" -> …<db_retrieve> The family register<db_end>. The family register lists Bila …
```
| load | fabrication rate on a miss | no value asserted | in-store control (retrieves correct) |
|---|---|---|---|
| n50k  | **0.0%** | 99.3% | 100% |
| n200k | **0.0%** | 97.3% | 99.3% |
| n800k | **0.0%** | 100% | 100% |

**Why it can't confabulate — same mechanism as H3.** The value slot (right after
`<db_retrieve>`) was **loss-masked in training**, so the model was *never trained to generate
values at all* (this is also why store-OFF recall is 0%). With no learned value distribution to
draw on, a miss falls back to generic prose rather than a made-up value. The very mechanism
that externalizes the facts is what prevents the model from fabricating them. **Contrast with
dense:** when the dense model doesn't know a fact it *does* confabulate a format-correct but
wrong value — so on unknown facts the arms behave oppositely (dense = confidently wrong; split
= declines).

**The one residual "confidently wrong" path is mis-addressing, and it's rare.** The guardrail
above assumes the lookup *misses*. It could instead **hit the wrong row** if the model emits a
wrong-but-well-formed key. From `keyguess` this is quantified and small: keys are correct
≈99–100% of the time; the specific dangerous case — a **right-person / wrong-attribute** key,
which still hits that person's row and returns a real (but wrong-attribute) value — occurred
**1 in 1,200 at n200k and 0 elsewhere** (≈0.08%). Keys that are wrong on *both* halves (the
other ≈0.1–0.25%) almost always **miss** (→ filler, no fabrication) rather than collide. So the
end-to-end "returns a real but wrong value due to mis-addressing" rate is **well under 0.1%**
here.

**Net guardrail picture:** on a genuine knowledge gap the split model **declines rather than
fabricates** (0% invented values), and the only confidently-wrong failure — mis-addressing to a
real row — is rare (<0.1%). Two operational caveats: (1) the model **doesn't explicitly refuse**
(it emits filler), so a caller should treat a **store miss** as "no answer" rather than parsing
the text; (2) the guarantee is conditional on the lookup missing — see the mis-addressing rate
above.

## Caveats
- **Split arm only** — there is no dense comparison (dense has no table), so read this as
  "the split architecture works as intended," not a contrast.
- **Exact-match table**, trained people, one seed. Edit-success rides on the model emitting a
  well-formed lookup for these (in-distribution) people — which it does ≈100% of the time
  (`keyguess`); the *edit* result is specifically that changing the value redirects the
  answer, and that unrelated lookups are unaffected.
- **"Add" is bounded by what the model can *address*, not by the table.** Adding works for
  **in-distribution** (synthetic) newcomers, whom the model can key correctly (`keyguess`).
  Adding a **real-world** person (e.g. "Barack Obama") would fail — not because the table can't
  hold the row, but because the model can't generate a correct key for out-of-vocabulary names
  (`popqa_keyguess`). So add-success here is "add + retrieve for people the model can address."
- **Delete guarantees the true value isn't produced, not a clean refusal.** After a miss the
  model may emit filler; it just won't surface the deleted value. Explicit "I don't know"
  behavior would need to be added/trained.
- **Behavioral** (what the model outputs) — the right level for a "facts are external and
  editable" claim; no internal probing is needed or used.

## Provenance
**Update:** `scripts/run_editability.py --records-jsonl …` (`evals/editability.py`).
**Add / delete:** `scripts/run_add_delete_edit.py --records-jsonl …` (inserts fresh
in-distribution people, or removes keys, from a copy of the organizer). Both on the frozen
trained people (`outputs/_frozen_data/<load>_recall.jsonl`), split arm, weights frozen. Raw
per-load JSON in this folder as `{load}_editability.json` (update) and `{load}_add_delete.json`
(add/delete). No training run was changed.
