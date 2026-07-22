# PopQA key-generation — does the addressing skill transfer to the *real world*?

> **Entity population: real-world (PopQA), i.e. maximally OUT-of-distribution.** Every
> other probe here uses the study's synthetic people; this one uses **real Wikidata
> entities** (from PopQA) to stress-test how far the split model's external-memory skill
> generalizes. Background on populations: [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md).

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **split** twin doesn't store fact values — it writes a "look-it-up"
query (an address like "`{person}, {attribute}`") and an external table returns the value.*

**What this probe asks, in plain terms.** The companion [`../keyguess/`](../keyguess/) probe
showed the split twin can write the correct lookup address for **unseen synthetic people**.
This probe pushes on the obvious next question: **does that skill reach real-world people?**
We take real facts from **PopQA** (e.g. "Serjik Teymourian — place of birth — …") and ask
the split twin to produce the lookup address, exactly as before.

**Why this is a hard, honest stress test (mind the mismatch).** Our model was trained *only*
on **6 made-up relations** and **made-up people**. PopQA has **16 real relations** and real
names. So there are two separate mismatches: (1) **relation mismatch** — only PopQA's
"place of birth" loosely maps to our `birth_city`; the model has never seen the other 15
relation words; and (2) **entity mismatch** — real names (e.g. "Marnus Labuschagne") are
built from tokens outside the model's learned name vocabulary. Because of this, the only
signals that can possibly transfer are (a) *does the lookup machinery fire at all* on
real-world prompts, and (b) *does the name-copy step handle real names*. Exact keys are
essentially impossible by construction, and we read the numbers in that light.

**How to read the numbers.** `lookup_fired` = fraction of prompts where the model emitted a
well-formed address; `name_exact (given fired)` = of those, how often the address contained
the *correct real* name; `place-of-birth fired` = firing rate on the one relation the model
has a word for.

**Why it matters — significance & nuance.** This is the **transfer/scope test** for the
whole external-memory idea: an addressing skill is only useful at scale if it works on
entities and relations the model wasn't spoon-fed. A strong result here would say "the split
model can address open-world knowledge"; the actual (near-floor) result says the opposite for
*this* model, and — crucially — pins down *why* (relation-gated firing + regenerate-not-copy
names). **Nuance:** low numbers are **not** evidence against the memory-split idea — our model
was trained on a closed synthetic world, so this is a statement about **its scope**, and the
right subject for a real-world *claim* is a model trained on real text (see Next steps).

**Which model / checkpoint / data.** The **split** twin at all three loads
(50k/200k/800k), finished checkpoint `snapshots/step0006100.pt`; 1,534 PopQA facts
(≈100 per relation × 16). One seed.

## Result
| load | lookup fires (overall) | never emits a lookup | correct real name \| given it fired | place-of-birth fires |
|---|---|---|---|---|
| n50k  | 2.5% | 97.5% | **0%** | 39% |
| n200k | 0.4% | 99.6% | **0%** | 5% |
| n800k | 0.07% | 99.9% | **0%** | 1% |

Two facts dominate everything:
1. **The lookup machinery almost never triggers on real-world prompts** — and it triggers
   *less* as the model is trained on more people (2.5% → 0.4% → 0.07%). Breaking it down by
   relation is stark: **every lookup that fired came from "place of birth"** (the one
   relation whose phrase, "birth city", the model actually knows). For the other **15
   relations the firing rate is exactly 0%** — the model never even attempts a lookup when
   the relation word is one it wasn't trained on.
2. **When a lookup does fire, the model does not copy the real name — it invents a synthetic
   one.** `name_exact` is **0% at every load**. The emitted addresses make this vivid:

   | real person in the prompt | address the model emitted |
   |---|---|
   | Serjik Teymourian | `Breda Ilec Winley, birth_city` |
   | Mariano Chao | `Baelian Joren Daleward, birth_city` |
   | Marnus Labuschagne | `Breda Fenec Hawkgate, birth_city` |
   | Hendrik Gerritsz Pot | `Bredis Holia Claymere, birth_city` |

   It keeps the *shape* ("First Middle Last, birth_city") and the correct relation, but
   **replaces the real name wholesale with a made-up name drawn from its training
   vocabulary.** Even a *partial* copy essentially never happens: the emitted address
   matches the real person's surname only **≈2.6%** of the time (n50k) — no better than
   chance overlap — so this is genuine substitution, not a garbled copy.

## Interpretation — what this says about our model's scope
**The externalization skill our model learned is real but distribution-bound; it does not
transfer to the real world.** Put together with the synthetic [`../keyguess/`](../keyguess/)
result (≈100% correct addresses for unseen *synthetic* people), the picture is precise:

- **The lookup is *cued by* the relation vocabulary.** The machinery fires only when the
  prompt contains a relation the model was trained on. Unseen relation words (occupation,
  director, author, …) don't trigger it at all — so the model can't address attributes it
  was never taught, even if it could copy the entity.
- **The "name copy" is really "regenerate an in-vocabulary name," not a literal copy.** On
  synthetic people this is invisible, because unseen synthetic names are novel *combinations
  of in-vocabulary tokens*, which the model reproduces perfectly. But real names are made of
  **out-of-vocabulary tokens**, and there the mechanism is exposed: it cannot carry those
  tokens through, so it substitutes a plausible synthetic name. This refines the keyguess
  conclusion — the addressing procedure generalizes across **new arrangements of familiar
  vocabulary**, not across **genuinely new tokens**.
- **It gets *more* locked-in with scale, not less.** Firing drops sharply from n50k to
  n800k. A model trained on more (synthetic) people is even more strongly tuned to its own
  distribution, so it is even less likely to engage its lookup on foreign input.

None of this is a failure of the *idea* — it is the expected behaviour of a model trained
end-to-end on a narrow synthetic world. It says the **capability is there but its
"trigger" and its "copy" are both tied to the training distribution**, which is exactly the
kind of thing you'd want to broaden before deploying such a system on open-domain text.

**One caveat on the mismatch working in our favour and against us.** We prompted with our
training-style stub ("`{name}'s birth city is`") rather than PopQA's natural question
("What is {name}'s place of birth?"). Our stub is the *most* favourable phrasing for
triggering our model (it matches training), and even so firing is rare — a natural QA
prompt would almost certainly fire even less. So these numbers are, if anything, an
*optimistic* ceiling on real-world transfer for this model.

## Next steps — what it would take to generalize further
If the goal is an external-memory model that addresses **real** knowledge, the changes are
concrete and independent of one another:
1. **Broaden the relation set and its phrasings.** Train with many relations (dozens+) and
   varied natural-language cues per relation, so the lookup triggers on relation *meaning*,
   not a memorized phrase. This directly fixes the "0% firing on 15/16 relations" problem.
2. **Break the entity-vocabulary dependence.** Train on realistic/diverse entity names (or
   real corpora), and/or give the model an explicit **copy/pointer mechanism** so the
   address-name is *copied from the prompt tokens* rather than *generated* from a learned
   name distribution. This is what would let it carry "Marnus Labuschagne" into a key.
3. **Train with question/instruction formats**, not just inline biography, so the lookup
   fires on the prompt styles real users write.
4. **Validate on the collaborator's intended setting** — an LMLM-style model trained on
   real text — where PopQA is in-distribution; our synthetic model is the wrong subject for
   a real-world claim, but the *harness* here (fetch → prompt → capture address → score
   name/relation halves + taxonomy) transfers directly.
5. **Add a real-name copy diagnostic** on synthetic training that deliberately inserts
   out-of-vocabulary tokens, to measure copy-vs-regenerate directly and track it over
   training.

## Caveats
- **Our model is the wrong subject for a real-world *claim*** — it was trained on synthetic
  data, so low numbers are expected and are a statement about *scope/transfer*, not a
  weakness of the memory-split idea. The value here is the precise *shape* of the failure
  (relation-gated firing; regenerate-not-copy names).
- **Only "place of birth" has a valid mapping** to our relations, so exact-key is only
  meaningfully defined there (and is 0% because the name never transfers).
- **Prompt phrasing is our training stub**, the most favourable case for firing; natural QA
  phrasing would be harder. One seed; split arm only.

## Provenance
`scripts/run_popqa_keyguess.py` + `evals/keyguess.py` (address extraction), PopQA pulled to
`outputs/_popqa/popqa_test.tsv`. Per-load raw JSON here as `{load}_popqa.json` (includes
per-relation breakdowns and example emitted addresses). No training run was changed.
