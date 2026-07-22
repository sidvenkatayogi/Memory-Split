# Entity populations: SEEN vs UNSEEN (read this before trusting any probe number)

Every probe here runs the model on **fact prompts about specific synthetic people**.
*Which* people — ones the model was trained on, or ones it wasn't — determines what a
probe's number *means*. This note defines the two populations, records a reproducibility
fact you must know to run these probes, and states which population is authoritative for
each probe.

## The two populations

- **SEEN** — the exact people the checkpoints were **trained on**. The faithful source is
  the **frozen data** written alongside the training tokens (`eval/recall.jsonl`,
  `organizer.jsonl`); we load these via `evals/frozen.py`
  (`records_from_recall_jsonl`), pulled to `outputs/_frozen_data/<load>_recall.jsonl`.
  **Use SEEN for any question about what the model *stored*.**
- **UNSEEN** — novel, in-distribution people the model **never saw** (same name/value
  pools, same generation process, a different draw). **Use UNSEEN for questions about
  *generalization* or the model's *prior* over values.** UNSEEN says nothing about
  memorization (there is nothing stored to find).

## Reproducibility fact: regenerating people ≠ the trained people
You **cannot** recover the trained people by calling `corpusgen.bios.generate_records`
at probe time — the repo's generator produces a *different draw* than the corpus the
checkpoints trained on:

| | frozen trained data | current repo `generate_records(seed=0)` |
|---|---|---|
| person 0     | `Brelian Lorec Cliffyard` | `Bremelle Joron Swiftwell` |
| person 31352 | `Berian Aris Birchby` (b. `July 1, 1994`) | `Buselle Pela Langdale` (b. `September 16, 1951`) |

The **pools are identical** (all these names/values exist in both) — only the RNG
sequence differs, so the two generators diverge from person 0, and **no seed reproduces**
the frozen set. **Therefore: to probe SEEN people, always load the frozen data; a
"regenerate the people" call yields UNSEEN people instead.** Every probe here is run on
the population that is correct for its question (below), and most are run on **both** so
the trained-vs-stranger contrast is available.

## Which population answers which question
- **"Is the fact stored in the weights?" (memorization / storage)** — **SEEN.** On
  strangers the model has nothing stored, so an UNSEEN run reads ≈0 and says nothing
  about storage; it serves only as a control that isolates entity-specific stored
  knowledge from generic format/prior effects.
- **"Does the skill generalize to new people?" (generalization)** — **UNSEEN is the
  point.** `keyguess` is this: emitting the right key for an unseen person is the
  measurement (a memorize-the-keys model would fail on strangers).
- **"What does the model know about the value distribution?" (prior)** — **FRESH/UNSEEN
  by design.** `ppl_slices` fact-value NLL uses held-out people to read the value prior.
- **"What's in the weight matrices?" (weights only)** — **population-independent.**
  `weight_spectral` never runs a forward pass.
- **"Where does an injected value enter the stream?" (mechanism)** — **entity-agnostic:**
  the value is supplied by the harness, so `value_injection` is unaffected by membership.

## Per-probe: authoritative population and current reading

| probe | authoritative population | current reading |
|---|---|---|
| `extractability` | **SEEN** | On trained people the fact *is* recoverable (battery `recall.json`: dense n50k = 62.4%). The stranger run is a control (recall ≈0 / recognition ≈chance, as expected). |
| `mechanism` — Finding 1 (localization) | SEEN | Dense's fact units are ≈5–6× less fact-selective in the split arm (3.4× at n800k); the trained-people and stranger numbers match, because selectivity tracks the fact-shaped prompt. |
| `mechanism` — Finding 2 (recall table; fact-use QA) | SEEN (battery) | The core H3 result: split ≈100% with its table, 0% without; dense closed-book recall collapses with load (62.4% → ≈1%). |
| `mechanism` — Finding 3 (linear probe-bits) | SEEN | Underpowered even on trained people (dense-n50k `major` ≈0.05) — do not cite. |
| `superposition` (participation ratio) | SEEN | Dense collapses and dense ≪ split at every load, on both populations; the trained-people ordering favours the "give-up collapse" reading. |
| `geometry_cka` (cross-arm CKA) | SEEN | Low cross-arm CKA at all loads on both populations; the argmin-layer is noisy. |
| `value_injection` (split) | entity-agnostic | The injected value's effect builds across depth (peaks near the output) — read the shape, not raw magnitudes. |
| `weight_spectral` (MLP eff-rank/norm) | N/A (weights only) | Clean null: no dense-vs-split difference at this coarse whole-matrix measure. |
| `fact_weight_attribution` | SEEN | No dense-vs-split localization on trained people (dense n50k 18.7k ≈ split 21.7k); only a within-dense, across-load pattern holds. Do not cite a cross-arm ratio. |
| `ppl_slices` — fact-value NLL | FRESH | Split (store OFF) sits at the "no idea" ceiling at every load; dense's value prior sharpens with more people. |
| `ppl_slices` — reasoning bpb | N/A (generated problems) | Dense ≈ split — no reasoning-side regression from splitting. |
| `keyguess` | UNSEEN | ≈100% correct key on unseen people ⇒ addressing is a general copy+select procedure, not memorized keys. |

## Where the numbers live
For the storage probes we keep both populations on disk: `<load>_seen.json` (trained
people, authoritative) and `<load>_unseen.json` (strangers, control); `mechanism` uses
`mechanism/<load>/` and `mechanism/<load>_seen/`. Each probe's `RESULTS.md` presents the
current reading with both populations woven in.
