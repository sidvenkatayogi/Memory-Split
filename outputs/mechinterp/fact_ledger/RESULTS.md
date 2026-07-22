# Fact ledger: how many bits of fact does the dense model actually store?

> **Entity population: SEEN (frozen trained people).** "How much did it store?" is only
> defined for people it trained on; run via `evals/frozen.py`. See
> [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md).

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one line
you need: the **dense** twin memorizes facts in its weights.*

**What this probe asks, in plain terms.** Several probes estimate "how much fact information
is in the weights," and they don't always agree. This one **reconciles them into a single
ledger** per model: how many bits of fact the model has stored, according to (a) what it can
actually **generate** (recall), and (b) what a **linear probe** can read out of its
activations — and it makes the gap between the two explicit.

**How to read the numbers.**
- **recall-bits** — total fact information the model can *produce* (from generative recall,
  converted to bits). This is the trustworthy figure.
- **probe-bits** — total fact information a *linear read-out* can recover from activations.
- **bits per entity** — recall-bits ÷ number of people, so loads are comparable (each person
  carries up to ≈53 bits of fact across the 6 attributes).
- **extractability gap** = probe-bits − recall-bits. Its *sign* is the story: in the classic
  case a positive gap means "stored but not generatively producible"; a large *negative* gap
  (our case) means **the linear probe is far weaker than generation** — i.e. the probe
  under-reads, so trust recall-bits.

**Why it matters — significance & nuance.** It turns the scattered storage numbers into one
auditable statement per model and per fact load, and it quantifies the **memorization
collapse** (Q3): how storage per person falls off as more people compete for the same training
budget. *Nuance:* this is **pure aggregation** — it inherits the strengths/limits of its
inputs. Our probe-bits come from a probe we've shown is **underpowered**, so the ledger's
honest use is "recall-bits is the storage estimate; the probe-bits column documents that the
linear probe can't recover it." (The companion `capacity_scaling` idea — bits vs parameters —
needs the 1B model and isn't computed here.)

**Which model / checkpoint / people.** Dense at all three loads (50k/200k/800k), final
checkpoint `snapshots/step0006100.pt`, trained people. One seed.

## Result
| load | closed-book recall | recall-bits (total) | recall-bits **per entity** | probe-bits (total) |
|---|---|---|---|---|
| n50k  | 0.622 | **1,657,145** | **33.1 / entity** | 124 |
| n200k | 0.009 | 49,273 | 0.25 / entity | 17 |
| n800k | 0.008 | 161,252 | 0.20 / entity | 20 |

## Interpretation
**Read the "per entity" column — it's the clean statement of the memorization wall.** At
50k people the dense model stores ≈**33 bits of the ≈53 bits per person** (it genuinely
memorized most facts); at 200k and 800k it stores ≈**0.2 bits per person** — essentially
nothing. That is the dose effect the whole sweep is built to show: with a fixed training
budget, more people means fewer exposures each, and dense memorization falls off a cliff
between 50k and 200k.

**Watch out for the total-bits column, which is misleading on its own:** n800k's total
(161k) is *larger* than n200k's (49k) even though its per-person recall is lower — simply
because n800k has 4× more people, so tiny per-person storage × many people still sums to
more raw bits. The per-entity figure removes that confound.

**The probe-bits are ≈10,000× smaller than recall-bits** (e.g. 124 vs 1.66M at n50k), giving
a large *negative* extractability gap. This is **not** "facts are hidden from generation" — it
is the opposite: generation recovers far more than the linear probe, confirming (as the
mechanism probe found) that **the linear probe is underpowered** and should not be used to
quantify storage. The reliable storage number is recall-bits.

## Caveats
- **Aggregation only** — inherits its inputs' limits. Probe-bits come from an underpowered
  linear probe (documented in `mechanism/`), so treat them as a lower bound / instrument
  check, not a storage estimate.
- **MC-recognition omitted** (the earlier MC run was on the wrong population); adding a
  trained-people MC term would complete the three-way reconciliation.
- **`capacity_scaling` (bits vs params) not computed** — needs the 1B dense model; only 160M
  is available locally. One seed.

## Provenance
`scripts/run_causal_probes.py --do ledger` (`probe.ledger.fact_info_ledger`); recall-bits via
`evals.recall.bits_in_weights` on the frozen trained people
(`outputs/_frozen_data/<load>_recall.jsonl`); probe-bits read from the `mechanism/<load>_seen`
reports. Raw per-load JSON in this folder as `{load}_ledger.json`. No training run was changed.
