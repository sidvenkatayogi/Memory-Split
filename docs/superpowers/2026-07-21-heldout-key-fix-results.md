# Held-Out Key Generalization — Fix Results (Copy-Constrained Decoding)

2026-07-21. Local CPU experiment, single seed (seed bundle 0). Follow-up to
`2026-07-20-heldout-key-generalization-results.md` (the 2.5% name-half
failure). The lost harness was recreated from that doc's protocol and is now
committed (`corpusgen/realfact.py`, `evals/keyguess.py`, `evals/constrain.py`,
`scripts/run_keyguess_local.py`); reproduce with:
`scripts/fetch_realfacts.py` then `scripts/run_keyguess_local.py --stage all`.
Replication seeds run via `--seed N` (FarmShare: `cluster/slurm/keyguess_cpu.sbatch`).

## Question

Does making the copy route structural fix the measured failure — the model
memorizing entity names into keys instead of copying them from the prompt?
Two candidate fixes, isolated and combined:

- **Inference fix:** copy-constrained query decoding — during
  `<|db_start|>…<|db_retrieve|>`, the name half may only extend a span of the
  live prompt (token trie with boundary-punctuation/possessive variants), the
  relation half comes from the closed 16-relation grammar. Non-copyable names
  become unemittable.
- **Training fix:** copy-dominance data — counterfactual name substitution on
  ~50% of real-fact traces (name swapped consistently across question + key
  within relation, per-exposure permutations) + 2,400 fresh-name flood docs
  (each name appears exactly once).

## What we did

Protocol as on 07-20, recreated: PopQA raw 14,267 → cleaned 13,063 → capped
3,000; per-relation floor split **2,394 seen / 606 held-out** (07-20 snapshot:
2,399/601 — upstream dataset drift, documented in `data_manifest.json`);
6 exposures per seen fact as single-hop Question/Reasoning/Answer traces with
values loss-masked; shared synthetic base (2,700 docs; 300 bios entities x 6
exposures + 900 factqa docs); toy 4L/256d ~29M, ctx 192, 800 steps, CPU;
organizer holds all 3,000 facts at eval. Corpus A ≈ 1.07M tokens; corpus C
(substitution + flood) ≈ 1.21M tokens; masks verified byte-exact around
`<|db_retrieve|>` spans in review.

Four arms; B/D differ from A/C only at decode time:

| Arm | Training corpus | Decoding |
|---|---|---|
| A | baseline (as 07-20) | free argmax |
| B | A's checkpoint | copy-constrained |
| C | + substitution + flood | free argmax |
| D | C's checkpoint | copy-constrained |

Gold-key emittability through the span trie (hard gate for B/D): **804/806 =
99.75%** — 2 structural misses (11-word subjects beyond the n-gram cap), so
span extraction caps constrained-arm ceiling at 99.7%, not a model limit.
Both trainings kept the mechanism intact (masked-value CE ≈ 10 vs general
loss ≈ 0.7 at step 800: fact values were never learned into weights).

## Results — emitted-key decomposition (held-out, n = 606, Wilson 95%)

| Arm | Full key | Name-half | Relation-half | Answer | No-lookup | Wrong-in-context name |
|---|---|---|---|---|---|---|
| A | 0.0 [0.0, 0.6] | 0.0 [0.0, 0.6] | 96.0 [94.2, 97.3] | 2.1 [1.3, 3.6] | 1.0 | 5.4 |
| B | **24.3 [21.0, 27.8]** | 24.4 [21.2, 28.0] | 97.2 [95.6, 98.2] | **26.2 [22.9, 29.9]** | 1.0 | 74.6 |
| C | 0.2 [0.0, 0.9] | 0.2 [0.0, 0.9] | 94.2 [92.1, 95.8] | 2.1 [1.3, 3.6] | 2.1 | 4.0 |
| D | 23.6 [20.4, 27.1] | 24.6 [21.3, 28.2] | 94.1 [91.9, 95.7] | 25.7 [22.4, 29.4] | 2.1 | 73.3 |

Seen split (n = 200): A 5.0% full key, B 63.0%, C 4.0%, **D 71.5%** (all
relation-half ≥ 93.5%).

## Results — the governance view (ship-on-store-hit, no gold labels)

The deployable policy "splice whatever the store returns on a hit" separates
the arms far more sharply than raw accuracy:

| Arm | Coverage (hit rate) | Precision among shipped | Silent-wrong (all items) | Wrong-referent keys among hits |
|---|---|---|---|---|
| A | 34.5% | **2.9%** | **33.5%** | 209/209 |
| B | 25.4% | **95.5%** | **1.16%** | 7 |
| C | 35.1% | 3.8% | 33.8% | 212/213 |
| D | 24.6% | **96.6%** | **0.83%** | 6 |

The baseline is not merely unhelpful — it is actively dangerous: every one of
its 209 held-out store hits is a memorized key for the WRONG referent, so a
naive splice pipeline would ship silently wrong values on a third of all
queries. The copy constraint converts that into a selective system: a quarter
of queries answered at ~96% precision, silent error collapsed **33.5% → 0.83%
(40x)**, and the residual six wrong-referent hits are exactly the
valid-but-wrong-key class that the proposal's mention-similarity +
discriminator verification targets. Under the oracle name-check (ship iff
emitted name == gold subject), precision is 96.6–100% at the same coverage —
the gap between 95.5% and 100% is the measured value of one verification vote.

## Reading

1. **Arm A reproduces the failure signature.** Name-half 0.0 [0.0, 0.6] vs
   2.5 [1.5, 4.1] on 07-20 (overlapping intervals; different PopQA snapshot,
   seeds, and batch schedule); relation-half 96.0 vs 98.5. Relation transfers,
   name-copy does not. Harness validated.
2. **The inference-side constraint is the fix that matters at this scale:**
   full-key 0.0% → 24.3% and end-to-end answer 2.1% → 26.2% from the SAME
   checkpoint — the failure was emission, not knowledge, for a quarter of
   items. The training-side fix alone moved nothing held-out (C ≈ A,
   0.2%): at 29M/800 steps, substitution data does not teach unconstrained
   copying. It does help WITH the constraint on seen facts (D 71.5% vs
   B 63.0%): substitution improves span ranking where knowledge exists.
3. **The bottleneck relocated, as designed.** With unemittable junk removed,
   74.6% of items now fail by picking the WRONG in-context span ("The,
   author"; "Question: Who, author") — the model cannot rank candidate spans
   it never learned to score. This is a capability gap at 29M, not an
   architecture gap: the 07-20 doc records the project's 160M model reaching
   100% on held-out synthetic names, and the span-ranking signal (pointer
   loss over spans) is exactly what the full Tier-S trains.
4. **Go-gate verdict, honest:** the pre-registered go threshold (verified
   end-to-end ≥60%, Wilson LB ≥55%) is **NOT met at 29M** — 26.2 [22.9,
   29.9]. What IS established: the structural claim (0 → 24 full-key from
   the same weights; silent error 33.5% → 0.83%) and the verification story
   (95.5% → 96.6% → 100% precision as checks stack). The kill-lever does not
   fire either (it targets Tier-C prefix accuracy, not Tier-S). Next rung per
   the experiment ladder: 160M with the pointer-ranking loss and seeds 1–4
   (FarmShare CPU jobs for the 29M seeds are packaged and handed off).

## Caveats

Single seed at toy scale (seeds 1–4 dispatched to a collaborator; directional
until they land). Ship-on-hit precision uses answer-string match against
PopQA possible_answers (aliases may undercount). The 2 emittability misses
are counted as failures in all rates. Single-hop extraction traces only; the
organizer is exact-match (no fuzzy/alias resolution). Raw numbers:
`data/keyguess_local/summary.json`, per-item `records_{A..D}.jsonl`,
console table in `eval.console.log`.
