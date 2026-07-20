# Preregistration — Memory Split battery (FROZEN 2026-07-20)

Committed before any battery run starts. Amends the design spec's §3.4
per the gate-phase outcomes (three gate-A rounds) and the owner's decision
of 2026-07-20. The analysis code is the repo at the commit carrying this
document; any later analysis-code change must be reported in the final
writeup.

## Frozen corpus recipe

Mixture: bed 0.54 / bios 0.23 / igsm 0.12 / deduction 0.08 / factqa 0.03.
iGSM: train op 1-4, 1/op-weighted, <=1 distractor at op<=2; eval ID op 1-4
uniform, OOD op 5-8 report-only. Deduction: train depth 1-2, small bases
(<=6 facts, <=4 rules); OOD depth 3-4 report-only. Loads: sweep n50k /
n200k / n800k at 3.2B tokens (160M); 1B tier n800k_1b / n4m_1b at 10B
tokens. All battery corpora rebuilt on this recipe before use (earlier
corpora are void).

## Hypotheses and endpoints

- **H1 (primary).** At matched params/tokens/data-order, the split arm
  beats the dense arm on **fact-use QA** — held-out questions requiring
  retrieval of two stored facts plus an inference step (date comparison,
  city equality) or one fact plus restatement (extraction), scored by
  exact answer match, split arm evaluated with organizer ON (its designed
  operating mode), dense closed-book. This measures reasoning-over-
  knowledge at the system level; it was chosen as primary after
  knowledge-free tasks failed gate A three times (chance-level at pilot
  budget across shares 12-20%, op>=1, two curricula) — that failure is
  itself reported as a finding.
- **H1 tests.** (a) 1B confirmation contrast at the calib1b-selected top
  load: per-item paired bootstrap (10k resamples) clustered by
  `meta.template`, 2 seed pairs; (b) 160M sweep interaction: difference in
  slope of fact-use accuracy vs log N between arms (6 pairs).
- **Decision rule.** Positive iff the 1B confirmation delta (split minus
  dense, fact-use QA) exceeds max(2 x sigma_pool, 1.0 pt) with the same
  sign in both seed pairs AND H2 holds. sigma_pool = seed standard
  deviation of the per-pair delta computed across the six 160M sweep
  pairs (pre-specified estimator; ddof=1 over pair-deltas within load,
  pooled across loads). Defensible null iff the 95% clustered CI of the
  confirmation delta lies within (-margin, +margin) and H2/H3 hold.
  Otherwise: inconclusive, failure mode named.
- **H2 (guardrail, unchanged).** Split-with-organizer recall >= dense
  closed-book recall - 2 pts on the identical probe set. If violated, H1
  is void regardless of its delta.
- **H3 (mediation, unchanged).** Split store-OFF recall < 5% absolute AND
  split bits-in-weights < 10% of its dense twin's. Establishes that any
  H1 delta co-occurs with facts genuinely living outside the weights.
- **H4 (secondary).** Tokens-to-milestone on fact-use QA training curves
  (checkpoint evals): split reaches each 10-pt accuracy band in fewer
  tokens than dense.
- **Emergence watch (exploratory secondary, pre-specified trigger).**
  iGSM and deduction remain in the corpus and are evaluated at every
  snapshot. If ANY full-budget dense run reaches iGSM >= 13% (3x chance)
  or deduction >= 65%, the knowledge-free composite is additionally
  reported with the same margin formula — labeled exploratory (post-gate
  endpoint change), never substituted for H1.
- **Supporting only (no decisions):** natural benchmarks (HellaSwag,
  ARC-E, PIQA, WinoGrande, LAMBADA; cloze, acc + correct-prob),
  fresh-entity lookup accuracy, lookup hit/malformed rates, training
  curves.

## Battery (frozen)

160M sweep: 3 loads x 2 arms x 2 seeds (0,1) = 12 runs, 3.2B tokens.
calib1b: dense 1B at n800k_1b and n4m_1b, 1.5B tokens each; top load =
the one with LOWER dense closed-book recall (the dose that binds); tie ->
n4m_1b. 1B confirmation: top load x 2 arms x 2 seeds = 4 runs, 10B tokens,
submitted as 4-link dependency chains. Kill order under schedule pressure:
drop one sweep load (n50k first); drop confirmation to 1 seed pair +
restore via <= $300 RunPod burst (whole pairs single-platform). The 1B
top-load paired contrast is protected last.

## Exclusions and integrity

- Eval sets are held out by construction (structure-hash exclusion for
  reasoning; prompt-disjointness for factqa; fresh entities only in
  organizer_fresh). No eval item or its hash may enter any training
  stream.
- Runs excluded only for infrastructure failure (non-convergence from
  NaN/hardware), never for their results; every exclusion reported.
- All six sweep pairs and both confirmation pairs are reported regardless
  of outcome; no per-load cherry-picking.
- Seed-sigma estimator, margins, and cluster key are fixed above and may
  not be revised after the first battery eval is read.
