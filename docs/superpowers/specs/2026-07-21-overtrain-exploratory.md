# Exploratory tier: the organizer above Chinchilla optimal

2026-07-21. Status: EXPLORATORY — explicitly outside the frozen
preregistration (2026-07-20-preregistration.md). Nothing here feeds the
H1 decision rule; results are reported descriptively and labeled
exploratory. This tier must never displace battery runs: it uses spare
capacity only (AWS GPUs 4-7 or idle FarmShare slots after the sweep).

## Question

The battery trains at or below Chinchilla optimal (20 tokens/param at
160M, 10 t/p at 1B), a regime where the dense arm is exposure-starved:
gate pilots showed dense closed-book recall near floor at a quarter
budget. Modern small-model practice overtrains heavily (SmolLM2 at
~65-1000 t/p), and Physics of LM 3.3 says parametric fact capacity grows
with exposures (1 bit/param at ~100 exposures, 2 at ~1000). So the
regime where memorization is CHEAP is the strongest test of the external
database: when the dense arm can afford to memorize, does the organizer's
advantage shrink, persist, or grow?

Specific readouts (all descriptive, both arms, as functions of tokens
seen):

1. Dense closed-book recall and bits-in-weights: how fast does parametric
   storage fill with exposures, and does it saturate below the ~2
   bits/param ceiling at this dose (200k entities x ~53 bits = 10.6 Mbit
   demanded vs ~324 Mbit ceiling: capacity is NOT binding, exposures
   are)?
2. Split-arm mechanism stability: does bits-in-weights stay at 0 and
   store-off recall at 0 even after ~840 exposures per entity (any
   leakage of facts into weights despite the mask, e.g. via unmasked
   restatements in factqa chains)?
3. The fact-use QA gap: trajectory of split-minus-dense through 4x
   Chinchilla. Hypothesis-friendly outcome: gap persists because lookup
   beats recall even when recall is cheap. Hypothesis-hostile outcome:
   dense converges to the split arm as memorization completes.
4. Emergence watch at 4x budget: do iGSM/deduction leave chance with
   16x the gate-pilot reasoning tokens? (Trigger thresholds from the
   preregistration apply for reporting.)

## Design

One pair (dense + split), 160M class, seed 0, n200k dose, 12.8B tokens
(80 t/p, 4x Chinchilla), fresh corpus built at full length (stage
"over" -> data dir n200k_over): fresh bed text and fresh paraphrase
exposures rather than epoch repetition, so exposure count rises without
verbatim-repeat confounds (~840 exposures/entity vs ~210 at battery
budget). Snapshots every 5% (20 per arm) for trajectory evals.

Caveat to carry into any writeup: mid-run checkpoints are not equivalent
to fully annealed shorter runs (cosine schedule differs), so trajectory
points are descriptive; the annealed 20 t/p comparison comes from the
battery sweep's n200k pair.

## Execution

Corpus (~1 h on 16 cores), then the pair:

```bash
# corpus (FarmShare data_prep or AWS box):
PYTHONPATH=$PWD python scripts/build_corpus.py --out-root <DATA_ROOT> --stage over --workers 16

# configs + manifest:
PYTHONPATH=$PWD python scripts/make_manifest.py --stage overtrain --data-root <DATA_ROOT>

# AWS (2 spare GPUs; ~8 h per arm on H100):
PYTHONPATH=$PWD nohup python scripts/run_local_gpus.py \
    --manifest outputs/manifests/overtrain.tsv --gpus 6,7 > launcher_over.out 2>&1 &

# FarmShare alternative (2 jobs, ~35-40 L40S-h each, chained on the build):
# submit train_single.sbatch per config as usual.

# trajectory evals (fast battery per snapshot; full battery on the final):
for run in outputs/d160m_*_n200k_s0_over; do
  for snap in "$run"/snapshots/step*.pt; do
    PYTHONPATH=$PWD python scripts/run_evals.py --run "$run" \
        --ckpt "snapshots/$(basename "$snap")" --limit 400
  done
  PYTHONPATH=$PWD python scripts/run_evals.py --run "$run" --limit 1500
done
```

Note: run_evals writes evals/ per invocation; per-snapshot results land
in the same directory and are distinguished by the "ckpt" field in
summary.json — collect trajectories by reading that field. Budget: ~16
H100-hours training + ~10 GPU-hours of snapshot evals for the pair.

## Cost and placement

AWS spare GPUs (preferred, tonight): fits alongside the confirmation
(GPUs 0-3) and 410M tier (4-5), using 6-7. FarmShare fallback: after the
sweep drains, ~80 L40S-hours total.
