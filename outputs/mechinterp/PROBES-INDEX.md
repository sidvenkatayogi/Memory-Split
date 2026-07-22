# Probe index — 20 probes, one subfolder each

> **New to this project? Read [`START-HERE`](START-HERE.md) first** (the dense-vs-split
> experiment in plain terms — every probe's summary box assumes it).
>
> **Before citing any probe: read [`ENTITY-POPULATIONS.md`](ENTITY-POPULATIONS.md).**
> The repo entity generator drifted from the trained corpus, so most early probes
> ran on **UNSEEN** (untrained) entities. Each `RESULTS.md` now carries a
> seen/unseen banner; storage probes need SEEN re-runs (in progress).

Each subfolder collects that probe's outputs + a `RESULTS.md` (interpretation +
caveats, per the standing rule) and any `.png` graphs. Status legend:
**NOW** = runnable on finished runs with no deferred deps; **GATED** = runs but
signal needs reasoning > chance (currently at chance); **§2** = needs
`corpusgen meta["solution"]` (deferred). Contrast of interest is always split−dense.

| # | folder | Q | probe | status |
|---|---|---|---|---|
| 1 | `mechanism` | Q2/Q3 | NR-5 fact-neuron localization + linear probe-bits (paired) | **NOW** ✅ have n50k,n200k |
| 2 | `extractability` | Q3 | NR-4 generative recall vs MC recognition (stored-vs-extractable) | **DONE** ✅ SEEN, 6 runs — dense n50k extractable (recall .64 / MC .90) |
| 3 | `ppl_slices` | Q3 | NR-6 bpb slices + fact-value NLL on fresh entities | **DONE** ✅ all 6 runs (final ckpt) |
| 4 | `reasoning_probe` | Q1 | NR-7 latent-answer probe + logit-lens (end-of-prompt primary) | NOW (after-gold-CoT part needs §2) |
| 5 | `ood_reasoning` | Q1 | E1 train-easy→eval-hard per-op/depth accuracy | GATED (reasoning at chance) |
| 6 | `editability` | Q2/H3 | E2 edit-success + locality (split follows edited store) | **DONE** ✅ 3 split loads (edit 0.99–1.0, locality 1.0) |
| 7 | `continuous_metrics` | Q1 | NR-1 M1/M2/M3 reference-trace / answer-given-CoT NLL | §2 |
| 8 | `geometry_cka` | Q2 | cross-arm CKA per layer (where capacity diverges) | **NOW** |
| 9 | `weight_spectral` | Q3 | MLP effective-rank / Frobenius norm dense vs split | **NOW** (weights only) |
| 10 | `neuron_overlap` | Q2 | fact-neuron vs reasoning-neuron top-k overlap | GATED (needs reasoning selectivity) |
| 11 | `subspace_overlap` | Q2 | fact-probe vs reasoning-probe subspace overlap | GATED |
| 12 | `fact_weight_attribution` | Q3 | which weights store a given fact | **DONE** ✅ seen+unseen, 3 loads |
| 13 | `double_dissociation` | Q2/Q3 | ablate fact-neurons → recall↓ (causal, recall side) | **DONE** ✅ n50k (recall −0.31 vs random −0.02); reasoning side GATED |
| 14 | `superposition` | Q3 | participation ratio / fact crowding vs dose (50k→200k→800k) | **NOW** |
| 15 | `value_injection` | Q2 | where an external value enters the residual stream (split) | **NOW** |
| 16 | `attention_heads` | Q1 | which layers' attention produce a stored fact | **DONE** ✅ n50k fact-use (peak L8–9); reasoning side GATED |
| 17 | `intermediate_value` | Q1 | are per-step DAG values computed internally | GATED (informative even at chance) |
| 18 | `faithfulness` | Q1 | does answer follow tampered CoT step (vs post-hoc) | §2 |
| 19 | `step_patch` | Q1 | causal step localization via activation patching | GATED — deduction's answer is produced only AFTER its CoT (0/40 decisive at prompt-end), so patching needs the post-chain "Answer:" position + the generated CoT (partial §2), not just "reasoning > chance" |
| 20 | `fact_ledger` | Q3 | reconcile recall/probe bits; per-entity storage | **DONE** ✅ 3 loads (33 bits/entity @n50k → 0.2 @n200k/800k) |
| 21 | `keyguess` | Q3 | key-gen generalization to unseen (synthetic) entities | **DONE** ✅ 3 split loads (final ckpt) |
| 22 | `popqa_keyguess` | Q3 | key-gen transfer to REAL-world entities/relations (PopQA) | **DONE** ✅ 3 split loads (final ckpt) |
| 23 | `editability/hallucination` | guardrail | does the split model fabricate a value on a store miss? | **DONE** ✅ 0% fabrication — written up in `editability/` (guardrail section) |
| 24 | `h1_deduction` | Q1/H1 | split-vs-dense on deduction (paired McNemar + CI) — the H1 test | **DONE** ✅ no consistent advantage; sign flips with load; single-seed ⇒ null |

**Added (not in the original 20):** `keyguess` — in-schema OOD-entity key-generation eval
(`evals/keyguess.py`, `scripts/run_keyguess_eval.py`), split arm only: ≈100% correct
addresses for unseen synthetic people. `popqa_keyguess` — the real-world stress test
(`scripts/run_popqa_keyguess.py`): the lookup machinery rarely fires on real entities and
never copies a real name, so the addressing skill is distribution-bound (see its RESULTS.md).

## Per-snapshot (training-trajectory) note
Every probe accepts a checkpoint, so a probe can be looped over the 10 snapshots
(`snapshots/step0000610.pt … step0006100.pt` ≈ 0.32B…3.2B tokens) to get an
over-training curve. That's the H4-style view; the automated snapshot sweep
(`eval_all_snapshots`) additionally needs the deferred §3 `run_evals --ckpt`
edit, but individual probes can loop `--ckpt` today.

## Runs available (seed-0)
160M pairs: `d160m_{dense,split}_{n50k,n200k,n800k}_s0` (all finished).
1B: `d1b_dense_{n800k,n4m}_s0_gate` (dense-only, still training) — dense-side probes only.
