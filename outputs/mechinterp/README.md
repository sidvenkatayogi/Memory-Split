# mechinterp/ — mech-interp & extra-eval results

> **Newcomers start with [`START-HERE.md`](START-HERE.md)** — the dense-vs-split experiment
> in plain terms (every probe's summary box assumes it), then
> [`PROBES-INDEX.md`](PROBES-INDEX.md) and [`ENTITY-POPULATIONS.md`](ENTITY-POPULATIONS.md).

Home for everything produced by the probe toolkit + the added eval instruments
(`probe.zip` / `evals_and_edits`). Kept separate from the battery's per-run
`outputs/<run>/evals/` so handback and mech-interp analysis stay tidy.

Populated on the cluster (probes run there as their own Slurm GPU job — NOT
co-scheduled with the battery); pull down with `cluster/sync_pull.sh` or rsync.

## Layout
```
outputs/mechinterp/
  reasoning/     NR-7 latent-reasoning probe (Q1)   ← scripts/run_reasoning_probe.py --out
  mechanism/     NR-5 fact-neuron + probe-bits (Q2/Q3) ← scripts/run_mechanism.py --out
  ood/           OOD reasoning generalization       ← scripts/run_ood_evals.py
  editability/   H3-backing edit success/locality   ← scripts/run_editability.py
  probes/        probe.* module outputs (geometry/attention/weights/splice/ledger/…)
```

Note: `run_extractability.py` and `run_ppl_slices.py` write into each run's own
`outputs/<run>/evals/{extractability.json,ppl_slices.json}` by design; and
`eval_all_snapshots.py` writes `outputs/<run>/evals/ckpt-*/summary.json`
(the latter needs the deferred §3 `run_evals --ckpt` edit — see
`docs/deferred/DEFERRED-eval-analysis-edits.md`). Copy/symlink those into
`mechinterp/` at handback time if you want everything in one tree.

## Example invocations (from repo root on the cluster, `PYTHONPATH=.`)
```bash
python scripts/run_reasoning_probe.py --run outputs/d160m_dense_n200k_s0 \
    --split-run outputs/d160m_split_n200k_s0 --out outputs/mechinterp/reasoning
python scripts/run_mechanism.py --dense outputs/d160m_dense_n200k_s0 \
    --split outputs/d160m_split_n200k_s0 --out outputs/mechinterp/mechanism
```

## Reporting convention (REQUIRED — user instruction 2026-07-21)
Every probe run must be saved **with interpretation and caveats**, not just raw
JSON/PNG. For each run write a `RESULTS.md` next to the outputs (e.g.
`mechinterp/mechanism/RESULTS.md`) containing:
1. **What/where:** probe name, exact run IDs + checkpoint (`ckpt.pt` or
   `snapshots/stepN.pt`), tokens = step × tokens_per_step, date, git/commit state.
2. **Numbers:** the key metrics, and the **split−dense contrast** (the
   on-hypothesis quantity), not just per-arm values.
3. **Interpretation:** what it means for Q1/Q2/Q3, in plain language.
4. **Caveats (always include the ones that apply):**
   - single seed (seed-0 only) ⇒ **directional/suggestive, not proven**;
   - probes show **decodability/correlation, not causal use** (only
     ablation/patching show use);
   - **reasoning-circuit causal probes are gated** until reasoning clears chance
     (iGSM/deduction were at chance) — until then they're noise;
   - **positive control** status (did the probe find a feature the model
     demonstrably has, before trusting any null?);
   - linear probes ⇒ a linear null doesn't exclude nonlinear encoding;
   - any dependence on deferred edits (e.g. snapshot curves need §3 `run_evals --ckpt`).
5. **Provenance for handback:** note that these are analysis-code additions
   (report per the preregistration).

## Status
Created 2026-07-21. Empty until probes are run (integration is done + tested
offline; nothing has been run against checkpoints yet).
