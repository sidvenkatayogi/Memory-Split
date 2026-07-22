# probe/ — mechanistic-interpretability & probing toolkit

Purpose: answer three questions about the dense-vs-split models, from checkpoints
only (no retraining), rigorously:

- **Q1 — How is reasoning done?** Where/whether the model computes the answer
  internally; is failure a *readout* or a *learning* problem; is the CoT faithful.
- **Q2 — How is reasoning separated from facts?** Do facts and reasoning live in
  different parameters/subspaces; does the split arm free and repurpose the
  capacity dense spends on memorization.
- **Q3 — How do the weights store things?** Where facts are stored, how much
  (bits), how storage crowds/superposes as fact-load grows, and whether it scales.

**New here? Read `docs/HANDOFF.md`** — a plain-language tour of every probe: what
it does, the hypothesis it tests, how to read the result, and how to run it.

### Layout (self-contained export bundle)
```
probe/
  README.md            ← you are here
  <probe>.py           the NEW probes, importable as `probe.<name>` (geometry,
  __init__.py          faithfulness, intermediate, attention, weights, splice,
                       ledger, causal_steps)
  docs/                METHODS.md · NEW-PROBES.md · HANDOFF.md · CLUSTER.md
  code/                reference copies of the REUSED library modules
                       (canonical in evals/): mechanism, reasoning_probe,
                       interp, continuous
  runners/             reference copies of the CLI drivers (canonical in
                       scripts/): run_reasoning_probe, run_mechanism,
                       run_extractability, run_ppl_slices, eval_all_snapshots,
                       plot_run_curves
```
`code/` and `runners/` are pinned copies so the bundle reads standalone; the
runnable canonical versions live in `evals/` and `scripts/`. To *execute*, drop
this folder back in the repo (deps: `train/`, `corpusgen/`, `organizer/`,
`evals/`) and use `PYTHONPATH=.`. Tests: `tests/test_probe_*.py`,
`tests/test_{mechanism,reasoning_probe,mechanism_bits,interp,continuous}.py`.
Specs: `../replication/specs/nr{1,5,7}-*.md`.

## Inventory (built) → which question

| Tool | File (canonical `evals/`) | Q | Needs |
|---|---|---|---|
| Latent-reasoning probe + logit-lens (NR-7) | `reasoning_probe.py` | Q1 | ckpt; pair for contrast |
| Fact-neuron localization + probe-bits (NR-5) | `mechanism.py` | Q2, Q3 | pair (dense+split) |
| Causal harness: ablation, activation patching (NR-5+) | `interp.py` *(other agent)* | Q1(gated), Q2, Q3 | ckpt |
| Continuous reasoning metrics (NR-1, feeds probes) | `continuous.py` *(other agent)* | Q1 | pair |

`interp.py` and `continuous.py` were authored by the parallel agent; copied here
(attributed) because the toolkit is incomplete without them. `mechanism.py` and
`reasoning_probe.py` are this workstream's.

## Inventory (built in THIS package, `probe/*.py` — importable)

The full NEW-PROBES backlog is now implemented as tested modules. Estimator /
assumptions / how-to-read for each: `docs/METHODS.md` §5. Status per idea:
`docs/NEW-PROBES.md` (top table).

| Module.function | Q | What it measures |
|---|---|---|
| `geometry.cross_arm_cka` | Q2 | per-layer dense↔split representational divergence (where capacity is freed) |
| `geometry.weight_spectral` / `effective_rank` | Q3 | MLP rank/norm dense vs split (fact-laden vs lean) |
| `geometry.topk_neuron_overlap` | Q2 | fact-neurons vs reasoning-neurons disjoint? dense's repurposed in split? |
| `geometry.subspace_overlap` | Q2 | fact-probe vs reasoning-probe subspace overlap |
| `faithfulness.cot_faithfulness` | Q1 | does the answer follow the stated CoT (vs post-hoc)? |
| `intermediate.intermediate_value_probe` | Q1 | are per-step DAG values computed internally (non-copy) |
| `attention.attention_weights` / `head_ablation_effect` | Q1 | dependency-tracing heads; causal head ablation |
| `weights.fact_weight_attribution` | Q3 | which weights store a given fact (dense localized vs split diffuse) |
| `weights.double_dissociation` | Q2 | ablate fact-neurons → recall↓, reasoning unchanged |
| `weights.participation_ratio` / `fact_superposition` | Q3 | superposition/crowding of fact storage vs dose |
| `splice.value_injection_profile` | Q2 | where an external value enters the residual stream |
| `ledger.fact_info_ledger` / `capacity_scaling` | Q3 | reconcile recall/probe/MC bits; bits-vs-params scaling |
| `causal_steps.step_patch_effect` | Q1 (gated) | causal step localization via activation patching |

Tests: `tests/test_probe_geometry.py`, `test_probe_ledger.py`, `test_probe_model.py`
(pure-math + toy-GPT). Full suite green (241 passed).

## How to run

**Snapshots + checkpoints live on the cluster, so probes run there as Slurm jobs.**
Full submit guide + resource sizing + the before-pruning workflow: **`docs/CLUSTER.md`**.
One-liner (from `$FS_REPO_DIR`):

```bash
sbatch --export=ALL,PAIRS="d160m_dense_n200k_s0:d160m_split_n200k_s0",SNAP=1 \
    cluster/slurm/probe_runs.sbatch
```
(160M sweep run IDs have **no** `_gate` suffix; 1B calib runs do — `…_s0_gate`.)

`SNAP=1` freezes per-snapshot summaries first (do before HANDOFF §7 pruning). The
job runs, per pair: `eval_all_snapshots` → `run_reasoning_probe` (Q1) →
`run_mechanism` (Q2/Q3, paired) → `run_extractability` + `run_ppl_slices` (Q3).

Equivalent manual invocations (e.g. on an interactive `salloc` node):

```bash
export PYTHONPATH=.
python scripts/run_reasoning_probe.py --run <dense> --split-run <split> --out out/probe   # Q1
python scripts/run_mechanism.py --dense <dense> --split <split> --out out/mech            # Q2/Q3
python scripts/run_extractability.py --run <dense>                                        # Q3 (L11)
# Q2/Q3 causal (fact-side, now): ablate dense's top memorization neurons, watch
# recall collapse while reasoning is untouched — see docs/METHODS.md §3.
```
(The same drivers are pinned under `runners/` for standalone reading.)

## Rigor stance (read before trusting any result)
- **Correlational vs causal.** Probes/logit-lens show *decodability*, not use;
  causal claims need `interp.py` ablation/patching. Reasoning-circuit causal uses
  are **gated until reasoning clears chance** (L10) — the NR-7 probe is the
  pre-req that says whether there's anything to find.
- **Always run a positive control** (a feature the model demonstrably has:
  operator `op`, the fact-lookup skill) before trusting a *null* — else "not
  decodable" may mean "weak probe."
- **The on-hypothesis quantity is the split−dense contrast**, not per-arm numbers;
  and (with seed-0 only) treat everything as directional (L16).
- **Linear probes by design** (bounded capacity ⇒ measure linear decodability; a
  linear null doesn't exclude nonlinear encoding).

See `docs/METHODS.md` for the per-tool estimator/assumptions/confounds, and
`docs/NEW-PROBES.md` for the ranked backlog.
