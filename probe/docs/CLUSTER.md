# Running the probes on FarmShare (where the snapshots live)

Snapshots + checkpoints are on the cluster (`/scratch/users/<sunet>/memorysplit/
outputs/<run>/`), so all probing runs there as Slurm jobs — never on the login node
under load, and never on a GPU a training job is using. Job template:
`cluster/slurm/probe_runs.sbatch` (mirrors `eval_runs.sbatch`: sources
`config.env`, loads the venv, `gpu:1` L40S).

## Snapshot freezing (pruning is currently DISABLED — no rush, still useful)
Per the training host, **pruning is disabled and all snapshots are retained**, so
the earlier "now-or-never before §7" urgency does not apply — you can probe any
snapshot whenever. `SNAP=1` (default) still runs `eval_all_snapshots.py` first to
precompute a tiny `evals/ckpt-step*/summary.json` per snapshot; that's what makes
the H4 / over-training trajectories reconstructable and small enough for handback
(instead of shipping ~80 GB of `.pt`). Idempotent (`--skip-existing`); re-run at
each check-in to pick up newly-written snapshots. If pruning is ever re-enabled,
run `SNAP=1` before it.

Snapshot cadence (verified): 160M = 10 snapshots at steps 610…6100
(524,288 tok/step ⇒ ≈0.32B…3.2B tokens); 1B = every 286 steps (≈0.15B…1.5B),
appearing as they train. `evals.curves.snapshot_step` parses these; tokens =
step × `tokens_per_step` (in `config.yaml`).

## Submit

From `$FS_REPO_DIR` on the login node:

Run IDs (verified on cluster): 160M sweep = `d160m_{dense,split}_{n50k,n200k,n800k}_s0`
(**no `_gate`**); 1B calib = `d1b_dense_{n800k,n4m}_s0_gate` (**with `_gate`**).

```bash
# paired 160M (one dose)
sbatch --export=ALL,PAIRS="d160m_dense_n200k_s0:d160m_split_n200k_s0",SNAP=1 \
    cluster/slurm/probe_runs.sbatch

# all three 160M dose pairs in one job
sbatch --export=ALL,PAIRS="d160m_dense_n50k_s0:d160m_split_n50k_s0 \
d160m_dense_n200k_s0:d160m_split_n200k_s0 \
d160m_dense_n800k_s0:d160m_split_n800k_s0",SNAP=1 \
    cluster/slurm/probe_runs.sbatch

# 1B dense-only calib (no split twin in Account-A scope)
sbatch --export=ALL,PAIRS="d1b_dense_n800k_s0_gate d1b_dense_n4m_s0_gate",SNAP=1 \
    cluster/slurm/probe_runs.sbatch
```

Vars: `SNAP` (1=freeze snapshots first), `OUT_ROOT` (default `outputs/probe`),
`TASKS` (`"igsm deduction"`), `PROBE_LIMIT` (items/task, default 500).

## Resource notes
- Default `--mem=48G --time=08:00:00 --gres=gpu:1`. Fine for 160M. For **1B**,
  bump if needed: `sbatch --mem=96G --time=16:00:00 ...` (the 1B checkpoint + KV
  is larger; `--gres=gpu:1` L40S 48GB still loads it in bf16, but give headroom).
- `SNAP=1` dominates runtime (it evals ~10 snapshots × tasks per run). If you only
  want final-checkpoint probes and have already frozen snapshots, submit with
  `SNAP=0`.
- **Do not co-schedule with the battery on the same GPU.** Submit as its own job;
  QOS allows up to 4 concurrent GPU jobs — leave room for the battery.

## Outputs
Per dense run under `$OUT_ROOT/<dense>/`: `reasoning/reasoning_probe.json`
(+ `probe_*.png`), `mechanism/mechanism.json` (+ `probe_acc.png`); per arm
`outputs/<run>/evals/extractability.json`, `ppl_slices.json`, and
`evals/ckpt-step*/summary.json` (snapshot curves → feed `scripts/analyze.py` /
`plot_run_curves.py`). Fold these into `handback_A.tgz` alongside the eval JSONs.

## Loader compatibility (verified against a real snapshot)
Snapshots are `torch.save({"model": state_dict, "step": int, "model_cfg": {...}})`,
weights-only model (no optimizer). Our loaders (`run_evals.load_model`,
`run_mechanism._load_model`, `run_reasoning_probe._load_model`) do
`torch.load(..., weights_only=False)` then `load_state_dict(state["model"])` and
read `state.get("step")` — **compatible as-is**. RoPE `rope_cos/rope_sin` are
non-persistent buffers (regenerated at construction), so they're absent from the
state_dict and `load_state_dict` does not expect them — no strict-load error.
To probe a specific snapshot pass `--ckpt snapshots/stepNNNNNNN.pt` to any runner
(they all accept `--ckpt`); the reasoning-probe/mechanism **trajectory over
training** = loop that flag over the 10 snapshots (thin wrapper; NEW-PROBES 1.13).

Data paths (per dose): `config.yaml:data_dir` →
`…/memorysplit_data/n{load}/{organizer.jsonl, eval/*.jsonl, organizer_fresh.jsonl}`.
The runners read `data_dir` from each run's `config.yaml`, so no path edits needed.

## What runs vs what's gated
Runnable now on any finished run: NR-7 reasoning probe, NR-5 mechanism (paired),
NR-4 extractability, NR-6 ppl/value-NLL, and the H4 snapshot curves. The causal
`interp.py` fact-side ablation is runnable now too (add a small driver if you want
it as a job); reasoning-*circuit* causal work stays gated until the NR-7 verdict
says reasoning is decodable (see `METHODS.md`, `NEW-PROBES.md`).
