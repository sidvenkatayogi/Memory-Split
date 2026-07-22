# Relational launch runbook

This runbook is the execution gate for the standard-GPT relational experiment.
The checked-in `configs/160m.tsv` and `configs/360m.tsv` are platform-neutral:
FarmShare and AWS consume those same YAML bytes. Each YAML contains only
`data_rel` and `out_rel`; launchers resolve them against `DATA_ROOT` and
`OUT_ROOT` immediately before `scripts/run_train.py`.

Do not hand-edit a generated config, replace a missing seed, use Spot, or run a
launcher before its preflight is entirely green. Launch commands are dry-runs
unless `--execute` is present.

## 1. Local required

Run from a clean checkout with the project virtual environment active.

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/relational_smoke_test.py \
  --device cpu --out outputs/relational-smoke
.venv/bin/python scripts/make_relational_manifest.py generate
```

The manifest command above is a dry-run. To regenerate intentionally, write to
a temporary root with `--execute`, compare it byte-for-byte with `configs/`,
and commit any approved difference before packaging.

The smoke report must have two Dense steps, two Split steps, both memory modes,
complete pairs, a shared stream, and exact resume. Build the real Task 5 bundle
only from a clean tree:

The tiny smoke uses the committed explicit policy fixture in
`tests/fixtures/relational-smoke-route-policy.json`; it does not recalibrate.
That fixture only keeps the 32-entity smoke evaluable and is not valid for a
protected corpus. Protected builds must use `configs/route-policy.json`.

```bash
.venv/bin/python scripts/package_relational_run.py \
  --out artifacts/relational-run.tar.gz
.venv/bin/python scripts/platform_preflight.py \
  --platform local \
  --bundle artifacts/relational-run.tar.gz
```

Packaging uses the checked-in 15-run and 6-run manifests, the frozen
`configs/route-policy.json`, and
`outputs/relational-smoke/smoke-report.json`. Local preflight validates every
archive member hash, every source/config byte, the exact matrices, the policy
hash, dependencies, and the real smoke report. The bundle `manifest.json`
indexes both tracked GPT-2 tokenizer cache blobs under `vendor/tiktoken/` with
their byte counts and SHA-256 hashes. Tokenizer startup uses only those
vendored assets; packaging and preflight must not fetch or require network.

Stop if any local command exits nonzero.

## 2. FarmShare required: 29M pilot and fifteen 160M jobs

FarmShare is the required platform for:

1. the separate 29M Dense/Split learnability pair;
2. all fifteen protected 160M jobs; and
3. their evaluation and analysis.

The 29M pair uses one paired seed, 299,892,736 raw tokens (572 steps), and the
same stream for Dense and Split. Both runs must exceed 75% paired accuracy in
each of path composition, date ordering, and balanced equality. An absent or
failed 29M arm blocks the protected manifest; it is not replaced by a 160M
result.

Set platform roots on FarmShare. They are launch-time values and never belong
in YAML:

```bash
export DATA_ROOT="$SCRATCH/relational-data"
export OUT_ROOT="$SCRATCH/relational-runs"
export RELATIONAL_VENV="$SCRATCH/venvs/memorysplit"
mkdir -p "$DATA_ROOT" "$OUT_ROOT"
```

Stage the pinned FineWeb-Edu JSONL and build six 160M corpora: loads `n50k` and
`n800k`, each with data seeds 10000, 10001, and 10002. For each load/seed:

```bash
"$RELATIONAL_VENV/bin/python" scripts/build_relational_corpus.py \
  --out "$DATA_ROOT/n50k_ds10000" \
  --entities 50000 \
  --tokens 1599602688 \
  --data-seed 10000 \
  --bed-jsonl "$DATA_ROOT/fineweb-edu.jsonl" \
  --route-policy configs/route-policy.json \
  --route-policy-sha256 \
    0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058
```

Repeat with the exact directory, entity count, and data seed named by each
config. Do not copy one seed's corpus under another name. Retain each corpus
`manifest.json`, route-policy, report, graph, sidecars, and eval data.
Every build must read the same committed policy file and pass the
`route_policy_sha256` frozen in its run config. Never recalibrate policy by
load or data seed; a policy mismatch must stop before any corpus output is
created. Apply the same policy arguments when building the three `n1p8m`
corpora for the 360M runs.

Run FarmShare preflight inside a one-L40S Slurm allocation so the GPU check is
real. It also requires Slurm commands, 500 GB writable free space, all relevant
corpus hashes, a 40–60% route rate, at least 80% tail externalization, at least
80% rule/top-centrality internalization, the frozen policy hash, and exact
two-step checkpoint/resume:

Route-audit accounting treats each graph fact as one route-rate unit. For the
`rules_top_centrality` internalization stratum, each top-centrality fact is one
unit and each generated world contributes one additional always-internal rule
unit. That rule unit is per world, not per emitted rule training record.

```bash
"$RELATIONAL_VENV/bin/python" scripts/platform_preflight.py \
  --platform farmshare \
  --bundle artifacts/relational-run.tar.gz \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT"
```

Preview all fifteen one-GPU jobs:

```bash
"$RELATIONAL_VENV/bin/python" scripts/make_relational_manifest.py \
  farmshare configs/160m.tsv
```

Only after local gates, the 29M gate, and FarmShare preflight pass, submit one
L40S Slurm job per config:

```bash
"$RELATIONAL_VENV/bin/python" scripts/make_relational_manifest.py \
  farmshare configs/160m.tsv --execute
```

The resulting commands are of the form:

```bash
sbatch --export=ALL,CONFIG_REL=configs/160m/<run>.yaml \
  cluster/slurm/relational_train.sbatch
```

The Slurm script resolves roots into a runtime config and invokes
`scripts/run_train.py --resume auto`. Checkpoints are atomic and the job is
requeue-enabled. Rerunning the same relative config resumes its existing
`OUT_ROOT/out_rel/ckpt.pt`.

## 3. AWS preferred: six concurrent 360M jobs

AWS is preferred only for the six 360M confirmation runs. Use one
`p5.48xlarge` with exactly eight visible H100 80 GB GPUs. Use On-Demand or an
EC2 Capacity Block; never use Spot for these protected runs. This runbook does
not provision or contact AWS.

The eight-GPU requirement is intentionally fixed even though the queue has six
jobs: it freezes the `p5.48xlarge` platform topology rather than adapting to
available hardware. Launcher and preflight therefore fail closed for any
count other than eight unique visible GPUs.

Stage the three `n1p8m` corpora for data seeds 10000–10002 on local NVMe. Each
uses 1,800,000 entities and 3,599,761,408 raw tokens. Set:

```bash
export DATA_ROOT=/local-nvme/relational-data
export OUT_ROOT=/local-nvme/relational-runs
export AWS_CAPACITY_TYPE=on-demand
mkdir -p "$DATA_ROOT" "$OUT_ROOT"
```

Preview the six-way, eight-worker 200-step queue:

```bash
.venv/bin/python cluster/aws/run_relational_manifest.py \
  configs/360m.tsv --gpus 8 --steps 200
```

After confirming the dry-run, execute the resumable probe:

```bash
.venv/bin/python cluster/aws/run_relational_manifest.py \
  configs/360m.tsv --gpus 8 --steps 200 --execute
```

Then require the complete AWS preflight:

```bash
.venv/bin/python scripts/platform_preflight.py \
  --platform aws \
  --bundle artifacts/relational-run.tar.gz \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --runs-root "$OUT_ROOT" \
  --capacity-type "$AWS_CAPACITY_TYPE"
```

It fails closed unless there are exactly eight H100s with at least 72 GiB free
each, 1 TB writable NVMe free, six successful 200-step statuses, mean
post-step-50 throughput of at least 60,000 raw tokens/s/GPU, peak observed
memory at most 72 GiB per run, next-loss resume delta at or below `1e-5`,
matching corpus/bundle hashes, and a non-Spot declaration.

The 60,000-token/s gate is deliberately conservative: its logged timing
windows include checkpoint and snapshot overhead. Resume success is based on
`next_loss_delta <= 1e-5`; the separately reported bit-exact flag is diagnostic.

Preview, then explicitly start the full six-run queue:

```bash
.venv/bin/python cluster/aws/run_relational_manifest.py \
  configs/360m.tsv --gpus 8
.venv/bin/python cluster/aws/run_relational_manifest.py \
  configs/360m.tsv --gpus 8 --execute
```

A failed config is recorded in
`$OUT_ROOT/aws-launch-status.json`. Other independent workers continue; the
launcher exits nonzero after they finish. Rerun the same command to use
existing checkpoints.

## Evaluation, analysis, and checkpoint sync

For every completed run, evaluate both memory modes with the committed
evaluator:

```bash
.venv/bin/python scripts/run_relational_evals.py \
  --run "$OUT_ROOT/<run_id>" \
  --device cuda
```

After all 21 protected runs are evaluated:

```bash
.venv/bin/python scripts/analyze_relational.py \
  --runs-root "$OUT_ROOT" \
  --out "$OUT_ROOT/protected-analysis"
```

The analyzer requires the exact frozen matrix. A missing seed-2 run is never
substituted with another seed, a duplicate, a lower scale, or a rerun chosen
after inspecting outcomes. Report an incomplete matrix as pending.

The 360M validation margin is `max(0.02, 2 * pooled_sigma)`.
`pooled_sigma` is estimated only from the paired 160M Split-minus-Dense
effects: three seeds at `n50k` and three seeds at `n800k`, pooled within load
with four total degrees of freedom. It is a preregistered 160M seed-noise scale
applied to the 360M confirmation mean, not a variance estimate from the 360M
runs themselves.

Synchronize checkpoints, logs, runtime configs, eval summaries, corpus
manifests, the bundle SHA-256, and `aws-launch-status.json` to durable storage.
Verify hashes after every transfer before deleting scratch or NVMe copies.
