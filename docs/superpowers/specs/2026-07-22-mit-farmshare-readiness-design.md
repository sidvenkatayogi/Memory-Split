# FarmShare and Generic MIT Slurm Readiness Design

**Date:** 2026-07-22
**Status:** approved in session; written review pending
**Scope:** launch preparation only; no model, corpus, metric, or protected
outcome changes

## 1. Goal

Complete the platform layer for the approved Relational MemorySplit study:

- local tests, smoke, bundle, and local preflight run on the Mac;
- FarmShare receives the missing 29M learnability-gate manifest plus the
  existing fifteen 160M jobs; and
- the six existing 360M configs run on an initially unknown MIT student Slurm
  cluster instead of AWS.

No command in this phase submits work unless the user explicitly adds
`--execute`.

## 2. Unchanged experiment

The following remain byte-identical:

- `train.model.GPT`;
- the frozen route policy and SHA-256;
- 160M and 360M YAML configs;
- shared corpus and weight sidecars;
- optimizer, token budgets, seeds, evaluation, and verdict; and
- portable source bundle.

MIT and FarmShare launch the same relative configs. Runtime paths are resolved
only from `DATA_ROOT` and `OUT_ROOT`.

## 3. FarmShare completion

Add `configs/29m.tsv` with exactly two runs:

- `toy_dense_gate_s0`;
- `toy_split_gate_s0`.

Both use:

- the existing toy GPT;
- one shared gate corpus;
- 299,892,736 raw tokens;
- 524,288 tokens/step (572 steps);
- identical initialization/data seed;
- context 1,024; and
- one L40S per run.

The existing FarmShare launcher remains one-config-per-Slurm-job and
checkpoint/requeue safe. FarmShare preflight must include:

- the two 29M configs and shared corpus hash;
- all fifteen 160M configs/corpora;
- one visible L40S;
- exact resume;
- route and mask guardrails; and
- writable scratch.

The 160M manifest is blocked until both 29M runs exceed 75%
counterfactual-consistent accuracy in all three primary strata.

## 4. MIT discovery profile

Because the exact MIT system is unknown, do not hard-code a partition, GPU,
account, wall time, or module stack.

Add a read-only discovery command:

```bash
python cluster/mit/probe_cluster.py --out mit-cluster-probe.json
```

It records:

- Slurm version;
- visible partitions, GRES strings, limits, nodes, memory, and default status;
- available module command and Python executables;
- environment scratch candidates;
- login-node filesystem free space; and
- current Git/bundle SHA-256.

It does not request an allocation or contact a paid service.

The user converts one discovered option into a frozen profile:

```json
{
  "schema_version": 1,
  "partition": "chosen_partition",
  "account": null,
  "qos": null,
  "gres": "gpu:1",
  "gpu_name_regex": "H100|H200|L40S|A100",
  "cpus": 16,
  "memory_gb": 96,
  "wall_minutes": 360,
  "python": "${RELATIONAL_VENV}/bin/python"
}
```

Every field is required except `account` and `qos`. Profiles containing shell
metacharacters, path traversal, or absolute repository/data/output paths are
rejected.

## 5. MIT probe and launcher

The MIT launcher consumes:

- the frozen profile;
- the existing `configs/360m.tsv`;
- `DATA_ROOT`, `OUT_ROOT`, and `RELATIONAL_VENV`; and
- the portable bundle/hash.

It renders `sbatch` arguments at launch time and defaults to dry-run.
`--execute` is required to submit.

First submit one 200-step resumable probe for each Dense/Split seed pair. The
probe must report:

- actual GPU name/count/memory;
- steps completed;
- mean post-warmup raw tokens/s;
- projected full-run hours;
- peak allocated memory;
- checkpoint presence; and
- next-loss resume delta.

MIT preflight fails unless:

- Slurm/profile/bundle/corpus hashes match;
- exactly one allowed GPU is visible per job;
- no OOM occurs;
- 200 steps complete;
- next-loss resume delta is at most `1e-5`; and
- the projected runtime fits the profile through checkpointed resubmission.

There is no fixed H100 throughput threshold on unknown hardware. Throughput is
reported, not used to alter data, seeds, or scientific interpretation.

After preflight, submit all six 360M configs. Independent submission failures
are collected; all configs are attempted, then the launcher exits nonzero if
any failed.

## 6. Tests and deliverables

Local tests cover:

- exact two-run 29M manifest and pairing;
- FarmShare dry-run commands;
- MIT `sinfo`/Slurm parsing via fixtures;
- profile schema and injection/path rejection;
- deterministic MIT `sbatch` rendering;
- dry-run default and explicit execute;
- all-six submission attempts after injected failures;
- 200-step evidence parsing and resume tolerance;
- unchanged 160M/360M YAML bytes; and
- bundle/platform documentation.

Deliverables:

- `configs/29m.tsv` and two relative YAMLs;
- `cluster/mit/probe_cluster.py`;
- `cluster/mit/run_relational_manifest.py`;
- `cluster/slurm/relational_mit_train.sbatch`;
- MIT profile schema/example;
- updated platform preflight and runbook; and
- passing full local suite, smoke, package, and local preflight.

Real FarmShare and MIT GPU preflights remain external actions and are not
claimed complete by local tests.
