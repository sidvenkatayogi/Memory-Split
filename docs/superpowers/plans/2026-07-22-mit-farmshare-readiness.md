# MIT and FarmShare Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Add the missing 29M FarmShare gate and a dry-run-first generic MIT Slurm discovery/profile/launcher without changing experiment configs or science.

**Architecture:** Extend the existing platform-neutral manifest and preflight modules. FarmShare keeps its one-L40S Slurm path. MIT consumes the unchanged six 360M YAMLs through a validated profile generated after a read-only discovery probe.

**Tech Stack:** Python 3.12, PyYAML, pytest, Slurm CLI, existing standard GPT pipeline.

## Global Constraints

- Authoritative design: `docs/superpowers/specs/2026-07-22-mit-farmshare-readiness-design.md`.
- No model, corpus, metric, route-policy, seed, or 160M/360M YAML changes.
- No command submits unless `--execute` is explicit.
- MIT partition/account/QOS/GRES are profile data, never hard-coded.
- FarmShare and MIT resolve only relative configs against runtime roots.
- All submission failures are collected after every independent config is attempted.

---

### Task 1: Add the 29M FarmShare Learnability Gate

**Files:**
- Modify: `scripts/make_relational_manifest.py`
- Create: `configs/29m.tsv`
- Create: `configs/29m/toy_dense_gate_s0.yaml`
- Create: `configs/29m/toy_split_gate_s0.yaml`
- Modify: `scripts/platform_preflight.py`
- Modify: `tests/test_relational_manifest.py`
- Modify: `tests/test_platform_preflight.py`

- [ ] Write failing tests asserting exactly two paired 29M jobs, one shared
  corpus/data seed, 299,892,736 tokens, 572 steps, and relative paths.
- [ ] Add scale settings:

```python
"29m": {
    "model": "toy",
    "total_tokens": 299_892_736,
    "micro_batch_size": 16,
    "lr": 1.5e-3,
}
```

- [ ] Generate only Dense/Split seed 0 at `n_gate`; add the entity count and
  shared `data_rel`.
- [ ] Extend FarmShare preflight to require both completed 29M eval summaries
  above 0.75 in all three strata before accepting the 160M manifest.
- [ ] Run focused tests and full suite; commit.

---

### Task 2: Add Generic MIT Slurm Discovery, Profile, and Launcher

**Files:**
- Create: `cluster/mit/probe_cluster.py`
- Create: `cluster/mit/run_relational_manifest.py`
- Create: `cluster/slurm/relational_mit_train.sbatch`
- Create: `schemas/mit-cluster-profile-v1.schema.json`
- Create: `cluster/mit/profile.example.json`
- Modify: `scripts/platform_preflight.py`
- Create: `tests/test_mit_cluster.py`

- [ ] Write failing fixture tests for Slurm discovery parsing, profile schema,
  shell/path rejection, deterministic `sbatch` rendering, dry-run default,
  all-six attempt behavior, and probe evidence.
- [ ] Implement discovery with bounded read-only commands:

```python
COMMANDS = (
    ("slurm_version", ("sinfo", "--version")),
    ("partitions", ("sinfo", "-h", "-o", "%P|%G|%l|%D|%m")),
    ("config", ("scontrol", "show", "config")),
)
```

- [ ] Validate profile fields and reject shell metacharacters, path traversal,
  and embedded data/output roots.
- [ ] Render one-GPU `sbatch` commands from profile + existing
  `configs/360m.tsv`; default dry-run, explicit `--execute`.
- [ ] Add a generic Slurm script that resolves `DATA_ROOT`, `OUT_ROOT`, and
  `RELATIONAL_VENV`, records GPU evidence, writes runtime YAML, and invokes
  `scripts/run_train.py --resume auto`.
- [ ] Extend preflight with `--platform mit --profile PROFILE`; require one
  profile-matching GPU, 200 steps, no OOM, checkpoint, and resume delta <=1e-5.
  Report throughput/projected hours without a hardware-specific threshold.
- [ ] Run focused tests and full suite; commit.

---

### Task 3: Integrate Bundle, Runbook, and Dry-Run Evidence

**Files:**
- Modify: `scripts/package_relational_run.py`
- Modify: `cluster/RELATIONAL-RUNBOOK.md`
- Modify: `.gitignore`
- Modify: `tests/test_relational_bundle.py`
- Create: `tests/test_mit_runbook.py`

- [ ] Add `artifacts/` to `.gitignore` so creating a bundle does not dirty the
  source tree.
- [ ] Include 29M configs, MIT scripts/profile schema/example, and their hashes
  in the portable bundle.
- [ ] Replace the AWS-required section with:
  - MIT discovery;
  - profile freeze;
  - six-job 200-step dry-run/probe;
  - MIT preflight;
  - six full resumable jobs.
  Keep AWS documented as an optional legacy launcher, not the chosen platform.
- [ ] Run:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/relational_smoke_test.py --device cpu
.venv/bin/python scripts/package_relational_run.py \
  --out artifacts/relational-run.tar.gz
.venv/bin/python scripts/platform_preflight.py \
  --platform local --bundle artifacts/relational-run.tar.gz
.venv/bin/python scripts/make_relational_manifest.py \
  farmshare configs/29m.tsv
.venv/bin/python scripts/make_relational_manifest.py \
  farmshare configs/160m.tsv
.venv/bin/python cluster/mit/run_relational_manifest.py \
  configs/360m.tsv --profile cluster/mit/profile.example.json
```

Expected: tests/smoke/local preflight pass; all launch commands print dry-run
plans and submit nothing.

- [ ] Commit and request final review.
