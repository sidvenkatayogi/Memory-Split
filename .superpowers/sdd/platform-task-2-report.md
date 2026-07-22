# Platform Task 2 implementation report

**Date:** 2026-07-22
**Branch:** `feat/relational-core`

## Delivered

- Added bounded read-only MIT discovery for the three approved Slurm commands,
  plus local module, Python, scratch, filesystem, Git revision, and bundle
  facts. Command runners and local probes are injectable.
- Added a closed v1 MIT profile schema and loader. It requires partition,
  one-GPU GRES, GPU-name regex, CPUs, memory, wall time, and the frozen
  `${RELATIONAL_VENV}/bin/python` template; account and QoS are optional.
  Unknown fields, duplicate keys, invalid types, shell injection, newlines,
  traversal, embedded roots, unsafe files, and non-one-GPU GRES are rejected.
- Added deterministic six-job `sbatch` argv rendering from the unchanged
  `configs/360m.tsv` matrix. The launcher is dry-run by default, never invokes
  a shell, requires `--execute` to submit, attempts every job, and reports all
  independent submission failures before exiting nonzero.
- Added a resource-neutral Slurm script. Submission argv supplies all cluster
  resources; the script resolves only `DATA_ROOT`, `OUT_ROOT`, and
  `RELATIONAL_VENV`, verifies exactly one profile-matching GPU, writes a
  runtime YAML and atomic evidence JSON, monitors peak GPU memory, and runs
  the standard GPT with `scripts/run_train.py --resume auto`.
- Extended platform preflight with `--platform mit --profile PROFILE`.
  Preflight verifies the strict profile, Slurm commands, bundle/source/config/
  corpus hashes, current and per-job GPUs, six exact 200-step probes, runtime
  configs, no OOM, positive peak memory, checkpoints, and next-loss resume
  delta at or below `1e-5`. It reports post-warmup raw throughput, projected
  full-run hours, and projected checkpointed resubmissions with no fixed
  hardware throughput threshold.
- Included all MIT launcher/profile/schema/test files in the portable bundle
  and added schema files to checkout-versus-bundle byte verification.
- Updated `cluster/RELATIONAL-RUNBOOK.md` so generic MIT Slurm is the approved
  360M route and every launch remains preview-first.

## TDD and verification

- Initial focused red run: `50 failed`, all due to absent Task 2 behavior.
- Discovery/profile red-green cycle: `29 passed` after implementation.
- Launcher focused cycle: `4 passed`.
- Complete MIT test file after implementation: `50 passed`; two additional
  self-review regressions and one missing discovery assertion were then added
  and driven red-to-green.
- Final focused platform/bundle/manifest suite:
  `116 passed in 24.54s`.
- Final full suite:
  `448 passed, 2 deselected in 62.01s`.
- Python compilation and `bash -n` passed for all changed executable files.
- `git diff --check` passed.
- `git diff -- configs/360m.tsv configs/360m` was empty; the six frozen YAML
  files and their manifest were not modified.

The full suite retains one pre-existing `dateutil` deprecation warning from
`tests/test_stats.py`.

## Self-review

- Confirmed discovery has no allocation, submission, or network operation.
- Confirmed no launcher path uses `shell=True`; all subprocess calls receive
  argv lists.
- Confirmed dry-run does not call the injected submitter and execute mode
  attempts all six commands after both nonzero results and exceptions.
- Confirmed profile strings cannot inject into Slurm exports. Regex
  alternation (`|`) remains allowed because it is required by the approved
  example and is passed only as an argv/environment value, never parsed by a
  shell.
- Confirmed evidence and runtime paths reject symlinks, root escape, duplicate
  JSON keys, mismatched hashes, incomplete steps, OOM, missing checkpoints,
  and unsupported GPUs.
- Fixed three issues found during review: login-node filesystem capacity was
  not initially recorded, bundled schemas were not initially in source-byte
  comparison, and the first memory-monitor loop could delay trainer reaping.

## External concerns

- No real Slurm allocation or submission was made.
- MIT partition/account/QoS/GRES/module choices remain intentionally
  undiscovered locally. A user must run the read-only probe on MIT, commit a
  reviewed profile, stage the bundle and corpora, and complete the real
  one-GPU 200-step preflight before any full launch.
- Hardware throughput is reported rather than gated, as required.
