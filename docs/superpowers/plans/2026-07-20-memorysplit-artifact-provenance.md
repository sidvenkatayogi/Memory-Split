# MemorySplit Artifact Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make it impossible to train, resume, or evaluate a protected battery run against the wrong corpus, config, checkpoint, or source revision.

**Architecture:** Fingerprint immutable corpus artifacts and scientific config, derive versioned corpus/run identities, carry provenance through YAML and checkpoints, validate before any output mutation, and treat all pre-schema artifacts as evaluation-only legacy data. Stamp the source revision during rsync because `.git` is intentionally not copied to FarmShare.

**Tech Stack:** Python 3.12 standard library, PyTorch checkpoint files, YAML/JSON, Bash/Slurm, pytest, SHA-256.

## Global Constraints

- Commit `b805919` is the frozen preregistration baseline; protected builds must use a clean descendant.
- Do not backfill provenance into old corpora or checkpoints.
- Legacy checkpoints are never resumable.
- Legacy artifacts may be evaluated only with an explicit opt-in and must be labeled `legacy-unverified`.
- `workers` is operational and excluded from the scientific build-config fingerprint.
- Artifact paths are location metadata and excluded from the run-config fingerprint.
- Artifact bytes, scientific configuration, source revision, model/arm/seed, and token budget are authoritative.
- Validation happens before directory creation, config writes, log appends, model-state loading, or optimizer-state loading.
- Dense runs pass `train_mask=None`; a configured missing mask is always an error.
- Full SHA-256 values are stored; abbreviated values are used only in directory names.
- No timestamp enters a fingerprint or deterministic report.

---

### Task 1: Implement canonical provenance primitives

**Files:**
- Create: `provenance.py`
- Create: `tests/test_provenance.py`

**Interfaces:**
- Consumes: mappings, `BuildCfg`, repository/stamp state, artifact paths.
- Produces: canonical fingerprints and compatibility validation.

- `SourceRevision(git_commit: str, git_dirty: bool = False)`
- `canonical_fingerprint(value: Mapping[str, Any]) -> str`
- `resolve_source_revision(repo_root: Path) -> SourceRevision`
- `build_config_fingerprint(cfg: BuildCfg) -> str`
- `corpus_build_identity(cfg: BuildCfg, source: SourceRevision) -> str`
- `hash_corpus_artifacts(root: Path) -> dict[str, dict[str, int | str]]`
- `corpus_fingerprint(artifacts: Mapping[str, Mapping[str, Any]]) -> str`
- `run_config_fingerprint(cfg: Mapping[str, Any]) -> str`
- `run_identity(training_source, corpus_fp, config_fp) -> str`
- `assert_resume_compatible(expected, actual, path) -> None`

- [ ] **Step 1: Write failing canonical-hash and source tests**

Create tests:

```python
def test_canonical_fingerprint_ignores_mapping_order():
    assert canonical_fingerprint({"a": 1, "b": [2, 3]}) == (
        canonical_fingerprint({"b": [2, 3], "a": 1})
    )


def test_build_config_fingerprint_excludes_workers():
    a = BuildCfg(n_entities=50, total_tokens=1000, seed=7, workers=1)
    b = dataclasses.replace(a, workers=16)
    assert build_config_fingerprint(a) == build_config_fingerprint(b)


def test_run_config_fingerprint_excludes_locations():
    base = minimal_run_cfg()
    moved = {
        **base,
        "run_id": "moved",
        "out_dir": "/other/out",
        "data_dir": "/other/data",
        "train_bin": "/other/data/train.bin",
        "train_mask": "/other/data/train.mask.bin",
    }
    assert run_config_fingerprint(base) == run_config_fingerprint(moved)
```

Also test:

```text
- a scientific BuildCfg field changes the build fingerprint;
- model, arm, seed, LR, or token budget changes the run fingerprint;
- invalid commit strings and dirty protected revisions are rejected;
- stamp-file resolution works without `.git`;
- compatibility errors name the mismatched field and both values.
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_provenance.py -q
```

Expected: FAIL because `provenance.py` does not exist.

- [ ] **Step 3: Implement canonical JSON and source resolution**

Use:

```python
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

SCHEMA_VERSION = 1
STAMP_NAME = ".source-revision.json"


@dataclass(frozen=True)
class SourceRevision:
    git_commit: str
    git_dirty: bool = False

    def __post_init__(self):
        if len(self.git_commit) != 40 or any(
            c not in "0123456789abcdef" for c in self.git_commit
        ):
            raise ValueError(f"invalid full git commit: {self.git_commit!r}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_fingerprint(value: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def resolve_source_revision(repo_root: Path) -> SourceRevision:
    stamp = repo_root / STAMP_NAME
    if stamp.exists():
        row = json.loads(stamp.read_text())
        return SourceRevision(
            git_commit=row["git_commit"],
            git_dirty=bool(row["git_dirty"]),
        )
    commit = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    )
    return SourceRevision(commit, dirty)
```

- [ ] **Step 4: Implement config and identity fingerprints**

Use:

```python
def build_config_fingerprint(cfg: BuildCfg) -> str:
    value = dataclasses.asdict(cfg)
    value.pop("workers", None)
    return canonical_fingerprint(value)


def corpus_build_identity(cfg: BuildCfg, source: SourceRevision) -> str:
    if source.git_dirty:
        raise ValueError("protected corpus builds require a clean source revision")
    return canonical_fingerprint(
        {
            "schema_version": SCHEMA_VERSION,
            "git_commit": source.git_commit,
            "build_config_fingerprint": build_config_fingerprint(cfg),
        }
    )


RUN_LOCATION_KEYS = {
    "run_id", "out_dir", "data_dir", "train_bin", "train_mask", "provenance"
}


def run_config_fingerprint(cfg: Mapping[str, Any]) -> str:
    scientific = {
        key: value for key, value in cfg.items() if key not in RUN_LOCATION_KEYS
    }
    return canonical_fingerprint(scientific)
```

`run_identity()` hashes schema version, training commit, corpus fingerprint,
and run-config fingerprint.

- [ ] **Step 5: Implement streaming artifact hashes**

Declare the exact post-alignment artifact set:

```python
CORPUS_ARTIFACTS = (
    "dense/train.bin",
    "dense/train.mask.bin",
    "split/train.bin",
    "split/train.mask.bin",
    "organizer.jsonl",
    "organizer_fresh.jsonl",
    "eval/igsm.jsonl",
    "eval/igsm_ood.jsonl",
    "eval/deduction.jsonl",
    "eval/deduction_ood.jsonl",
    "eval/factqa.jsonl",
    "eval/factqa_fresh.jsonl",
    "eval/recall.jsonl",
)
```

Hash each file in 8 MiB chunks. Reject missing and unexpected protected
artifacts. Define `corpus_fingerprint()` over sorted relative path, byte size,
and full file hash. Exclude `report.json` to avoid self-reference.

- [ ] **Step 6: Implement compatibility errors**

`assert_resume_compatible()` must raise messages beginning with:

```text
Refusing resume for <path>: checkpoint has no provenance (legacy/pre-v1).
Refusing resume for <path>: corpus fingerprint mismatch
Refusing resume for <path>: run-config fingerprint mismatch
Refusing resume for <path>: run identity mismatch
```

- [ ] **Step 7: Run tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_provenance.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add provenance.py tests/test_provenance.py
git commit -m "feat: add immutable artifact provenance primitives"
```

### Task 2: Publish versioned corpus directories and reports

**Files:**
- Modify: `corpusgen/build.py:53-88,199-203,402-461`
- Modify: `scripts/build_corpus.py:29-100`
- Modify: `scripts/smoke_test.py`
- Modify: `tests/test_build.py:180-351`
- Create: `tests/test_build_corpus_script.py`

**Interfaces:**
- Consumes: Task 1 fingerprints and the frozen corpus-alignment artifacts.
- Produces: `<legacy-leaf>__p1-<16hex>/report.json`.

- [ ] **Step 1: Write failing report and immutability tests**

Test:

```text
- report contains schema version, clean source commit, config fingerprint,
  artifact hashes, and aggregate corpus fingerprint;
- serial and parallel builds have identical corpus fingerprints;
- one-byte artifact mutation fails verification;
- missing/extra protected artifacts fail verification;
- failed build never publishes the final directory;
- existing final destination is refused unless `--reuse-verified`;
- reused destination is revalidated and not rewritten;
- `report.json` is written last and excluded from the corpus fingerprint.
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_build.py tests/test_build_corpus_script.py -q
```

Expected: FAIL because reports have no provenance and paths are flat.

- [ ] **Step 3: Add source revision to the builder contract**

Change:

```python
def build_corpus(
    cfg: BuildCfg,
    tok,
    bed_iter,
    out_dir: Path | str,
    *,
    source_revision: SourceRevision,
) -> dict:
```

After all writers close and eval JSONL files exist:

```python
artifacts = hash_corpus_artifacts(Path(out_dir))
report["provenance"] = {
    "schema_version": SCHEMA_VERSION,
    "git_commit": source_revision.git_commit,
    "git_dirty": source_revision.git_dirty,
    "build_config_fingerprint": build_config_fingerprint(cfg),
    "artifacts": artifacts,
    "corpus_fingerprint": corpus_fingerprint(artifacts),
}
```

Update direct test/smoke callers to pass a fixed clean test revision:

```python
SourceRevision("0" * 40, False)
```

- [ ] **Step 4: Derive and atomically publish the leaf directory**

In `scripts/build_corpus.py`:

```python
source = resolve_source_revision(Path(__file__).resolve().parents[1])
cfg = BuildCfg(
    n_entities=LOADS[load],
    total_tokens=total,
    seed=args.seed,
    workers=args.workers,
)
identity = corpus_build_identity(cfg, source).removeprefix("sha256:")
legacy_leaf = load + tag
leaf = f"{legacy_leaf}__p1-{identity[:16]}"
final_dir = Path(args.out_root) / leaf
temp_dir = final_dir.with_name(f".{leaf}.tmp-{os.getpid()}")
```

Add `--reuse-verified`. Default behavior:

```text
- refuse an existing final directory;
- remove a same-process empty temp directory only;
- build into the temp directory;
- write report.json last;
- fsync/close;
- atomically rename temp to final;
- never mutate a published final directory.
```

- [ ] **Step 5: Run build tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_build.py tests/test_build_corpus_script.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add corpusgen/build.py scripts/build_corpus.py scripts/smoke_test.py \
  tests/test_build.py tests/test_build_corpus_script.py
git commit -m "feat: publish provenance-stamped corpora"
```

### Task 3: Validate corpus reports and derive run identities in manifests

**Files:**
- Modify: `scripts/make_manifest.py:28-130`
- Create: `tests/test_manifest.py`

**Interfaces:**
- Consumes: versioned corpus reports from Task 2.
- Produces: provenance-bearing YAML and versioned run IDs.

- [ ] **Step 1: Write failing manifest tests**

Test:

```text
- exactly one matching corpus is resolved by load, stage tag, token budget,
  frozen recipe, and clean source;
- zero or multiple matching corpora fail before any config/manifest write;
- config embeds the complete corpus provenance;
- path relocation does not change config fingerprint;
- model/arm/seed/LR/token budget changes run identity;
- run ID ends in `__p1-<16hex>`;
- generated output path can never equal the old flat run path;
- no partial config files remain after a validation failure.
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_manifest.py -q
```

Expected: FAIL because `make_manifest.py` does not inspect reports.

- [ ] **Step 3: Add corpus discovery and validation**

Implement:

```python
def resolve_corpus_dir(
    data_root: Path,
    legacy_leaf: str,
    *,
    n_entities: int,
    total_tokens: int,
    training_source: SourceRevision,
) -> tuple[Path, dict]:
    candidates = sorted(data_root.glob(f"{legacy_leaf}__p1-*"))
    matches = []
    for candidate in candidates:
        report = json.loads((candidate / "report.json").read_text())
        cfg = report["cfg"]
        if (
            cfg["n_entities"] == n_entities
            and cfg["total_tokens"] == total_tokens
            and tuple(cfg["igsm_op"]) == (1, 4)
            and tuple(cfg["deduction_depth"]) == (1, 2)
            and report["provenance"]["git_commit"]
            == training_source.git_commit
        ):
            matches.append((candidate, report))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one verified corpus for {legacy_leaf}; "
            f"found {len(matches)}"
        )
    return matches[0]
```

Also validate mixture shares and schema version. Requiring the corpus and
manifest to come from the same clean source commit intentionally fails closed
after any code change; rebuild or check out the recorded commit rather than
guessing compatibility.

- [ ] **Step 4: Embed run provenance and derive IDs**

Build the scientific config first, then:

```python
config_fp = run_config_fingerprint(cfg)
identity = run_identity(
    training_source,
    report["provenance"]["corpus_fingerprint"],
    config_fp,
)
cfg["provenance"] = {
    "schema_version": 1,
    "training_git_commit": training_source.git_commit,
    "corpus": {
        "git_commit": report["provenance"]["git_commit"],
        "fingerprint": report["provenance"]["corpus_fingerprint"],
        "build_config_fingerprint": (
            report["provenance"]["build_config_fingerprint"]
        ),
        "build_config": report["cfg"],
    },
    "config_fingerprint": config_fp,
    "run_identity": identity,
}
run_id = f"{legacy_run_id}__p1-{identity.removeprefix('sha256:')[:16]}"
cfg["run_id"] = run_id
cfg["out_dir"] = str(Path(out_root) / run_id)
```

Write all configs into a temporary config directory and atomically replace the
manifest only after every job validates.

- [ ] **Step 5: Run manifest tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_manifest.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/make_manifest.py tests/test_manifest.py
git commit -m "feat: derive provenance-safe run manifests"
```

### Task 4: Enforce masks, immutable configs, and compatible resume

**Files:**
- Modify: `train/data.py:20-44`
- Modify: `train/trainer.py:39-138,229-235`
- Modify: `scripts/run_train.py`
- Modify: `tests/test_data.py:59-74`
- Modify: `tests/test_trainer.py:22-78`

**Interfaces:**
- Consumes: provenance-bearing run config from Task 3.
- Produces: validated full checkpoints, snapshots, and resume behavior.

- [ ] **Step 1: Write failing mask and resume tests**

Add:

```python
def test_configured_missing_mask_is_fatal(tmp_path):
    bp, _ = make_shards(tmp_path)
    with pytest.raises(FileNotFoundError, match="Configured loss mask"):
        PackedShards(
            bp, tmp_path / "missing.mask.bin",
            ctx=16, batch_size=2, device="cpu",
        )
```

Trainer tests:

```text
- exact run provenance resumes cursor/RNG;
- corpus fingerprint mismatch refuses before state load;
- config fingerprint mismatch refuses;
- legacy checkpoint without provenance refuses;
- snapshots carry provenance;
- resume does not rewrite config.yaml;
- `resume=none` refuses a nonempty output directory;
- auto mode with config/logs but no checkpoint refuses ambiguous partial output;
- fresh run writes config once after validation.
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_data.py tests/test_trainer.py -q
```

Expected: FAIL on missing-mask and provenance cases.

- [ ] **Step 3: Make configured masks mandatory**

Replace the mask branch with:

```python
if mask_path is None:
    self.mask = None
else:
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Configured loss mask does not exist: {mask_path}"
        )
    self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
    if len(self.mask) != len(self.tokens):
        raise ValueError("mask/token length mismatch")
```

- [ ] **Step 4: Validate output state before constructing `Trainer`**

Add:

```python
@dataclass(frozen=True)
class RunStart:
    resume_checkpoint: Path | None
    write_config: bool


def validate_run_start(cfg: dict, resume: str) -> RunStart:
    if resume not in {"auto", "none"}:
        raise ValueError(f"unknown resume mode: {resume!r}")
    out_dir = Path(cfg["out_dir"])
    ckpt = out_dir / "ckpt.pt"
    config_path = out_dir / "config.yaml"
    existing = list(out_dir.iterdir()) if out_dir.exists() else []
    if resume == "none":
        if existing:
            raise RuntimeError(
                f"Refusing fresh run: {out_dir} already contains run artifacts"
            )
        return RunStart(None, True)
    if ckpt.exists():
        if not config_path.exists():
            raise RuntimeError(
                f"Refusing resume: {ckpt} exists without config.yaml"
            )
        saved_cfg = yaml.safe_load(config_path.read_text())
        assert_resume_compatible(
            cfg["provenance"],
            saved_cfg.get("provenance"),
            config_path,
        )
        sidecar = json.loads(
            (out_dir / "checkpoint.provenance.json").read_text()
        )
        assert_resume_compatible(cfg["provenance"], sidecar, ckpt)
        return RunStart(ckpt, False)
    if existing:
        raise RuntimeError(
            f"Refusing --resume auto for ambiguous partial output: {out_dir}"
        )
    return RunStart(None, True)
```

Rules:

```text
- `auto` + valid checkpoint/config -> resume, no config rewrite;
- `auto` + empty/nonexistent output -> fresh;
- `auto` + partial output without checkpoint -> refuse;
- `none` + any existing artifact -> refuse;
- any missing/legacy/mismatched provenance -> refuse before Trainer init.
```

Store `checkpoint.provenance.json` beside `ckpt.pt` so compatibility can be
checked without deserializing model/optimizer tensors. The sidecar and embedded
checkpoint provenance must match.

- [ ] **Step 5: Carry provenance through checkpoints**

Full checkpoint:

```python
state = {
    "model": raw.state_dict(),
    "opt": self.opt.state_dict(),
    "data": self.data.state_dict(),
    "step": self.step,
    "rng_torch": torch.get_rng_state(),
    "rng_cuda": (
        torch.cuda.get_rng_state_all() if self.device == "cuda" else None
    ),
    "cfg": self.cfg,
    "provenance": self.cfg["provenance"],
}
```

Snapshot:

```python
{
    "model": raw.state_dict(),
    "step": self.step,
    "model_cfg": raw.cfg.__dict__,
    "provenance": self.cfg["provenance"],
}
```

`load_ckpt()` calls `assert_resume_compatible()` before loading model,
optimizer, data cursor, or RNG.

- [ ] **Step 6: Run trainer/data tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_data.py tests/test_trainer.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add train/data.py train/trainer.py scripts/run_train.py \
  tests/test_data.py tests/test_trainer.py
git commit -m "feat: refuse incompatible training resumes"
```

### Task 5: Validate checkpoint consumers and preserve legacy evaluation

**Files:**
- Modify: `scripts/run_evals.py`
- Modify: `scripts/local_gate_check.py`
- Modify: `scripts/analyze.py`
- Modify: `tests/test_run_evals.py`
- Modify: `tests/test_analyze.py`

**Interfaces:**
- Consumes: Task 4 checkpoint/config provenance.
- Produces: provenance-labeled eval and analysis outputs.

- [ ] **Step 1: Write failing consumer tests**

Test:

```text
- new checkpoint/config mismatch is rejected;
- legacy checkpoint is rejected by default;
- `--allow-legacy-unverified` permits evaluation and labels the summary;
- eval summary copies full run/corpus provenance;
- local snapshot check labels legacy mode;
- analysis reads arm/load/seed from config, not only the run-name regex;
- `n4m` and versioned run IDs are accepted.
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_run_evals.py tests/test_analyze.py -q
```

Expected: FAIL.

- [ ] **Step 3: Validate new artifacts and explicitly gate legacy artifacts**

Add `--allow-legacy-unverified` to eval/local scripts. New summaries include:

```python
"provenance": cfg["provenance"],
"verification": "verified-v1",
```

Legacy opt-in summaries include:

```python
"provenance": None,
"verification": "legacy-unverified",
```

Never guess a commit for legacy files.

- [ ] **Step 4: Remove run-name parsing as the authoritative source**

In analysis, load `config.yaml` and use:

```python
arm = cfg["arm"]
load = cfg["load"]
seed = int(cfg["seed"])
preset = cfg["model"] if isinstance(cfg["model"], str) else "custom"
```

The directory name remains display metadata only.

- [ ] **Step 5: Run consumer tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_run_evals.py tests/test_analyze.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_evals.py scripts/local_gate_check.py scripts/analyze.py \
  tests/test_run_evals.py tests/test_analyze.py
git commit -m "feat: verify provenance in evaluation consumers"
```

### Task 6: Stamp remote source and sequence cluster operations safely

**Files:**
- Modify: `cluster/sync_push.sh:1-21`
- Modify: `cluster/stage.sh:1-80`
- Modify: `cluster/slurm/data_prep.sbatch`
- Modify: `cluster/slurm/train_single.sbatch`
- Create: `tests/test_cluster_scripts.py`

**Interfaces:**
- Consumes: clean local Git state and Tasks 2–4.
- Produces: remote source stamp and build-before-manifest ordering.

- [ ] **Step 1: Write shell-contract tests**

Create tests that inspect/run dry paths and assert:

```text
- sync refuses a dirty worktree;
- sync refuses HEAD not descended from b805919;
- stamp contains full commit and `git_dirty: false`;
- data job logs the stamp;
- train job logs the run identity;
- stage no longer generates a manifest before report.json exists;
- scripts remain `bash -n` clean.
```

- [ ] **Step 2: Run tests to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_cluster_scripts.py -q
```

Expected: FAIL.

- [ ] **Step 3: Guard and stamp `sync_push.sh`**

Before rsync:

```bash
if [[ -n "$(git -C "$REPO_DIR" status --porcelain)" ]]; then
    echo "ERROR: protected sync requires a clean worktree" >&2
    exit 1
fi
git -C "$REPO_DIR" merge-base --is-ancestor b805919 HEAD || {
    echo "ERROR: HEAD must descend from frozen preregistration b805919" >&2
    exit 1
}
COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
```

After rsync, write remote `.source-revision.json` atomically:

```json
{"git_commit":"<40 hex>","git_dirty":false}
```

- [ ] **Step 4: Remove premature manifest generation from `stage.sh`**

`stage.sh` may submit data builds, but it must stop before manifest generation
and print the report-validation command. Manifest and training submission are a
separate human checkpoint after successful builds.

- [ ] **Step 5: Log source/run identity in Slurm**

`data_prep.sbatch` prints `.source-revision.json`. `train_single.sbatch`
prints `cfg["provenance"]["run_identity"]` before training.

- [ ] **Step 6: Run script tests and syntax checks**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_cluster_scripts.py -q
bash -n cluster/sync_push.sh cluster/stage.sh \
  cluster/slurm/data_prep.sbatch cluster/slurm/train_single.sbatch
```

Expected: PASS and shell exit 0.

- [ ] **Step 7: Commit**

```bash
git add cluster tests/test_cluster_scripts.py
git commit -m "feat: stamp and sequence protected cluster runs"
```

### Task 7: Replace manual runbook safeguards with enforced provenance

**Files:**
- Modify: `cluster/RUNBOOK.md`
- Modify: `docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md:395-413`

**Interfaces:**
- Consumes: Tasks 1–6 and the documentation-sync plan.
- Produces: executable rebuild/submit instructions for v1 provenance.

- [ ] **Step 1: Document the v1 directory and migration policy**

State:

```markdown
- Every protected corpus directory ends in `__p1-<identity>`.
- Every protected run directory ends in `__p1-<identity>`.
- Flat legacy directories are evaluation-only and require
  `--allow-legacy-unverified`.
- Missing provenance is never reconstructed or backfilled.
- Manifests are generated only after all required reports validate.
```

- [ ] **Step 2: Add exact protected workflow**

Document:

```bash
git status --porcelain
git merge-base --is-ancestor b805919 HEAD
bash cluster/sync_push.sh

sbatch --export=ALL,BUILD_ARGS="--stage full" \
  cluster/slurm/data_prep.sbatch
sbatch --export=ALL,BUILD_ARGS="--stage full1b" \
  cluster/slurm/data_prep.sbatch

python scripts/make_manifest.py --stage sweep --data-root "$FS_DATA"
python scripts/make_manifest.py --stage calib1b --data-root "$FS_DATA"
```

State that manifest generation must fail closed if reports are absent,
ambiguous, legacy, or mismatched.

- [ ] **Step 3: Update the architecture spec safeguard status**

Mark provenance safeguards implemented and point to `provenance.py`. Do not
change the frozen preregistration.

- [ ] **Step 4: Verify docs**

```bash
rg -n "__p1-|allow-legacy-unverified|b805919|provenance.py" \
  cluster/RUNBOOK.md \
  docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md
git diff --check
```

Expected: all markers found, no whitespace errors.

- [ ] **Step 5: Commit**

```bash
git add cluster/RUNBOOK.md \
  docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md
git commit -m "docs: document enforced artifact provenance"
```

### Task 8: Full verification and migration rehearsal

**Files:**
- Test: complete repository and temporary synthetic artifacts

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces: verified fail-closed provenance behavior.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_provenance.py tests/test_build.py \
  tests/test_build_corpus_script.py tests/test_manifest.py \
  tests/test_data.py tests/test_trainer.py \
  tests/test_run_evals.py tests/test_analyze.py \
  tests/test_cluster_scripts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full offline and smoke suites**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
PYTHONPATH=. .venv/bin/python -m pytest tests/test_smoke_pipeline.py -m slow -q
```

Expected: PASS.

- [ ] **Step 3: Rehearse a protected toy build and manifest**

```bash
tmp=$(mktemp -d)
printf 'One local bed document.\n\nAnother document.\n' > "$tmp/bed.txt"
PYTHONPATH=. .venv/bin/python scripts/build_corpus.py \
  --out-root "$tmp/data" --stage gates --loads n50k \
  --bed-file "$tmp/bed.txt" --total-tokens 240000 --workers 1
PYTHONPATH=. .venv/bin/python scripts/make_manifest.py \
  --stage gates --gate-loads n50k \
  --gate-tokens 240000 --data-root "$tmp/data" --out-root "$tmp/outputs"
```

Expected: corpus and run paths contain `__p1-`; reports/configs contain full
fingerprints.

- [ ] **Step 4: Prove mismatch rejection**

Copy the toy run config, alter its corpus fingerprint, and invoke the
validation entry point.

Expected stderr begins:

```text
Refusing resume
```

No config, log, checkpoint, or output file is modified.

- [ ] **Step 5: Verify cluster syntax and Git diff**

```bash
bash -n cluster/sync_push.sh cluster/stage.sh \
  cluster/slurm/data_prep.sbatch cluster/slurm/train_single.sbatch
git diff --check
git status --short
```

Expected: shell exit 0, no whitespace errors, only intended changes.

- [ ] **Step 6: Commit verification corrections if necessary**

If verification required changes:

```bash
git add provenance.py corpusgen train scripts cluster tests docs
git commit -m "test: verify artifact provenance enforcement"
```

Otherwise, do not create an empty commit.

## Self-Review

- Spec coverage: source stamping, corpus hashes, run identity, checkpoint
  compatibility, mask enforcement, legacy policy, cluster ordering, and
  migration are covered.
- Completeness scan: every step contains concrete content.
- Type consistency: `schema_version`, fingerprint fields, and identity names
  are stable across reports, YAML, checkpoints, evals, and analysis.
