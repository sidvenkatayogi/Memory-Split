# MemorySplit Frozen Corpus Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generated corpora, eval artifacts, and the smoke pipeline exactly match the frozen Option-B recipe before any battery rebuild.

**Architecture:** Keep the training generators general-purpose, but pin `BuildCfg` to the frozen in-distribution ranges and emit separate report-only OOD files from disjoint difficulty bands. Evaluate OOD files without feeding them into H1, the emergence trigger, or the knowledge-free ID composite.

**Tech Stack:** Python 3.12, deterministic corpus generators, JSONL, pytest.

## Global Constraints

- Frozen mixture: bed 0.54, bios 0.23, iGSM 0.12, deduction 0.08, factqa 0.03.
- Frozen iGSM train/ID: op 1–4, `1/op` training weighting, at most one distractor for op <= 2.
- Frozen iGSM OOD: op 5–8, report-only.
- Frozen deduction train/ID: depth 1–2, 4–6 facts, 3–4 rules.
- Frozen deduction OOD: depth 3–4, report-only.
- CoT plus final `Answer:` remains unchanged.
- OOD item counts equal the corresponding 10,000-item ID counts unless the preregistration is amended before battery data exists.
- OOD metrics never enter H1, H2, H3, H4, or the emergence-watch trigger.
- Same seed and config must remain byte-identical across worker counts.
- Do not change the frozen preregistration.

---

### Task 1: Pin the frozen builder defaults with tests

**Files:**
- Modify: `corpusgen/build.py:53-88`
- Modify: `corpusgen/igsm_lite.py:1-8,153-168,265-309`
- Modify: `tests/test_build.py`
- Modify: `tests/test_igsm.py`
- Modify: `tests/test_deduction.py`

**Interfaces:**
- Consumes: existing general generator APIs.
- Produces: tested frozen `BuildCfg` defaults and accurate documentation.

- [ ] **Step 1: Write failing frozen-default tests**

Add:

```python
def test_build_cfg_defaults_match_frozen_preregistration():
    cfg = BuildCfg(n_entities=50, total_tokens=100_000, seed=1)
    assert cfg.shares() == {
        "bed": 0.54,
        "bio": 0.23,
        "igsm": 0.12,
        "deduction": 0.08,
        "factqa": 0.03,
    }
    assert cfg.igsm_op == (1, 4)
    assert cfg.deduction_depth == (1, 2)
```

Add iGSM tests:

```python
def test_easy_option_b_problems_have_at_most_one_distractor():
    rng = random.Random(20260720)
    for op in (1, 2):
        for _ in range(100):
            p = generate_problem(op, rng)
            names = _defined_names(p.prompt)
            distractors = [
                name for name in names
                if f"The number of {name} is " not in p.cot
            ]
            assert len(distractors) <= 1


def test_option_b_training_weights_low_operations():
    docs = generate_igsm_docs(2000, 1, 4, seed=17)
    counts = Counter(d.meta["op"] for d in docs)
    assert counts[1] > counts[2] > counts[3] > counts[4]
```

Add deduction tests asserting depth <=2 yields at most six facts and four
rules. Keep the existing broad-range generator tests because the API must still
support OOD depths.

- [ ] **Step 2: Run tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_build.py tests/test_igsm.py tests/test_deduction.py -q
```

Expected: existing defaults pass; the inaccurate iGSM module documentation or
any unpinned behavior fails the new assertions.

- [ ] **Step 3: Correct the iGSM module contract**

State:

```python
"""iGSM-lite deterministic modular-arithmetic generator.

The generator supports arbitrary positive operation bands. Under the frozen
Option-B builder defaults, training uses op 1-4 with 1/op sampling; op <= 2
has at most one distractor, while harder/OOD problems may have up to three.
"""
```

Do not hard-code Option-B ranges inside `generate_problem`; the builder owns
the frozen selection.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_build.py tests/test_igsm.py tests/test_deduction.py -q
git add corpusgen/build.py corpusgen/igsm_lite.py \
  tests/test_build.py tests/test_igsm.py tests/test_deduction.py
git commit -m "test: pin the frozen corpus recipe"
```

### Task 2: Emit deterministic OOD eval artifacts

**Files:**
- Modify: `corpusgen/build.py:402-461`
- Modify: `tests/test_build.py:279-351`

**Interfaces:**
- Consumes: frozen ID bands and general eval generators.
- Produces: `eval/igsm_ood.jsonl` and `eval/deduction_ood.jsonl`.

- [ ] **Step 1: Extend the failing eval-file test**

Use:

```python
expected = {
    "igsm": CFG.n_igsm_eval,
    "igsm_ood": CFG.n_igsm_eval,
    "deduction": CFG.n_deduction_eval,
    "deduction_ood": CFG.n_deduction_eval,
    "factqa": CFG.n_factqa_eval,
    "factqa_fresh": CFG.n_fresh_eval,
    "recall": CFG.n_entities * 6,
}
```

Assert:

```python
for row in load_jsonl(out / "eval/igsm_ood.jsonl"):
    assert 5 <= row["meta"]["op"] <= 8
for row in load_jsonl(out / "eval/deduction_ood.jsonl"):
    assert 3 <= row["meta"]["depth"] <= 4
```

Assert all OOD structure hashes are absent from training and ID sets.

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_build.py::test_eval_files_exist_and_parse -q
```

Expected: FAIL because OOD files do not exist.

- [ ] **Step 3: Generate OOD sets with dedicated seeds**

Extend `evals`:

```python
"igsm_ood": igsm.generate_igsm_eval(
    cfg.n_igsm_eval,
    5,
    8,
    cfg.seed * 1000 + 45,
    igsm_hashes,
),
"deduction_ood": deduction.generate_deduction_eval(
    cfg.n_deduction_eval,
    3,
    4,
    cfg.seed * 1000 + 56,
    ded_hashes,
),
```

Keep ID seeds 44/55 unchanged. The bands are disjoint, but still pass training
hashes to preserve the exclusion invariant.

- [ ] **Step 4: Run build determinism tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_build.py -q
```

Expected: PASS, including serial/parallel byte identity.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/build.py tests/test_build.py
git commit -m "feat: emit frozen OOD eval sets"
```

### Task 3: Score OOD sets without contaminating decisions

**Files:**
- Modify: `scripts/run_evals.py:8-10,90-138`
- Modify: `tests/test_run_evals.py`

**Interfaces:**
- Consumes: OOD JSONL from Task 2.
- Produces: `igsm_ood` and `deduction_ood` exact/diagnostic summaries labeled report-only.

- [ ] **Step 1: Write failing role-isolation tests**

Assert:

```text
- OOD files are scored when present;
- summaries contain `"role": "report_only_ood"`;
- OOD rows never enter `composite_knowledge_free`;
- OOD results never affect the emergence-watch trigger;
- missing OOD files are an error for provenance-v1/frozen corpora;
- legacy corpora may omit OOD files only under explicit legacy mode.
```

- [ ] **Step 2: Run test to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_run_evals.py -q
```

Expected: FAIL.

- [ ] **Step 3: Add OOD tasks to the eval loop**

Use task metadata:

```python
TASKS = (
    ("igsm", False, "id_emergence_watch"),
    ("igsm_ood", False, "report_only_ood"),
    ("deduction", False, "id_emergence_watch"),
    ("deduction_ood", False, "report_only_ood"),
    ("factqa", None, "primary"),
    ("factqa_fresh", None, "supporting"),
)
```

Set `summary[task]["role"] = role`. Compute
`composite_knowledge_free` only from `igsm` and `deduction`.

If the evaluation-and-diagnostics plan has already landed, attach soft
diagnostics to both ID and OOD reasoning tasks, still with OOD marked
report-only.

- [ ] **Step 4: Run tests and commit**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_run_evals.py -q
git add scripts/run_evals.py tests/test_run_evals.py
git commit -m "feat: report frozen OOD evaluations"
```

### Task 4: Make the smoke pipeline exercise frozen defaults

**Files:**
- Modify: `scripts/smoke_test.py:97-117`
- Modify: `tests/test_smoke_pipeline.py:32-88`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a cheap end-to-end check using Option-B ID ranges and OOD files.

- [ ] **Step 1: Write a failing smoke-config assertion**

In the smoke test, assert:

```python
assert cfg.igsm_op == (1, 4)
assert cfg.deduction_depth == (1, 2)
```

After corpus build:

```python
assert (data_dir / "eval/igsm_ood.jsonl").exists()
assert (data_dir / "eval/deduction_ood.jsonl").exists()
```

- [ ] **Step 2: Run smoke tests to verify failure**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_smoke_pipeline.py -q
```

Expected: FAIL because `scripts/smoke_test.py` overrides the frozen ranges.

- [ ] **Step 3: Remove stale difficulty overrides**

Construct `BuildCfg` without overriding `igsm_op` or `deduction_depth`. Keep
small eval counts and token budget:

```python
cfg = BuildCfg(
    n_entities=40,
    total_tokens=240_000,
    seed=7,
    workers=1,
    n_igsm_eval=60,
    n_deduction_eval=60,
    n_factqa_eval=40,
    n_fresh_entities=20,
    n_fresh_eval=20,
    n_recall_entities=20,
)
```

- [ ] **Step 4: Run smoke and full corpus tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_smoke_pipeline.py tests/test_build.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_test.py tests/test_smoke_pipeline.py
git commit -m "test: exercise the frozen recipe in smoke runs"
```

### Task 5: Full scientific-recipe verification

**Files:**
- Test: corpus generators, builder, eval runner, smoke pipeline

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: verified frozen recipe before provenance hashing/rebuild.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_igsm.py tests/test_deduction.py tests/test_build.py \
  tests/test_run_evals.py tests/test_smoke_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete offline suite**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Expected: PASS in the documented offline budget.

- [ ] **Step 3: Build and inspect a frozen toy corpus**

Run:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import itertools
import tempfile
from pathlib import Path

from corpusgen.build import BuildCfg, build_corpus
from scripts.smoke_test import toy_bed
from train.tokenizer import get_tok

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    cfg = BuildCfg(
        n_entities=40,
        total_tokens=240_000,
        seed=7,
        workers=1,
        n_igsm_eval=60,
        n_deduction_eval=60,
        n_factqa_eval=40,
        n_fresh_entities=20,
        n_fresh_eval=20,
        n_recall_entities=20,
    )
    report = build_corpus(
        cfg, get_tok(), itertools.cycle(toy_bed()), out
    )
    assert (out / "eval/igsm_ood.jsonl").exists()
    assert (out / "eval/deduction_ood.jsonl").exists()
assert report["cfg"]["igsm_op"] == [1, 4]
assert report["cfg"]["deduction_depth"] == [1, 2]
assert report["eval_counts"]["igsm_ood"] == report["cfg"]["n_igsm_eval"]
assert report["eval_counts"]["deduction_ood"] == report["cfg"]["n_deduction_eval"]
PY
```

Expected: PASS.

- [ ] **Step 4: Check preregistration immutability**

```bash
git diff b805919 -- docs/superpowers/specs/2026-07-20-preregistration.md
git diff --check
```

Expected: no preregistration diff and no whitespace errors.

- [ ] **Step 5: Commit verification corrections if necessary**

If verification required changes:

```bash
git add corpusgen scripts tests
git commit -m "test: verify frozen corpus alignment"
```

Otherwise, do not create an empty commit.

## Self-Review

- Spec coverage: frozen shares, ID bands, OOD bands, weighting, distractors,
  small deduction bases, smoke defaults, and report-only isolation are covered.
- Completeness scan: every step contains concrete content.
- Boundary check: no endpoint, threshold, optimizer, or model change.
