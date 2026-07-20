# MemorySplit Evaluation and Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align evaluation/analysis with the frozen fact-use-QA preregistration and add nonblocking iGSM/deduction soft diagnostics at every checkpoint.

**Architecture:** Preserve exact generative scoring as the source of all decisions. Add a separate teacher-forced diagnostics module, checkpoint-specific output directories, and a preregistration-aware analysis layer that computes fact-use H1, H2/H3, the pooled seed margin, dose interaction, H4 curves, and the exploratory emergence watch.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, JSONL/JSON, matplotlib.

## Global Constraints

- `docs/superpowers/specs/2026-07-20-preregistration.md` is authoritative.
- H1 uses fact-use QA exact accuracy; soft metrics never enter a gate or verdict.
- Positive H1 requires delta `> max(2 × sigma_pool, 0.01)`, positive deltas in both confirmation seed pairs, and H2.
- Defensible null requires the combined 95% clustered CI inside the margin and H2/H3.
- H2 is split organizer-ON recall `>= dense closed-book recall - 0.02`.
- H3 is split store-OFF recall `< 0.05` and split fact bits `< 0.10 × dense fact bits`.
- Emergence watch triggers if any full-budget dense run has iGSM `>= 0.13` or deduction `>= 0.65`.
- Exact per-item scoring remains paired by `qid` and clustered by `meta.template`.
- Existing eval JSONL without new optional fields must remain loadable.
- Diagnostics unavailable on legacy data must produce explicit `null`/status fields, never NaN.
- No optimizer, training-loss, model-architecture, or tokenizer change.

---

### Task 1: Preserve canonical reference reasoning in eval items

**Files:**
- Modify: `corpusgen/records.py:52-58`
- Modify: `corpusgen/igsm_lite.py:301-328`
- Modify: `corpusgen/deduction.py:291-318`
- Test: `tests/test_igsm.py:113-136`
- Test: `tests/test_deduction.py:140-163`
- Test: `tests/test_build.py:282-299`

**Interfaces:**
- Consumes: generator `p.cot` strings ending in `\nAnswer: {answer}`.
- Produces: `QAItem.reference_reasoning: str | None`.

- [ ] **Step 1: Write failing data-contract tests**

Add to both generator test files:

```python
def _assert_reference_completion(item):
    assert item.reference_reasoning
    assert "\nAnswer:" not in item.reference_reasoning
    canonical = (
        item.prompt
        + " "
        + item.reference_reasoning
        + "\nAnswer: "
        + item.answer
    )
    assert item.prompt.endswith("Reasoning:")
    assert canonical.count("\nAnswer:") == 1
    assert canonical.endswith(f"\nAnswer: {item.answer}")


def test_eval_items_preserve_reference_reasoning():
    items = generate_igsm_eval(8, 1, 4, seed=91, exclude=set())
    for item in items:
        _assert_reference_completion(item)
```

Use `generate_deduction_eval(8, 1, 2, seed=91, exclude=set())` in the
deduction file.

Change the build-file schema assertion to:

```python
assert set(row) == {
    "qid", "task", "prompt", "answer", "reference_reasoning", "meta"
}
if stem in {"igsm", "deduction"}:
    assert row["reference_reasoning"]
else:
    assert row["reference_reasoning"] is None
```

- [ ] **Step 2: Run the tests to verify failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_igsm.py::test_eval_items_preserve_reference_reasoning \
  tests/test_deduction.py::test_eval_items_preserve_reference_reasoning \
  tests/test_build.py::test_eval_files_exist_and_parse -q
```

Expected: FAIL because `QAItem` has no `reference_reasoning`.

- [ ] **Step 3: Add the optional field and strict extraction helper**

Add to `corpusgen/records.py`:

```python
@dataclass
class QAItem:
    qid: str
    task: str
    prompt: str
    answer: str
    reference_reasoning: str | None = None
    meta: dict = field(default_factory=dict)


def extract_reference_reasoning(cot: str, answer: str) -> str:
    suffix = f"\nAnswer: {answer}"
    if not cot.endswith(suffix):
        raise ValueError(
            f"reference CoT must end with {suffix!r}; got {cot[-80:]!r}"
        )
    reasoning = cot[: -len(suffix)]
    if not reasoning:
        raise ValueError("reference reasoning body is empty")
    return reasoning
```

In each reasoning eval generator, pass:

```python
reference_reasoning=extract_reference_reasoning(p.cot, str(p.answer)),
```

Import the helper from `corpusgen.records`. Nonreasoning generators rely on
the default `None`.

- [ ] **Step 4: Run focused and build tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_igsm.py tests/test_deduction.py tests/test_build.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add corpusgen/records.py corpusgen/igsm_lite.py corpusgen/deduction.py \
  tests/test_igsm.py tests/test_deduction.py tests/test_build.py
git commit -m "feat: preserve reasoning references for eval diagnostics"
```

### Task 2: Implement teacher-forced reasoning diagnostics

**Files:**
- Create: `evals/diagnostics.py`
- Create: `tests/test_diagnostics.py`
- Read only: `evals/natural.py:23-56`
- Read only: `train/model.py:170-204`

**Interfaces:**
- Consumes: `list[QAItem]`, a model returning `[B,T,V]` logits, tokenizer, device.
- Produces: per-`qid` diagnostic dictionaries and aggregate summaries.

```python
DiagnosticScalar = float | int | bool | str | None
ReasoningDiagnostic = dict[str, DiagnosticScalar]
```

- `score_reasoning_diagnostics(model, tok, items, device, batch_size=4) -> dict[str, ReasoningDiagnostic]`
- `summarize_reasoning_diagnostics(diagnostics) -> dict[str, float | int | None | str]`

- [ ] **Step 1: Write failing scorer tests**

Create a zero-logit model:

```python
from types import SimpleNamespace

import pytest
import torch

from corpusgen.records import QAItem
from evals.diagnostics import (
    score_reasoning_diagnostics,
    summarize_reasoning_diagnostics,
)
from train.tokenizer import get_tok


class UniformModel:
    def __init__(self, vocab_size=50304, ctx=256):
        self.cfg = SimpleNamespace(vocab_size=vocab_size, ctx=ctx)

    def forward(self, x):
        logits = torch.zeros(
            x.shape[0], x.shape[1], self.cfg.vocab_size, device=x.device
        )
        return logits, None


def item(task="igsm", answer="12345678901234567890"):
    return QAItem(
        qid=f"{task}-0",
        task=task,
        prompt="Question\nReasoning:",
        answer=answer,
        reference_reasoning="Compute the result.",
        meta={"template": "t"},
    )


def test_uniform_multitoken_answer():
    tok = get_tok()
    out = score_reasoning_diagnostics(
        UniformModel(), tok, [item()], "cpu", batch_size=1
    )["igsm-0"]
    expected = torch.log(torch.tensor(float(tok.VOCAB_SIZE))).item()
    assert out["status"] == "ok"
    assert out["answer_token_count"] >= 2
    assert out["final_answer_nll"] == pytest.approx(expected)
    assert out["reasoning_trace_nll"] == pytest.approx(expected)
    assert out["correct_answer_logprob_per_token"] == pytest.approx(-expected)


def test_uniform_deduction_probability_and_brier():
    out = score_reasoning_diagnostics(
        UniformModel(), get_tok(), [item("deduction", "yes")], "cpu"
    )["deduction-0"]
    assert out["correct_answer_log_odds"] == pytest.approx(0.0)
    assert out["correct_answer_probability"] == pytest.approx(0.5)
    assert out["brier"] == pytest.approx(0.25)
```

Add tests for:

```text
- leading-space tokenization (`" " + answer`);
- disjoint trace and answer masks;
- batch sizes 1 and 4 producing identical rows;
- right padding not affecting scores;
- prompt-left-truncation setting `context_truncated=true`;
- suffix longer than context returning `sequence_too_long`;
- missing reference returning `missing_reference_reasoning`;
- duplicate qids raising ValueError;
- summaries averaging only finite `status == "ok"` rows;
- no JSON field containing NaN or Infinity.
```

- [ ] **Step 2: Run tests to verify import failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_diagnostics.py -q
```

Expected: FAIL because `evals.diagnostics` does not exist.

- [ ] **Step 3: Implement causal span scoring**

Create `evals/diagnostics.py` with:

```python
from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from corpusgen.records import QAItem

ROLE = "supporting_nonblocking"
VERSION = 1


def _model_ctx(model) -> int:
    raw = getattr(model, "_orig_mod", model)
    return int(raw.cfg.ctx)


def _encode_item(tok, item: QAItem) -> tuple[list[int], list[int], list[int]]:
    prompt_ids = tok.encode(item.prompt) or [tok.EOT]
    trace_ids = tok.encode(" " + item.reference_reasoning + "\nAnswer:")
    answer_ids = tok.encode(" " + item.answer)
    if not trace_ids or not answer_ids:
        raise ValueError("trace and answer must each encode to at least one token")
    return prompt_ids, trace_ids, answer_ids


def _token_logprobs(model, seqs, targets, device, pad_id):
    max_len = max(len(s) for s in seqs)
    x = torch.full(
        (len(seqs), max_len), pad_id, dtype=torch.long, device=device
    )
    for i, seq in enumerate(seqs):
        x[i, : len(seq)] = torch.tensor(seq, device=device)
    with torch.no_grad():
        logits, _ = model.forward(x)
    rows = []
    for row, specs in enumerate(targets):
        values = []
        for position, token_id in specs:
            score = logits[row, position - 1].float()
            values.append(
                float(score[token_id] - torch.logsumexp(score, dim=-1))
            )
        rows.append(values)
    return rows
```

Call `_token_logprobs(model, seqs, targets, device, pad_id=tok.EOT)`. For each item:

1. Build `prompt_ids + trace_ids + answer_ids`.
2. If the suffix fits but the full sequence does not, left-truncate only
   `prompt_ids`.
3. Score trace positions and answer positions from logits at `j - 1`.
4. Compute sums/means in float32.
5. For deduction, score both `" yes"` and `" no"` under the same
   teacher-forced answer context and normalize with `logaddexp`.

Return exactly:

```python
{
    "version": 1,
    "role": ROLE,
    "status": "ok",
    "correct_answer_logprob": answer_lp_sum,
    "correct_answer_logprob_per_token": answer_lp_mean,
    "final_answer_nll": -answer_lp_mean,
    "answer_token_count": len(answer_ids),
    "reasoning_trace_nll": -trace_lp_mean,
    "reasoning_token_count": len(trace_ids),
    "correct_answer_log_odds": deduction_log_odds_or_none,
    "correct_answer_probability": deduction_probability_or_none,
    "brier": deduction_brier_or_none,
    "context_truncated": context_truncated,
}
```

Unavailable rows use one of:

```text
missing_reference_reasoning
invalid_reference
sequence_too_long
```

All numeric fields are `None` when unavailable.

- [ ] **Step 4: Implement aggregation**

`summarize_reasoning_diagnostics()` returns:

```python
def mean_or_none(field: str):
    values = [
        float(row[field])
        for row in diagnostics.values()
        if row["status"] == "ok" and row[field] is not None
    ]
    return sum(values) / len(values) if values else None


{
    "role": ROLE,
    "version": VERSION,
    "n_items": len(diagnostics),
    "n_scored": n_ok,
    "n_unavailable": len(diagnostics) - n_ok,
    "n_context_truncated": n_truncated,
    "mean_correct_answer_logprob_per_token": mean_or_none(
        "correct_answer_logprob_per_token"
    ),
    "mean_final_answer_nll": mean_or_none("final_answer_nll"),
    "mean_reasoning_trace_nll": mean_or_none("reasoning_trace_nll"),
    "mean_correct_answer_log_odds": mean_or_none(
        "correct_answer_log_odds"
    ),
    "mean_correct_answer_probability": mean_or_none(
        "correct_answer_probability"
    ),
    "mean_brier": mean_or_none("brier"),
}
```

Reject nonfinite values instead of serializing them.

- [ ] **Step 5: Run diagnostics tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_diagnostics.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add evals/diagnostics.py tests/test_diagnostics.py
git commit -m "feat: add nonblocking reasoning diagnostics"
```

### Task 3: Make checkpoint evaluation additive instead of destructive

**Files:**
- Modify: `scripts/run_evals.py:41-142`
- Create: `tests/test_run_evals.py`
- Modify: `tests/test_smoke_pipeline.py`

**Interfaces:**
- Consumes: exact scorer and Task 2 diagnostics.
- Produces: final and checkpoint-specific eval trees.

```text
<run>/evals/summary.json
<run>/evals/checkpoints/step0001520/summary.json
<run>/evals/checkpoints/step0001520/{igsm,deduction,factqa}.jsonl
```

- [ ] **Step 1: Write failing output-routing and integration tests**

Extract and test:

```python
def checkpoint_step(state: dict) -> int:
    if "step" not in state:
        raise ValueError("checkpoint has no training step")
    return int(state["step"])

def eval_output_dir(
    run_dir: Path, ckpt_rel: str | None, step: int
) -> Path:
    if ckpt_rel is None:
        return run_dir / "evals"
    return run_dir / "evals" / "checkpoints" / f"step{step:07d}"
```

Test:

```text
- final checkpoint writes to `<run>/evals`;
- explicit snapshot writes to its step directory;
- two snapshots cannot overwrite each other;
- full summaries retain exact `acc`;
- iGSM/deduction rows receive nested `diagnostics`;
- factqa rows do not receive reasoning diagnostics;
- legacy items produce unavailable diagnostics without failing exact scoring.
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_run_evals.py -q
```

Expected: FAIL because the helpers and nested diagnostics are absent.

- [ ] **Step 3: Return checkpoint metadata from model loading**

Change:

```python
def load_model(
    run_dir: Path, ckpt_rel: str | None, device: str
) -> tuple[GPT, int]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    model_cfg = (
        PRESETS[cfg["model"]]
        if isinstance(cfg["model"], str)
        else GPTConfig(**cfg["model"])
    )
    model = GPT(model_cfg)
    ckpt_path = run_dir / (ckpt_rel or "ckpt.pt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, int(state["step"])
```

- [ ] **Step 4: Attach diagnostics before saving reasoning rows**

For iGSM/deduction:

```python
diagnostics = score_reasoning_diagnostics(
    model, tok, items, device, batch_size=args.diagnostic_batch_size
)
for row in rows:
    row["diagnostics"] = diagnostics[row["qid"]]
task_summary["diagnostics"] = summarize_reasoning_diagnostics(diagnostics)
```

Add:

```python
ap.add_argument("--diagnostic-batch-size", type=int, default=4)
```

Summary root fields:

```python
{
    "run": run_dir.name,
    "arm": arm,
    "ckpt": args.ckpt or "ckpt.pt",
    "checkpoint_step": step,
    "diagnostics_role": "supporting_nonblocking",
}
```

Do not use diagnostics in `composite_knowledge_free`.

- [ ] **Step 5: Add sorted all-checkpoint execution**

Add:

```python
def discover_checkpoints(run_dir: Path) -> list[str | None]:
    snapshots = sorted(
        run_dir.glob("snapshots/step*.pt"),
        key=lambda p: int(p.stem.removeprefix("step")),
    )
    return [str(p.relative_to(run_dir)) for p in snapshots] + [None]
```

Expose `--all-checkpoints`. Move the current one-checkpoint body
(`run_evals.py:72-138`) into
`evaluate_checkpoint(args, run_dir, ckpt_rel) -> dict`, replacing only model
loading, output-directory selection, diagnostic attachment, and the returned
summary described in Steps 3–4. The function writes its JSONL/JSON outputs,
prints the summary, and returns that same summary dictionary.

When `--all-checkpoints` is present, evaluate every snapshot and final
checkpoint. If a snapshot step equals the final checkpoint step, evaluate it
once. Full checkpoint runs include exact iGSM, deduction, and factqa scoring so
the preregistered emergence watch and H4 curves are available.

- [ ] **Step 6: Run script and smoke tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_run_evals.py tests/test_smoke_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_evals.py tests/test_run_evals.py tests/test_smoke_pipeline.py
git commit -m "feat: preserve checkpoint eval histories"
```

### Task 4: Implement the frozen primary analysis and emergence watch

**Files:**
- Modify: `evals/stats.py:8-95`
- Modify: `evals/figures.py:1-52`
- Modify: `scripts/analyze.py:1-131`
- Modify: `tests/test_stats.py:1-157`
- Create: `tests/test_analyze.py`

**Interfaces:**
- Consumes: final and checkpoint summaries from Task 3.
- Produces: `analysis.json`, fact-use dose figure, verdict, H2/H3, H4 curves, and exploratory emergence output.

- [ ] **Step 1: Write failing preregistration-statistic tests**

Add to `tests/test_stats.py`:

```python
def test_pooled_within_load_sigma():
    groups = {
        "n50k": [0.10, 0.14],
        "n200k": [0.20, 0.24],
        "n800k": [0.30, 0.34],
    }
    out = pooled_within_group_sigma(groups)
    expected = np.sqrt(
        sum((len(v) - 1) * np.var(v, ddof=1) for v in groups.values())
        / sum(len(v) - 1 for v in groups.values())
    )
    assert out == pytest.approx(expected)


def test_confirmation_verdict_positive_requires_both_signs_and_h2():
    result = confirmation_verdict(
        per_seed_deltas=[0.05, 0.04],
        ci=(0.03, 0.06),
        sigma_pool=0.01,
        h2=True,
        h3=True,
    )
    assert result["margin"] == pytest.approx(0.02)
    assert result["verdict"] == "positive"


def test_confirmation_null_requires_ci_inside_margin_and_h3():
    result = confirmation_verdict(
        per_seed_deltas=[0.002, -0.001],
        ci=(-0.006, 0.007),
        sigma_pool=0.002,
        h2=True,
        h3=True,
    )
    assert result["margin"] == pytest.approx(0.01)
    assert result["verdict"] == "defensible_null"
```

Also test:

```text
- one nonpositive seed makes a positive verdict inconclusive;
- H2 failure voids positive;
- H3 failure prevents defensible null;
- CI crossing the margin is inconclusive;
- pooled sigma rejects groups with fewer than two seeds;
- slope interaction is split slope minus dense slope over log entity count.
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_stats.py -q
```

Expected: FAIL because the preregistration helpers are absent.

- [ ] **Step 3: Add frozen-statistic helpers**

Add:

```python
def pooled_within_group_sigma(groups: dict[str, list[float]]) -> float:
    numerator = denominator = 0.0
    for name, values in groups.items():
        if len(values) < 2:
            raise ValueError(f"{name} needs at least two seed-pair deltas")
        numerator += (len(values) - 1) * float(np.var(values, ddof=1))
        denominator += len(values) - 1
    if denominator == 0:
        raise ValueError("no pooled degrees of freedom")
    return float(np.sqrt(numerator / denominator))


def confirmation_verdict(
    per_seed_deltas: list[float],
    ci: tuple[float, float],
    sigma_pool: float,
    h2: bool,
    h3: bool,
) -> dict:
    margin = max(2.0 * sigma_pool, 0.01)
    same_positive_sign = bool(per_seed_deltas) and all(
        delta > 0 for delta in per_seed_deltas
    )
    mean_delta = float(np.mean(per_seed_deltas))
    if mean_delta > margin and same_positive_sign and h2:
        verdict = "positive"
    elif ci[0] >= -margin and ci[1] <= margin and h2 and h3:
        verdict = "defensible_null"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "margin": margin,
        "mean_delta": mean_delta,
        "same_positive_sign": same_positive_sign,
        "h2": h2,
        "h3": h3,
        "ci_lo": ci[0],
        "ci_hi": ci[1],
    }
```

Add `dose_slope_interaction(points)` using `np.polyfit(log(n_entities), acc, 1)`
for each arm and returning `split_slope - dense_slope`.

- [ ] **Step 4: Generalize the figure API**

Change:

```python
def dose_response_figure(
    points: list[dict],
    out_png,
    *,
    metric_key: str = "accuracy",
    y_label: str = "fact-use QA accuracy",
) -> Path:
```

Read `p[metric_key]`, and update its test to use `"accuracy"`. Remove the
module docstring's claim that the plot is necessarily a reasoning composite.

- [ ] **Step 5: Write failing end-to-end analysis tests**

Create synthetic dense/split run directories with two seeds per load and one
confirmation load. Assert:

```text
- dose_response.png uses factqa accuracy;
- primary pair deltas equal paired_delta(factqa split, dense);
- sigma_pool uses six 160M pair deltas grouped by load;
- confirmation verdict uses the 1.0-point floor;
- H2/H3 are reported per pair and aggregated with all-pairs semantics;
- emergence watch is false below 0.13/0.65 and true at either threshold;
- knowledge-free composite is labeled exploratory only when triggered;
- checkpoint factqa summaries are step-sorted for H4;
- no diagnostic metric appears in `verdict` inputs;
- summary Markdown says "supporting nonblocking" for soft metrics.
```

- [ ] **Step 6: Refactor and align `scripts/analyze.py`**

Extract the current `main()` analysis body into
`analyze_runs(root: Path, out: Path) -> dict`; `main()` only parses paths,
calls it, and prints the two written output paths.

Use fact-use rows as primary:

```python
factqa = paired_delta(rows_split_factqa, rows_dense_factqa)
primary_delta = factqa["delta"]
```

Build dose points:

```python
{
    "n_entities": LOADS[load],
    "arm": arm,
    "seed": seed,
    "preset": preset,
    "accuracy": summary["factqa"]["acc"],
}
```

Aggregate:

```python
result = {
    "primary": "factqa",
    "sweep": {
        "pair_deltas": sweep_pair_deltas,
        "sigma_pool": sigma_pool,
        "arm_x_log_load": dose_slope_interaction(dose_points),
    },
    "confirmation": {
        "per_seed_deltas": confirmation_pair_deltas,
        "clustered_ci": [combined["ci_lo"], combined["ci_hi"]],
        "h2": h2_all_pairs,
        "h3": h3_all_pairs,
        "verdict": confirmation_verdict(
            confirmation_pair_deltas,
            (combined["ci_lo"], combined["ci_hi"]),
            sigma_pool,
            h2_all_pairs,
            h3_all_pairs,
        ),
    },
    "emergence_watch": {
        "triggered": emergence_trigger_run is not None,
        "trigger_run": emergence_trigger_run,
        "knowledge_free": exploratory_knowledge_free,
        "role": "exploratory_secondary",
    },
    "checkpoint_curves": checkpoint_curves,
    "diagnostics_role": "supporting_nonblocking",
}
```

For the combined confirmation CI, prefix each row's qid and cluster with its
seed before calling `paired_delta`; this makes `(seed, template)` the cluster
and prevents cross-seed qid collisions.

H2 holds only if every confirmation pair satisfies the two-point guardrail.
H3 holds only if every split confirmation run has store-OFF recall `< 0.05`
and split bits `< 0.10 ×` its dense twin.

- [ ] **Step 7: Run analysis tests**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_stats.py tests/test_analyze.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add evals/stats.py evals/figures.py scripts/analyze.py \
  tests/test_stats.py tests/test_analyze.py
git commit -m "feat: implement the frozen MemorySplit analysis"
```

### Task 5: Update local and cluster evaluation callers

**Files:**
- Modify: `scripts/local_gate_check.py:1-61`
- Modify: `cluster/slurm/eval_runs.sbatch:10-28`
- Modify: `scripts/smoke_test.py`
- Modify: `cluster/RUNBOOK.md:84-90`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: operator-visible diagnostics and checkpoint histories.

- [ ] **Step 1: Change the local script from a gate to an emergence diagnostic**

Keep its filename for compatibility, but change its description and output:

```python
diagnostics = score_reasoning_diagnostics(
    model, tok, items, device, batch_size=args.diagnostic_batch_size
)
summary = summarize_reasoning_diagnostics(diagnostics)
print(
    f"{task}: acc={acc:.4f} "
    f"answer_lp={summary['mean_correct_answer_logprob_per_token']} "
    f"trace_nll={summary['mean_reasoning_trace_nll']} "
    "(supporting; nonblocking)"
)
```

Add `--diagnostic-batch-size`. Do not add pass/fail exits.

- [ ] **Step 2: Make Slurm evaluate checkpoint histories explicitly**

Add an environment-controlled branch:

```bash
if [[ "${ALL_CHECKPOINTS:-0}" == "1" ]]; then
    EXTRA_ARGS="--all-checkpoints ${EVAL_ARGS:-}"
else
    EXTRA_ARGS="${EVAL_ARGS:-}"
fi
"$VENV/bin/python" -u scripts/run_evals.py \
    --run "outputs/$run" $EXTRA_ARGS
```

Document quoting expectations for `EVAL_ARGS`. Run `bash -n`.

- [ ] **Step 3: Exercise diagnostics in the smoke script**

Score four iGSM and four deduction items. Assert:

```python
assert all(row["status"] == "ok" for row in diagnostics.values())
assert all(
    math.isfinite(row["final_answer_nll"])
    for row in diagnostics.values()
)
```

Do not assert that diagnostic quality improves in a toy run.

- [ ] **Step 4: Update the runbook commands**

Add:

```bash
sbatch --export=ALL,RUNS="<run ids>",ALL_CHECKPOINTS=1 \
  cluster/slurm/eval_runs.sbatch
```

State that soft metrics are supporting and fact-use exact accuracy remains H1.

- [ ] **Step 5: Run caller checks**

Run:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_diagnostics.py tests/test_run_evals.py tests/test_smoke_pipeline.py -q
bash -n cluster/slurm/eval_runs.sbatch
```

Expected: PASS and shell exit 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/local_gate_check.py scripts/smoke_test.py \
  cluster/slurm/eval_runs.sbatch cluster/RUNBOOK.md
git commit -m "feat: expose checkpoint diagnostics operationally"
```

### Task 6: Full verification

**Files:**
- Test: complete repository

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: verified eval/analysis implementation.

- [ ] **Step 1: Run focused tests**

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_igsm.py tests/test_deduction.py tests/test_build.py \
  tests/test_diagnostics.py tests/test_run_evals.py tests/test_stats.py \
  tests/test_analyze.py -q
```

Expected: PASS.

- [ ] **Step 2: Run the complete offline suite**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Expected: all tests pass in under the repository's documented one-minute
offline budget.

- [ ] **Step 3: Run the slow smoke pipeline**

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_smoke_pipeline.py -m slow -q
```

Expected: PASS.

- [ ] **Step 4: Run an existing-checkpoint diagnostic smoke**

```bash
PYTHONPATH=. .venv/bin/python scripts/local_gate_check.py \
  --snapshot outputs/pulled/step0001520.pt \
  --eval-dir outputs/pulled \
  --limit 32 \
  --batch-size 4 \
  --diagnostic-batch-size 2
```

Expected: exact accuracy plus nonblocking diagnostic summaries. If the pulled
legacy eval JSONL lacks `reference_reasoning`, exact accuracy still runs and
diagnostics explicitly report unavailable.

- [ ] **Step 5: Check frozen-threshold isolation**

Run:

```bash
rg -n "0\.13|0\.65|0\.01|supporting_nonblocking|factqa" \
  scripts/analyze.py evals tests
rg -n "diagnostic.*verdict|brier.*margin|logprob.*margin" scripts evals
git diff --check
```

Expected: thresholds appear only in preregistration-aware analysis/tests; the
second command finds no diagnostic-to-verdict coupling; no whitespace errors.

- [ ] **Step 6: Commit validation corrections if necessary**

If verification required changes:

```bash
git add corpusgen evals scripts tests cluster
git commit -m "test: verify frozen evaluation and diagnostics"
```

Otherwise, do not create an empty commit.

## Self-Review

- Spec coverage: fact-use H1, H2/H3, pooled margin, sweep interaction, H4,
  emergence watch, exact checkpoint outputs, and soft diagnostics are mapped.
- Completeness scan: every step contains concrete content.
- Type consistency: `reference_reasoning`, diagnostic row fields, summary
  fields, and analysis inputs use the same names throughout.
