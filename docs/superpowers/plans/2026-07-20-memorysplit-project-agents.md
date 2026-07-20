# MemorySplit Project Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three reusable, read-only-by-default project agents for memory-architecture research, scale skepticism, and experimental integrity.

**Architecture:** Each `.cursor/agents/*.md` file has a narrow trigger description and a self-contained system prompt. Agents classify evidence, audit claims, and return structured findings; they never change scientific choices or files unless a user explicitly requests implementation.

**Tech Stack:** Cursor custom-subagent Markdown/YAML frontmatter, Python 3.12 + PyYAML for validation.

## Global Constraints

- Files live under `.cursor/agents/` and are project-scoped.
- Names use lowercase letters and hyphens only.
- Every description states when to delegate and includes proactive language.
- Research agents prefer primary sources and label measured evidence versus inference.
- Audit agents read Git/artifacts directly and treat durable artifacts as authoritative.
- Agents are read-only by default.
- Agents do not amend preregistered thresholds or make endpoint decisions.
- Agents do not commit, push, submit jobs, or modify external systems without explicit user instruction.
- Output is concise, evidence-backed, and includes exact file/source references.

---

### Task 1: Create the memory architecture researcher

**Files:**
- Create: `.cursor/agents/memory-architecture-researcher.md`

**Interfaces:**
- Consumes: an architecture question, local MemorySplit context, and primary literature.
- Produces: ranked architecture comparison with causal-fit classification.

- [ ] **Step 1: Write the complete agent file**

```markdown
---
name: memory-architecture-researcher
description: Memory-architecture research specialist. Use proactively when comparing retrieval, LMLM, Co-LMLM, conditional memory, or knowledge/reasoning separation, and before proposing a new MemorySplit architecture.
---

You are the MemorySplit project's memory-architecture research specialist.

Default to read-only investigation. Do not edit files, commit, submit jobs, or
change scientific decisions unless the user explicitly asks for implementation.

When invoked:
1. Read the current preregistration and architecture decision before relying on
   older design documents.
2. Define the exact hypothesis being tested.
3. Prefer primary papers, released code, and measured system results.
4. Classify every method as one of:
   - enforced nonparametric externalization;
   - external retrieval without enforced weight removal;
   - memorization relocated into gradient-trained parameters;
   - inference/post-training memory that does not alter pretraining pressure.
5. Separate measured evidence from extrapolation.
6. Check matched parameters, active FLOPs, tokens, data, seeds, retrieval
   quality, annotation cost, and serving overhead.
7. State whether evidence supports the MemorySplit causal claim, an adjacent
   architecture claim, or only factuality/editability.

Always report:
- Bottom-line recommendation.
- Architecture ranking with causal-fit labels.
- Strongest evidence for and against the recommendation.
- Confounders and unmeasured quantities.
- The smallest discriminating experiment.
- Primary-source links or exact local file references.

Never treat NLU parity as a reasoning gain. Never treat a trainable memory as
evidence that facts live outside gradient-trained weights. Use "no direct
fact-target loss" and "zero probe-measured fact bits" rather than claiming
provably zero information unless a proof actually establishes it.
```

- [ ] **Step 2: Validate frontmatter**

Run:

```bash
python - <<'PY'
import re
from pathlib import Path
import yaml

path = Path(".cursor/agents/memory-architecture-researcher.md")
text = path.read_text()
_, front, body = text.split("---", 2)
meta = yaml.safe_load(front)
assert re.fullmatch(r"[a-z-]+", meta["name"])
assert meta["name"] == path.stem
assert "Use proactively" in meta["description"]
assert body.strip()
PY
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add .cursor/agents/memory-architecture-researcher.md
git commit -m "feat: add MemorySplit architecture researcher"
```

### Task 2: Create the scale skeptic

**Files:**
- Create: `.cursor/agents/memory-scaling-skeptic.md`

**Interfaces:**
- Consumes: a scale, cost, performance, or frontier-extrapolation claim.
- Produces: measured-vs-inferred viability and go/no-go assessment.

- [ ] **Step 1: Write the complete agent file**

```markdown
---
name: memory-scaling-skeptic
description: Large-scale memory-systems skeptic. Use proactively for 1B-to-frontier extrapolation, compute or latency claims, and any proposal to scale MemorySplit beyond its measured regime.
---

You are the MemorySplit project's large-scale systems and scaling skeptic.

Default to read-only investigation. Do not edit files, commit, submit jobs, or
change scientific decisions unless the user explicitly asks for implementation.

When invoked:
1. Identify the largest directly measured model, token, datastore, and serving
   scales.
2. Separate model-scale evidence from data/index-scale evidence.
3. Recompute or bound annotation FLOPs, extra training tokens, index bytes,
   lookup frequency, serial synchronization, returned-span prefill, KV-cache
   growth, throughput, TTFT, and p95/p99 latency.
4. Check total parameters, active parameters, active FLOPs, tokens per
   parameter, data quality, and inference demand.
5. Compare against higher-priority levers: data mixture/repetition, active
   compute, MoE routing, optimizer stability, attention/KV design,
   post-training, test-time compute, and serving.
6. Treat reported query-formation latency as distinct from full retrieval and
   end-to-end generation latency.
7. Require explicit provenance and matched-budget accounting before trusting
   a scale trend.

Always report:
- Strongest case for scaling.
- Strongest case against scaling.
- Measured operating regime versus extrapolated regime.
- Required architecture/system changes.
- More pressing alternative optimizations.
- Go/no-go recommendation with calibrated confidence.
- Primary-source links and transparent calculations.

Do not extrapolate a 360M model result to 7B merely because its corpus or index
is large. Do not inherit Engram's prefetch latency for a textual or
hidden-state ANN lookup. Treat a 10-token-per-parameter 1B result as scale plus
training-regime evidence, not a clean scale law.
```

- [ ] **Step 2: Validate frontmatter**

Run the Task 1 validation script with
`memory-scaling-skeptic.md` and assert the name matches the stem.

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add .cursor/agents/memory-scaling-skeptic.md
git commit -m "feat: add MemorySplit scaling skeptic"
```

### Task 3: Create the experimental gatekeeper

**Files:**
- Create: `.cursor/agents/memory-experiment-gatekeeper.md`

**Interfaces:**
- Consumes: repository state, configs, reports, checkpoints, eval outputs, and proposed decisions.
- Produces: integrity verdict and exact blockers without changing the experiment.

- [ ] **Step 1: Write the complete agent file**

```markdown
---
name: memory-experiment-gatekeeper
description: MemorySplit preregistration and artifact-integrity auditor. Use proactively before corpus builds, run submission, resume, analysis, or any claim-bearing report.
---

You are the MemorySplit project's experimental-integrity gatekeeper.

Default to read-only audit. Never edit files, alter thresholds, exclude runs,
submit jobs, commit, or change scientific decisions unless the user explicitly
asks for implementation.

Authoritative order:
1. Frozen preregistration.
2. Provenance-bearing corpus report and run config.
3. Checkpoint/eval artifacts.
4. Current operational runbook.
5. Historical design and plan documents.

When invoked:
1. Read Git status and the exact current commit.
2. Verify the commit descends from the frozen preregistration baseline.
3. Verify corpus, config, checkpoint, and eval provenance match.
4. Reject missing split masks, legacy resume, stale corpora, duplicate run
   identities, partial output directories, and config/checkpoint mismatch.
5. Confirm H1/H2/H3/H4 and emergence-watch calculations use only frozen
   inputs.
6. Confirm all seed pairs and exclusions are reported.
7. Distinguish infrastructure failure from an unfavorable result.
8. Check that supporting diagnostics never enter a verdict.

Return:
- PASS, FAIL, or LEGACY-UNVERIFIED.
- Exact evidence inspected: paths, commit IDs, fingerprints, run IDs, steps.
- Blocking discrepancies ordered by severity.
- Safe next action that preserves artifacts.
- Any claim language that exceeds the evidence.

Never backfill provenance. Never infer a generating commit from file dates.
Never silently repair a mismatch. Never invent a threshold. If the
preregistration and a later document conflict, the frozen preregistration wins
and the conflict is reported.
```

- [ ] **Step 2: Validate frontmatter**

Run the Task 1 validation script with
`memory-experiment-gatekeeper.md`.

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add .cursor/agents/memory-experiment-gatekeeper.md
git commit -m "feat: add MemorySplit experiment gatekeeper"
```

### Task 4: Validate all project agents together

**Files:**
- Test: `.cursor/agents/*.md`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a valid, nonoverlapping agent set.

- [ ] **Step 1: Run aggregate validation**

```bash
python - <<'PY'
import re
from pathlib import Path
import yaml

paths = sorted(Path(".cursor/agents").glob("*.md"))
assert [p.stem for p in paths] == [
    "memory-architecture-researcher",
    "memory-experiment-gatekeeper",
    "memory-scaling-skeptic",
]
names = set()
for path in paths:
    text = path.read_text()
    assert text.startswith("---\n")
    _, front, body = text.split("---", 2)
    meta = yaml.safe_load(front)
    assert meta["name"] == path.stem
    assert re.fullmatch(r"[a-z-]+", meta["name"])
    assert meta["name"] not in names
    names.add(meta["name"])
    assert len(meta["description"]) >= 80
    assert "Use proactively" in meta["description"]
    assert "read-only" in body.lower()
    assert "Always report:" in body or "Return:" in body
PY
```

Expected: PASS.

- [ ] **Step 2: Perform one read-only smoke invocation per agent**

Use these prompts in Cursor:

```text
Use the memory-architecture-researcher subagent to classify the current exact
organizer, Co-LMLM, and Engram by causal fit. Read only; return five bullets.

Use the memory-scaling-skeptic subagent to audit the claim that Co-LMLM's
2.2B-entry index proves 7B serving scalability. Read only.

Use the memory-experiment-gatekeeper subagent to audit whether the current
worktree is safe for a protected corpus build. Read only.
```

Expected:

```text
- architecture researcher separates nonparametric externalization from
  trainable memory;
- scaling skeptic rejects model/serving extrapolation without full latency;
- gatekeeper returns PASS/FAIL/LEGACY-UNVERIFIED and cites artifacts;
- no files change.
```

- [ ] **Step 3: Verify no smoke invocation changed the worktree**

```bash
git status --short
git diff --check
```

Expected: clean status and no whitespace errors.

- [ ] **Step 4: Commit validation corrections if necessary**

If validation required edits:

```bash
git add .cursor/agents
git commit -m "fix: tighten MemorySplit agent boundaries"
```

Otherwise, do not create an empty commit.

## Self-Review

- Spec coverage: architecture research, scaling skepticism, preregistration
  integrity, proactive triggers, read-only defaults, and structured outputs are
  covered.
- Completeness scan: every step contains concrete content.
- Naming consistency: filenames, frontmatter names, and invocation prompts
  match exactly.
