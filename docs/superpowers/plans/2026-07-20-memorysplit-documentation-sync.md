# MemorySplit Documentation Synchronization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every current-facing MemorySplit document agree with the third Gate-A result and the frozen fact-use-QA preregistration while preserving dated history.

**Architecture:** Treat `docs/superpowers/specs/2026-07-20-preregistration.md` at commit `b805919` as authoritative for the battery. Add explicit supersession notices instead of rewriting historical plans or research dossiers, and keep operational instructions separate from historical incident notes.

**Tech Stack:** Markdown, Git, Python 3.12 standard library, ripgrep.

## Global Constraints

- The frozen preregistration is authoritative for endpoints, thresholds, corpus recipe, and battery.
- H1 is fact-use QA; iGSM/deduction are emergence-watch secondaries.
- The H1 margin is `max(2 × sigma_pool, 1.0 point)`, not 0.5 point.
- H3 is split store-OFF recall `< 5%` and split fact bits `< 10%` of dense.
- Preserve round-one, round-two, and round-three pilot results as dated evidence.
- Do not edit the three `docs/superpowers/research/2026-07-17-memory-split/wave*.md` files.
- Do not backfill provenance into legacy artifacts or claim their generating commit.
- Do not change Python or shell behavior in this plan.
- Every current status statement must cite a file, run ID, or commit.

---

### Task 1: Reconcile the two design specifications with the freeze

**Files:**
- Modify: `docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md:1-176,363-474`
- Modify: `docs/superpowers/specs/2026-07-17-memory-split-design.md:1-39,86-151,263-326`
- Read only: `docs/superpowers/specs/2026-07-20-preregistration.md`

**Interfaces:**
- Consumes: frozen endpoint and battery definitions from `2026-07-20-preregistration.md`.
- Produces: one current architecture decision and one clearly historical original design.

- [ ] **Step 1: Write a failing documentation-state check**

Run:

```bash
python - <<'PY'
from pathlib import Path

root = Path("docs/superpowers/specs")
architecture = (root / "2026-07-20-memory-architecture-scale-decision-design.md").read_text()
original = (root / "2026-07-17-memory-split-design.md").read_text()

assert "written-spec review pending" not in architecture
assert "mark preregistration as pending" not in architecture
assert "max(2 × pooled seed sigma, 0.5 point)" not in architecture
assert "FROZEN 2026-07-20" in architecture
assert "fact-use QA" in architecture
assert "2026-07-20-preregistration.md" in original
assert "Frozen amendment" in original
PY
```

Expected: FAIL on the first assertion because the architecture spec still says review is pending.

- [ ] **Step 2: Add the post-approval/freeze amendment to the architecture decision**

Change the status block to:

```markdown
Status: approved by the owner 2026-07-20

> **Battery amendment (frozen later on 2026-07-20):** Gate-A round three
> followed the "neither task exceeds 90%" branch below. The owner froze
> fact-use QA as H1, with iGSM/deduction retained only as emergence-watch
> secondaries. For any battery detail, threshold, or endpoint,
> `docs/superpowers/specs/2026-07-20-preregistration.md` supersedes this
> document.
```

Update the current-program sections so they state:

```markdown
- Gate A failed in all three pilot rounds.
- H1 is now split-versus-dense fact-use QA.
- The H1 margin floor is 1.0 point.
- H3 uses the frozen `<5%` and `<10% of dense` thresholds.
- The preregistration is complete; the next scientific step is rebuilding
  all battery corpora on the frozen recipe.
- Soft iGSM/deduction diagnostics support the emergence watch and never
  replace H1.
```

Retain the approved future hybrid architecture and large-scale conclusions.
Label the old per-task 90% branch as the pre-round-three decision framework,
resolved by branch three.

- [ ] **Step 3: Add a frozen-amendment ledger to the original design**

Insert after the status paragraph:

```markdown
## Frozen amendment ledger

- **2026-07-18 — scale plan:** 1B became the confirmation tier; 410M became
  optional.
- **2026-07-19 — first Gate-A remediation:** mixture changed to
  54/23/12/8/3 and iGSM narrowed to op 2-6.
- **2026-07-20 — difficulty floor:** iGSM changed to op 1-4 with `1/op`
  training mass and reduced easy distractors; deduction changed to depth
  1-2 with small bases.
- **2026-07-20 — endpoint freeze:** after all three Gate-A rounds remained
  at chance, fact-use QA became H1 and knowledge-free reasoning became an
  emergence-watch secondary. The authoritative definition is
  `docs/superpowers/specs/2026-07-20-preregistration.md`.
```

Update current-facing summaries, mixture, schedule, risks, and deliverable
language. Keep original endpoint prose under its existing supersession marker.
Replace the stale preregistration path `2026-07-22-preregistration.md` with the
actual frozen file.

- [ ] **Step 4: Run the documentation-state check**

Run the Step 1 command again.

Expected: PASS with no output.

- [ ] **Step 5: Verify no frozen number was changed**

Run:

```bash
rg -n "1\.0 point|store-OFF recall < 5%|10% of.*dense|iGSM >= 13%|deduction >= 65%" \
  docs/superpowers/specs/2026-07-20-preregistration.md \
  docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md
git diff --check
```

Expected: both files expose the frozen values; `git diff --check` exits 0.

- [ ] **Step 6: Commit**

```bash
git add \
  docs/superpowers/specs/2026-07-17-memory-split-design.md \
  docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md
git commit -m "docs: reconcile MemorySplit designs with the freeze"
```

### Task 2: Update the entry documents and dated report

**Files:**
- Modify: `README.md:1-12,42-61`
- Modify: `docs/README-SHARE.md:7-38,55-58`
- Modify: `docs/superpowers/2026-07-20-interim-report.md:3-8,12-45,233-257,296-363`
- Read only: `outputs/cluster-summaries/d160m_{dense,split}_n200k_s0_gate.v3.summary.json`

**Interfaces:**
- Consumes: current status from the preregistration and checked-in v3 summaries.
- Produces: accurate project entry points without erasing historical rounds.

- [ ] **Step 1: Capture the v3 numbers used by current-facing prose**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/cluster-summaries")
dense = json.loads((root / "d160m_dense_n200k_s0_gate.v3.summary.json").read_text())
split = json.loads((root / "d160m_split_n200k_s0_gate.v3.summary.json").read_text())

assert round(dense["igsm"]["acc"] * 100, 1) == 3.2
assert round(split["igsm"]["acc"] * 100, 1) == 5.6
assert round(dense["deduction"]["acc"] * 100, 1) == 53.7
assert round(split["deduction"]["acc"] * 100, 1) == 53.1
assert round(dense["factqa"]["acc"] * 100, 1) == 31.2
assert round(split["factqa"]["acc"] * 100, 1) == 58.1
PY
```

Expected: PASS.

- [ ] **Step 2: Replace the README pitch and status**

Use:

```markdown
Does loss-masking externally supplied facts change what a small language
model learns at fixed decoder parameters and raw-token budget? MemorySplit
runs a 160M fact-load sweep and a two-seed-pair 1B confirmation. After three
pilot curricula left knowledge-free symbolic tasks at chance, the frozen H1
is fact-use QA; iGSM and deduction remain emergence-watch secondaries.
```

Add links to:

```markdown
- Frozen preregistration
- Architecture and scale decision
- Original design (historical, amended)
- Original implementation plan (historical)
```

Correct the smoke description to the behavior actually asserted by
`scripts/smoke_test.py`; do not claim 500 entities or a dense bio-specific
loss assertion unless the script contains those checks.

- [ ] **Step 3: Rewrite the share package's current status**

The read order must place the frozen preregistration immediately after the
interim report. Replace "failed twice / decision pending" with:

```markdown
- Gate A failed in three 0.8B-token pilot rounds, including the op-1-4 /
  depth-1-2 difficulty floor.
- Latest checked-in v3 pair: iGSM 3.2% dense / 5.6% split; deduction 53.7% /
  53.1%; fact-use QA 31.2% / 58.1%.
- H1 is frozen on fact-use QA. Knowledge-free reasoning remains an
  emergence-watch secondary under the preregistered trigger.
```

Change the result-file section to `outputs/cluster-summaries/`. Label
unversioned summaries as earlier rounds and `.v3` summaries as round three.

- [ ] **Step 4: Add a post-report freeze block to the interim report**

Near the top, add:

```markdown
> **Post-report update, 2026-07-20:** Gate-A round three also remained at
> chance. The owner froze fact-use QA as H1 and retained iGSM/deduction as
> emergence-watch secondaries in
> `docs/superpowers/specs/2026-07-20-preregistration.md`. All earlier battery
> corpora are void and must be rebuilt on the frozen recipe.
```

Preserve the round-one and round-two sections. Label the headline fact-use
number with its round/run; use the checked-in v3 pair for current status.
Replace the stale "retry in flight" heading and schedule sequence with the
completed freeze followed by rebuild → sweep → calibration → confirmation.

- [ ] **Step 5: Check current-facing prose against the artifacts**

Run:

```bash
rg -n "failed twice|Decision pending|retry in flight|2026-07-22-preregistration|160M/410M scale with" \
  README.md docs/README-SHARE.md docs/superpowers/2026-07-20-interim-report.md
```

Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/README-SHARE.md docs/superpowers/2026-07-20-interim-report.md
git commit -m "docs: publish the frozen MemorySplit status"
```

### Task 3: Supersede the historical plan and make the runbook operational

**Files:**
- Modify: `docs/superpowers/plans/2026-07-17-memory-split.md:1-17,314-340`
- Modify: `cluster/RUNBOOK.md:1-134`
- Read only: `docs/superpowers/specs/2026-07-20-preregistration.md`

**Interfaces:**
- Consumes: frozen battery order and current cluster constraints.
- Produces: a historical implementation record plus a safe current runbook.

- [ ] **Step 1: Add a supersession banner to the historical plan**

Insert after the title:

```markdown
> **Historical implementation plan.** The system described below was built,
> but its mixture, Gate-A path, endpoint, preregistration path, and battery
> counts were superseded on 2026-07-20. Current scientific choices are frozen
> in `docs/superpowers/specs/2026-07-20-preregistration.md`; current operations
> are in `cluster/RUNBOOK.md`. The original task body is retained for
> provenance and must not be executed literally.
```

Do not rewrite the old task body.

- [ ] **Step 2: Separate the dated incident history from current instructions**

Move the July 17–18 network and node history under:

```markdown
## Dated bring-up incidents
```

Start the runbook with:

```markdown
Status: preregistration frozen at commit `b805919`; no battery run may use
pre-freeze corpora. Before submission, rebuild sweep and 1B corpora and use
new output directories. Until provenance enforcement lands, archive or rename
every legacy run directory before invoking `--resume auto`.
```

- [ ] **Step 3: Replace the gate workflow with the frozen battery workflow**

Document exact order:

```text
1. Warm SSH control socket and sync a clean descendant of b805919.
2. Rebuild full sweep corpora: n50k, n200k, n800k.
3. Rebuild full1b corpora: n800k, n4m.
4. Verify each report.json records 54/23/12/8/3, op 1-4, depth 1-2.
5. Generate and submit the 12-run sweep.
6. Evaluate snapshots and finals.
7. Generate and submit calib1b.
8. Select the lower-recall load, tie -> n4m.
9. Generate and submit four chained confirmation runs.
```

Mark `cluster/stage.sh` as a historical bring-up helper that must not be used
for the frozen battery until the provenance plan changes its sequencing.

- [ ] **Step 4: Add a manual pre-provenance safety checklist**

Include commands:

```bash
git status --porcelain
git merge-base --is-ancestor b805919 HEAD
rg -n '"igsm_op": \[1, 4\]|"deduction_depth": \[1, 2\]' \
  "$FS_DATA"/n*/report.json
```

Require the operator to move old `outputs/d160m_*` and `outputs/d1b_*`
directories to an explicitly named legacy archive before submission. State
that this manual step is replaced by the artifact-provenance plan.

- [ ] **Step 5: Verify historical/current boundaries**

Run:

```bash
rg -n "Historical implementation plan|FROZEN|b805919|stage\.sh.*historical|legacy" \
  docs/superpowers/plans/2026-07-17-memory-split.md cluster/RUNBOOK.md
rg -n "2026-07-22-preregistration|op 2-6" cluster/RUNBOOK.md
git diff --check
```

Expected: the first command finds all safety markers; the second finds no
matches; `git diff --check` exits 0.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/plans/2026-07-17-memory-split.md cluster/RUNBOOK.md
git commit -m "docs: align the runbook with the frozen battery"
```

### Task 4: Validate the complete documentation set

**Files:**
- Test: all modified Markdown files

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: a clean, cross-linked documentation state.

- [ ] **Step 1: Scan for stale current-facing claims**

Run:

```bash
rg -n "preregistration pending|Decision pending|retry in flight|failed twice|2026-07-22-preregistration|op 2-6" \
  README.md docs/README-SHARE.md cluster/RUNBOOK.md \
  docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md
```

Expected: no matches. Historical documents may contain old terms only inside
explicitly labeled retained history.

- [ ] **Step 2: Validate local Markdown links**

Run:

```bash
python - <<'PY'
import re
from pathlib import Path

files = [
    Path("README.md"),
    Path("docs/README-SHARE.md"),
    Path("cluster/RUNBOOK.md"),
    Path("docs/superpowers/2026-07-20-interim-report.md"),
    Path("docs/superpowers/specs/2026-07-17-memory-split-design.md"),
    Path("docs/superpowers/specs/2026-07-20-memory-architecture-scale-decision-design.md"),
    Path("docs/superpowers/plans/2026-07-17-memory-split.md"),
]
missing = []
for file in files:
    text = file.read_text()
    for target in re.findall(r"`((?:docs|cluster)/[^` ]+\.md)`", text):
        if not Path(target).exists():
            missing.append((str(file), target))
assert not missing, missing
PY
```

Expected: PASS.

- [ ] **Step 3: Review final diff and whitespace**

Run:

```bash
git diff --check
git status --short
git log --oneline -4
```

Expected: no whitespace errors; only intended documentation changes remain.

- [ ] **Step 4: Commit any validation-only corrections**

If Step 3 required corrections:

```bash
git add README.md docs cluster/RUNBOOK.md
git commit -m "docs: finish MemorySplit documentation synchronization"
```

Otherwise, do not create an empty commit.

## Self-Review

- Spec coverage: current status, third Gate-A result, frozen H1/H2/H3/H4,
  stale corpora, historical preservation, and runbook safety are covered.
- Completeness scan: every step contains concrete content.
- Boundary check: no Python or shell behavior changes are included.
