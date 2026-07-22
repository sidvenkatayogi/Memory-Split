# Core Measurement Cleanup Design

**Date:** 2026-07-21  
**Status:** approved in session  
**Scope:** the original dense-versus-split MemorySplit hypothesis

## 1. Scientific claim

The experiment tests whether moving arbitrary fact values out of model weights
allows a matched language model to learn reusable reasoning procedures better
under increasing fact load.

The comparison remains a paired dense-versus-split intervention:

- same architecture, parameter count, initialization, token budget, corpus
  components, and underlying-record schedule;
- dense fact values receive ordinary language-model loss;
- split fact values are loss-masked and available through the organizer; and
- fact load varies while the reasoning distribution stays fixed.

Pairing must be enforced by an explicit shared record schedule; recomputing
component order independently from each arm's token deficits is not acceptable.
Training seeds must control and record both initialization and data order.

The cleaned experiment does not use organizer-assisted versus closed-book
accuracy as evidence for improved reasoning. Both arms receive identical
information for the claim-bearing reasoning evaluation. Organizer-assisted and
store-off runs are manipulation checks only.

## 2. Retained measurement contract

### 2.1 Primary endpoint

Use equal-information two-fact inference. At evaluation, both arms receive the
same exact gold fact text. Report these strata separately:

1. date ordering; and
2. balanced equality.

Every item has a counterfactual twin in which one fact changes and the correct
answer flips. Claim-bearing scoring requires both exact answer accuracy and
counterfactual-pair consistency. Extraction is not a reasoning endpoint, and
the old aggregate fact-use score is removed.

No aggregate headline is formed unless both inference strata clear a
predeclared above-floor validity gate.

### 2.2 Conditional secondary endpoints

iGSM and deduction remain only if a blinded pilot demonstrates that at least
one arm is measurably above chance at the planned budget. Otherwise they are
removed before the protected battery and retained only in the historical note.

### 2.3 Required intervention guardrails

Retain:

- exact trained-fact recall;
- counterbalanced candidate recognition to detect knowledge hidden by
  generation-format failure;
- split store-off leakage;
- per-item lookup traces: exact key, correct referent, hit, returned value,
  malformed call, and extra call;
- masked-value and unmasked-restatement audits; and
- held-out shared-text bits per byte with a predeclared non-inferiority margin.

Dense fact memory must be demonstrably above chance; otherwise no memorization
burden was established. Split store-off claims use confidence bounds rather
than point estimates.

Remove the recall-derived quantity previously called “bits in weights.” It is
not an information estimate with defensible uncertainty.

## 3. Statistical contract

- Use at least three paired training seeds per claim-bearing condition.
- Training-seed paired deltas carry inference; item bootstraps describe only
  evaluation-set uncertainty.
- Report date and equality strata separately with balanced accuracy where
  applicable.
- Estimate a fact-load interaction only when every retained load has at least
  three paired seeds.
- Describe the dose axis as a load/exposure interaction because increasing the
  number of entities also reduces exposures per entity.
- Freeze thresholds, exclusions, and analysis code before reading protected
  battery results.

## 4. Files and measurement families to retain

Retain the minimal core:

- synthetic record, biography, reasoning, and redesigned equal-information
  inference generation;
- organizer and matched training machinery;
- exact generation, recall/recognition, leakage, shared-text, and paired
  statistical analysis;
- provenance manifests, non-overwriting checkpoint outputs, and expected-count
  validation;
- focused tests for generation, masking, lookup interception, counterfactual
  pairs, exact scoring, pairing, statistics, and artifact provenance; and
- one concise historical note covering the three gate rounds and why their
  measurements were not decision-grade.

Sound infrastructure may be transplanted from
`feat/memorysplit-freeze-hardening`: seed-level statistics, provenance checks,
OOD/reference integrity, and non-overwriting outputs. Its current
organizer-versus-closed-book endpoint is not retained.

## 5. Measurement families to delete

Delete each family atomically with its code, tests, scripts, documents, and
outputs.

### 5.1 Mechanistic probe family

- `probe/`
- `scripts/probe_local.py`
- `evals/continuous.py`
- `evals/interp.py`
- `evals/mechanism.py`
- `evals/reasoning_probe.py`
- probe reports, copied checkpoints, and probe outputs

This removes attention ablations with an invalid scorer, floor-effect unit
ablations, CKA, effective rank, participation ratio, gradient attribution, and
the unsupported fact-information ledger.

### 5.2 Keyguess and real-fact addressing family

- `corpusgen/realfact.py` and real-fact snapshots;
- `evals/keyguess.py` and `evals/constrain.py`;
- keyguess fetch, orchestration, and analysis scripts;
- keyguess/realfact/constrain tests, Slurm jobs, role documents, reports, and
  outputs.

This is a separate addressing study and does not measure the original
capacity-reallocation hypothesis.

### 5.3 Low-quality or peripheral measurements

Remove:

- the aggregate organizer-assisted fact-use endpoint and extraction score;
- the recall-derived “bits in weights” estimator;
- the natural-benchmark evaluator, tests, and benchmark-only dependencies;
- H4 milestone/sample-efficiency analysis and broken curve/diagnostic runners;
- the exploratory overtraining tier;
- Entity-First and product-architecture material;
- preliminary `paper/` artifacts;
- Canvas/project-agent plans and other peripheral experiment plans; and
- `data/smoke/`, smoke logs, and `outputs/cluster-summaries/` after the
  historical note is verified.

## 6. Legacy-result handling

Before deleting raw pilot outputs, write one concise historical note containing:

- the three Gate-A rounds and their floor/answer-prior failures;
- the fact-use subtype breakdown;
- enough run identifiers and dates to trace the corresponding Git history;
- an explicit `legacy-unverified` label for results lacking full configuration,
  corpus, and revision provenance; and
- a statement that none of these values carries the new decision.

Git history remains the provenance record. No duplicate raw archive remains in
the active repository.

## 7. Execution order

1. Add tests for the replacement endpoint, manipulation checks, pairing,
   expected counts, and provenance.
2. Implement the retained measurement contract and compact analysis path.
3. Replace README, design, preregistration, runbook, and analysis documentation
   with one consistent experiment definition.
4. Write and verify the historical note.
5. Delete side-study families and remove shared imports or command branches
   surgically.
6. Delete raw smoke/gate/probe/keyguess outputs.
7. Run focused tests, the full offline suite, and a toy end-to-end smoke.
8. Search tracked, untracked, ignored, Canvas, and plan files for deleted
   concepts and dangling references.

If a low-quality feature shares a core module, remove only its code path rather
than deleting the shared file.

Do not remove `feat/memorysplit-freeze-hardening` or its worktree until the
retained statistics, provenance, and output-integrity changes have been
transplanted and independently tested.

## 8. Verification

Completion requires:

- all focused measurement tests passing;
- the full offline unit suite passing;
- a toy paired run producing the new per-stratum and counterfactual metrics;
- no batch-composition dependence in generation scoring;
- expected evaluation counts enforced;
- three-seed statistics tested on synthetic fixtures;
- no imports, commands, files, ignored outputs, canvases, or plans from deleted
  measurement families; and
- documentation and executable analysis agreeing on the same primary endpoint,
  guardrails, and decision rule.

## 9. Non-goals

This cleanup does not claim:

- that existing pilots establish improved reasoning;
- that organizer access itself demonstrates freed capacity;
- that mechanistic probes identify where capacity was reused;
- that natural-benchmark deltas carry the result; or
- that the experiment validates a practical memory architecture.
