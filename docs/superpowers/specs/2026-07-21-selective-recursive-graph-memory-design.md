# Relational MemorySplit — Vanilla Transformer Design

**Date:** 2026-07-22
**Status:** revised design approved in session; written-spec review pending
**Scope:** a simple controlled Dense-versus-Split experiment, not a production
retrieval architecture

## 1. Scientific question

The experiment tests one operational hypothesis:

> At fixed standard-Transformer parameters, raw training tokens,
> initialization, record order, and inference steps, removing direct
> next-token loss from arbitrary relational facts improves acquisition of
> reusable relational reasoning procedures.

The claim-bearing models are ordinary decoder-only Transformers. The external
graph, route labels, and fixed reasoning traces are data/evaluation interfaces,
not trainable neural modules.

The hypothesis predicts:

1. Split outperforms Dense on equal-information fresh-graph reasoning.
2. The Split advantage is larger at high arbitrary-fact load than low load.
3. Selective fact masking outperforms an equal-mass random-target mask.
4. Split still performs the factual job when memory is attached.

The design may reject this hypothesis in the tested 160M/360M regime. It does
not claim to isolate bit-capacity pressure from optimization interference; the
fact-dose axis remains a load/exposure interaction.

## 2. Why the design is simpler

The superseded design added a recurrent ponder module, a selector MLP, custom
cache semantics, and a separate trainer. Those changes made a result hard to
attribute to fact offloading.

The revised design removes all four:

- no recurrent neural blocks;
- no learned selector parameters;
- no graph encoder, GNN, query adapter, or semantic retriever; and
- no separate model family.

Dense, Split, and random-mask conditions instantiate the same `train.model.GPT`
class. The only claim-bearing training difference is the per-token target
weight.

The already measured exact-key failure still motivates pointer-like graph
actions: the prior real-fact study recovered the relation half at 98.5% but the
entity-name half at 2.5%. Initial entities are therefore represented by fixed
working slots linked deterministically to question mentions; the model never
generates an entity name as an address.

## 3. Standard Transformer and recursive graph trace

### 3.1 Matched model

Every condition uses the existing pre-norm decoder:

- RMSNorm, RoPE, causal attention, and SwiGLU;
- no bias or dropout;
- untied input/output embeddings;
- vocabulary size 50,304;
- context length 1,024; and
- identical initialization and data-order seeds within each paired condition.

The protected scales are:

- **29M development:** existing 4-layer, width-256 toy GPT
  (28,969,216 parameters);
- **160M:** existing 12-layer, width-768 GPT
  (162,220,800 parameters); and
- **360M:** 20 layers, width 1,024, 16 heads
  (356,033,536 parameters).

No graph-related trainable parameters are added. Graph-control tokens use
already padded vocabulary IDs 50,261–50,291, so the embedding matrices remain
the same size in every arm.

### 3.2 Atomic graph memory

Memory is a non-trainable directed typed graph. Each row is:

`(source_id, relation_id, direction, target_kind, target, qualifiers, provenance_id)`.

For protected data, every `(source_id, relation_id, direction)` address
resolves to exactly one row or `MISS`. One read returns one row. The store
never:

- expands a neighborhood;
- ranks candidates;
- infers or verifies a path;
- composes relations; or
- returns a final answer.

The graph is immutable within a run and is reported separately in rows and
bytes.

### 3.3 Six-step autoregressive recursion

Relational reasoning is represented as six fixed text-generation steps. At
each step the standard Transformer predicts:

`GRAPH_START, source_slot, relation, direction, READ|NOOP|HALT, GRAPH_END`.

If it predicts `READ`, the exact atomic row is appended to the context. The
next action therefore conditions on all previous actions and returned facts.
This is the Tiny-Recursive-Model inspiration: repeated state refinement and a
fixed compute budget, implemented entirely through ordinary autoregressive
context rather than new recurrent weights.

`HALT` is followed by no-op steps so every example still contains six action
slots and six provisional answer slots. Dense and Split receive identical
action and answer supervision.

During training, gold actions and graph returns are teacher-forced in every
condition. During evaluation, actions are model-generated and reads are
resolved by the selected memory mode.

## 4. Selective internal-versus-external policy

There is no selector network. A frozen cost rule assigns each fact:

\[
C_{\mathrm{predict},i}
=
\frac{H_i}{\max(E_i,1)}
\]

and:

\[
C_{\mathrm{external},i}
=
c_w + 0.25Q_i + 0.25D_i,
\]

where:

- `H_i` is known payload entropy in bits;
- `E_i` is scheduled exposure count;
- `Q_i` is expected use count in generated reasoning questions; and
- `D_i` is expected graph-hop contribution.

Fact `i` is external when
`C_predict,i > C_external,i`.

Calibrate only `c_w` on disjoint development worlds over
`{0.25, 0.5, 1, 2, 4, 8}`. Choose the value whose route rate is closest to
50% while remaining inside 40–60%; ties choose the smaller `c_w`. Freeze the
value and route-policy hash before protected corpus generation.

No semantic class label is an input. The expected behavior follows from the
costs:

- rare high-entropy leaves are expensive to predict and cheap to read;
- repeated low-entropy rules are cheap to predict;
- central facts are expensive to read repeatedly.

The Transformer learns when to emit `READ` versus `NOOP` through ordinary
next-token loss on the cost-derived route traces. Held-out combinations of
entropy, exposure, and centrality test whether this routing policy
generalizes.

## 5. Training streams and loss

### 5.1 One shared stream

One underlying schedule fixes:

- component;
- record ID;
- paraphrase;
- graph actions and returns;
- curriculum position; and
- packing boundary.

All conditions consume the same token IDs in the same order. Three sidecars
provide target weights:

- **Dense:** all target weights are one.
- **Split:** weights are zero only on every direct occurrence of
  cost-routed external payloads.
- **Random-mask:** factual payloads retain loss; an equal token mass of
  matched non-factual spans receives zero weight.

Random-control spans are matched by token length and relative document
position and exclude rules, actions, provisional answers, and final answers.

### 5.2 All-position normalization

The existing GPT forward path gains an optional target-weight argument but no
new parameters:

\[
\mathcal L
=
\frac{1}{T}
\sum_{t=1}^{T}
w_t[-\log p_\theta(x_t\mid x_{<t})].
\]

The denominator is always the original raw target count `T`. Masking therefore
does not increase the weight of every remaining Split target.

Returned payloads remain teacher-forced inputs to later predictions. The
defensible intervention is “no direct target loss on selected payloads,” not
“no gradient path.” Store-off recognition bounds indirect learning.

### 5.3 Corpus mixture

Every protected stream uses:

- **45% natural bed:** one pinned FineWeb-Edu JSONL snapshot;
- **30% graph exposure:** 70% peripheral facts, 20% central facts, 10%
  relation/comparison rules; and
- **25% relational reasoning:** path composition, date ordering, and
  counterfactually balanced equality.

The fixed curriculum is:

- first 20% of raw tokens: one-hop traces;
- next 30%: one- and two-hop traces; and
- final 50%: one- through four-hop traces.

## 6. Development and protected battery

### 6.1 Local smoke gate

The mandatory local gate uses a tiny deterministic corpus and two optimizer
steps. It verifies:

- atomic store save/load and misses;
- graph token IDs and fixed action grammar;
- counterfactual evidence changes the oracle answer;
- shared token-stream identity;
- Split and random sidecar alignment;
- all-position loss normalization;
- memory ON/OFF decoding;
- exact checkpoint/resume; and
- portable relative config paths.

This gate is implementation validation, not a scientific result.

### 6.2 29M learnability pilot

Train one paired Dense/Split seed for 299,892,736 raw tokens
(572 steps at 524,288 tokens/step). Both models must exceed 75%
counterfactual-consistent accuracy in each primary reasoning stratum.

Failure is an endpoint/instrument failure. It does not support or reject the
memory hypothesis.

### 6.3 Initial protected battery

Use three paired initialization/data seeds.

**160M dose:**

- low load: 50,000 entities;
- high load: 800,000 entities;
- 1,599,602,688 raw tokens per run;
- Dense and Split at both loads: 12 runs total; and
- random-mask control at high load: 3 runs.

**360M confirmation:**

- 1.8M entities;
- 3,599,761,408 raw tokens per run; and
- Dense/Split × three seeds: 6 runs.

The initial battery is **21 runs and approximately 45.6B raw tokens**.

The middle 200k load and 1B tier are removed from the initial claim. They may
be added only after the frozen initial verdict.

## 7. Evaluation and decision

### 7.1 Primary equal-information endpoint

Both twins receive the same fresh graph with memory ON. Supporting arbitrary
facts are routed external by the frozen policy; relation and comparison rules
remain internal.

Evaluate 10,000 counterfactual pairs in each stratum:

1. path composition;
2. two-branch date ordering; and
3. two-branch balanced equality.

A pair scores correct only when both the original and changed-evidence twin
are correct. Report strata separately. Form an equal-weight composite only
because the development gate requires every stratum above 75%.

Also report:

- exact action-path accuracy;
- per-hop action and referent accuracy;
- `MISS`, malformed, and excess-read rates;
- final answer accuracy after gold-path replay; and
- five/six-hop, entity-renamed, and graph-isomorphic generalization.

### 7.2 Memory factorial

Evaluate both trained twins with memory OFF and ON:

1. value loss ON, memory OFF: Dense closed-book;
2. value loss ON, memory ON: retrieval-only;
3. value loss OFF, memory OFF: mask-only; and
4. value loss OFF, memory ON: Split system.

The schema claim compares value loss OFF versus ON under the same fresh
memory-ON evidence. Other cells are manipulation checks.

### 7.3 Required guardrails

- Frozen route rate is 40–60%.
- At least 80% of low-use high-entropy facts route external.
- At least 80% of rules and top-centrality facts route internal.
- Dense four-way recognition lower 95% bound exceeds 30% (25% chance).
- Split store-OFF recognition upper 95% bound is below 30%.
- Split memory-ON exact factual recall is within two points of Dense
  memory-OFF recall.
- Split internal-rule/central-fact accuracy is within two points of Dense.
- Shared natural-text bits per byte is non-inferior within 1% relative.
- A mask ledger proves zero unmasked direct occurrences of selected external
  payloads and zero masked rule/action/answer targets.

Run shuffled-store, relevant-edge, irrelevant-edge, gold-path,
entity-renaming/isomorphism, no-query, and memory-OFF controls.

### 7.4 Frozen verdict

Let `delta_scale,seed` be Split minus Dense primary composite.

Validate only if:

1. all three 360M deltas are positive;
2. mean 360M delta exceeds
   `max(2 points, 2 × pooled seed sigma)`;
3. every 360M stratum has positive mean delta;
4. the paired difference-in-differences
   `(Split-Dense)_high - (Split-Dense)_low` has a 95% lower bound above zero;
5. Split exceeds random-mask at 160M high load by more than one point in all
   three paired seeds; and
6. every guardrail passes.

Reject a practically meaningful effect only if all gates pass and:

1. the upper 95% bound on 360M delta is below two points;
2. the upper 95% bound on the high-minus-low difference-in-differences is at
   or below zero; and
3. selective masking does not beat random masking by more than one point.

Anything between those boundaries is reported as inconclusive. A failed
instrument gate is reported as invalid, not reinterpreted.

## 8. Local, FarmShare, and AWS execution

### 8.1 Portable artifact contract

`scripts/package_relational_run.py` produces
`artifacts/relational-run.tar.gz` containing:

- source Git revision and clean-tree assertion;
- frozen route-policy JSON and SHA-256;
- relative YAML config templates;
- expected run manifests;
- tiny deterministic smoke corpus;
- evaluator and analysis code;
- offline tests; and
- an artifact manifest with SHA-256 for every member.

No config in the bundle contains a user home directory, scratch path, or
cloud-specific absolute path. Runtime paths come from `DATA_ROOT` and
`OUT_ROOT`.

Full FineWeb/graph corpora and checkpoints are not placed in the source
bundle. They are generated or staged separately and verified by hash.

### 8.2 Local commands

The Mac must pass:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/relational_smoke_test.py --device cpu
.venv/bin/python scripts/package_relational_run.py \
  --out artifacts/relational-run.tar.gz
```

Local execution covers the full offline suite, two-step training/evaluation,
memory modes, resume equivalence, manifest counts, and bundle portability.
Optional MPS runs are diagnostics only.

### 8.3 FarmShare

FarmShare carries:

- the 29M/0.3B learnability pair;
- all 15 protected 160M jobs; and
- 160M evaluation/analysis.

Use one L40S per run and the existing atomic checkpoint/resume pattern. Before
submission:

1. run a Slurm GPU smoke job;
2. verify corpus, policy, code, and evaluator hashes;
3. submit the relative manifest through a FarmShare wrapper; and
4. refuse resume on any provenance mismatch.

With four concurrent L40S slots, the 160M battery is approximately four
training waves after data staging.

FarmShare may run the six 360M jobs, but their two long waves are a schedule
risk and are not the preferred Thursday path.

### 8.4 AWS

One `p5.48xlarge` is the preferred 360M platform:

- eight H100 80GB GPUs;
- six 360M jobs run concurrently;
- each run occupies one GPU; and
- no distributed training is required.

Run 200 resumable steps first. Proceed only if:

- all eight GPUs are visible and are H100s;
- 360M mean throughput is at least 60,000 raw tokens/s/GPU after 50 warmup
  steps;
- peak allocated memory is at most 72GB/GPU;
- checkpoint/resume reproduces the next loss within `1e-5`; and
- corpus/policy/code hashes match every config.

Use On-Demand capacity or an EC2 Capacity Block, not Spot, for
deadline-critical protected runs.

The AWS and FarmShare configs are byte-identical. Only their launchers differ:

- FarmShare submits one config per Slurm job.
- AWS assigns configs to a shared eight-worker `CUDA_VISIBLE_DEVICES` queue.

If AWS is unavailable, complete the 160M verdict on FarmShare and label 360M
confirmation pending; do not reduce the seed count.

## 9. Explicit non-goals

The initial experiment excludes:

- custom recurrent/ponder blocks;
- trainable selector or router modules;
- graph neural networks or semantic retrieval;
- neighborhood/path computation in memory;
- adaptive test-time compute;
- RL routing, voting, or verifier selection;
- natural-world entity linking;
- product latency/governance claims; and
- mechanistic claims about where freed capacity resides.

These exclusions keep the comparison interpretable against a standard
Transformer baseline.

## 10. Primary sources

- LMLM: arXiv:2505.15962.
- Tiny Recursive Models: arXiv:2510.04871.
- Recurrent Relational Networks: arXiv:1711.08028.
- Differentiable Neural Computer: Nature 538 (2016).
- CLUTRR: arXiv:1908.06177.
- Internal causal standard:
  `docs/superpowers/specs/2026-07-21-core-measurement-cleanup-design.md`.
