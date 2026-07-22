# Selective Recursive Graph Memory (SRGM) Design

**Date:** 2026-07-21
**Status:** approved design; implementation plan pending written-spec review
**Scope:** a controlled Dense-versus-Split test of capacity allocation, not a
production retrieval proposal

## 1. Scientific question

The experiment tests one operational hypothesis:

> At fixed trainable parameter count, raw training tokens, initialization,
> record order, and recurrent compute, removing direct next-token loss from
> selectively externalized arbitrary relational facts improves acquisition of
> reusable relational reasoning procedures. The improvement should grow as
> arbitrary-fact load rises.

The experiment must be able to reject this hypothesis in the tested regime.
Organizer-assisted accuracy by itself is not evidence for better reasoning.
The claim-bearing evaluation therefore gives both trained twins the same fresh
graph and the same memory interface. The treatment is whether selected
fact-value targets received direct language-model loss during pretraining.

The design intentionally combines two possible sources of relief:

1. reduced parametric storage pressure; and
2. reduced optimization interference from high-surprise arbitrary targets.

It does not claim to separate those mechanisms. Because distinct-fact load is
varied at fixed fact-token share, the dose is described as a **load/exposure
interaction**, not a pure bit-capacity intervention.

## 2. Why a new architecture is needed

The previous organizer treated each fact as an isolated
`(entity, relation) -> value` row. Its real-fact held-out-key study recovered
the relation half at 98.5% but the entity-name half at 2.5%: the model learned
which relation to request but substituted memorized entity names instead of
constructing the live referent. That result is recorded in
`2026-07-20-heldout-key-generalization-results.md`.

The replacement must satisfy four constraints:

- represent connected facts as a graph and make the model choose every hop;
- avoid generated string keys by pointing to entities already present in the
  question or a prior memory result;
- learn which facts are worth internalizing without using hand-provided
  “rule” versus “arbitrary” labels; and
- give Dense and Split identical neural architecture, recursive compute,
  traversal supervision, token streams, and evidence on the reasoning
  endpoint.

The current repository is reusable infrastructure, not a valid implementation
of this specification. In particular, protected runs may not begin until the
independently assembled cross-arm record schedules are replaced by one shared
schedule and the equal-information endpoint is implemented.

## 3. Architecture design

### 3.1 Matched neural core

Both arms use the same decoder-only pre-norm Transformer family:

- RMSNorm, RoPE, causal multi-head attention, and SwiGLU;
- no biases or dropout;
- untied token input and output embeddings;
- vocabulary size 50,304 and context length 1,024; and
- graph-control symbols allocated from already padded vocabulary IDs, so they
  do not enlarge one arm relative to the other.

Two distinct Transformer blocks form a **ponder module whose weights are tied
across recursive steps**. The two blocks do not share weights with each other.
Ordinary backbone blocks run once; the complete two-block ponder module is
reused for six recursive steps. The unique trainable block counts and exact
base parameter counts are:

- **30M development class:** 4 ordinary blocks + 2 ponder blocks, width 256,
  4 heads: 30,575,872 base parameters;
- **160M class:** 10 ordinary blocks + 2 ponder blocks, width 768, 12 heads:
  162,220,800 base parameters;
- **360M class:** 18 ordinary blocks + 2 ponder blocks, width 1,024, 16 heads:
  356,033,536 base parameters; and
- **1B class:** 20 ordinary blocks + 2 ponder blocks, width 1,792, 14 heads:
  1,030,667,008 base parameters.

The frozen selector is a five-input, 32-hidden-unit MLP with 225 parameters.
It is present byte-for-byte in every arm, bringing the development, 160M,
360M, and 1B totals to 30,576,097, 162,221,025, 356,033,761, and
1,030,667,233. The external graph has no trainable parameters and is reported
separately in rows and bytes.

### 3.2 Typed atomic graph memory

The external memory is a non-trainable directed typed graph. Each row is:

`(source_id, relation_id, direction, target_id_or_literal, qualifiers, provenance_id)`.

For the protected synthetic experiment, every
`(source_id, relation_id, direction)` address is functional: it resolves to
exactly one row or `MISS`. One read returns exactly one atomic edge. The store
does not:

- expand a neighborhood;
- rank candidates;
- run message passing;
- infer a path;
- compose relations; or
- return a final answer.

This boundary is essential. Returning a neighborhood or path would move
reasoning computation into the external system and invalidate the capacity
claim.

Initial entity arguments are pointers to unambiguous spans in the live
question. A deterministic linker maps a selected span to a stable graph ID.
Targets returned by one edge may populate a working-node slot and become the
source of the next read. The model never generates an entity name as an
address.

The graph is immutable within a run. Dense and Split use byte-identical graph
snapshots. Memory availability can be toggled at evaluation without changing
weights.

### 3.3 Recursive graph controller

The controller carries:

- the encoded question `x`;
- four working-node slots;
- a latent reasoning state `z_t`;
- a provisional answer state `y_t`; and
- the previous atomic memory result `m_t`.

At each of six steps it emits:

`(source_slot, relation, direction, read_or_noop, halt_or_continue)`.

The recurrent updates are:

\[
z_{t+1}=P_\theta(x,y_t,z_t,m_t), \qquad
y_{t+1}=P_\theta(y_t,z_{t+1}),
\]

where the same two-layer `P_theta` is reused at every step. Graph actions and
the provisional answer are serialized and supervised after every step,
following Tiny Recursive Model-style iterative refinement and deep
supervision.

`HALT` changes subsequent reads to no-ops, but all examples in all arms still
execute six ponder steps. Claim-bearing runs therefore have fixed realized
recursive FLOPs. Adaptive early exit, stochastic rollouts, answer voting, and
arm-specific computation are excluded.

The hop curriculum is fixed by raw-token position:

- first 20% of training: one-hop traces;
- next 30%: one- and two-hop traces;
- final 50%: one- through four-hop traces.

No arm advances based on its measured performance.

### 3.4 Learned selective internalization

The selector is learned on graph worlds disjoint in entities, values, surface
names, and protected evaluation templates. It receives no semantic label
saying that a fact is a rule, a hub fact, or an arbitrary leaf. Relation
symbols are permuted between worlds.

For each fact it consumes five transferable statistics:

1. log exposure count;
2. payload entropy;
3. payload token length;
4. expected query count; and
5. edge path-centrality in the generated training-query distribution.

On the selector curriculum, the observed cumulative payload NLL supplies the
estimated internal cost. A hard-concrete gate minimizes:

\[
\mathcal L_{\mathrm{select}}
=
\mathbb E_i\left[
(1-g_i)C_{\mathrm{internal},i}
+g_i\left(
c_{\mathrm{write}}
+c_{\mathrm{read}}\widehat q_i
+c_{\mathrm{hop}}\widehat h_i
\right)
\right].
\]

Rare high-entropy values remain costly to predict and cheap to retrieve, while
rules and central facts become predictable and would incur repeated read
costs. The expected optimum is therefore arbitrary facts outside and reusable
structure inside.

The three cost coefficients and the deterministic threshold `g_i >= 0.5` are
calibrated once on the disjoint selector curriculum. Calibration must yield
40–60% externalization there. The selector is then frozen before any
claim-bearing model initialization or protected data generation. Protected
results may not be used to retune it.

The selector-calibration backbone is discarded. Only the 225 selector
parameters and a manifest containing its cost coefficients, feature
normalizers, threshold, training-data hash, and state-dict hash transfer into
the protected experiment. Those artifacts are frozen before protected graph
generation.

### 3.5 Training and evaluation conditions

There are two claim-bearing trained twins:

- **Dense / value-loss ON:** every factual payload target receives ordinary
  next-token loss.
- **Split / value-loss OFF:** direct target loss is removed only from all
  occurrences of payloads selected external by the frozen selector.

The serialized graph statements, lookup calls, returned payloads, provisional
answers, and final answers are otherwise identical.

During pretraining, gold graph actions and their atomic returns are
teacher-forced from the immutable graph snapshot in every condition. The
runtime may verify those returns against the store, but no arm receives an
extra online retrieval or planning operation. At evaluation, actions are
model-generated and each valid read is resolved by the selected ON/OFF memory
condition.

Each trained checkpoint is evaluated with graph memory both OFF and ON,
forming a value-loss by memory-availability factorial without adding neural
training arms:

1. value-loss ON, memory OFF: Dense closed-book;
2. value-loss ON, memory ON: retrieval-only;
3. value-loss OFF, memory OFF: mask-only; and
4. value-loss OFF, memory ON: Split system.

The reasoning claim compares value-loss OFF against value-loss ON while both
models use the same fresh evaluation graph. Memory ON/OFF contrasts are
manipulation checks.

One additional 160M high-load control is trained for three paired seeds:

- **Random-target mask:** factual payloads retain loss, while an equal token
  mass of non-factual targets in the same graph documents is masked. Spans
  are matched by length and relative document position and may not include
  rules, graph actions, provisional answers, or final answers.

This control tests whether a gain is specific to removing arbitrary-fact
targets rather than merely reducing supervised-token mass.

## 4. Training and loss modifications

### 4.1 Graph-world data

Each synthetic entity has six candidate facts:

- four functional typed edges to other entities; and
- two literal attributes, one date-like and one categorical.

Worlds also contain reusable relation-composition, inverse, equality, and
ordering rules. Generator metadata marks facts as peripheral, central, or
structural only for blinded auditing; those labels are not selector inputs.
Entity names, edge assignments, literal values, and surface templates are
seeded independently.

The reasoning families are:

1. **path composition:** follow two to four typed edges and infer the composed
   relation;
2. **two-branch date ordering:** follow two independent paths and select the
   entity whose terminal date is earlier; and
3. **two-branch balanced equality:** follow two paths and decide whether the
   terminal categorical literals match.

Every protected evaluation item has a counterfactual twin in which one
supporting edge or literal changes and the correct answer flips. Irrelevant
facts are resampled while preserving graph size and degree statistics.

### 4.2 Byte-identical rendering

One underlying record schedule is generated per load and seed. It determines
component, record ID, paraphrase, graph trace, and packing position. All
training conditions consume that exact schedule and exact token IDs.

A sidecar target-weight array is the only arm-specific training artifact:

- Dense weights all targets by one;
- Split sets selected payload targets to zero; and
- random-target control sets matched non-factual targets to zero.

The generator must tag every direct occurrence of an externalized payload,
including declarations, lookup returns, copied echoes, and repeated mentions.
Claim-bearing final answers are derived relation, pointer, comparison, or
Boolean outputs and never direct payload copies.

Returned payloads remain teacher-forced inputs to later predictions. The
defensible intervention claim is therefore “no direct next-token target loss
on selected payloads,” not “no gradient path” or “zero information in
weights.” Store-off recognition bounds indirect learning.

### 4.3 Main objective

For a packed sequence of `T` raw tokens:

\[
\mathcal L_{\mathrm{main}}
=
\frac{1}{T}
\sum_{t=1}^{T}
w_t\left[-\log p_\theta(x_t\mid x_{<t})\right].
\]

The denominator is always the original raw-token count `T`, not the number of
active targets. Thus masking does not increase the weight of every remaining
Split target. Graph actions, halt decisions, provisional answers, and derived
final answers are trained through the same LM head and require no
arm-specific auxiliary loss.

Masked positions still participate in forward and backward computation as
inputs; only their direct target terms are multiplied by zero. Optimizer
steps, learning rates, gradient accumulation, and recurrent passes are
identical within every paired condition.

### 4.4 Corpus mixture

Every protected stream has the following raw-token mixture:

- **45% natural bed:** one fixed FineWeb-Edu sample and shard order;
- **30% relational graph exposure:** 70% peripheral arbitrary facts, 20%
  high-use central facts, and 10% rules/axioms; and
- **25% relational reasoning:** 10% path composition, 7.5% date ordering,
  and 7.5% balanced equality.

The graph and reasoning shares remain fixed across loads. Structural-rule and
reasoning record distributions are fixed; only the number of distinct
arbitrary facts changes.

### 4.5 Optimizer and pairing

All scales use:

- AdamW with betas `(0.9, 0.95)`;
- weight decay 0.1 on matrix parameters only;
- gradient clipping at 1.0;
- 300 warmup steps and cosine decay to 10% of peak LR;
- peak LR 1.5e-3 at 160M, 1.0e-3 at 360M, and 6.0e-4 at 1B;
- bf16 training with TF32 matrix multiplication; and
- 524,288 raw tokens per optimizer step.

For every seed, all conditions start from the same serialized state dict and
use the same data-order RNG state. The manifest records separate
initialization and data-order seeds, but both values are matched within a
pair. Checkpoints record state-dict hash, corpus manifest hash, tokenizer
hash, graph snapshot hash, Git revision, evaluator revision, and optimizer
cursor. Resume is refused on any mismatch.

## 5. Experimental setup

### 5.1 Development gate

A paired 30M Dense/Split development run receives 299,892,736 raw tokens
(572 optimizer steps per arm) on worlds disjoint from all protected data. It
is non-claim-bearing and establishes:

- exact mask coverage and all-token loss normalization;
- byte-identical cross-condition token streams and record order;
- pointer linking and atomic read correctness;
- both development twins above 75% in all three primary strata;
- valid graph actions above 95%;
- monotonic predicted halt depth with gold path length; and
- a final-only-supervision ablation showing whether deep supervision is
  necessary for learnability.

Failure triggers implementation or data repair before protected runs. Gate
results may not choose a favorable primary endpoint.

### 5.2 Initial protected battery

The 160M sweep uses 1,599,602,688 raw tokens per run (3,051 optimizer steps):

- 50,000 entities;
- 200,000 entities; and
- 800,000 entities.

With six facts per entity, these are 0.3M, 1.2M, and 4.8M candidate facts.
The matrix is 3 loads × 2 arms × 3 paired seeds = **18 runs**.

The random-target control is added at 160M/800k for three matched seeds:
**3 runs**.

The Thursday confirmation uses the 360M model, 3,599,761,408 raw tokens per
run (6,866 optimizer steps), and 1.8M entities. The normalized high dose keeps
candidate-fact count per parameter approximately aligned with the
160M/800k condition. The matrix is 2 arms × 3 paired seeds = **6 runs**.

The initial battery is therefore 27 training runs and 55.2B nominal raw
tokens. All runs and exclusions are reported.

### 5.3 Later 1B replication

The later replication uses:

- 9,999,745,024 raw tokens per run (19,073 optimizer steps);
- 5M entities;
- two arms; and
- three paired seeds.

It tests scale robustness but does not retroactively change the frozen
160M/360M verdict.

### 5.4 AWS execution envelope

The Thursday plan assumes one AWS `p5.48xlarge`:

- eight NVIDIA H100 80GB GPUs;
- 640GB aggregate HBM3;
- 2TiB host memory; and
- eight local 3.84TB NVMe devices.

Each run occupies one H100; no distributed training is required. Corpora,
containers, and evaluation assets must be staged before billable training
begins.

Before the battery, run 200 steps at both scales, discarding the first 50
steps from throughput measurement. Proceed only if:

- 360M sustains at least 60,000 raw tokens/s/GPU over the remaining steps;
- neither scale exceeds 72GB peak allocated GPU memory;
- checkpoint/resume reproduces the next loss within 1e-5; and
- all eight GPUs can run independent jobs without dataloader starvation.

One `p5.48xlarge` is sufficient only if this gate passes and capacity is
available. P5 quota and an On-Demand instance or Capacity Block must be
confirmed before launch. Spot capacity is excluded from the
Thursday-critical battery. If the throughput or capacity gate fails,
Thursday output is preliminary and cannot carry the verdict.

## 6. Evaluation and guardrails

### 6.1 Primary equal-information endpoint

Both twins receive the same fresh graph, containing entities and arbitrary
values absent from every training and selector-calibration world. Both use
memory ON, the same deterministic linker, and the same six recursive steps.
Primary items require arbitrary supporting edges that the frozen selector
routes external; reusable relation and comparison rules remain internal.
The route decisions and graph bytes are identical for both twins.

Evaluate 10,000 counterfactual pairs in each stratum:

1. path composition;
2. two-branch date ordering; and
3. two-branch balanced equality.

The primary metric is **counterfactual-consistent pair accuracy**: a pair
receives credit only when both the original and answer-flipped twin are
correct. Report strata separately. Form an equal-weight composite only because
the development gate requires all three strata above 75% before protected
runs.

Secondary schema tests use:

- five- and six-hop paths;
- unseen compositions of known relation rules;
- larger graphs at fixed path length;
- fresh entity-name permutations; and
- graph-isomorphic relabelings and reorderings.

Also report full-path exact match, per-hop action accuracy, correct-referent
rate, valid-call rate, `MISS` rate, excess reads, and predicted halt step.

### 6.2 Selector guardrails

On a blinded audit set, the frozen selector must satisfy:

- 40–60% of all candidate facts external;
- at least 80% of bottom-quartile-use random facts external;
- at least 80% of rules and top-quartile-centrality facts internal; and
- unchanged routing under entity and relation-token renaming.

Factorial selector probes hold surface text fixed while changing exposure or
path centrality. The routing decision must follow the amortized cost change,
not the lexical form. Failure invalidates the selective-architecture
interpretation and stops protected training.

### 6.3 Memorization-burden and leakage guardrails

Use counterbalanced four-way recognition on selected arbitrary facts. In
every paired training seed, compute a Wilson interval over the fixed probe:

- chance is 25%;
- Dense’s lower 95% confidence bound must exceed 30%; and
- Split store-OFF’s upper 95% confidence bound must be below 30%.

If Dense is not above this burden gate, there is no demonstrated memorization
job to relieve. If Split exceeds the leakage bound, selected facts were not
successfully externalized.

For the factual-job guardrail:

- Split memory-ON exact recall must be no more than two points below Dense
  memory-OFF exact recall on the same selected-fact probe;
- Split memory-OFF accuracy on internal rules and central facts must be
  within two points of Dense memory-OFF; and
- shared natural-text bits per byte must be non-inferior within 1% relative.

Every run also emits a mask ledger proving:

- all direct selected-payload targets have weight zero in Split;
- no selected payload is accidentally restated as an unmasked target;
- every retained rule and central fact remains supervised; and
- random-target masks match Split’s masked token mass and span-length
  histogram within 1%.

### 6.4 Causal negative controls

Run the following interventions on the same checkpoints:

1. **Memory OFF:** separates parametric recall from external use.
2. **Shuffled store:** preserves read count and format while permuting target
   rows; relational accuracy must collapse.
3. **Relevant-edge swap:** changing one supporting edge must flip the answer.
4. **Irrelevant-edge swap:** changing a distractor edge must not change the
   answer.
5. **Gold-path replay:** both arms receive the identical correct path. A
   remaining gap localizes the effect to composition; a disappearing gap
   localizes it to navigation.
6. **Entity renaming and graph isomorphism:** accuracy must remain within one
   point, ruling out memorized names and serialization layouts.
7. **No-query controls:** ordinary language, internal-rule questions, and
   zero-hop answers must not degrade by more than two points.
8. **Recursion check:** predicted halt depth and first-correct step must
   increase monotonically with gold path length.

An apparent end-to-end advantage cannot carry the schema claim if it survives
shuffled memory, disappears on the common-fresh-graph primary endpoint, or is
reproduced by the random-target mask control.

### 6.5 Statistical unit and decision rule

Training seeds carry inference. Item bootstraps are clustered jointly by graph
world and template and describe only evaluation-set uncertainty.

Let `Delta_s,l` be Split minus Dense counterfactual-consistent composite
accuracy for paired seed `s` and 160M load `l`. Estimate:

- the 160M arm × log-load interaction with three paired seeds at every load;
- pooled seed standard deviation `sigma_pool` from the nine 160M paired
  deltas; and
- the three paired 360M high-load deltas.

The 160M interaction is the slope in
`Delta_s,l = beta_0 + beta_load log(N_l) + gamma_s + epsilon_s,l`, fit by
ordinary least squares with seed fixed effects and a two-sided t interval
using its residual degrees of freedom. Compute
`sigma_pool = sqrt(sum_l sum_s (Delta_s,l - mean_s Delta_s,l)^2 / 6)`.
The 360M confidence interval is the paired-seed mean plus or minus
`t(0.975, 2) * sd(Delta_s) / sqrt(3)`. The same paired-seed t construction is
used for Split minus random-target masking at 160M high load.

The hypothesis is **validated in the tested regime** only if:

1. mean 360M delta exceeds `max(2 points, 2 × sigma_pool)`;
2. all three 360M paired deltas are positive;
3. every primary stratum has a positive 360M mean delta;
4. the 95% confidence interval for the 160M arm × log-load interaction lies
   above zero;
5. at 160M high load, selective Split exceeds the random-target control by
   more than one point, with positive differences in all three paired seeds;
   and
6. every selector, burden, leakage, factual-job, language, and causal-control
   guardrail passes.

The practically meaningful hypothesis is **rejected in the tested regime**
only if all manipulation and learnability gates pass and:

1. the upper 95% bound on the 360M Split-minus-Dense delta is below two
   points;
2. the upper 95% bound on the 160M arm × log-load interaction is at or below
   zero; and
3. selective masking does not beat the random-target mask control by more
   than one point.

Any failure of task learnability, Dense memorization burden, selector
selectivity, leakage control, stream pairing, or scorer validity makes the
experiment invalid for this hypothesis. Such a result is reported as an
instrument failure, not converted post hoc into support or rejection.

## 7. Interpretation boundaries

- A gain only in Split memory-ON versus Dense memory-OFF is retrieval
  assistance, not evidence of freed reasoning capacity.
- A gain reproduced by random-target masking is generic target reduction,
  not selective fact offloading.
- A gain that vanishes under common fresh memory is unequal factual access,
  not a schema difference.
- A null with Dense recognition at chance does not test capacity relief,
  because Dense never carried the factual burden.
- A positive equal-information, dose-responsive result with successful
  offloading and preserved factual/language performance validates the
  operational hypothesis at 160M and 360M.
- The later 1B tier tests scale robustness. No result licenses claims about
  frontier models, natural-world annotation coverage, production retrieval,
  or universal bit-capacity limits.

## 8. Explicit non-goals

The protected experiment excludes:

- approximate nearest-neighbor search or learned semantic retrieval;
- a GNN, trainable graph encoder, query adapter, or per-fact embedding;
- neighborhood, path, proof, or answer computation inside the store;
- writable memory or test-time weight updates;
- adaptive-compute differences between arms;
- RL lookup planning, answer voting, or verifier selection;
- product latency, deletion, provenance-policy, or governance claims; and
- mechanistic claims about where freed capacity resides in the weights.

These may be studied only after the causal Dense/Split result is known.

## 9. Primary sources

- LMLM: *Pre-training Limited Memory Language Models with Internal and
  External Knowledge*, arXiv:2505.15962.
- Tiny Recursive Models: *Less is More: Recursive Reasoning with Tiny
  Networks*, arXiv:2510.04871.
- Recurrent Relational Networks, arXiv:1711.08028.
- Differentiable Neural Computer: *Hybrid computing using a neural network
  with dynamic external memory*, Nature 538 (2016).
- CLUTRR: *A Diagnostic Benchmark for Inductive Reasoning from Text*,
  arXiv:1908.06177.
- Adaptive Computation Time, arXiv:1603.08983; used only as motivation for
  a halt prediction, not variable claim-bearing compute.
- Semi-parametric Language Model with Selective Memory, OpenReview
  `5a2H3HEN61`; an under-review precedent for loss-based selective masking,
  not evidence for the present reasoning hypothesis.
- Internal causal standard:
  `docs/superpowers/specs/2026-07-21-core-measurement-cleanup-design.md`.
