# MemorySplit Architecture and Scale Decision

Date: 2026-07-20
Owner: Stephen Zhang
Status: approved section-by-section in session; written-spec review pending

## 1. Decision

Protect the current MemorySplit experiment. Keep the exact-key, loss-masked
organizer as the claim-bearing intervention for:

1. the Option-B difficulty-floor re-gate;
2. the 160M fact-load sweep;
3. the 1B dose calibration and confirmation.

Add only nonblocking soft-answer diagnostics now. Do not add Co-LMLM, Engram,
memory layers, or another training arm to the protected battery.

This is a split recommendation:

- **Best causal instrument:** the current exact-key LMLM-style intervention.
- **Best production follow-up:** a selective hybrid memory hierarchy combining
  continuous queries, a canonical exact store, and a free-form span index.
- **Best evidence for trainable conditional memory:** Engram, which tests a
  different hypothesis because its memory remains gradient-trained.

MemorySplit has plausible large-scale value, but not as an unchanged
frontier-scale architecture. The current experiment should first establish
whether removing direct fact-target pressure changes knowledge-free reasoning.

## 2. Question and hypothesis

The experiment asks whether a decoder trained without direct loss on arbitrary
fact values reallocates effective parametric capacity toward reusable,
knowledge-free reasoning.

The approved hypotheses remain:

- **H1:** the split arm improves knowledge-free reasoning at matched decoder
  parameters, raw training-token budget, data order, and seed, with the
  split–dense gap increasing with fact load.
- **H2:** the split system with its organizer attached remains within two
  points of dense closed-book factual recall.
- **H3:** the split arm stores fewer probe-recoverable fact bits in its
  weights, supported by store-ON/store-OFF behavior.
- **H4:** the split arm reaches reasoning milestones in fewer training tokens.

Two mechanisms remain intentionally combined:

1. literal fact-storage competition; and
2. optimization interference from rare, high-loss factual targets.

The experiment does not require the dense model to approach the synthetic
two-bits-per-parameter ceiling. At the current exposure counts, a positive
effect would more plausibly establish reduced optimization interference unless
additional evidence demonstrates storage saturation.

## 3. Current evidence and claim boundary

The 160M pilots establish the intervention, not H1:

- split factual recall is approximately 99.8% with the organizer and 0% without
  it;
- measured split fact bits in weights are zero under the current probe;
- fresh-entity lookups generalize perfectly in the measured pilot;
- fact-use QA strongly favors the split system;
- the first two versions of iGSM and deduction remained at chance.

Claims must use precise language:

- defensible: **no direct next-token loss is applied to returned fact values**;
- defensible: **zero fact bits were recovered by the specified probe**;
- defensible: **store-OFF factual behavior collapsed in the measured runs**;
- too strong: **facts receive provably no gradient of any kind**;
- too strong: **the weights contain provably zero information about facts**.

Returned fact-value tokens are teacher-forced inputs to later unmasked
predictions, so indirect gradient paths remain possible. The store-OFF and
capacity probes measure whether those paths produce functionally recoverable
knowledge.

## 4. Why the current architecture remains the scientific instrument

The exact-key intervention changes the fewest causal variables:

- identical decoder architecture and parameter count;
- identical underlying records, reasoning data, order, and paired seeds;
- explicit masking of fact-value targets;
- non-gradient, editable external storage;
- exact retrieval, avoiding retriever quality as a treatment variable;
- direct store-ON/store-OFF mediation;
- complete synthetic annotation, avoiding leakage from missed facts.

Its remaining fairness limitations must be reported:

- lookup markup changes tokenization and sequence composition;
- masking reduces the number of supervised target tokens;
- the split rendering uses context positions for lookup syntax and returned
  values;
- fixed total raw tokens do not imply identical numbers of supervised tokens.

A future mask-only control can distinguish gradient removal from useful
external memory, but it is not added to the protected battery.

## 5. Gate A amendment

The 2026-07-20 Option-B implementation changes the reasoning distribution:

- iGSM trains and evaluates in-distribution on operation counts 1–4;
- iGSM training mass is proportional to `1 / op`;
- iGSM problems with operation count at most two contain at most one
  distractor;
- deduction trains and evaluates in-distribution at depth 1–2;
- shallow deduction instances contain 4–6 facts and 3–4 rules;
- chain-of-thought plus final `Answer:` formatting remains unchanged.

The following per-task learnability rule is approved before the re-gate:

1. **Both tasks exceed 90% exact accuracy:** retain their mean as the
   knowledge-free primary composite.
2. **Exactly one task exceeds 90%:** freeze the passing task alone as the
   knowledge-free primary endpoint before the sweep.
3. **Neither task exceeds 90%:** do not claim a valid knowledge-free reasoning
   test. Freeze fact-use QA as a system-level fallback, explicitly report the
   knowledge-free endpoint learnability failure, and do not use the fallback
   to support H1.

Correct-answer log probability and calibration metrics explain trajectories
near the gate, but never override these branches.

## 6. Nonblocking diagnostic addition

Evaluate every available and future checkpoint with soft answer scoring:

- iGSM: length-normalized log probability or log-odds of the correct numeric
  answer;
- deduction: correct yes/no log-odds and Brier score;
- where inexpensive, separate final-answer negative log-likelihood from
  teacher-forced reasoning-trace negative log-likelihood.

These metrics:

- expose learning before exact generation crosses a threshold;
- improve checkpoint-curve resolution;
- help distinguish a true floor from formatting failure;
- remain supporting diagnostics, not new success gates.

The approved exact-accuracy endpoint and seed-level decision rule remain
unchanged and will be recorded at the preregistration freeze.

## 7. Current battery and inference rules

The protected sequence is:

1. rebuild Option-B gate data under commit `45befa8` or a descendant;
2. run the paired dense/split re-gate;
3. apply the Gate-A branch and freeze the resulting endpoint;
4. write the preregistration;
5. rebuild all sweep and 1B corpora under the frozen implementation;
6. run the 160M sweep at 3.2B tokens, approximately 20 tokens per parameter;
7. calibrate the 1B fact dose;
8. run the 1B confirmation at 10B tokens, approximately 10 tokens per
   parameter;
9. analyze using the approved paired, clustered, seed-aware statistics.

The H1 decision remains:

- 1B split–dense delta exceeds `max(2 × pooled seed sigma, 0.5 point)`;
- both seed-pair deltas have the same sign;
- H2 passes.

The 1B token budget is deliberately under compute-optimal training. It remains
a valid confirmation under the approved design, but it is not a clean
scale-law point relative to the 160M sweep.

## 8. Outcome interpretation

### 8.1 Positive at 160M and 1B

If H1 is positive at both scales with H2 and H3 intact:

- conclude that enforced fact externalization improved reasoning in the tested
  regimes;
- run a natural/noisy-store replication;
- only then design a structured 7B hybrid-memory pilot.

### 8.2 Positive at 160M, null at 1B

Scale and undertraining are confounded because the 160M sweep uses about 20
tokens per parameter and the 1B confirmation uses about 10. Before claiming
that the effect decays with scale, run a compute-optimal 1B follow-up near 20
tokens per parameter.

### 8.3 Defensible null after Gate A passes

Conclude that the intervention improves factual control and system-level fact
use but does not measurably improve knowledge-free reasoning under the tested
conditions.

### 8.4 Gate A failure

Report endpoint learnability failure. This is not evidence for or against the
memory hypothesis.

### 8.5 H2 failure

Void a positive H1 interpretation because the split system bought reasoning by
failing the factual job.

### 8.6 Fact-use gain without knowledge-free gain

Position MemorySplit as a tool-use/external-memory architecture, not a
freed-reasoning-capacity result.

## 9. Architectural alternatives

### 9.1 Co-LMLM

Co-LMLM is the strongest practical successor to relational LMLM:

- a single hidden-state query replaces decoded textual query strings;
- free-form factual spans expand coverage beyond entity–relation tuples;
- an InfoNCE objective trains the retrieval representation;
- a source-faithful annotation pipeline scales beyond Wikipedia;
- returned knowledge remains textual, attributable, and editable.

Measured evidence at 135M/360M and up to 90B training tokens shows large
factuality and perplexity gains, but no reasoning gain. Its 2.2B-entry index
establishes index construction at meaningful scale, not low-latency production
serving. Query timing, end-to-end retrieval latency, paraphrase-complete edits,
and index drift after post-training remain open.

Co-LMLM is therefore the first follow-up architecture, not a current arm.

### 9.2 Engram and trainable conditional memory

Engram provides the strongest large-scale evidence that specialized lookup
capacity can improve reasoning. At 27B total and 3.8B active parameters, its
iso-parameter/iso-FLOP comparison reports gains on BBH, ARC-Challenge, code,
and math. Its deterministic n-gram addressing also supports prefetch and host
memory offload.

However:

- its memory embeddings are gradient-trained parameters;
- facts are not externally editable or attributable in the MemorySplit sense;
- local lexical caching can directly improve benchmark performance;
- its result tests sparse-capacity allocation, not facts outside weights.

Engram is an adjacent production component and control, never evidence for H1.

### 9.3 RETRO, Memory Layers, Memory3, and hierarchical memories

These architectures establish that retrieval and sparse memory can scale, but
they do not isolate MemorySplit's hypothesis:

- RETRO and related systems retain ordinary fact-prediction pressure;
- Memory Layers and hierarchical memories relocate storage into trainable
  parameters;
- Memory3 externalizes latent key/value memories but lacks the matched,
  loss-masked causal contrast.

They inform serving and architecture design rather than replace the current
experiment.

## 10. Approved future architecture

If the current program warrants a follow-up, build a selective hybrid memory
hierarchy.

### 10.1 Components

1. **Backbone**
   - Stores procedures, schemas, language, and common stable knowledge.
   - Retains sufficient active width/depth for reasoning.

2. **Continuous router**
   - Emits a hidden-state query in one generation step.
   - Includes calibrated lookup timing, abstention, and miss behavior.

3. **Canonical exact store**
   - Uses typed entity and relation identifiers.
   - Supports aliases, temporal qualifiers, set-valued facts, provenance,
     confidence, and versioned snapshots.
   - Handles authoritative, high-stakes, and frequently edited facts.

4. **Dense free-form span store**
   - Covers evidence that does not fit a fixed schema.
   - Returns human-readable, attributable text.

5. **Optional conditional-memory cache**
   - Uses an Engram-like trainable tier for hot, stable local patterns.
   - Is treated as an efficiency component, not nonparametric fact storage.

### 10.2 Training data flow

1. Select facts based on rarity, volatility, privacy, attribution value, or
   expected update frequency.
2. Canonicalize records and retain source spans.
3. Mask direct targets for externalized values.
4. Train continuous retrieval queries and exact-route classification.
5. Train misses, stale values, conflicts, counterfactual swaps, and abstention.
6. Version the retrieval projection and index with the model checkpoint.

### 10.3 Inference data flow

1. The router decides whether and where to retrieve.
2. Independent queries are batched.
3. The exact store serves authoritative matches; the span index supplies
   broader evidence.
4. Retrieved values carry provenance and version metadata.
5. The model resumes generation from the returned evidence.
6. Retrieval traces remain available for audit and deletion verification.

### 10.4 Serving requirements

- one request-level memory sidecar rather than one RPC per tensor-parallel rank;
- hot HBM/DRAM cache with slower tiers for the long tail;
- asynchronous prefetch where addresses are predictable;
- batched lookup and returned-span prefill;
- p50/p95/p99 latency, time-to-first-token, throughput, and KV-cache accounting;
- snapshot consistency and explicit cache invalidation on edits;
- end-to-end cost accounting including annotation, indexing, training, and
  serving.

## 11. Scale assessment

### 11.1 1–3B

This is the strongest regime for observing effective-capacity competition.
Small models have limited active compute and factual capacity. Ordinary
inference-time retrieval shows its largest marginal gains below roughly 3B;
whether enforced training-time externalization follows the same trend is
exactly what remains unmeasured. The current experiment is well targeted to
this regime.

### 11.2 7B–50B

A selective design remains plausible for rare, volatile, private, or
provenance-sensitive facts. A structured pilot is justified only after the
current causal experiment and a natural/noisy-store replication.

The unchanged textual exact-call design is not recommended because serial
query generation, annotation density, KV-cache growth, and retrieval pauses
become material serving costs.

### 11.3 Frontier MoE

Conditional memory may become an important sparse-capacity axis, but
MemorySplit-style fact externalization is not currently a first-order general
reasoning optimization. Higher-priority levers include:

1. data quality, mixture, repetition, and curriculum;
2. active FLOPs, model shape, and tokens per parameter;
3. MoE sparsity, routing, expert placement, and interconnect;
4. optimizer and numerical stability;
5. attention/KV-cache design and serving throughput;
6. post-training and adaptive test-time compute.

External memory remains first-order when freshness, deletion, provenance, or
private knowledge—not benchmark reasoning—is the product requirement.

## 12. Documentation synchronization

The first implementation change is documentation-only.

Update:

- `README.md`;
- `docs/README-SHARE.md`;
- `docs/superpowers/2026-07-20-interim-report.md`;
- `docs/superpowers/specs/2026-07-17-memory-split-design.md`;
- `docs/superpowers/plans/2026-07-17-memory-split.md`;
- `cluster/RUNBOOK.md`.

Preserve as dated research records:

- `waveA_architectures.md`;
- `waveB_prior_evidence.md`;
- `waveC_eval_methodology.md`.

The synchronization must:

- distinguish the 2026-07-18 scale-plan “Option B” from the 2026-07-20
  Gate-A “Option B”;
- record both failed Gate-A attempts and the selected difficulty floor;
- mark preregistration as pending;
- mark all pre-`45befa8` corpora/checkpoints stale;
- make the remaining schedule relative to a successful re-gate;
- label the historical implementation plan as superseded where necessary;
- preserve historical result sections with explicit run provenance;
- avoid silently rewriting dated reports as though later decisions were known
  at publication time.

## 13. Operational safeguards

The current run IDs and `--resume auto` behavior create a data/checkpoint
mixing risk. Before any Option-B run:

- use versioned data and output roots, or archive old outputs;
- include the generating git commit in corpus provenance;
- refuse resume when checkpoint provenance and corpus provenance differ;
- rebuild every referenced pre-`45befa8` corpus before use;
- freeze the amended Gate-A rule before submitting the re-gate.

No result should be quoted without:

- run ID;
- arm, scale, dose, seed, and checkpoint;
- generating code commit;
- corpus build configuration;
- organizer mode;
- evaluation version.

## 14. Reusable project agents

After written-spec approval, create three project-level agents:

1. **Memory architecture researcher**
   - Tracks new primary literature.
   - Classifies whether a method externalizes facts or relocates memorization.
   - Separates measured evidence from extrapolation.

2. **Scaling skeptic**
   - Audits compute, annotation, index, latency, and serving claims.
   - Checks model-scale evidence and matched-budget fairness.
   - Challenges frontier extrapolation.

3. **Experimental gatekeeper**
   - Audits run provenance, preregistration, endpoint validity, and stale
     artifacts.
   - Prevents post hoc threshold changes.
   - Reports blockers without modifying scientific decisions.

Each agent is project-specific and lives under `.cursor/agents/`.

## 15. Visual artifact

After written-spec approval, create a Cursor Canvas containing:

- the causal-versus-production architecture split;
- the evidence hierarchy;
- the current battery and decision branches;
- scale-specific recommendations;
- the future hybrid memory data flow;
- measured evidence clearly separated from inference.

The Canvas embeds reviewed data and citations; it performs no network calls.

## 16. Verification

Documentation phase:

- compare every status claim against git history and run artifacts;
- search for stale mixture, scale, and Gate-A statements;
- validate internal links and paths;
- review the final diff for preservation of dated history;
- confirm a clean worktree after commit.

Agent phase:

- validate YAML frontmatter and lowercase-hyphen names;
- ensure descriptions state precise delegation triggers;
- ensure agents are read-only by default for research and audit tasks.

Canvas phase:

- pass the Canvas TypeScript check;
- verify visual hierarchy, source labels, and absence of unsupported imports;
- include no empty sections or placeholder content.

No model-training code is changed in the documentation phase. Code drift,
including smoke-test overrides and missing intended OOD artifacts, belongs in
the implementation plan after this specification is reviewed.

## 17. Primary sources

- LMLM: https://arxiv.org/abs/2505.15962
- Co-LMLM: https://arxiv.org/abs/2607.07707
- Auditing LMLM forgetting: https://arxiv.org/abs/2607.00605
- Engram: https://arxiv.org/abs/2601.07372
- Memory Layers at Scale: https://arxiv.org/abs/2412.09764
- Hierarchical parametric memories: https://arxiv.org/abs/2510.02375
- Memory3: https://arxiv.org/abs/2407.01178
- RETRO: https://arxiv.org/abs/2112.04426
- Retrieval/pretraining scaling: https://arxiv.org/abs/2604.00715
- More Room for Language: https://arxiv.org/abs/2404.10939
- Great Memory, Shallow Reasoning: https://arxiv.org/abs/2408.11815
- Knowledge capacity scaling: https://arxiv.org/abs/2404.05405
- Compute-optimal skills: https://arxiv.org/abs/2503.10061
- Optimal MoE sparsity for reasoning: https://arxiv.org/abs/2508.18672
- Capacity-aware mixture optimization: https://arxiv.org/abs/2603.08022
