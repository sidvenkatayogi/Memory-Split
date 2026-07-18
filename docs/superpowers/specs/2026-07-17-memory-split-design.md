# Memory Split: Schemas in Weights, Facts in the Organizer — Design Spec

2026-07-17. Owner: Stephen Zhang. Status: approved design (brainstormed and
approved section-by-section in-session); preregistration values marked
"pinned at freeze" are finalized by the day-5 preregistration document, not
this spec.

## 1. Summary

Test the CLS-inspired hypothesis that a small language model trained with
facts offloaded to an external knowledge organizer learns better
reasoning/procedures ("schemas") than a dense twin of the same parameter and
token budget. The mechanism is LMLM-style fact externalization (lookup-call
tokens, fact values loss-masked; arXiv 2505.15962); the measurement is a
fact-load dose-response at 160M scale plus a multi-seed confirmation at 410M,
with one 1B pair as a stretch. Month-end deliverable: reasoning accuracy vs
fact load for both arms — a measured delta or a defensible null by July 31.

Hypotheses:

- **H1 (primary).** At matched params, tokens, and data order, the split arm
  beats the dense arm on knowledge-free reasoning, and the gap widens as
  fact load increases (arm x load interaction).
- **H2 (guardrail).** Split arm with organizer attached matches or beats
  dense closed-book on recall of trained facts. If H2 fails, H1 is void
  (the split "bought" reasoning by discarding the fact job).
- **H3 (mediation).** The split arm stores measurably fewer fact-bits in
  weights: store-unplugged recall collapses, and bits-in-weights accounting
  (Physics of LM 3.3-style, arXiv 2404.05405) shows a large dense-vs-split
  gap.
- **H4 (secondary; novel CLS rate prediction).** The split arm reaches
  reasoning-accuracy milestones in fewer training tokens (sample
  efficiency read off training-curve checkpoints).

Two crowding mechanisms are in scope and deliberately not separated: (i)
bit-capacity competition (2 bits/param ceiling, exposure-dependent), and
(ii) optimization interference from high-loss, never-converging rare-fact
tokens. The dose sweep measures their combined effect; mechanism attribution
beyond H3 is out of scope.

## 2. Background and gap (see docs/superpowers/research/2026-07-17-memory-split/)

- LMLM (2505.15962, ICLR 2026) pretrains 176M/382M models with entity facts
  wrapped in `<|db_start|> query <|db_retrieve|> value <|db_end|>` and value
  tokens masked from the loss; facts provably live outside (unplugged
  factual precision collapses). They report NLU parity only — **no
  reasoning endpoint**. That measurement is this project's contribution.
- Closest evidence is mildly negative: "To Memorize or to Retrieve"
  (2604.00715) finds inference-time retrieval leaves reasoning tasks
  roughly unchanged at 30M-3B; "More Room for Language" (2404.10939) finds
  from-scratch retrieval LMs reallocate capacity to syntax with an NLU
  regression. phi-3 (2404.14219) states the crowding hypothesis verbatim
  but never ran the controlled ablation.
- Matched-budget reasoning-adjacent effect sizes in the literature are +2-3
  points against seed noise of 0.5-2 points on natural benchmarks at this
  scale — natural benchmarks cannot carry the claim. Synthetic testbeds can:
  iGSM-style math is learnable to 99% by 100M models (2407.20311), and
  10k-item held-out sets certify sub-point deltas.

## 3. Experimental design

### 3.1 Arms

Identical GPT decoders, identical corpus content, identical mixture
schedule and document order, identical init and data-order seeds per pair.
Single difference — the fact rendering:

- **Dense arm.** Synthetic fact text appears as plain prose. Loss on all
  tokens. Facts can only be used if memorized in weights.
- **Split arm.** Every attribute value inside synthetic fact text and QA
  chain-of-thought is wrapped: `<|db_start|> {entity}, {relation}
  <|db_retrieve|> {value} <|db_end|>`. Tokens strictly between
  `<|db_retrieve|>` and `<|db_end|>` are loss-masked (label -100);
  everything else, including the special tokens and the query, receives
  loss. The model is never rewarded for knowing a value, only for knowing
  when and how to ask.

No retrieval service runs during training (values sit inline, masked). The
**organizer** — an exact-match `(entity, relation) -> value` store built by
the corpus generator — is consulted only at inference/eval.

Natural-bed text is left untouched in both arms (limitation: incidental
natural-world facts are memorizable by both arms equally; the controlled
dose is the synthetic fact component).

### 3.2 Corpus mixture (token-budget-matched per component, both arms)

| Component | Share | Content |
|---|---|---|
| Natural bed | 62% | FineWeb-Edu sample-10BT, fixed shard order |
| Synthetic biographies (the dose) | 23% | bioS-style, N entities x 6 attributes, paraphrased exposures |
| Synthetic reasoning (fixed) | 12% | iGSM-lite math (7%) + rule deduction (5%) |
| Fact-use QA | 3% | extraction, date comparison, equality/multi-hop, with CoT |

Because wrapped biographies are ~40-60% longer than dense ones, matching
token budgets per component means the split arm sees fewer bio exposures.
This is by design and harmless to the contrast: exposures set dense in-weight
storage; the split arm's values are masked regardless. Preregistered choice:
match total tokens, steps, and per-component token budgets; report the
exposure difference.

### 3.3 Dose, scales, seeds

Fact load = number of distinct entities N at fixed fact-token share.
Planning levels N in {50k, 200k, 800k} (approx. 210/53/13 exposures per
entity at the 160M token budget) — pinned at gate B so the top level sits
past the dense arm's recall saturation.

| Stage | Scale | Tokens | Runs | Est. per-run |
|---|---|---|---|---|
| Dose sweep | 160M | 3.2B | 3 loads x 2 arms x 2 seeds = 12 | ~10-15 L40S-h |
| Confirmation | 410M | 8B | top load x 2 arms x 3 seeds = 6 | ~45 L40S-h |
| Stretch | 1B | 10B | top load x 2 arms x 1 seed = 2 | ~135 L40S-h (requeue chain or 4-GPU) |

### 3.4 Endpoint and decision rule

- **Primary endpoint:** knowledge-free reasoning composite = mean of
  iGSM-lite held-out accuracy and deduction held-out accuracy
  (in-distribution difficulty), >= 10k items each. Multi-hop fact-use
  accuracy (split scored with organizer; dense closed-book) is co-reported
  as the "system-level" reasoning-over-knowledge secondary — it conflates
  store access, so it does not carry H1 alone.
- **H1 tests:** (a) confirmation contrast at 410M top load: split - dense on
  the primary composite, 3 seeds, per-item paired bootstrap clustered by
  problem template; (b) sweep interaction: difference in slope of composite
  vs log N between arms at 160M.
- **Decision rule (structure fixed now; exact margins pinned at freeze from
  pilot seed-sigma):** positive iff confirmation delta > max(2 x pooled
  seed-sigma, 0.5 pt) with the same sign in all 3 seed pairs AND H2 holds
  (split-with-store recall >= dense closed-book - 2 pts). Defensible null
  iff the 95% CI of the confirmation delta lies within (-margin, +margin)
  and gates A-C passed. Anything else: report as underpowered/inconclusive
  with the failure mode named.
- Natural benchmarks (HellaSwag, ARC-E, PIQA, LAMBADA, WinoGrande — cloze,
  CORRECT-PROB continuous scoring and accuracy) and bits-per-byte on
  held-out slices (bed slice, bio slice, reasoning slice) are supporting
  evidence only.

## 4. Corpus specification

### 4.1 Biographies (bioS-style)

- Entity: unique full name (first x middle x last pools, collision-free).
  Attributes (6): birth_date, birth_city, university, major, employer,
  current_city. Values drawn uniformly from finite pools (dates from a
  ~36.5k-day range; ~200 cities; ~300 universities; ~100 majors; ~263
  employers) => ~53 bits/entity information content (15.2 + 7.6 + 8.2 +
  6.6 + 8.0 + 7.6).
- Each exposure renders the 6 attributes as one mini-biography: per-attribute
  sentence templates (>= 20 per attribute), sentence-order permutation, and
  name-form variation, seeded deterministically.
- Both renderings (dense prose / split wrapped+mask spans) derive from one
  underlying record; the organizer table and eval probes derive from the
  same records.

### 4.2 iGSM-lite (knowledge-free math)

Faithful-in-spirit reimplementation of iGSM (2407.20311): random dependency
DAGs over instance-quantities of abstract categories, arithmetic mod 23,
chain-of-thought solutions that introduce intermediate variables in
topological order, difficulty parameter op (number of operations). Train
op in {2..8}; eval in-distribution op in {2..8} (primary) and OOD op in
{9..12} (reported). Answers verifiable by a solver oracle; problems deduped
by structure hash; held-out set disjoint by construction.

### 4.3 Rule deduction (knowledge-free logic)

RuleTaker-style closed-world deduction over nonsense predicates: fact base
(4-10 atoms), rule base (3-8 Horn rules), question "is X F?" with CoT
forward-chaining derivation and yes/no answer. Train depth <= 4; eval
in-distribution depth <= 4 (primary) and OOD depth 5-6 (reported).
Forward-chaining oracle guarantees labels.

### 4.4 Fact-use QA

Over biography entities: (a) extraction ("What was {name}'s major?"), (b)
date comparison ("Who was born earlier, A or B?"), (c) equality/multi-hop
("Does A's employer's city match B's birth city?" - style chains over
stored attributes). All with short CoT that cites the needed facts then
concludes. In the split rendering the cited fact values are
lookup-wrapped and masked. Held-out QA items use trained entities in fresh
combinations; a further probe set uses fresh entities added to the
organizer only (tests the split arm's lookup-skill generalization; dense
cannot answer these — reported, not part of H2).

### 4.5 Sizing invariants

- Per-component token budgets equal across arms within 1%.
- Document order: single seeded interleave schedule shared by both arms
  (component pattern identical; bio doc k in dense = bio doc k in split,
  same underlying record).
- Held-out eval items never appear in either training stream (checked by
  ID and by string containment on a sample).

## 5. Training specification

- **Tokenizer:** GPT-2 BPE (tiktoken) + 4 special tokens
  (`<|db_start|>`, `<|db_retrieve|>`, `<|db_end|>`, `<|eot|>`) => vocab
  padded to 50304.
- **Architecture:** decoder-only, RMSNorm pre-norm, RoPE, SwiGLU MLP
  (8/3 ratio rounded to multiple of 64), no biases, untied embeddings,
  context 2048.
  - 160M-class: 12L x 768d x 12h (~162M total)
  - 410M-class: 24L x 1024d x 16h (~405M)
  - 1B-class: 22L x 1792d x 14h (~1.03B)
- **Optimizer:** AdamW (beta 0.9/0.95, wd 0.1 on matrices only), peak LR
  1.5e-3 (160M) / 1e-3 (410M) / 6e-4 (1B), 300-step warmup, cosine to 10%,
  grad clip 1.0, bf16 autocast, optional torch.compile (on for cluster).
- **Batch:** ~0.5M tokens/step (160M/410M) via grad accumulation sized to
  GPU; sequence packing with `<|eot|>` separators; loss mask stored
  alongside tokens (uint8), labels -100 where mask==0.
- **Determinism/pairing:** per pair, identical init seed and data-order
  seed; seeds vary across replicate pairs. All configs are YAML in
  configs/ and logged into the checkpoint.
- **Checkpoint/resume:** atomic checkpoint every ~30 min with model, opt
  state, dataloader cursor, RNG states; runs are requeue-safe (FarmShare
  2-day MaxWall). Eval checkpoints (H4 curves): every ~10% of steps a
  lightweight model-only snapshot.
- **Data format:** pre-tokenized uint16 memmap shards + parallel uint8
  mask shards per arm x load; bed shards shared across loads.

## 6. Evaluation specification

- **Generative scoring** (iGSM, deduction, QA): greedy decode, parse final
  answer after `Answer:`; exact match against oracle. Batched with KV
  cache.
- **Lookup interception (split arm, store ON):** when `<|db_start|>` is
  emitted, model free-decodes the query until `<|db_retrieve|>`; the
  organizer exact-matches (entity, relation); on hit, value tokens +
  `<|db_end|>` are force-decoded; on miss, model decodes freely and the
  miss is logged. Store OFF: no interception anywhere (H3 collapse
  measurement).
- **Recall probes (H2/H3):** per-attribute extraction on a fixed 2k-entity
  sample per load, dense closed-book vs split store-ON vs split store-OFF.
- **Bits-in-weights accounting (H3):** bits ~= N x sum over attributes of
  max(0, acc_a - guess_a)/(1 - guess_a) x log2 |V_a| using closed-book
  (dense) / store-OFF (split) accuracies — a simplified, clearly-labeled
  variant of Physics 3.3 accounting.
- **Natural benchmarks:** HellaSwag, ARC-E, PIQA, WinoGrande, LAMBADA from
  HF datasets, cloze log-likelihood (length-normalized accuracy +
  CORRECT-PROB). Contamination: report 13-gram overlap of eval strings
  against the bed sample (measure, don't filter, given supporting-only
  status).
- **Perplexity slices:** held-out bed shard, held-out bio text (dense
  rendering for both arms), held-out reasoning text; bits-per-byte.
- All metrics per checkpoint where cheap (composite, recall); full battery
  at final checkpoint.

## 7. Gates, kill conditions, schedule

Days count from 2026-07-18; all cluster work on FarmShare (login rice-04,
gpu partition, QOS gpu: 4 concurrent GPU jobs, 32 submitted, MaxWall
2 days; L40S 48GB; scratch at /scratch/users/syz).

- **Day 1-2 — build.** corpusgen + trainer + organizer + evals + tests;
  local (MPS/CPU) toy pilot proving: loss falls; split-arm masked-value
  loss stays high while dense bio loss falls (mechanism smoke = gate 0);
  FarmShare env build + data prep job.
- **Day 3 — gate A.** 160M dense pilot (short-token budget) learns
  iGSM-lite to >90% held-out at mixture ratios. Fail: raise reasoning
  share once; still failing: descope primary composite to deduction +
  fact-use.
- **Day 4 — gate B.** Dense recall across candidate N levels degrades
  (crowding regime found). If dense stores everything at every feasible N:
  escalate N / cut exposures once; if still saturated, report "no crowding
  at this scale/budget" as a finding and continue with top feasible N.
- **Day 4 — gate C.** Split pilot emits well-formed lookups (>95% parse
  rate on QA probes) and store-ON recall exceeds store-OFF by >30 pts.
  Fail: fix rendering/masking before any battery run.
- **Day 5 — preregistration freeze.** Margins, exact N levels, decision
  rule constants committed to git
  (docs/superpowers/specs/2026-07-22-preregistration.md) before any
  confirmation run starts.
- **Days 5-9 — sweep + confirmation.** 12 sweep runs then 6 confirmation
  runs, 4 concurrent; evals stream as checkpoints land.
- **Days 9-12 — stretch.** 1B pair only if sweep+confirmation are done and
  analyzed.
- **Days 12-14 — report.** Analysis, dose-response figure, month-end
  report.
- **Kill order under schedule pressure:** drop 1B stretch; drop one fact
  level; drop confirmation to 2 seeds. The 410M top-load multi-seed
  contrast is protected last; the <= $300 RunPod burst is the contingency
  for exactly that contrast.

Compute worst case ~730 L40S-hours vs ~1,300 nominally available at 4
concurrent GPUs over the window.

## 8. Risks and limitations (stated up front)

- Synthetic-heavy corpus: external validity argued, not demonstrated; the
  natural bed keeps language ability realistic but the dose is synthetic.
- Natural-bed facts are unmasked in both arms (uncontrolled background
  memorization pressure, equal by construction).
- Crowding may not bind at 160M/410M with feasible N (gate B may trigger
  its "no crowding" branch — that is a reportable result, not a failure).
- 1B stretch is single-seed: labeled as anecdote, never pooled with the
  confirmation statistics.
- iGSM-lite is a reimplementation, not the released iGSM; distributional
  drift risk is bounded by gate A.
- FarmShare is queue-shared and free; wall-clock risk handled by
  checkpoint/requeue and the kill order.

## 9. Deliverable

Month-end report (docs/superpowers/): dose-response figure (primary
composite vs log N, both arms, per-seed points), confirmation contrast with
CI and preregistered decision, H2 guardrail table, H3 collapse + bits
accounting, H4 curves, natural-benchmark table, limitations. Verdict is one
of: measured positive delta / defensible null / inconclusive-with-named-
failure-mode.

## 10. References

LMLM 2505.15962; Physics of LM 2.1 iGSM 2407.20311, 3.1 2309.14316, 3.3
2404.05405; To Memorize or to Retrieve 2604.00715; More Room for Language
2404.10939; RETRO 2112.04426; Memory Layers at Scale 2412.09764; kNN-LM
1911.00172 and limits 2408.11815; phi-3 2404.14219; RuleTaker 2002.05867;
seed-variance 2406.10229; DataDecide 2504.11393; error bars 2411.00640;
CLS: McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland
2016; Tse et al. 2007. Full annotated research dossier:
docs/superpowers/research/2026-07-17-memory-split/.
