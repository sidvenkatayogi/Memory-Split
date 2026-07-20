# Memory Split: Schemas in Weights, Facts in the Organizer

## Interim technical report and compute case — 2026-07-20

Stephen Zhang. Repo: `MemorySplit/` (spec `docs/superpowers/specs/2026-07-17-memory-split-design.md`,
preregistration pending gate review). Status at writing: build and gate
phase complete except the gate-A retry verdict (section 6.4), which this
document incorporates the moment it lands.

---

## 1. Executive summary

We are running the first controlled measurement of a hypothesis the field
keeps asserting and has never tested: **that a small language model wastes
parametric capacity memorizing facts, and that externalizing those facts to
a fast key-value store during pretraining frees capacity for reasoning.**
The phi-3 team states this as their design intuition; the LMLM paper (ICLR
2026) built the training mechanism and explicitly lists the measurement as
open; the closest adjacent studies find nulls or trade-offs but never ran
the controlled experiment. Whatever direction our result takes, it is the
first number of its kind.

In 48 hours of building on free compute (Stanford FarmShare, $0 cloud spend
to date) we have: a fully tested experiment system (124 unit tests, one
command from laptop to queued cluster battery); six purpose-built corpora
(~55 GB); and gate-phase pilot results that already prove the core
mechanism works at real scale:

- A 160M-parameter model trained with our organizer stores **zero measured
  fact-bits in its weights** (its dense twin stores tens of thousands) yet
  answers 99.8% of fact probes correctly when the organizer is attached —
  and 0.0% when unplugged. Facts demonstrably live outside the model.
- The same model treats lookup as a **generalizable skill**: on questions
  about entities it has *never seen in any training text*, it composes the
  correct organizer query 500/500 times (100%), with zero malformed calls
  in ~4,000 attempts.
- On fact-use QA it **more than doubles** its dense twin (70.6% vs 30.8%)
  at identical parameter count and token budget.

The remaining schedule runs the dose-response sweep at 160M and the
preregistered billion-parameter confirmation by month-end on free compute.
Section 8 lays out what marginal funding converts directly into: seeds,
compute-optimal token budgets, and a 3B scaling point that would turn a
month-end internal answer into a publication-grade scaling claim.

## 2. The question, and why nobody has answered it

**Hypothesis (CLS-inspired).** Complementary learning systems theory
(McClelland, McNaughton & O'Reilly 1995; Kumaran et al. 2016) argues
intelligent agents need two memory systems: a fast, item-specific store
(hippocampus) and a slow structure-learner (neocortex) that would otherwise
suffer interference from arbitrary facts. Mapped to small LMs: facts in
weights are the interference; an external organizer is the fast store; if
the mapping holds, removing the memorization burden should yield measurably
better reasoning ("schemas") at fixed parameters.

**Why it is open.** The literature triangulates the gap precisely:

- **phi-3 technical report** (2404.14219): deletes fact-heavy web data "to
  leave more model capacity for 'reasoning' for the mini size models" — a
  production bet on exactly our hypothesis, shipped without a controlled
  ablation.
- **LMLM** (2505.15962, ICLR 2026): invented the training mechanism we
  adopt (lookup calls with loss-masked values) and showed a 382M model can
  match Llama-2-7B *factual precision*. Their general-capability evaluation
  is five natural benchmarks reported as parity ("does not compromise
  overall model performance"), all near chance at their scale; their
  limitations section states that "distinct benchmarks for disentangling
  knowledge and reasoning remain underexplored." The reasoning delta is
  asserted as intuition ("dedicating model capacity to reasoning"),
  evidenced only by faster convergence and lower validation perplexity.
- **To Memorize or to Retrieve** (2604.00715): the closest measurement —
  finds inference-time retrieval leaves reasoning tasks essentially
  unchanged at 30M-3B. Critically, this is RAG *bolted onto conventionally
  pretrained models*: the weights already paid the memorization cost. It is
  the control condition for our experiment, not the experiment.
- **More Room for Language** (2404.10939): from-scratch retrieval-augmented
  MLMs reallocate freed capacity to *syntax* while broader understanding
  slightly regresses — a reallocation result at ≤98M scale with the wrong
  endpoint, and a warning that the freed capacity does not automatically go
  where one hopes.
- **kNN-LM limits** (2408.11815, NAACL 2025): post-hoc datastore
  interpolation actively *degrades* reasoning even with oracle retrieval —
  eliminating the cheapest alternative design.

No study combines (1) fact offloading enforced during pretraining, (2) a
matched-parameter, matched-data paired baseline, (3) reasoning as the
primary preregistered endpoint, and (4) a controlled dose axis. We do.

## 3. Experimental design

### 3.1 The intervention, exactly

Two arms per condition, identical in every respect — same decoder
architecture (RMSNorm, RoPE, SwiGLU; 160M: 12L x 768d; 1B: 22L x 1792d),
same tokenizer, same corpus content, same document order, same
per-component token budgets (within 1%), same init and data-order seeds —
except the rendering of fact values:

- **Dense arm**: "Kai Nakamura majored in Communications at ..." — plain
  prose, loss on every token. Facts are usable only if memorized.
- **Split arm**: "Kai Nakamura majored in `<|db_start|>`Kai Nakamura,
  major`<|db_retrieve|>` Communications`<|db_end|>` at ..." — the value
  tokens between `<|db_retrieve|>` and `<|db_end|>` are excluded from the
  loss (label -100). The model receives gradient signal for *when and how
  to ask* (the call tokens and the query get loss) and **provably none for
  the fact itself**. No retrieval runs during training; values sit inline,
  masked. At inference, emitting a query triggers an exact-match lookup in
  the organizer — a key-value table `(entity, relation) -> value` built by
  the corpus generator — whose result is force-decoded into the stream.

This is LMLM's mechanism by deliberate adoption (credited). Everything
around it is ours.

### 3.2 The corpus as a measuring instrument

Every training stream mixes five components (post gate-A remediation:
54% natural bed / 23% biographies / 12% iGSM math / 8% deduction /
3% fact-use QA):

- **Natural bed** — FineWeb-Edu, identical shards in identical order both
  arms; keeps language competence realistic.
- **Biographies (the dose)** — N synthetic entities x 6 attributes drawn
  from finite pools totaling ~53 bits/entity, rendered through 20+
  paraphrase templates per attribute, 8 sentence orderings, 3 name forms.
  Because we generate the facts, the fact load is a *controlled variable*
  — N sweeps 50k / 200k / 800k (160M) and 800k / 4M (1B) — and the split
  arm's annotations are complete by construction (no learned annotator, no
  leakage of unwrapped facts).
- **Reasoning (fixed across all conditions)** — two knowledge-free
  families with independent oracles: iGSM-style synthetic math (random
  dependency DAGs over nonsense quantities, arithmetic mod 23, chain-of-
  thought solutions, difficulty = operation count) and RuleTaker-style
  Horn-clause deduction over nonsense predicates (balanced yes/no,
  forward-chaining oracle, depth-controlled). Identical text in both arms.
- **Fact-use QA** — extraction, date comparison, and city-equality
  questions whose chain-of-thought cites stored facts then reasons; in the
  split rendering the citations are lookups. Held-out sets use trained
  entities in fresh combinations, plus a probe set of *fresh entities
  present only in the organizer* — the discriminator between "memorized
  answers" and "learned the lookup schema."

### 3.3 Endpoints, statistics, decision rule

- **Primary (H1)**: knowledge-free reasoning composite — mean of iGSM and
  deduction held-out accuracy, >=10k items each, so per-run CIs are
  sub-point; contrast = split minus dense at the top binding dose, plus
  the arm x load interaction across the sweep.
- **Guardrail (H2)**: split-with-organizer must match dense closed-book on
  fact recall (within 2 points).
- **Mediation (H3)**: bits-in-weights accounting (recall-derived, Physics
  of LM 3.3-style) plus the ON/OFF collapse — a positive H1 without H3
  would be uninterpretable; with it, the causal chain is measured at both
  ends.
- **Rate (H4, CLS's novel prediction)**: reasoning-milestone tokens from
  training curves — schema-consistent learning should be faster when
  interference is removed.
- **Inference**: per-item paired bootstrap clustered by problem template;
  seeds as replicates (2 pairs at 1B, sigma pooled from the 160M sweep's
  6 pairs, preregistered); decision rule frozen before confirmation runs:
  positive requires delta > max(2 x pooled seed-sigma, 0.5 pt) with sign
  consistency across seed pairs AND H2 intact; a tight CI around zero is
  a *defensible null* — explicitly a deliverable, not a failure.
- **Gates before compute**: A (reasoning learnable at pilot budget),
  B (dose actually binds on the dense arm), C (lookup mechanics work).
  Each gate has a prescribed remediation and a kill path, so GPU-days are
  never spent on an invalid configuration.

## 4. Anticipated objections (defense round)

**"Isn't this just LMLM?"** The mechanism is LMLM's; the *experiment* is
not, on six axes: (1) they never measure reasoning — verified against the
paper text, which concedes the gap in its limitations; (2) they have one
uncontrolled Wikipedia fact load, we have a swept, generated dose with
exact bit-content; (3) their annotations come from a fine-tuned Llama-3.1-8B
annotator with imperfect coverage (facts leak into unmasked text), ours are
complete by construction; (4) their retrieval is fuzzy embedding match over
54.6M noisy triplets, ours is exact-match over canonical keys — retrieval
noise cannot masquerade as a capacity effect (500/500 fresh-entity hits);
(5) we add mediation instruments (bits accounting, ON/OFF collapse,
fresh-entity probes) and a frozen decision rule; (6) we confirm at 1B.
A positive result *validates their open conjecture with its first
controlled measurement*; a null *bounds* it.

**"Isn't this just RAG?"** No — the division of labor is imposed during
pretraining. RAG on a conventionally trained model leaves memorization
pressure intact; that configuration was already measured (2604.00715) and
moves reasoning by roughly nothing. Our question is whether *never paying
the memorization cost* changes what the weights learn. kNN-LM-style
post-hoc memory actually hurts reasoning (2408.11815), which sharpens the
distinction.

**"Why not memory layers / Engram?"** Those relocate storage *inside* the
gradient loop (trainable KV slots) — a complementary but different claim
(architectural separation), and the store cannot be edited, audited, or
unplugged the way an external organizer can. Our design measures the
external-store version of CLS, which is also the deployable one for the
program (facts updated without touching weights).

**"Why synthetic reasoning tasks instead of GSM8K/MMLU?"** Power. At 1B
scale and our budgets, GSM8K sits at its floor and MMLU-multiple-choice at
chance (documented across SmolLM2/OLMo/loss-emergence studies); seed sigma
on natural MC benchmarks (0.4-0.9 pts) is the same order as the expected
effect (+2-3 pts in the best matched precedents). Our synthetic endpoints
are learnable at this scale (GPT-2-small reaches 99% on iGSM in the
original paper), knowledge-free by construction (mod-23 arithmetic,
nonsense predicates — no memorized co-occurrence shortcut), oracle-scored,
and sized (10k items) to certify sub-point deltas. Natural benchmarks are
reported as supporting evidence, exactly as the noise analysis demands.

**"The dense arm's facts might be under-trained, not crowded out."** This
is a real regime distinction and we measure it rather than assume it:
gate-B pilots show dense recall degrading monotonically in N at fixed
budget (interference regime), and the capacity accounting quantifies
stored bits directly. Both crowding mechanisms — bit-capacity competition
and optimization interference from never-converging rare-fact tokens — are
in scope; the preregistration names them and the dose sweep measures their
combined effect.

**"Two seeds at 1B?"** Seed sigma is estimated where seeds are cheap (six
160M pairs), pooled, and preregistered; the 1B contrast then requires sign
consistency across both pairs plus the margin, with per-item clustered CIs
carrying precision (10k items). The <= $300 burst reserve restores a third
pair if sigma comes in high — and section 8's ask makes that unnecessary.

**"Natural-bed facts are unmasked in both arms."** Correct — the dose is
the synthetic component only; incidental web facts exert equal pressure on
both arms by construction. This biases *against* finding an effect (both
arms carry some memorization burden), making a positive result
conservative.

## 5. What we built (and how fast)

Two days from empty repo to validated cluster battery, on free compute:

- **System**: seeded corpus generators with independent oracles; two-arm
  builder with byte-identical determinism (serial == parallel workers,
  tested); GPT trainer with loss masking, atomic checkpoint/requeue
  (survives FarmShare's 2-day walls via dependency chains), and a live
  mechanism metric (CE at masked fact positions); generative eval harness
  with mid-decode lookup interception; capacity accounting; clustered
  paired-bootstrap statistics; dose-response figure generation. 124 unit
  tests plus an end-to-end pipeline test; everything runs from one staging
  command.
- **Corpora**: six built (gate loads, three sweep loads, two 1B loads at
  10B tokens each) at ~2.5M tokens/s after profiling fixes; the 4M-entity
  1B corpus builds in ~2.5h.
- **Throughput**: 160M trains at ~130k tokens/s/L40S (planning-band MFU),
  putting a full 1B arm at ~5.8 GPU-days — measured, not estimated.
- **Operational hardening** (each found by the pipeline, fixed, tested,
  and documented same-day): a dead compute node, a submit-directory trap,
  a stage-collision in corpus paths, an interpreter-shutdown hang that
  zombified dependency chains, a quadratic index rebuild that would have
  made 1B corpora infeasible, and HF-ecosystem breakage (stale tokens,
  datasets-v5 naming). The marginal cost of the *next* experiment this
  team runs on this infrastructure is dramatically lower than the first.

## 6. Results to date

### 6.1 The mechanism works at real scale (gate C — passed)

160M split-arm pilot, 0.8B tokens, 200k entities:

| Measurement | Split arm | Dense twin |
|---|---|---|
| Recall, organizer attached | **99.8%** | — |
| Recall, organizer unplugged | **0.0%** | 0.8% closed-book |
| Measured fact-bits in weights | **0.0** | ~40,300 |
| Fact-use QA accuracy | **70.6%** | 32.9% |
| Fresh-entity QA (never in training text) | **100.0%** (500/500) | 0.2% |
| Malformed lookup calls | 0 of ~4,000 | — |

During training, the split arm's cross-entropy on masked fact values stays
at 9.45 — near the uniform-distribution ceiling — while its overall
training loss falls normally. Facts stay out of the weights *while
language and skills go in*: the fast/slow split is not a metaphor in this
system; it is a measured property.

The fresh-entity result deserves emphasis: the model composes correct
canonical queries — full name, attribute key — for entities that appear in
no training document. It has learned looking-things-up as a *schema*, the
precise behavior the hypothesis says freed capacity should buy.

### 6.2 The dose binds (gate B — informative pass)

Dense closed-book recall degrades monotonically with fact load at fixed
budget (1.1% at 50k -> 0.8% at 200k -> 0.3% at 800k entities), and stored
bits peak at 200k — direct evidence that fact pressure strains the dense
arm in the pilot regime, with the full-budget sweep (4x exposures) sizing
the effect properly. The binding mechanism at pilot budget is
exposure/interference rather than the 2-bits/param ceiling; the
preregistration will state this, and the 1B calibration stage picks the
dose that binds at scale before any confirmation GPU-day is spent.

### 6.3 Reasoning-task learnability (gate A — remediated, retry in flight)

First-round pilots put both knowledge-free reasoning tasks at their chance
floors (iGSM ~4% vs 4.3% chance; deduction ~44% vs 50%) — the 7%+5%
reasoning share at quarter budget was insufficient signal, exactly the
failure mode gate A exists to catch *before* battery compute is committed.
The spec-prescribed remediation was applied once: reasoning share raised
to 12%+8%, training difficulty band narrowed (op 2-6), corpus rebuilt, and
the dense/split pilot pair rerun.

**Retry verdict (evals completed 2026-07-20 ~10:00 PT).** The doubled
reasoning share did not unlock the tasks at pilot budget: dense iGSM 4.5%
(chance ~4.3%), deduction 45.1% (chance 50%); split 3.6% / 45.9%. Language
modeling clearly improved under the new mixture (final loss 2.60 -> 2.37)
and every mechanism measurement replicated (split: recall 99.8% ON / 0.0%
OFF, 0.0 fact-bits in weights, 100% fresh-entity lookups, fact-use QA
56.9% vs dense 27.1%) — the failure is specific: multi-step symbolic
computation does not emerge from a 20% in-mixture share at 0.8B tokens and
160M params, at least not at op>=2 / depth>=1 difficulty.

This is the gate system doing its job at pilot cost: roughly 8 GPU-hours
bought the finding before the ~550-GPU-hour battery was committed.

**Round three (difficulty floor; evals completed 2026-07-20 ~14:30 PT).**
Option (b) was executed same-day: iGSM floored to op 1-4 with 1/op-weighted
training mass and <=1 distractor on easy problems; deduction floored to
depth 1-2 with small fact/rule bases. The verdict is unambiguous: dense
iGSM 3.2% / split 5.6% (chance ~4.3%); deduction 53.7% / 53.1% (chance
50%). Even single-operation modular arithmetic does not become reliably
executable from a 20% in-mixture share at 0.8B tokens and 160M params —
consistent with the interpretation that mod-23 arithmetic *tables* (not
just multi-step composition) need drilling exposure far beyond one pass.
Meanwhile the mechanism results replicated a third time (split: 99.4%
recall ON / 0.0% OFF, 0.0 fact-bits, 99.6% fresh-entity lookups, fact-use
QA 58.1% vs dense 31.2%).

Three pilot rounds constitute a real, reportable secondary finding:
**in-mixture acquisition of knowledge-free symbolic reasoning does not
occur at pilot scale/budget**, across mixture shares (12-20%), difficulty
bands (op 1-8, depth 1-4), and curricula (uniform, low-weighted). The
remaining fork for the primary endpoint: (a) full-budget bet — the sweep's
4x tokens may drill the arithmetic through (measured for free either way,
since the reasoning components remain in the corpus); (c) the
preregistered fallback — re-anchor H1's primary on fact-use QA, which is
robustly above chance in every round and shows the largest, most stable
split-vs-dense separation (+25 to +38 points). Recommendation: adopt (c)
as primary with (a) as an emergence-watch secondary; freeze both in the
preregistration.

### 6.4 Honest ledger

The +2.3-point composite edge the split pilot showed over its dense twin
in round one is *not yet evidence* — both reasoning tasks sat at chance.
The claim-bearing numbers arrive with the sweep and the 1B confirmation,
under the frozen decision rule. What the gate phase has established is
that every instrument works: the intervention verifiably separates facts
from weights, the dose verifiably binds, the statistics and oracles are
tested, and the compute plan is measured rather than hoped.

## 7. Schedule

Gate-A resolution (section 6.3 decision) -> preregistration freeze ->
160M sweep (12 runs, ~1-2 days) + 1B dose calibration (2 short runs) ->
1B confirmation (4 chained runs, ~6 days across all four free GPUs) ->
analysis and month-end report with the measured delta or defensible null.
Land date ~Jul 28-30 on the free tier if the gate-A decision lands
2026-07-20/21; the difficulty-floor option (b) costs about one calendar
day, absorbed by CPU/GPU overlap.

## 8. The compute case

Everything above cost **$0 in cloud spend** — free-tier L40S time plus two
days of engineering. That freedom has a price: four concurrent GPUs, 2-day
walls, Sunday-night queue contention with other groups, and a 1B
confirmation that must run undertrained (10 tokens/param) with two seed
pairs to fit the calendar. Marginal dollars convert directly into
scientific strength:

| Tier | Spend | Buys | Converts |
|---|---|---|---|
| Burst | ~$0.5-1k | 8xA100 for 2-3 days | 1B confirmation in <2 days instead of 6, third seed pair restored — schedule risk retired |
| Rigor | ~$2-3k | compute-optimal 1B (20 t/p) x 3 seeds | removes the undertraining caveat reviewers will raise first |
| Scaling claim | ~$6-9k | +3B pair at top dose, natural-corpus replication arm | turns one scale point into a scaling trend + external-validity answer; publication-grade either direction |

The asymmetry that justifies the ask: the *question* is already paid for.
Design, instruments, corpora, statistics, and gates exist and are
validated; the hypothesis is one the field's leading small-model team bet
a product line on without measuring. Funding at this point buys certainty
per dollar at a rate that will not recur — every tier reuses the same
tested pipeline, and the null is as citable as the positive. For the
broader program, the same organizer infrastructure is the fast-memory
substrate the fast/slow architecture bet needs regardless of this
experiment's sign: the compute case and the platform case compound.

## References

LMLM 2505.15962 (v3, quotes verified against the HTML full text
2026-07-19); phi-3 2404.14219; To Memorize or to Retrieve 2604.00715; More
Room for Language 2404.10939; kNN-LM limits 2408.11815; Physics of LM 2.1
iGSM 2407.20311, 3.3 capacity 2404.05405; RETRO 2112.04426; Memory Layers
2412.09764; Engram 2601.07372; seed variance 2406.10229; error-bar
methodology 2411.00640; CLS: McClelland, McNaughton & O'Reilly 1995,
Kumaran, Hassabis & McClelland 2016; Tse et al. 2007. Full dossier:
docs/superpowers/research/2026-07-17-memory-split/.
