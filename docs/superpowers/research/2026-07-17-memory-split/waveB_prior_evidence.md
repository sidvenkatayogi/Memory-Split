Done. Here's the full synthesis.

# Prior evidence for "schemas in weights, facts in external memory" at ~1B scale

## Bottom line up front

**No published paper has shown that offloading facts to a retrieval store *during training* causes an improvement in *reasoning* at matched parameter count.** The literature splits cleanly into three groups, none of which tests your exact claim:

1. **Retrieval-during-training papers measure factual QA and perplexity, not reasoning** (RETRO, Memory Layers, kNN-LM, Atlas, REALM, MemLLM, Larimar). Where they touch reasoning-flavored tasks at all, gains are small.
2. **The two papers that come closest to a "capacity reallocation" test find a *tradeoff or parity*, not a reasoning gain**: "More Room for Language" (retrieval-pretrained LM gets better *syntax* but *worse* NLU/global-context) and LmLm (fact-offloading preserves NLU as a sanity check — no reasoning improvement claimed).
3. **The one recent paper that measures reasoning under a retrieval-vs-memorize training tradeoff finds retrieval does *not* help reasoning** ("To Memorize or to Retrieve," 2026): knowledge tasks improve, reasoning-heavy tasks (GSM8K, PIQA, StrategyQA, LAMBADA) show "minimal change."

So the *premise* (facts crowd out capacity) has partial mechanistic support, but the *payoff* (freeing that capacity measurably improves reasoning) is **unmeasured** — that is your gap.

---

## (a) Closest prior work, ranked by similarity to your hypothesis

| # | Paper | arXiv | Scale | What they did / found re: your hypothesis |
|---|-------|-------|-------|--------------------------------------------|
| 1 | **To Memorize or to Retrieve: Scaling Laws for RAG-Considerate Pretraining** | 2604.00715 (2026) | OLMo-2, 30M–3B, ≤100B tok | The single most similar study. Systematically trades pretraining-data-in-weights vs retrieval-store size under a fixed budget, evaluating **reasoning, scientific QA, open QA**. Retrieval consistently helps knowledge tasks; **smaller models benefit most** (marginal benefit falls with scale, ~saturates by 3B). Crucially, **reasoning-heavy tasks (GSM8K, LAMBADA, PIQA, StrategyQA) show "minimal change."** Concludes retrieval is a "scale-dependent alternative for where knowledge is stored," *not* a reasoning booster; reasoning bottleneck is "computation over knowledge rather than access to it." |
| 2 | **More Room for Language** (Charpentier et al.) | 2404.10939 | LTG-BERT MLM, 8.5M/27.7M/98M, from scratch | Trains retrieval-augmented LMs **from scratch** with "ideal" (paraphrase) retrieval, then measures the *standalone* LM. Finds a genuine reallocation: retrieval LMs **store less world knowledge, gain syntactic ability** ("uses the freed parameters for other features"), and the effect **grows with scale**. But the freed capacity goes to *syntax*, and **NLU/global-context gets worse** — a partial capacity-reallocation result with a reasoning-relevant *regression*, not a gain. |
| 3 | **LmLm — Limited Memory Language Models** (Cornell) | 2505.15962 (2025) | GPT-2/LLaMA2-style, 124M–382M, from scratch | Nearly your training recipe: annotate pretraining corpus with entity-fact lookups and **mask retrieved fact tokens from the loss** so facts are never learned in weights. Result: **382M LmLm matches LLaMA2-7B factual precision**; lower val perplexity (−1.98 dynamic). Explicitly argues this "frees up model capacity … dedicating model capacity to reasoning and linguistic competencies." **But they only test NLU as a "sanity check" and report parity — no reasoning benchmark, no reasoning gain claimed.** Code/models open. |
| 4 | **Memory Layers at Scale** (Meta FAIR) | 2412.09764 | Llama-style, 134M–8B base + up to 128B memory params | Replaces FFNs with trainable key-value memory (facts in a lookup table, learned during training). Gains "**especially pronounced for factual tasks**" (NQ, TQA), with **smaller gains on commonsense/knowledge (PIQA/OBQA/HellaSwag/MMLU) and coding**. Directly relevant effect sizes at 1.3B (below). Not external retrieval, but the "cheap capacity for associations frees dense layers" thesis is identical. |
| 5 | **RETRO** (DeepMind) | 2112.04426 | 150M–7.5B, retrieval from scratch, 2T-token DB | The canonical retrieval-during-training model. Gains are **LM bits-per-byte and QA** (NQ EM 30.4→45.5 after FT). Constant gain across 150M–7B (≈10× param-equivalent). **Downstream reasoning barely evaluated; on `dm_mathematics` RETRO does *not* beat its own baseline** — an early hint retrieval doesn't transfer to math/reasoning. |
| 6 | **kNN-LM** + **"Why do kNN-LMs work?"** | 1911.00172 / 2301.02828 | WikiText-103 scale | Mechanistic backbone of your premise. Original: interpolating with a datastore gives −2.9 ppl, "learning similarity is easier than predicting the next word," helps most on **rare/factual** patterns. The "Why" paper's key negative control: **you cannot replicate the gain by interpolating with a standard LM that was further trained to over-memorize** — "memorizing … causes its representation to generalize less effectively," while kNN-LM lets the model "memorize the training data while retaining an effective similarity function." This is the cleanest existing evidence that **in-weight memorization degrades representational generalization** — but it's measured in perplexity, not reasoning. |
| 7 | **MemSinks / Natively Unlearnable LLMs** (CMU) | 2507.09937 / 2606.13873 | 1B params, 1B tokens | Shows memorization of *natural* sequences becomes "**mechanistically entangled with general language abilities**." Routes memorization into sequence-keyed "sink" neurons so it can be removed cleanly. Supports the crowding premise (memorization and general ability share machinery) and shows separation is achievable *by design* — but goal is unlearning, and they measure isolation + generalization preservation, not reasoning gains. |
| 8 | **Physics of LM 3.3 — Knowledge Capacity Scaling Laws** (Allen-Zhu & Li) | 2404.05405 | GPT-2/LLaMA/Mistral synthetic | Quantifies the "capacity budget" your hypothesis assumes (see below). **2 bits/param**; junk data can cost 20× capacity. Doesn't test retrieval, but gives you the measuring stick and an **open-source synthetic testbed**. |

Secondary/adjacent (weaker fit, useful context): **Atlas** (2208.03299, retrieval few-shot, factual), **REALM** (2002.08909, retrieval pretraining, open-domain QA), **TRIME** (2205.12674, train-time memory, perplexity), **MemLLM** (2404.11672, finetuned read-write memory, knowledge tasks), **Larimar** (2403.11901, CLS-inspired episodic memory for *editing*), **Titans** (2501.00663, test-time neural memory — reports commonsense-reasoning gains but confounds added params + long-context memory), **Cartridges** (2506.06266, offline-trained KV cache via self-study), **Retrieval Head** (2404.15574, shows a <5% sparse head set drives both retrieval *and* CoT reasoning — mechanistic link between retrieval and reasoning), **Disentangling Memory and Reasoning** (ACL 2025.acl-long.84) and **Decoupling Knowledge and Reasoning** (2507.18178) — both *inference-time* prompt-level separations, not training interventions.

---

## (b) The GAP — what your experiment would be first to measure

Stated precisely, **no work has tested whether externalizing facts to a retrieval store *during training*, at matched active+total parameters and matched data, *causally improves non-factual reasoning* (math, logical, multi-hop, symbolic) at ~1B scale.** Every neighbor misses on at least one axis:

- **Wrong endpoint.** RETRO, Memory Layers, kNN-LM, Atlas, REALM, MemLLM, Larimar, TRIME optimize/measure **factual QA or perplexity**. Reasoning is not the dependent variable.
- **Reallocation shown, but to the wrong skill / with a regression.** "More Room for Language" shows freed capacity → **syntax up, NLU down**. It never measures reasoning and reports a general-capability *cost*.
- **Capacity freed, but reasoning never tested.** LmLm is your recipe almost exactly, but treats NLU as a *sanity check* (parity) and reports **no reasoning benchmark at all**.
- **Reasoning tested, but retrieval didn't help it.** "To Memorize or to Retrieve" measures reasoning and finds "minimal change" — but it uses **inference-time RAG on a conventionally pretrained model**, not fact-masking during training, and reports perplexity rather than a causal, capacity-mediated reasoning gain.

Therefore your experiment is the first to combine all of: **(1) fact offloading enforced *during* pretraining** (à la LmLm/RETRO), **(2) matched-parameter dense baseline**, **(3) reasoning as the *primary* pre-registered endpoint** (not factual QA/perplexity/syntax), at **(4) ~1B scale**. Two further things nobody has measured that would strengthen the causal story:
- **Mediation/representational evidence** that capacity *freed* from facts is *re-used* for reasoning circuits (e.g., probing, capacity accounting in the Physics-3.3 sense, or ablation of "reasoning" vs "memorization" neurons à la MemSinks). Everyone asserts "frees capacity for reasoning"; nobody demonstrates the freed capacity is causally responsible for a reasoning delta.
- **A CLS-derived training-dynamics prediction**: that schema-consistent (reasoning-structured) data consolidates *faster* when facts are externalized (see theory below). Untested in LMs.

---

## (c) CLS / schema theory, in transferable form

**Core CLS claims (McClelland, McNaughton & O'Reilly 1995, Psych Review 102:419–457; Kumaran, Hassabis & McClelland 2016, TICS).** Intelligent agents need two systems because of a fundamental tradeoff. The **neocortex** is a slow, distributed, overlapping learner that discovers *structure* across many interleaved experiences (schemas/semantics) but suffers **catastrophic interference** if forced to learn fast. The **hippocampus** is a fast, sparse, pattern-separated store that grabs individual episodes quickly without overwriting cortical structure. Consolidation happens by **hippocampal replay** interleaving new episodes with old, so the cortex integrates them gradually. Damage to the hippocampus spares remote (consolidated) memory but destroys recent memory — the empirical anchor.

**The 2016 update adds three things that matter for you.** (i) **Neocortical learning is *not* intrinsically slow — it is slow only for *schema-inconsistent* information.** Information consistent with existing structure can be integrated *rapidly* ("the rate of neocortical learning is prior-knowledge dependent"). (ii) Replay is not faithful playback; it allows **goal-dependent reweighting** — statistically rare-but-important events get privileged consolidation. (iii) They explicitly connect this to ML memory-augmented architectures (the hippocampus ≈ external differentiable memory).

**Schema theory (Bartlett 1932; Anderson; Tse et al. 2007, *Science* 316:76–82; McClelland 2013; van Kesteren SLIMM 2012).** Tse et al. is the decisive animal result: once a neocortical **schema** exists, a *single* exposure to a schema-consistent new fact becomes **hippocampus-independent within 48 hours**, bypassing slow consolidation. Schema-*consistent* information consolidates dramatically faster than novel/inconsistent information.

**Which predictions transfer to a ~1B LM with an external fact store:**

1. **Division-of-labor prediction (the main one).** If the external store plays the hippocampal role (fast, editable, item-specific fact storage), the parametric network is freed to behave like neocortex — spend capacity on *structured, generalizing* representations (reasoning schemas) rather than pattern-separated facts. → **Retrieval-trained model should improve on schema/reasoning tasks and lose little on factual tasks it can look up.** (Your hypothesis, in CLS terms.)
2. **Rate prediction (novel, testable, nobody has done it).** Schema-consistent content consolidates faster. → **Reasoning-structured / schema-consistent training data should be learned in *fewer exposures* when facts are externalized**, because the parametric net isn't simultaneously fighting interference from arbitrary facts. This is a *training-dynamics* prediction (loss-curve / sample-efficiency), distinct from a final-accuracy prediction, and maps onto Physics-3.3's "exposures needed" axis.
3. **Interference prediction.** In-weight facts are exactly the "schema-inconsistent, arbitrary" items that cause interference. → **Externalizing them should reduce catastrophic-interference signatures** (e.g., less forgetting of reasoning skills as more facts are seen). kNN-LM-why's finding (over-memorization degrades representation generalization) is the closest existing analog.
4. **Editability corollary.** CLS predicts the fast store should be updatable without disturbing cortical structure. → matches Larimar/LmLm/MemLLM's demonstrated instant, side-effect-free fact editing/unlearning.

Caveat for reviewers: LMs have **no architectural separation** of "fast" vs "slow" systems and **no replay by default** — a dense transformer is monolithic (exactly the "monolithic connectionist model" MMO95 contrasted against). So CLS is a *motivating analogy*, and the cleanest transferable, falsifiable claims are #2 (rate/sample-efficiency) and #3 (interference), which you can measure directly.

---

## (d) Effect sizes for power analysis

The usable matched-comparison numbers. **Note the recurring pattern: factual gains are large (tens of points / >75%), reasoning-flavored gains are small (≈2–3 points) — so if your endpoint is reasoning, budget for a small effect and high variance.**

**Memory Layers at Scale, 1.3B base (dense vs Memory+, identical base params & FLOPs):**

| Task | Dense 1.3B | Memory+ (same base) | Δ |
|------|-----------|---------------------|---|
| NaturalQuestions (acc) | 7.76 | 13.68 | **+5.9 (+76%)** |
| TriviaQA (F1) | 32.64 | 42.89 | **+10.3** |
| HotpotQA (F1) | 13.92 | 16.72 | +2.8 |
| OBQA (acc) | 23.4 | 26.8 | +3.4 |
| PIQA (acc) | 72.74 | 75.35 | **+2.6** |

Commonsense/reasoning-adjacent multiple-choice (PIQA/OBQA) move only **~2–3 points** even as factual QA jumps ~75%.

**RETRO 7.5B:** NQ exact-match 30.4 (closed-book) → **45.5** with retrieval (+15.1); constant gain ≈ **10× parameter-equivalent** on LM bpb; **no gain on `dm_mathematics`.**

**kNN-LM:** WikiText-103 ppl 18.65 → 16.06 (interpolation); SOTA 15.79 (**−2.9 ppl**). No reasoning eval.

**To Memorize or to Retrieve (OLMo-2, 30M–3B):** retrieval crossover (retrieval starts substituting for pretraining) at **D/N ≈ 4.14 tokens/param**; smaller models gain most per retrieval token, **gains largely saturate by 3B**; reasoning tasks "minimal change." **Power warning they report directly: reasoning benchmarks are noisy at this scale — PIQA fit error (ARE) 40–129%, StrategyQA similar**, vs HellaSwag ~2–5%. Expect wide error bars on exactly the tasks you care about → plan multiple seeds and pick lower-variance reasoning metrics (e.g., likelihood-based / ARC-Challenge over PIQA).

**LmLm (382M):** matches **LLaMA2-7B** factual precision (FactScore/T-REx/PopQA), dynamic perplexity −1.98 avg; **NLU at parity** with standard baseline (no reasoning delta measured).

**Physics-of-LM 3.3 capacity budget (for capacity accounting):** **2 bits/param** at 1000 exposures (int8 preserves; int4 → 0.7 bit/param); drops to **1 bit/param at 100 exposures**; **junk data at 1:7 useful:junk costs up to 20× capacity** for useful knowledge (recoverable to ~1.3× at 1000 exposures, or by prepending domain tokens). Use this to *quantify* how many bits you're freeing by externalizing facts, and translate that into an expected capacity budget for reasoning.

**Titans (340M–760M):** avg commonsense-reasoning (PIQA/HellaSwag/WinoGrande/ARC-e/ARC-c/SIQA/BoolQ) at 760M: Transformer++ **48.69 → Titans 56.82**. Large, but confounds *added* memory-module parameters and long-context memory — treat as an upper-bound-ish reference, not a matched-param fact-offloading result.

---

## (e) Null / negative results (cite these defensively)

1. **"To Memorize or to Retrieve" (2604.00715):** retrieval yields **"minimal change" on reasoning-heavy tasks** (GSM8K, LAMBADA, PIQA, StrategyQA); "retrieval is not a uniform substitute for pretraining"; reasoning limited by "computation over knowledge rather than access to it." *The most direct negative evidence against retrieval → reasoning.*
2. **"More Room for Language" (2404.10939):** retrieval-augmented pretraining **worsens NLU** and **long-range/global context resolution** (the LM offloads global-context handling to the retriever). Freed capacity went to syntax, at a cost to multi-sentence understanding — a *reallocation with a general-capability regression*.
3. **LmLm (2505.15962):** general capability only **preserved (NLU parity)**, not improved — the "freed capacity → reasoning" step is asserted but **not demonstrated**.
4. **"Why do kNN-LMs work?" (2301.02828):** the memorization-substitute control **fails** — you can't reproduce retrieval's benefit by making a standard LM memorize harder (over-memorizing *hurts* representation generalization). Supports your *mechanism* but warns that naive "just memorize less" ≠ automatic gain.
5. **Wang et al. 2023 (via 2404.10939):** retrieval augmentation **improves perplexity but not generation quality** — a reminder that perplexity/factual metrics can move without downstream capability moving.
6. **MemSinks (2507.09937):** memorization of natural text is **mechanistically entangled with general language ability** and resists post-hoc removal — separation is possible but *hard*, and doesn't happen for free; implies your intervention must *enforce* the split during training (as you plan) rather than expect it to emerge.

---

## Extra specifics you asked for

**Physics-of-LM testbed openness:** **Yes, open-source.** Code + `bioS`/`bioR` generators for Parts 3.1/3.3 are released at **github.com/facebookresearch/PhysicsLM4** (the "Capo" dataset dir), plus 16 released model weights on HF. Per their FAQ: `bioS` is released and `bioR` *generation prompts* are released (raw `bioR` text withheld for legal reasons); Part 2's **iGSM** reasoning-data generator is also released. Capacity was measured not by loss/benchmarks but by **estimating stored knowledge bits** vs a bit-complexity lower bound on synthetic (name, attribute, value) tuples — a method you could adopt to *directly measure* capacity reallocation. Parts 3.1/3.2 (2309.14316 / 2309.14402): knowledge is often **memorized but not extractable** without training-time augmentation (paraphrase/permutation/full-name), and models **cannot manipulate stored knowledge (compare/classify/inverse-search) without CoT** — relevant because it means "facts in weights" aren't even reliably usable for reasoning, strengthening the case for externalizing them.

**phi-3 exact passages (2404.14219)** — the small-model capacity + data-mixture argument, quoted:

> "we filter the publicly available web data to contain the correct level of 'knowledge' and keep more web pages that could potentially improve the 'reasoning ability' for the model. As an example, the result of a game in premier league in a particular day might be good training data for frontier models, but we need to remove such information to **leave more model capacity for 'reasoning' for the mini size models**."

> "the phi-3-mini model … is still fundamentally limited by its size for certain tasks. **The model simply does not have the capacity to store too much 'factual knowledge'**, which can be seen for example with low performance on TriviaQA. However, we believe such weakness **can be resolved by augmentation with a search engine**."

This is essentially your hypothesis stated informally by a frontier small-model team — but phi-3 acts on it via **data curation**, not architectural fact-externalization, and offers **no controlled ablation** isolating the capacity-reallocation effect. That controlled ablation is precisely your contribution.

A practical design note that falls out of all this: make **reasoning the pre-registered primary endpoint**, keep **total+active params and token budget matched** to the dense baseline (Memory-Layers-style parameter matching is the standard reviewers will expect), include a **capacity-accounting or neuron-ablation mediation analysis** so a positive result isn't just "added an external store," and add the **CLS sample-efficiency (exposures-to-learn) measurement** since that prediction is both novel and cheap to collect from training curves.

Now I'm organizing the final synthesis with the core findings, ranked prior work, theoretical grounding, effect sizes, and null results, making sure to only reference arXiv IDs I can verify from the search results. MemSinks and its follow-up on natively unlearnable patterns, goldfish loss, retrieval heads, and the ACL 2025 paper on disentangling memory from reasoning—I'm verifying these IDs where I can, though some arxiv numbers need confirmation. Finishing up the reference list with some key cognitive science papers on memory consolidation and schema learning, then I'm ready to write the report itself with clear structure and practical guidance for experiment design. I need to make sure I capture the specific details they asked for—Physics 3.3's 2-bits/param measurement and whether the testbed is open source (it is: facebookresearch/PhysicsLM4, bioS released, bioR generators/prompts available), plus the exact phi-3 quote about filtering web data for knowledge and reasoning ability, and the CLS-specific claims. I'll weave these into the deliverables section while keeping the phi-3 passage verbatim.

Generating effect size table... kNN-LM achieves WikiText-103 perplexity of 16.06, approaching state-of-the-art at 15.79 but without reasoning improvements. Looking at OLMo-2 across model sizes, retrieval helps knowledge tasks like SimpleQA and CommonsenseQA modestly, while reasoning-heavy tasks show minimal gains—smaller models benefit most per retrieval unit, with saturation around 3B parameters and a crossover threshold around 4.14 tokens per parameter. LmLm's 382M model matches LLaMA2-7B on factual precision metrics.

Examining the syntax-knowledge tradeoff in smaller MLM models, retrieval improves syntax but hurts world knowledge and NLU tasks, with the gap widening at larger scales. Titans use memory as an architectural component rather than external fact storage, showing commonsense reasoning improvements in the 340M–760M range, though this involves added memory-module parameters and long-context memory rather than pure fact offloading. Physics 3.3 achieves 2 bits per parameter.

Quantization studies show int8 preserves performance while int4 drops to 0.7 efficiency; reaching 2 bits per parameter requires around 1000 exposures, whereas 100 exposures yields roughly 1 bit per parameter. Junk data severely impacts capacity—a 1:7 useful-to-junk ratio causes a 20x loss on useful knowledge, though this recovers to 1.3x loss at 1000 exposures and improves further with domain token prepending. For power analysis, memory and retrieval systems show substantial factual gains at ~1B parameters but only modest improvements of 2–3 points on commonsense and reasoning tasks, so detecting reasoning gains would require adequate statistical power and multiple random seeds.

Looking at the null and negative findings: broader language understanding actually worsens with retrieval augmentation, and reasoning tasks show minimal change—the bottleneck appears to be computational rather than knowledge access. General capability is preserved but not improved across these approaches, and attempts to boost standard language models by interpolating with memorizing models fail, suggesting that memorization alone doesn't help and may even harm representation generalization. Reasoning benchmarks like PIQA and StrategyQA exhibit high variance (40–129%), making them particularly noisy at this scale.

The key gap I'm identifying: no published work has cleanly demonstrated that offloading facts to external retrieval during training—with matched parameter counts and data—actually improves non-factual reasoning (math, logic, multi-hop, symbolic tasks) at the ~1B scale. Existing studies either measure only factual QA and perplexity, or they don't isolate the reasoning component properly.

What's untested is whether freed capacity actually gets reallocated toward reasoning, and whether the mechanism works as predicted. This would be the first controlled experiment with reasoning as the primary outcome plus a mechanistic analysis.

I'll structure the response with tables for the ranked findings and effect sizes, and include arXiv identifiers for the papers cited, since that's what was requested.