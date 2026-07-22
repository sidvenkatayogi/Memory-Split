# Training-Time Retrieval & Memory Architectures for a ~1B "Facts-External" Experiment

**Scope.** Architectures where the model can lean on an external store *during training* (not just inference-time RAG), assessed at 100M–1.5B scale for a 1-month, modest-budget project whose deliverable is a measured delta vs. a matched dense baseline.

**TL;DR recommendation.** The single most relevant discovery: your hypothesis has already been partially operationalized by **LMLM — "Pre-training Large Memory Language Models with Internal and External Knowledge" (arXiv 2505.15962, Cornell/Kilian Weinberger's group, ICLR 2026)** — GPT-2/Llama-style 176M and 382M models pretrained from scratch with an external triplet database, fact tokens masked from the loss, total cost ~8 H100-days, code released. It shows factual-precision gains and faster convergence but does **not** measure a reasoning delta — which is exactly the gap your project could fill. The cheapest credible plan is: LMLM-style recipe (or TRIME) as the treatment arm, dense baseline at matched params/tokens, on a nanochat/nanoGPT-class or OLMo-core stack; RETRO is the heavyweight alternative (+12–26% GPU-hours, deprecated reference code); memory layers (Meta, open code) are a strong third arm that tests "separated storage" *inside* the weights, with DeepSeek's Engram (Jan 2026) providing the strongest published evidence that offloading static knowledge frees capacity for reasoning.

---

## 1. RETRO family (chunked cross-attention into a corpus-scale store)

### 1.1 RETRO (DeepMind, 2021)

**(a) How it works.** The training corpus is split into 64-token chunks, each embedded with a *frozen* BERT. For every input chunk, the k nearest neighbor chunks (plus their 64-token continuations) are retrieved from the database, encoded by a small (~2-layer, ~19M-param) bidirectional encoder, and injected into the decoder via **chunked cross-attention (CCA)** placed in every third layer (layers 6, 9, 12 for the smallest model). Causality is preserved by letting chunk *u* attend only to the neighbors of chunk *u−1*. Because the retriever is frozen, all neighbors are **precomputed offline** and stored with the training data — retrieval never runs in the training loop. With a 2T-token database, a 7.5B RETRO matched GPT-3 175B on much of the Pile ("25× fewer parameters"); gains grow monotonically with database size up to 2T tokens.

**(b) Open-source implementations.**
- **No official DeepMind release** (confirmed in the RETRO++ paper).
- [lucidrains/RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch) (~880 stars): full model + `TrainingWrapper` that chunks a text folder, embeds with BERT, builds a faiss index via autofaiss, and precomputes kNNs. **Maturity: research toy** — single-GPU oriented, no distributed training, last release Oct 2023, deviates from paper (rotary embeddings, faiss instead of SCaNN). Fine for a 100M-scale prototype; you'd bolt it onto your own trainer for 1B.
- **NVIDIA Megatron-LM `tools/retro`**: the production implementation used for RETRO++/InstructRetro (full pipeline: chunk DB, BERT embedding, faiss index build, neighbor precompute, pretraining). **Critical 2026 flag: RETRO was fully deleted from Megatron-LM main on Jan 28, 2026 ([PR #3001](https://github.com/NVIDIA/Megatron-LM/pull/3001), −10,582 lines) as deprecated/unsupported.** You must pin a pre-2026 tag or the [`InstructRetro` branch](https://github.com/NVIDIA/Megatron-LM/tree/InstructRetro/tools/retro). NeMo's RETRO is likewise "legacy" in current docs. Maturity: was solid, now frozen/abandoned — expect dependency rot.

**(c) Cost vs. dense at ~1B.** Best public numbers (RETRO++ paper, A100 DGX-2H, 330B training tokens):

| Size | GPT | RETRO | Overhead |
|---|---|---|---|
| 148M | 1,240 GPU-h | 1,560 GPU-h | +25.8% |
| 410M | 3,600 GPU-h | 4,480 GPU-h | +24.4% |
| 1.5B | 12,000 GPU-h | 13,440 GPU-h | +12.0% |

InstructRetro reported only **+2.58% GPU-hours** when continuing to pretrain a 43B model with retrieval (encoder/CCA are small relative to a big decoder). Index build at their scale: 330B tokens → 5.3B chunks; faiss IVF(2²² centroids)+OPQ/PQ64+HNSW; ~4 ms/query batched on one DGX-2H with <400 GB RAM. **My estimate for your 10–50B-token corpus** (anchored to those numbers): 156M–780M chunks; BERT-class embedding ≈ 10–70 GPU-hours; index train/add: hours; neighbor precompute at 4 ms/chunk ≈ 170–870 node-hours on CPU faiss, cut ~5–10× with GPU faiss and trivially parallelizable. Realistically **1–4 days on one 8-GPU node for 10B tokens** — feasible but a real engineering line-item.

**(d) Hypothesis fit.** Good in principle: the model sees the neighbors *throughout training*, so the loss never pressures it to memorize what retrieval supplies. Two caveats. First, RETRO's own leakage analysis shows much of the gain comes from near-duplicate copying — you must dedupe train/eval overlap (they used 13-gram Jaccard filtering) or your "delta" is contaminated. Second, **InstructRetro's surprise finding: after retrieval-augmented pretraining you can ablate the encoder entirely and the bare decoder performs comparably on downstream tasks** — evidence that retrieval-pretraining acts partly as a *regularizer/curriculum* rather than a persistent external dependence. That is actually an interesting sub-hypothesis for you (does the benefit persist without the store?), but it complicates the "facts live outside" story.

**(e) Papers.** RETRO: arXiv **2112.04426** (Borgeaud et al., ICML 2022). RETRO++ / "Shall We Pretrain Autoregressive LMs with Retrieval?": arXiv **2304.06762** (Wang et al., EMNLP 2023) — reproduces RETRO at 148M/410M/1.5B/9.5B, exactly your scale band. InstructRetro: arXiv **2310.07713** (Wang et al., ICML 2024).

---

## 2. kNN-LM and training-time variants

### 2.1 Vanilla kNN-LM — and why it does *not* test your hypothesis

**(a)** Train a normal dense LM; afterward, run it over a corpus storing (hidden state → next token) pairs; at inference interpolate the parametric softmax with a kNN distribution over the datastore. **No training change whatsoever** — the model experienced full memorization pressure during training, so it cannot tell you whether externalizing facts frees capacity. Use it only as an *inference-time control arm*.

**(b)** Code: [neulab/knn-transformers](https://github.com/neulab/knn-transformers) (HF-based, cleanest; functional, not actively developed), original [urvashik/knnlm](https://github.com/urvashik/knnlm) (fairseq, stale). **(c)** No training cost; datastore is per-*token* (103M entries ≈ 200 GB of uncompressed keys for WikiText-103), so 10–50B tokens is multi-TB without PQ — and inference runs ~6× slower (50 vs 300 tok/s in TRIME's measurements). **(d)** Poor fit, and worse: **"Great Memory, Shallow Reasoning: Limits of kNN-LMs" (arXiv 2408.11815, NAACL 2025)** shows kNN-LM improves memory-intensive tasks but *degrades* reasoning (multi-hop, math), even with oracle retrieval. This is a directly relevant negative result: post-hoc memory doesn't buy reasoning, sharpening the case that the interesting question is training-time. **(e)** kNN-LM: arXiv **1911.00172** (Khandelwal et al., ICLR 2020); limits paper: arXiv **2408.11815**. Related inference-time datastore scaling: SILO arXiv **2308.04430**; MassiveDS/trillion-token datastore arXiv **2407.12854**.

### 2.2 TRIME — the cheap training-time variant (best effort/insight ratio in this family)

**(a)** TRIME ("Training Language Models with Memory Augmentation", Zhong, Lei & Chen, EMNLP 2022) replaces the LM objective with a contrastive one where the target's probability mass can come from output embeddings *or* from **in-batch token memories** (aligned hidden states of other positions/segments). Data batching is engineered so in-batch memory approximates the eventual test-time memory: **BM25-packed batches** of lexically similar segments stand in for external corpus memory. At test time you plug in local/long-term/external memories kNN-LM-style — but now the model was *trained to use them*.

**(b)** Code: [princeton-nlp/TRIME](https://github.com/princeton-nlp/TRIME) — fairseq-based, 247M-scale, unmaintained since ~2022. **Maturity: stale research code, but the method is two portable components** (a loss function + a batch sampler), reimplementable in a modern trainer in days, which is its real appeal.

**(c)** Cost: authors report **negligible training overhead** vs. standard LM training (no index in the training loop; BM25 batching is a cheap preprocessing pass). This is the lowest-cost training-time memory method that exists.

**(d)** Hypothesis fit: decent. During training the model can push probability onto memory tokens instead of memorizing them in weights — a soft version of "lean on the store". Results: 247M on WikiText-103, ppl 18.70 → 17.76 with in-batch memory alone, → **15.37/15.51** with long-term + external memory. Caveats: memory is *the training corpus itself* (not a curated fact store), the mechanism is output-distribution interpolation (shallow integration vs. RETRO's cross-attention), and everything is at 247M/WikiText scale — nobody has shown it at 1B on modern data. That's a gap you could exploit, but it makes the method less de-risked.

**(e)** TRIME: arXiv **2205.12674**. Related: NPM ("Nonparametric Masked Language Modeling", arXiv **2212.01349**, [facebookresearch/NPM](https://github.com/facebookresearch/NPM)) — trains an encoder-only model whose *entire* output space is retrieval over corpus phrases (fully external facts, but masked-LM/encoder-only, so a poor substrate for measuring generative reasoning). Memorizing Transformers (arXiv **2203.08913**, Wu et al., ICLR 2022) trains with kNN attention into stored *past activations* — training-time memory, but over the model's own long context rather than an external fact corpus ([lucidrains reimplementation](https://github.com/lucidrains/memorizing-transformers-pytorch)). Newer 2025 twist: **Memory Decoder** (arXiv **2508.09874**, NeurIPS 2025, [LUMIA-Group/MemoryDecoder](https://github.com/LUMIA-Group/MemoryDecoder)) pretrains a small decoder to *imitate* a kNN retriever's distributions and plugs into any same-tokenizer LM — clever, but it's domain adaptation for existing LLMs, not a from-scratch training-time store.

---

## 3. Memory layers (trainable sparse key-value memory *inside* the model)

**(a) How they work.** A product-key memory (PKM, Lample et al. 2019) replaces an FFN: the input produces a query, which is split in two and matched against two sets of √N "half-keys"; the Cartesian product gives N≈millions of addressable slots with only O(√N) key comparisons; the output is a weighted sum (EmbeddingBag) of the top-k *value embeddings*. Keys and values are ordinary trainable parameters updated sparsely. Meta's **Memory Layers at Scale** modernizes this: 3 memory layers sharing one value pool, swilu nonlinearity, custom CUDA kernels (3 TB/s vs <0.4 TB/s stock PyTorch), value pools sharded across GPUs. **PEER** pushes the same product-key routing to ~1M *single-neuron experts* instead of static value vectors. **UltraMem** (ByteDance) and **Engram** (DeepSeek) are 2025/2026 successors focused on inference efficiency and scale.

**(b) Implementations.**
- [facebookresearch/memory](https://github.com/facebookresearch/memory) — **official** Memory Layers at Scale reference, built on Meta Lingua (a clean, small-scale pretraining codebase). Maturity: research-grade but complete and genuinely usable at your scale; the paper's own scaling grid starts at **134M/373M/720m/1.3B base models** — exactly your band.
- PEER: **no official DeepMind code**; [lucidrains/PEER-pytorch](https://github.com/lucidrains/PEER-pytorch) (layer only) and [huyphan168/PEER](https://github.com/huyphan168/PEER) (partial training repo) are unofficial.
- UltraMem (arXiv 2411.12364, ICLR 2025): no official code, unofficial reimplementations only.
- **Engram: official code at [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)** (released Jan 2026).

**(c) Cost vs. dense at ~1B.** FLOPs overhead ≈ **zero** (lookups are memory-bandwidth-bound, not compute); the price is parameter/optimizer memory (a 1M-slot × 2048-dim value pool is ~2B extra params + Adam states — shard it or shrink it) and some wallclock overhead unless you use their kernels. No index to build, no retrieval infra at all — operationally the cheapest "memory" option after TRIME. Meta's headline: a **1.3B model + 128B memory parameters approaches Llama2-7B on factual QA (NQ/TQA), i.e. matches a model trained with ~10× the FLOPs**; memory models beat dense models given >2× their compute; gains concentrate on factual tasks.

**(d) Hypothesis fit — important conceptual distinction.** Memory layers do **not** externalize facts to a corpus: values are learned parameters, trained by the same gradient stream, and the model *can't* consult ground-truth text — so they don't remove memorization; they *relocate* it into a dedicated, sparsely-accessed, cheap-to-scale organ. That makes them a different (and complementary) operationalization of CLS: "separate storage system inside the brain" rather than "facts outside the organism". Two strong reasons to still include a memory-layer arm: (i) it cleanly tests whether *architectural separation of storage from computation* alone frees the dense backbone for reasoning at matched FLOPs; (ii) **DeepSeek's Engram (arXiv 2601.07372, Jan 12, 2026, with code)** provides the best current evidence for exactly your mechanism at 27B-MoE scale: adding O(1) hashed N-gram conditional memory under an iso-parameter, iso-FLOP budget improved *reasoning* (BBH +5.0, ARC-C +3.7, HumanEval +3.0) **more than knowledge** (MMLU +3.4), with mechanistic analysis showing memory "relieves early layers from static reconstruction, effectively deepening the network" — plus a "sparsity allocation law" (~20–25% of sparse budget to memory). Also relevant: **sparse memory finetuning** (Lin et al., arXiv **2510.15103**, Oct 2025) shows memory-layer slots localize knowledge well enough that updating only hot slots learns new facts with 11% forgetting vs 89% for full finetuning — supporting the "facts are separable" premise.

**(e) Papers.** Memory Layers at Scale: arXiv **2412.09764** (Berges, Oğuz et al., Meta FAIR). PKM: arXiv **1907.05242** (Lample et al.). PEER / Mixture of A Million Experts: arXiv **2407.04153** (He, DeepMind). UltraMem: arXiv **2411.12364** (ICLR 2025). Engram: arXiv **2601.07372** (DeepSeek + PKU, Jan 2026). Sparse memory finetuning: arXiv **2510.15103**.

---

## 4. Simplest engineering path: retrieval-into-context during pretraining

### 4.1 REALM and Atlas (the classics — probably not your tools)

**REALM** (arXiv **2002.08909**, Guu et al., ICML 2020): BERT-style masked LM that marginalizes over top-k retrieved Wikipedia docs; retriever trained end-to-end with **asynchronous index refresh** (periodically re-embed the whole corpus). Encoder-only, QA-focused, TensorFlow code in google-research/language — historically foundational, practically obsolete for your purpose. **Atlas** (arXiv **2208.03299**, Izacard et al., JMLR 2023): T5 encoder-decoder (770M–11B) + Contriever + Fusion-in-Decoder, jointly pretrained with retrieval; the honest cost accounting is useful: **refreshing a 37M-doc index every 1000 steps adds ~30% overhead**, avoidable via query-side-only retriever finetuning (~0 overhead). Code [facebookresearch/atlas](https://github.com/facebookresearch/atlas) is complete (pretraining, indices, checkpoints, PQ-compressed indexes that run on 2 GPUs) but stale (2022–23) and encoder-decoder — awkward for decoder-only reasoning benchmarks. The transferable lesson from both: **a trainable retriever is where the complexity explodes; freeze your retriever** (as RETRO did) and the whole problem becomes offline preprocessing.

### 4.2 In-Context Pretraining (data-ordering only — zero architecture change)

**(a)** ICP (Shi et al., arXiv **2310.10638**, ICLR 2024) keeps the standard training loop and simply **reorders documents** so each context window is a chain of semantically related docs: embed every doc with Contriever, faiss kNN (k=10), then a greedy max-TSP traversal chains related docs without repetition. The "retrieval" happens once, offline, in the data pipeline.

**(b)** Code: [swj0419/in-context-pretraining](https://github.com/swj0419/in-context-pretraining) (research-grade sorting pipeline). Since it's pure data prep, it composes with *any* trainer — maturity of the trainer is whatever you choose.

**(c)** Cost: for 235M docs / 306B tokens: kNN search **6 h on 32 GPUs**, graph traversal **12 h on 20 CPUs**, plus doc embedding. Scaled to your 10–50B tokens: roughly a day on a single 8-GPU node, and **zero training-FLOPs overhead**.

**(d)** Hypothesis fit: indirect. ICP doesn't give the model a store — it *teaches context reliance*, which shows up as +15% avg reading comprehension (HotpotQA 17.4→23.6 EM), +8% in-context learning, +9% better use of retrieved docs at eval, **verified down to 0.3B models** (0.3/0.7/1.5/7B grid — evidence it works at your scale). Facts in the stream still receive loss, so memorization pressure is *not* removed. Best used as: (i) a cheap additive ingredient for any arm, and (ii) the "make the dense baseline as strong as possible at using context" control. Related: RPT (arXiv **2306.13421**, TACL 2024) trains retriever+LM jointly from scratch, but retrieval is over the same long document — long-context modeling, not an external fact store.

### 4.3 LMLM — the 2025 recipe that *is* your hypothesis (flagged as the headline)

**(a)** **LMLM** ("Pre-training Large Memory Language Models with Internal and External Knowledge", Zhao et al., arXiv **2505.15962**, v3 Oct 2025; appearing at ICLR 2026): a small fine-tuned annotator model rewrites the pretraining corpus, wrapping entity-level facts in explicit lookup calls (`<|db_start|> entity, relation <|db_retrieve|> value <|db_end|>`) and depositing (entity, relation)→value triplets into an external database (54.6M triplets from a 3B-token Wikipedia corpus). Training is **standard next-token prediction with one change: retrieved value tokens are masked from the loss** — the model gets zero gradient signal to memorize facts and instead learns *when and how to query*. At inference it emits lookup calls, the DB returns the value (fuzzy MiniLM-embedding match or trie-constrained decoding), generation continues.

**(b)** Code: [kilian-group/LMLM](https://github.com/kilian-group/LMLM) (HF Accelerate-based; models on the Hub). Maturity: fresh single-paper research code, but the recipe itself is trivially portable — special tokens + a loss mask + a lookup service; no faiss-over-billions, no CCA, no retriever training.

**(c)** Cost: their entire grid (GPT-2- and Llama2-style, **176M and 382M from scratch**, 3B tokens × 8 epochs, batch 256, 105k steps) ran in **~8 H100-days** — squarely inside a one-month modest budget, with headroom to scale to ~1B. Hidden cost to plan for: **corpus annotation** (they used GPT-4o on a 1k-doc seed + a fine-tuned Llama-3.1-8B annotator over 3B tokens; annotating 10–50B tokens with an 8B model is order 10²–10³ GPU-hours and becomes your main data-engineering task).

**(d)** Hypothesis fit: the strongest of anything surveyed — facts are externalized *by construction* and the loss mask means the model **provably can't be rewarded for memorizing them** (they show loss on fact tokens stays high all through training, and factual precision collapses when the DB is unplugged — i.e., facts genuinely live outside). Results: LMLM-382M ≈ **Llama2-7B on FactScore factual precision** (+17.9 to +20.5 points over same-size dense), lower validation perplexity throughout training (−1.98 ppl dynamic), NLU parity, and instant unlearning by DB deletion. **What they did *not* measure is your deliverable: a reasoning/schema delta at matched params/tokens.** They report NLU parity and convergence speed, and explicitly hypothesize freed capacity — but no GSM8K-style, multi-hop, or procedural-generalization comparison. A 1-month project that pretrains LMLM-vs-dense at ~0.4–1B on 10–30B tokens and evaluates a reasoning battery is a genuine, well-scoped contribution sitting right on top of released code.

**(e)** LMLM: arXiv **2505.15962**. REALM: **2002.08909**. Atlas: **2208.03299**. ICP: **2310.10638**. RPT: **2306.13421**.

---

## 5. Training frameworks for 100M–1.5B in 2026, and retrieval extensibility

| Framework | State (mid-2026) | Sweet spot | Retrieval-extension effort |
|---|---|---|---|
| [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | Very active; GPT-2 (124M) speedrun record ~77–80 s on 8×H100 (June 2026); 350M track ~26 min | Fastest possible small-model iteration | Single file = easy to hack, but hyper-tuned (Muon, FP8, custom attention) — architecture edits can silently break the tuning; fine for 124–350M ablations |
| [nanochat](https://github.com/karpathy/nanochat) (Karpathy, Oct 2025) | Active, has its own speedrun leaderboard | Full pipeline (tokenizer→pretrain→SFT→eval→UI) on one 8×H100 node; GPT-2-grade d26 for ~$70–100, ~3 h; `--depth` dial sweeps a model miniseries | **Best harness for this project**: minimal, single-node, sane defaults, CORE-metric eval built in; adding LMLM-style special tokens + loss mask is a small diff; adding a CCA block + neighbor-tensor dataloader is days |
| [OLMo-core](https://github.com/allenai/OLMo-core) (Ai2) | Very active; OLMo 3 (7B/32B) Nov 2025, OLMo 3.1 Dec 2025; official pretraining scripts | Rigor + full openness: **OLMo 2 1B (Apr 2025) gives you a fully-open 1B dense reference** with data (Dolma), code, checkpoints | Clean PyTorch; moderate effort; the natural choice if you want your dense baseline anchored to a known-good open 1B recipe |
| [torchtitan](https://github.com/pytorch/torchtitan) (PyTorch) | Very active (v0.2.2 Feb 2026); best-in-class throughput — 3B trained at 41.4k tok/s/GPU, 45% MFU with FP8 (3× nanotron) | Production-grade multi-node 1B–405B | Plain-module model code, easy CCA insertion; dataloader customization for precomputed neighbors is straightforward; more infra than a 1-month solo project strictly needs |
| [nanotron](https://github.com/huggingface/nanotron) (HF) | Maintained (trained SmolLM1-3) but publicly benchmarked ~3× slower than torchtitan at 3B | HF-ecosystem pretraining | YAML-config architecture makes ad-hoc surgery clunkier than nanochat/titan |
| [litgpt](https://github.com/Lightning-AI/litgpt) | Maintained; 20+ models, recipe-driven | Beginner-friendly single-file model defs | Easy edits, but no speedrun-class efficiency; fine fallback |
| Megatron-LM `tools/retro` | **Removed Jan 2026**; pin old tag / `InstructRetro` branch | Only if you want the reference RETRO data pipeline | High friction; treat as a mining source for the chunk-DB/index/neighbor-precompute scripts, not a living framework |

**Least-effort ranking for the treatment arm:** (1) LMLM recipe on nanochat or OLMo-core — pure data + loss-mask change, no retrieval in the training loop; (2) TRIME objective on the same stack — loss + BM25 batch sampler, no index; (3) RETRO-style CCA with *frozen* retriever and fully precomputed neighbors — model change is one cross-attention block, data change is one extra tensor per batch, but you own the chunk/index/neighbor pipeline (budget the ~1–4 node-days above); (4) memory-layer arm via facebookresearch/memory (Lingua) or by porting their `product_key` module into your trainer. Avoid anything requiring trainable retrievers with index refresh (REALM/Atlas-style, ~30% overhead and lots of machinery).

---

## 6. Cross-cutting cautions for the experimental design

1. **Leakage is the #1 confound.** Retrieval-augmented models exploit train/eval overlap far more than dense ones (RETRO quantified this with 13-gram Jaccard filtering). Dedupe your eval sets against both the training corpus *and* the retrieval store, and report deltas on filtered evals.
2. **Decide what "matched budget" means.** RETRO adds ~10–25% FLOPs at your scale; memory layers add ~0 FLOPs but ~2× parameters-in-memory; LMLM adds annotation compute but ~0 training FLOPs. Report params, tokens, *and* wallclock/FLOPs; consider giving the dense baseline the retrieval arm's extra FLOPs as a control.
3. **Test store-dependence explicitly.** Evaluate each retrieval arm with the store unplugged (LMLM's Table 8 pattern, InstructRetro's encoder ablation). If performance survives without the store, facts leaked into weights — the hypothesis test is void.
4. **Measure the right delta.** Perplexity gains from retrieval are largely copying (kNN-LM limits paper). Your claim lives or dies on *reasoning/procedural* evals at matched factual access: e.g., multi-hop QA where facts are in the store, math/BBH-style tasks, and knowledge-free probes of schema learning. Engram's knowledge-vs-reasoning split (MMLU vs BBH) is a good template.
5. **Small-scale precedent exists for every arm**: RETRO++ at 148M–1.5B, Memory Layers at 134M–1.3B, LMLM at 176–382M, ICP at 0.3B — so a null result at 1B won't be dismissible as "too small to see the effect," which is good for publishability either way.