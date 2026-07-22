# NEW-PROBES — the backlog, by question (rigorous, ranked)

> **STATUS (2026-07-21): all implemented** as tested modules in `probe/*.py`
> (full suite 241 passed). Map idea → code:
>
> | # | module.function | runnable now? |
> |---|---|---|
> | 1.1 | `intermediate.intermediate_value_probe` | yes |
> | 1.2 | (folded into 1.1 needed-vs-distractor note) | partial |
> | 1.3 | `faithfulness.cot_faithfulness` | yes |
> | 1.4 | `attention.attention_weights` / `head_ablation_effect` | yes |
> | 1.5 | `causal_steps.step_patch_effect` | gated (needs reasoning>chance) |
> | 2.1 | `geometry.cross_arm_cka` | yes |
> | 2.2 | `geometry.subspace_overlap` | reasoning-dir gated |
> | 2.3 | `weights.double_dissociation` | fact-side yes |
> | 2.4 | `geometry.topk_neuron_overlap` | yes |
> | 2.5 | `splice.value_injection_profile` | yes |
> | 3.1 | `weights.fact_weight_attribution` | yes |
> | 3.2 | `weights.participation_ratio` / `fact_superposition` | yes |
> | 3.3 | `geometry.weight_spectral` / `effective_rank` | yes |
> | 3.4 | `ledger.fact_info_ledger` | yes |
> | 3.5 | `ledger.capacity_scaling` | yes |
>
> Rigor backing per module: `METHODS.md` §5. The design rationale below is the
> spec each was built to.


Probes not yet built, assuming you have **all snapshots + final checkpoints for
both arms across doses** (and 1B dense). Each: *method · what it shows
([MT]/[A]/[C]) · checkpoints · control · gated? · effort*. Tags: **pair** =
needs dense+split; **snap** = needs snapshots; **now** = runnable at chance.

Guiding discipline (applies to all): report the **split−dense contrast**, run a
**positive control** before any null, and keep probes **linear** unless testing
nonlinearity explicitly.

---

## Q1 — How is reasoning done?

### 1.1 Intermediate-quantity probe (per-step DAG values) — [now, pair] — S/M
**Method.** Parse the iGSM gold CoT into its dependency DAG; for each intermediate
node, probe the residual at the slot **after "= " but before the value is written**
(value not yet in context → non-copy) for that node's value (0–22). Report
decodable-fraction of needed nodes vs distractor nodes, per layer.
**Shows.** [C] whether the model computes intermediate results (partial dependency
tracing) even when the final answer fails — the finest-grained "hidden reasoning"
(Physics 2.1). Distractors as built-in negative control: needed decodable ≫
distractor ⇒ genuine tracing.
**Control.** Distractor-node values (should be *less* decodable if the model tracks
the needed chain). **Effort:** medium (CoT→DAG label extraction + tokenization-aligned
positions).

### 1.2 Dependency / necessity probe — [now, pair] — S
**Method.** At end-of-prompt, probe a per-quantity binary "is this on the needed
chain?" (label from the generator's `needed` set).
**Shows.** [C] whether the model represents the *plan* (which quantities matter)
before computing — planning vs blind. Non-copy (structure, not values).
**Control.** Shuffle labels ⇒ probe→chance.

### 1.3 CoT-faithfulness (causal-CoT) — [now, pair] — S/M
**Method.** Teacher-force the gold CoT but **perturb one intermediate value**;
measure whether the model's answer-slot distribution shifts *consistently* with the
perturbation.
**Shows.** [MT-ish] whether the stated CoT actually *drives* the answer or the
answer is computed independently (unfaithful/post-hoc CoT). Answer follows the
perturbation ⇒ faithful; ignores it ⇒ post-hoc. High value for "how reasoning is
done," and it's causal without needing above-chance final accuracy (you're
measuring sensitivity, not correctness).
**Control.** Perturb an *unused distractor* value ⇒ answer should NOT change.

### 1.4 Attention dependency-tracing + head ablation — [now, pair] — M
**Method.** Capture attention (needs an attention hook — small addition) and test
whether specific heads attend query→needed-operands along DAG edges; ablate those
heads and measure the effect on the answer logit / intermediate probes.
**Shows.** [C] the retrieval/induction heads implementing dependency lookup; causal
via head ablation. Mechanistic "how the trace is assembled."
**Control.** Ablate random heads (should hurt less). **Effort:** medium (add
attention capture; `interp.py` ablation extends to heads).

### 1.5 Step-localization activation patching — [gated, pair] — M
**Method.** Patch residual at each intermediate position from a correct run into a
wrong-operand corrupted run (metric = `Δ log p(gold)` so it works sub-threshold);
which positions/layers restore the answer = the compute path.
**Shows.** [C] the causal reasoning path. **Gated** (low-SNR until reasoning lifts;
use continuous readout). Harness ready (`interp.activation_patch_logits`).

---

## Q2 — How is reasoning separated from facts?

### 2.1 Cross-arm representational divergence (CKA) — [now, pair] — S
**Method.** Pairs share init+seed+doc-order, so dense/split are comparable
weight-for-weight. Compute **linear CKA** between dense and split residuals per
layer on shared inputs; also per-layer weight-delta norms.
**Shows.** [C] *where* the arms diverge — the layers where "freed capacity" actually
manifests. High CKA in early layers (shared language) + low CKA where facts/reasoning
live is the predicted signature. Cheap, novel, and needs no reasoning to work.
**Control.** CKA(dense, dense-other-seed) as an upper baseline if a second seed
exists (else interpret relatively across layers).

### 2.2 Fact-vs-reasoning subspace overlap — [pair; reasoning part gated] — M
**Method.** Fit a **fact direction** (probe for stored attribute values) and a
**reasoning direction** (probe for intermediate quantities, 1.1); measure principal
-angle / cosine overlap of the two probe subspaces per layer, per arm.
**Shows.** [C] the geometric form of capacity competition: in dense, fact and
reasoning subspaces overlap (interference); in split, the fact subspace is
empty/repurposed, leaving reasoning more room. Directly operationalizes
"separation." Reasoning-direction quality is the gated part.
**Control.** Random directions ⇒ overlap ≈ chected chance.

### 2.3 Causal double-dissociation — [now for fact-side; full gated] — M
**Method.** (a) Ablate fact neurons → recall↓, reasoning-probe unchanged. (b) Ablate
the reasoning direction → reasoning↓, recall unchanged.
**Shows.** [C] the gold-standard "two separate mechanisms" evidence. Half (a) is
runnable now; (b) needs a reasoning signal.
**Control.** Ablate random units of equal count (should not selectively hit either).

### 2.4 Memorization-vs-reasoning neuron overlap & repurposing — [pair] — S/M
**Method.** In dense, compute per-neuron fact-selectivity (NR-5) and
reasoning-relevance (probe-weight attribution). Measure overlap. In **split**, take
the neurons that were fact-neurons in dense — did they acquire reasoning-relevance?
**Shows.** [C] whether freed fact-capacity is *reused* for reasoning (the L10
headline) at the neuron level, and whether the two functions share units in dense.
**Control.** Random neuron sets.

### 2.5 Store-ON/OFF splice-point localization (split) — [now] — S
**Method.** On fact-use items, diff the split arm's residual with the store ON vs
OFF; localize the layer/position where the retrieved value enters the stream.
**Shows.** [C] *mechanically how external facts are injected* vs how weights would
have supplied them — the "separation" made concrete.
**Control.** Non-fact tokens (no ON/OFF diff expected).

---

## Q3 — How do parameters/weights store things?

### 3.1 Weight-level fact localization — [now, pair] — M
**Method.** Gradient attribution (integrated gradients) of a single fact's recall
loss onto weights → the parameters that, if perturbed, change *that* fact; and/or
causal tracing (`activation_patch_logits` on a fact probe) → the storage layer.
Compare dense (localized) vs split (none).
**Shows.** [C] *which weights* hold a fact (not just which activations). MemSinks /
ROME-tracing precedent. Fact-side ⇒ runnable now.
**Control.** A fact the model doesn't know (attribution should be diffuse).

### 3.2 Superposition / crowding vs dose — [now, dense, snap-optional] — M
**Method.** Across doses {50k,200k,800k}, measure per-neuron polysemanticity /
effective #facts-per-neuron (e.g., how many distinct facts load onto a fact-neuron;
interference between fact readouts).
**Shows.** [C] the *mechanism* of crowding: as N grows, more facts share fewer
units (superposition ↑), which is the capacity pressure the whole thesis invokes.
Your 3-dose axis is built for this.
**Control.** Same measure on reasoning-irrelevant neurons.

### 3.3 Weight spectral / effective-rank analysis — [now, pair] — S
**Method.** Per-layer MLP weight-matrix spectra (effective rank, norm), dense vs
split.
**Shows.** [C] whether removing memorization lowers the rank/norm dense spends on
facts (fuller in dense, leaner in split ⇒ capacity freed). Cheap, weights-only.
**Control.** Attention weights (less fact-bearing) as a comparison.

### 3.4 Unified fact-information ledger — [now, pair] — S
**Method.** Combine recall-bits + probe-bits (NR-5) + MC-recognition (NR-4) into one
per-arm accounting: total fact info, in-weights (dense) vs in-store (split), and the
extractability gap.
**Shows.** [C] a single reconciled H3 statement that resolves L11 (bits caveat).

### 3.5 Capacity scaling on our stack (160M→1B dense) — [now, dense] — S
**Method.** Measured fact-bits vs params (dense n800k at 160M vs 1B; plus n4m at 1B).
**Shows.** [C] whether storage tracks ~2 bits/param on *our* models — anchors the
CASM memorization-wall arithmetic with our numbers, not cited ones. Also the 1B
long-tail cliff (exposure-stratified recall).
**Control.** n/a (a scaling measurement).

---

## Suggested build order (value ÷ effort, runnable-now first)
1. **2.1 CKA cross-arm divergence** — cheapest structural "separation" signal, now.
2. **1.3 CoT-faithfulness** — causal, high-value for "how reasoning is done", now.
3. **3.1 + 2.3(a) fact-weight localization + causal ablation dissociation** — the
   "facts live in these weights, and they're not reasoning" core, now (fact-side).
4. **1.1 intermediate-quantity probe** — the finest "hidden reasoning" signal (needs
   CoT→DAG labels).
5. **3.2 superposition-vs-dose** and **3.3 spectral** — the storage *mechanism*.
6. Gated (post-P0): 1.5 step-patching, 2.2/2.4 reasoning-subspace & repurposing.

Each new probe should ship like NR-5/NR-7: a spec in `../../replication/specs/`, a
module in `evals/`, a runner in `scripts/`, tests on a toy GPT, and a copy pinned
here. Ask and I'll build them in this order.
