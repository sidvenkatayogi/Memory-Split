# METHODS — what each built probe measures, and why it's valid

Rigor tags: **[MT]** mathematical/definitional truth · **[A]** assumption (empirical
premise; where a skeptic pushes) · **[C]** claim that holds only if its [A]s do.
Every method lists: *estimator · assumptions · confounds · control · output · how
to read*.

---

## 1. Latent-reasoning probe (NR-7) — `evals/reasoning_probe.py` — Q1

**Question.** Is reasoning at chance because the model never computes the answer
(**learning-limited**) or computes-but-can't-emit it (**readout-limited**)?

**Estimator.** For a probe site (a context string per item), capture the
last-token residual per layer (`mechanism.capture_last_token`), then per layer fit
a held-out linear classifier for the answer class (`fit_linear_probe`); report
`acc_by_layer` vs `chance = 1/#classes`. Also a **logit-lens**: apply the model's
own `lm_head(ln_f(h_ℓ))` and read `p(gold first token)` per layer (no fitting).

**Two sites, and the copy trap [MT].** The iGSM gold CoT contains the answer
verbatim just before `Answer:`, so probing the answer at the **after-gold-CoT**
slot measures *copying*. Therefore the **end-of-prompt** site (answer not yet in
context) is the copy-free primary signal; after-gold-CoT is kept only as an
execution/copy positive-control. Probing a hidden *vector* for a *class* (not a
next token) also sidesteps BPE-boundary hazards.

**Assumptions.** [A] the answer, if computed, is *linearly* decodable (linear
probe); [A] the eval set's answer set defines the class space (chance = 1/#classes).

**Confounds & controls.** Copy (handled by site choice). Probe over-capacity
(handled: linear, held-out by item, report chance + n_test). **Positive control
required**: probe a known-present feature (operator `op`) to show the probe *can*
recover before trusting a null.

**Output / how to read (decision rule).** With `E` = best end-of-prompt acc, `C` =
best after-CoT acc, `g` = greedy acc:
- `E` decodable ⇒ **readout-limited** (computes latently, can't verbalize) — fix is
  decoding/format, not more training.
- `E` at chance, `C` decodable ⇒ **CoT-generation-limited** (can execute given the
  CoT, can't produce it).
- both at chance ⇒ **execution/learning-limited** (P0 ladder).
The module emits this `verdict` string plus the raw `acc_by_layer`.

**Status.** Correlational. Causal follow-up (ablate the decoded direction, does
greedy move?) needs above-chance behavior + `interp.py`.

---

## 2. Fact-neuron localization + probe-bits (NR-5) — `evals/mechanism.py` — Q2, Q3

**Question.** Where are facts stored, and are those parameters absent/repurposed in
the split arm?

**Estimator (localization).** Capture last-token MLP gated activations
(`block.mlp.w2` input) on fact-retrieval prompts vs neutral control; per neuron
**Cohen's d** selectivity `(mean_fact − mean_ctrl)/pooled_sd` [MT]. Rank top-k. The
**claim-bearing statistic is the cross-arm ratio**: take dense's top-k neuron
indices and compare their |d| in dense vs in split — high ratio ⇒ the fact-storage
units are silent in split.

**Estimator (probe-bits, L11).** A held-out **linear probe** on residual states
predicts the fact value; probe accuracy → bits via the *same* clamped estimator as
recall-bits (`recall.bits_in_weights`). `dense(probe) − dense(recall)` isolates
*stored-but-not-generatively-extractable* facts; `dense(probe) − split(probe)` is
the mechanistic H3 gap.

**Assumptions.** [A] memorization concentrates in identifiable MLP units
(MemSinks/knowledge-neuron precedent); [A] fact value is linearly decodable from
residual; [A] the clamped estimator's chance-subtraction is the right zero.

**Confounds & controls.** Per-arm selectivity conflates prompt-structure with
retrieval (**fixed**: use structure-matched fresh-entity control, `control_records`,
L26); the cross-arm ratio is the controlled quantity. Probe-bits `n_eval` must be
the **entity count**, not entity×attr (L25, fixed).

**Output / how to read.** `localization_report` → per-arm top-|d| and
`cross_arm_ratio`; `probe` → per-attribute probe acc + bits per arm +
`probe_bits_gap_dense_minus_split`. Expect: dense localizes fact neurons and stores
bits; split ≈ 0/flat; memorization-neuron count grows with dose N.

---

## 3. Causal harness — `evals/interp.py` (parallel agent) — Q1(gated), Q2, Q3

**Tools.** `ablate_mlp_neurons` (zero SwiGLU units via forward-pre-hook),
`neuron_ablation_effect` (baseline−ablated on any scorer), `activation_patch_logits`
(patch a layer's residual from a source run into a target at chosen positions),
`capture_residual_at`.

**Runnable NOW (fact-side) [C given the localization [A]].** Ablate dense's top
memorization neurons (`mechanism.top_neurons`) and measure: **recall collapses,
reasoning unchanged** → causal evidence that those units store facts (not reason).
Repeat across dose to show the memorization footprint grows with N. This is the
capacity-competition test and does **not** need reasoning to work.

**Gated (reasoning-side).** Patching/ablation *for correct reasoning* needs
above-chance behavior (nothing to move at chance). The harness is generic and
tested; run it the moment NR-7 says reasoning is decodable.

**Confounds.** Ablation effects are causal but *component-level* (a unit's effect
in-context, not a full circuit). Patching identity check: source==target ⇒
patched==clean (verifies the hook).

---

## 4. Continuous metrics (NR-1) — `evals/continuous.py` (parallel agent) — Q1 input

M1 trace-NLL, M2 answer-given-gold-CoT, M3 gen-slot per-token prob. These give the
**graded** reasoning signal the probes and patching use as a target when argmax is
at the floor (patch on `Δ log p(gold)`, not on correct/wrong). M2 in particular is
the final-layer version of NR-7's after-gold-CoT readout.

---

## 5. Newly built probes (`probe/*.py`) — estimator · assumptions · how to read

All reuse the capture/probe infra above; validated in `tests/test_probe_*`.

**`geometry.cross_arm_cka`** (Q2). Estimator: linear CKA per layer between dense &
split residuals on **shared inputs** [MT] (invariant to rotation/scale). [A] pairs
share init+seed+doc-order so same-index comparison is meaningful. Read: low CKA at
a layer = the arms represent inputs differently there (freed-capacity locus).
Control: CKA(dense, dense-other-seed) as an upper baseline (needs a 2nd seed).

**`geometry.weight_spectral` / `effective_rank`** (Q3). Effective rank = exp(entropy
of normalized singular values) [MT]. Read: split MLP leaner (lower rank/norm) than
dense's fact-laden matrices ⇒ capacity freed. Descriptive (weights only).

**`geometry.topk_neuron_overlap`** (Q2). Jaccard of top-k |selectivity| sets.
Read: fact vs reasoning neurons disjoint (low overlap) ⇒ separable; dense's fact
neurons acquiring reasoning-selectivity in split ⇒ repurposing.

**`geometry.subspace_overlap`** (Q2). Mean squared principal-angle cosine of two
orthonormalized probe-weight subspaces [MT]. [A] the fact/reasoning directions are
the probe weight matrices. Reasoning direction is gated on a reasoning probe.

**`faithfulness.cot_faithfulness`** (Q1). Perturb the **final** stated "= v"→"= w"
and read p(w) vs p(v) at the Answer slot; control = perturb an **earlier** step
(answer should stay v). [MT] measures answer's *causal sensitivity* to its own
stated CoT — faithful ⇒ follows final, ignores earlier. Runnable at chance
(sensitivity, not correctness). Confound: only meaningful where the answer is
CoT-derived (iGSM); needs ≥2 numeric steps for the control.

**`intermediate.intermediate_value_probe`** (Q1). Per-layer linear probe for each
compute-step value, at the slot **before** the value (non-copy [MT]). [A] value
linearly decodable. Read: above-chance ⇒ the model computes intermediates
(hidden reasoning) even if the final greedy answer fails. Control: shuffle labels
⇒ chance; distractor nodes (from the prompt) as negatives (future).

**`attention.attention_weights` / `head_ablation_effect`** (Q1). Weights recomputed
as softmax(QKᵀ/√d) with RoPE [MT] (rows sum to 1). Ablation zeros a head's
contribution and scores the causal effect. Read: heads attending query→needed
operands, confirmed causal by ablation. Control: ablate random heads.

**`weights.fact_weight_attribution`** (Q3). First-order |grad ⊙ W| of a fact's
recall NLL onto MLP weights [MT-approx: 1st-order Taylor of the loss]. Read: dense
concentrates attribution in specific layers (storage sites); split diffuse/near-0.
Control: a fact the model doesn't know ⇒ diffuse.

**`weights.double_dissociation`** (Q2). Ablate the dense arm's top memorization
neurons; report recall Δ vs reasoning Δ. Read: recall collapses, reasoning intact
⇒ facts and reasoning use different units (fact-side of the dissociation; the
reasoning-side ablation is gated).

**`weights.participation_ratio` / `fact_superposition`** (Q3). PR = (Σλ)²/Σλ² of the
activation covariance [MT] (effective #directions). Read: PR falls as dose N grows
⇒ facts increasingly superposed (crowding) — the mechanism the thesis invokes.

**`splice.value_injection_profile`** (Q2). Per-layer L2 residual diff between
value-present vs value-absent contexts. Read: the peak layer is where an external
value is incorporated — how the split arm *splices* vs how dense would *store*.
Proxy (context presence), not the live lookup path; documented as such.

**`ledger.fact_info_ledger` / `capacity_scaling`** (Q3). Pure aggregation:
recall/probe/MC bits into one statement (`extractability_gap = probe−recall`, L11);
bits-vs-params log-log slope (≈const bits/param ⇒ the 2-bit/param regime).

**`causal_steps.step_patch_effect`** (Q1, **GATED**). Δ log p(gold) from patching a
layer clean→corrupt (continuous metric, works sub-threshold) [MT identity: source
==target ⇒ no change]. Informative only once reasoning > chance (no clean/corrupt
gap at the floor).

## Cross-cutting rigor rules
1. **Positive control before any null.** Probe/ablate a known-present feature first.
2. **Contrast, not level.** Report split−dense; per-arm numbers are descriptive.
3. **Correlational ⇏ causal.** Decodability needs an ablation/patch to become
   "the model uses it."
4. **Seed-0 only ⇒ directional.** No confirmatory claims from one seed (L16).
5. **Snapshots are consumable.** Trajectory versions of every probe (over training
   tokens) require the snapshots — score them before pruning (see
   `../../replication/future-directions.md`).
