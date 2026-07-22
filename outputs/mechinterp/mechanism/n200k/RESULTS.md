# NR-5 mechanism (Q2/Q3) — n200k pair — RESULTS

> **Entity population:** the localization / probe-bits numbers in this per-pair record are
> the **UNSEEN** (stranger) run; the **SEEN** (trained-people) run gives a near-identical
> cross-arm ratio (6.37 vs 6.38 here), so the localization holds on both populations, and
> the probe-bits are near-chance on both (underpowered). The recall figure in the caveats
> (n200k closed-book 0.5%) is from the battery's frozen eval = SEEN. Full discussion:
> [`../RESULTS.md`](../RESULTS.md); background:
> [`../../ENTITY-POPULATIONS.md`](../../ENTITY-POPULATIONS.md).

## What / where
- Probe: `evals.mechanism.fact_mechanism_report` via `scripts/run_mechanism.py`.
- Runs: `d160m_dense_n200k_s0` vs `d160m_split_n200k_s0`, **snapshot `step0006100.pt`**
  (= final, 3.2B tokens). Seed 0.
- Ran **locally on Mac (MPS, MPS→CPU fallback)** on pulled config+snapshot, capped
  `--n-entities 800` for speed (`--probe-layer -1`, `--k 64`); pulled checkpoints
  deleted after. Date 2026-07-21. Cluster copies untouched.
- Output: `mechanism.json`, `probe_acc.png` (this dir).

## Numbers (the split−dense contrast is the on-hypothesis quantity)
**Memorization-neuron localization (Cohen's d of top-64 fact-selective neurons):**
- dense top |d| = **9.69**; split top |d| = 9.00.
- **Cross-arm:** dense's top fact-neurons score |d|=9.69 in dense but only **1.52 in
  split** ⇒ **cross_arm_ratio = 6.38**. I.e. the dense arm's memorization neurons
  are ≈6.4× *less* fact-selective in the split arm.

**Linear probe-bits (last-layer residual → attribute value class):**
- dense total_bits = 19.97; split total_bits = 6.15; **gap (dense−split) = 13.83 bits**.
- BUT probe accuracies are **near chance for BOTH arms** (e.g. dense birth_city
  0.0125 vs chance 0.005; most attrs 0). n_probes=4000, n_eval=240.

> **What "bits" mean here (two different bit quantities — don't conflate them):**
> A **bit** = one yes/no question's worth of information (halving the uncertainty).
> (1) *Probe-bits* above = how much a linear read-out reduces uncertainty about an
> attribute's value, summed over the 6 attributes — so ≈20 bits *would* mean "the
> last layer linearly reveals ≈20 bits about a person's facts." But because the
> probe is at chance, **treat these as ≈0 / noise, not real recovered information.**
> For scale, pinning down one value out of ≈200 cities is ≈7.6 bits (log₂200), so a
> *believable* full-fact probe would be tens of bits — these near-chance numbers are
> far below that and unreliable. (2) The separate **"bits≈14k"** in the caveats is a
> *different* accounting: total information the whole model memorized = (facts
> recalled) × (bits per fact) across all 200k people — a corpus-level tally, not a
> per-token or per-probe number. It is small here precisely because dense n200k
> barely memorized (0.5% recall).

## Interpretation
- **Q2 (separation):** the localization cross-arm ratio (6.38) is the clean signal
  here — the fact-selective machinery in dense is largely **quiet in split**,
  consistent with the split arm not building in-weight fact neurons.
- **Q3 (storage):** dense shows more probe-recoverable fact-bits than split
  (directionally on-hypothesis), but see the big caveat below — at n200k the dense
  arm itself barely memorized, so this is a weak/near-floor regime.

## Caveats (read before trusting)
- **Single seed (seed-0) ⇒ directional/suggestive, not proven.**
- **Correlational**, not causal: probes/localization show decodability/selectivity,
  not that these neurons are *used*. Causal test = `interp.ablate_mlp_neurons` (fact
  side), not yet run.
- **n200k is the dose-bound regime:** from the sweep evals, dense n200k closed-book
  recall is only **0.5%** (bits≈14k) — the dense arm *failed to memorize* at this
  load — so probe-recoverable bits are near-floor for both arms and the 13.8-bit gap
  is small/low-confidence. **The informative Q3 mechanism contrast is n50k** (dense
  recall 62%), where a large dense≫split probe-bits gap is expected — pending the
  cluster probe job.
- **Positive control NOT established:** the linear probe barely recovers facts even
  from dense; partly because dense didn't store them at n200k, but also the probe is
  underpowered (last layer only; 100–300-way labels; only 800 records/240 eval).
  Before trusting any null, re-run with more records, sweep `--probe-layer`, and use
  n50k dense (which demonstrably recalls) as the control.
- **Linear probe** ⇒ a linear null doesn't exclude nonlinear encoding.
- Local run capped at 800 entities (cluster default 2000); numbers will tighten there.

## Provenance
Analysis-code addition (probe toolkit) — report per the preregistration. Not a
frozen-code change; training untouched.
