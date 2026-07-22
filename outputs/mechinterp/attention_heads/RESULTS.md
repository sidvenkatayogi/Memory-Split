# Attention heads (fact-use side): which layers' attention help produce a stored fact?

> **Entity population: SEEN (frozen trained people).** This only means something where the
> model actually stored the fact, so we run it on the frozen trained people via
> `evals/frozen.py`. See `[../ENTITY-POPULATIONS.md](../ENTITY-POPULATIONS.md)`.

## Summary box — read this first

*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one line
you need: the **dense** twin memorizes facts in its weights.*

**What this probe asks, in plain terms.** To say a memorized fact out loud ("Kai's employer is
→ Cascade Logistics") the model has to move information between token positions — e.g. carry
"who we're talking about" and "which attribute" to the spot where it writes the answer.
**Attention is the only mechanism that moves information between positions.** This probe asks:
**which layers' attention does the model actually lean on when producing a stored fact?**

**How we measure it.** We go layer by layer. For each layer we **switch off all of that
layer's attention heads** (zero their contribution, leaving everything else — the residual
stream, the MLPs — intact) and measure how much worse the model gets at the fact. "Worse" is
the **fact-value NLL**: the model's *surprise* at the correct value (in nats; lower = it
predicts the value confidently, higher = it struggles). The number we report per layer is
**ΔNLL = (NLL with that layer's attention removed) − (baseline NLL)** — how much removing that
layer's attention hurt. It's a single forward pass, so it's exact and cheap.

**Why it matters — significance & nuance.** The storage probes say facts are *in the weights*;
this asks *how the model gets them out* — where the cross-position routing happens. It's the
**fact-use** half of the attention analysis. Two limits up front: (1) the **reasoning-side**
version (which heads do multi-step dependency lookup) is **gated** until reasoning clears
chance; (2) as detailed below, this measures where attention is **needed to produce the fact**
— which is **not** the same as "where the fact is stored," and (crucially) we have **not**
shown the importance is *fact-specific* rather than *generally important layers*.

**Which model / checkpoint / people.** Dense **n50k** — the one load where the model memorized
(fact-value NLL on trained people is **0.85 nats**, versus ≈5 on strangers, which confirms it
really did store these facts). Final checkpoint `snapshots/step0006100.pt`, 200 trained people,
one seed.

## How to read a number here (worked example)

Baseline fact-value NLL is **0.85** (the model is quite confident of the values it memorized).
Layer 8's ΔNLL is **1.55**, meaning: switch off layer 8's attention and the surprise jumps to
0.85 + 1.55 = **2.40 nats** — the model becomes markedly worse at producing the value. Layer
1's ΔNLL is **0.16**, meaning switching off layer 1's attention barely changes anything
(0.85 → 1.01). So **bigger ΔNLL = that layer's attention mattered more** for getting the fact
out.

## Result

ΔNLL when each layer's attention heads are ablated (baseline NLL = 0.85; bigger = more needed):


| layer | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | **8**    | **9**    | 10   | 11   |
| ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | -------- | ---- | ---- |
| ΔNLL  | 0.95 | 0.16 | 0.77 | 0.14 | 1.04 | 1.13 | 0.73 | 1.07 | **1.55** | **1.24** | 1.05 | 0.54 |


Ranked: **8 (1.55) > 9 (1.24) > 5 (1.13) ≈ 7 (1.07) ≈ 10 (1.05) ≈ 4 (1.04) > 0 (0.95) > 2
(0.77) ≈ 6 (0.73) > 11 (0.54) ≫ 1 (0.16) ≈ 3 (0.14)**.

## What it means 

**1. Attention is broadly important for fact production, with a mild mid/late emphasis — not a
sharp "fact circuit" at one layer.** Removing almost *any* single layer's attention hurts:
ΔNLL is between 0.5 and 1.5 for ten of the twelve layers, i.e. knocking out one layer's
attention roughly doubles-to-triples the model's surprise at the value. The mid/late layers
(8, 9, and the cluster 4–7, 10) hurt a bit more; two early layers (1 and 3) barely matter at
all. So the honest headline is **"producing a stored fact relies on cross-position attention
spread across most layers, weighted toward the middle-and-late ones,"** not "the fact is
fetched at layer 8." Do **not** read the mild peak as a localized retrieval circuit.

**2. The mid/late emphasis lines up with the other probes — suggestively.** The
fact-weight-attribution probe put a memorized fact's weight-sensitivity at **layer 9**, and
this independent, causal attention measurement is heaviest at layers **8–9** too. That
convergence is encouraging. But see caveat below: "late layers matter most" is *also* the
generic expectation for producing any output token, so the agreement is consistent with a
fact-specific late-fetch **and** with "late layers are just generally load-bearing."

**3. Two early layers (1, 3) are nearly free to remove for this task.** Their attention
contributes almost nothing to producing these values — a small, concrete negative result
(some layers' attention just isn't on the fact-production path).

**A tentative mechanistic story, offered as a hypothesis only:** early layers assemble the
"(name, relation)" query into the residual stream, and mid/late-layer attention carries the
stored value to the output position, which is why removing late attention hurts most. This
probe **motivates** that picture but cannot **prove** it — the causal test would be activation
patching (`step_patch`), whose fact-use use is future work and whose reasoning use is gated.

## The caveat that matters most: is this *fact-specific* or just *generally important layers*?

This is the biggest reason not to over-read the table. We ablated each layer's attention and
measured damage **only on fact production**. We did **not** run the control of measuring the
same ablations' damage on **non-fact text** (e.g. the knowledge-free reasoning slice). Removing
a whole layer's attention is a large, generic perturbation — the layers that hurt fact
production most (8, 9) may simply be the layers that hurt *everything* most. **Until we compare
against a non-fact control, we cannot claim these layers are doing anything fact-specific** —
only that they are important for this (and possibly any) output. The clean follow-up: run the
identical per-layer attention ablation with a bits-per-byte scorer on the reasoning slice; the
*difference* between the two Δ-profiles (fact minus non-fact) is the genuinely fact-specific
signal. That is not done here.

## Other limitations

- **Layer-level, not head-level.** We zero all 12 heads in a layer at once, so this localizes
to a *layer*, never to individual "dependency-lookup" heads; a per-head sweep (supported by
the harness) is the refinement.
- **Importance ≠ storage.** Attention *routes* information; the value itself may be stored in
MLP weights at other layers. "Layer 8 attention is needed to produce the fact" does not mean
"the fact lives in layer 8."
- **Causal but blunt, and single-config.** Zeroing an entire layer's attention is a heavy
intervention; ΔNLL is a directional importance signal, not a precise circuit. Dense **n50k
only** (the memorized load — elsewhere there's no stored fact to route), one seed.
- **NLL, not generation.** We score the model's likelihood of the correct value, not free
generation; the two usually agree but aren't identical.

## Provenance

`scripts/run_causal_probes.py --do attention` using `probe.attention.head_ablation_effect`
with a fact-value-NLL scorer, on the frozen trained people
(`outputs/_frozen_data/n50k_recall.jsonl`). Raw numbers in `n50k_attention.json`. No training
run was changed.