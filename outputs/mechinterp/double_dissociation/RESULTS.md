# Double dissociation (recall side): are the "fact neurons" causally responsible for recall?

> **Entity population: SEEN (frozen trained people) — required.** This asks whether specific
> units *store* facts, which is only defined for people the model was trained on. Run on the
> frozen trained people via `evals/frozen.py`. See
> [`../ENTITY-POPULATIONS.md`](../ENTITY-POPULATIONS.md).

## Summary box — read this first
*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment; the one
line you need: the **dense** twin memorizes facts in its weights.*

**What "double dissociation" means (the term).** It's the gold-standard way to show two
functions are handled by two different parts: demonstrate that *knocking out part A breaks
function X but not Y*, **and** that *knocking out part B breaks Y but not X*. Here the two
functions are **fact recall** and **reasoning**, and the claim we're building toward is that
the split model frees the "fact part" for reasoning. This probe delivers the **first half** —
knocking out the fact-part breaks recall — because the reasoning half can't be run yet (the
model's reasoning is at chance, so there's no reasoning to break).

**What this probe asks, in plain terms.** Other probes found internal units that *correlate*
with fact recall (we call them "fact neurons" — individual units inside the network whose
activity spikes specifically when the model is recalling a fact). Correlation isn't cause.
This is the **causal** test: **switch those units off and see whether recall actually
breaks.** If it does — and switching off the *same number of random units* doesn't — then
those units are genuinely carrying the stored facts.

**How we measure it.** In the dense model we (1) identify the top-64 fact neurons (the units
most fact-selective, by the same Cohen's-d measure the `mechanism` probe uses), (2) **ablate**
them — i.e. force their activation to zero during the forward pass, with no weight change — and
re-measure closed-book recall, and (3) as a control, ablate **64 random** units and
re-measure. The gap between the fact-neuron ablation and the random-neuron ablation is the
causal signal (the random one accounts for any generic damage from just removing 64 units).

**Why it matters — significance & nuance.** This is the strongest form of evidence we can get
short of retraining: it moves the fact-neuron story from "these units *light up* on facts" to
"these units are *necessary* for producing facts" — direct mechanistic support for **H3**
(facts are stored in specific dense weights). *Nuance:* this is the **recall half** of a full
double dissociation. The complete claim also needs the converse — *ablating fact-neurons
leaves reasoning intact* — which is **gated** until the model reasons above chance, so we
report only the recall half here. Also, ablation is blunt (zeroing units can have side
effects); the **random-ablation control** is what makes the result interpretable.

**Which model / checkpoint / people.** Dense **n50k** — the one load where the model actually
memorized (62% recall), so there is recall to break. Final checkpoint
`snapshots/step0006100.pt`, trained people, 150 entities × 6 = 900 recall probes. One seed.
(n200k/n800k dense recall is ≈0, so there is nothing to ablate there.)

## Result
| condition | closed-book recall |
|---|---|
| baseline (nothing ablated) | **0.657** |
| ablate the 64 **fact neurons** | **0.351**  (drop **−0.306**, ≈47% of recall gone) |
| ablate 64 **random** neurons | 0.640  (drop −0.017, essentially unchanged) |

The fact-neuron ablation removes **≈18× more recall** than the random-neuron ablation.

## Interpretation
**The identified fact-neurons are causally necessary for recall, not just correlated with
it.** Knocking out 64 specific units (out of the ≈24,000 MLP units in the network) cuts the
dense model's memorized-fact recall nearly in half, while knocking out 64 random units does
essentially nothing. That is a genuine causal dissociation on the recall side: it confirms
that the "fact machinery" the correlational probes pointed to is where a meaningful chunk of
the stored facts actually live (H3), and it validates the fact-neuron identification used by
the mechanism/localization probe.

That recall drops by ≈half (not to zero) from ablating only the top 64 units is consistent
with storage being **concentrated but distributed** — the strongest 64 units carry a large
share, and the remainder is spread across more units (ablating more would remove more). This
also dovetails with the attention and attribution probes, which place the fact-producing
machinery in the mid-to-late layers.

## Caveats
- **Recall half only.** The full double dissociation needs the reasoning-side control
  (ablate fact-neurons → reasoning unchanged), which is **gated** until reasoning clears
  chance. Until then we've shown "necessary for recall," not "necessary for recall *and* not
  for reasoning."
- **n50k only** (the memorized load); at higher loads there is no recall to ablate.
- **Ablation is blunt** — zeroing units perturbs the network broadly; the random-neuron
  control (which barely moves recall) is what licenses attributing the drop to *these* units
  specifically. One seed.
- Fact-neurons identified from fact-vs-neutral selectivity (Cohen's d, top-64), the same
  definition as the mechanism probe.

## Provenance
`scripts/run_causal_probes.py --do dissociation` using `evals.interp.ablate_mlp_neurons` +
`evals.mechanism` (neuron selectivity) on the frozen trained people
(`outputs/_frozen_data/n50k_recall.jsonl`). Raw numbers in `n50k_dissociation.json`. No
training run was changed.
