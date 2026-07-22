# Hallucination / guardrail check (part of `editability/`)

This is a sub-probe of editability — "what does the split model do when the store has **no
answer**?" — so it lives here. The full write-up and interpretation are in the parent doc
**[`../RESULTS.md`](../RESULTS.md)** (section: *"Guardrail: what happens when the store has no
answer"*).

**Headline:** on a store miss the split model fabricates a plausible fact value **0% of the
time** (all 3 loads); it emits generic filler instead. Mechanistic reason: value tokens were
loss-masked in training, so the model never learned to generate values at all. The only rare
"confidently wrong" path is mis-addressing to a real row (<0.1%, from `../../keyguess/`).

**Raw data** stays here: `{n50k,n200k,n800k}_hallucination.json`
(produced by `scripts/run_hallucination_check.py`). Background on populations:
[`../../ENTITY-POPULATIONS.md`](../../ENTITY-POPULATIONS.md).
