# MemorySplit — team share package (2026-07-20)

One-line pitch: the first controlled test of "schemas in weights, facts in
an external organizer" — do small LMs reason better when pretraining never
rewards fact memorization?

## Read in this order

1. `docs/superpowers/2026-07-20-interim-report.md` — the whole story:
   hypothesis, design, defense-style objection handling, results to date,
   compute case. **Start here.**
2. `docs/superpowers/specs/2026-07-17-memory-split-design.md` — the
   authoritative experimental design (arms, dose, endpoints, decision
   rule, gates).
3. `docs/superpowers/plans/2026-07-17-memory-split.md` — implementation
   plan; `cluster/RUNBOOK.md` — how the battery actually runs on FarmShare.
4. `docs/superpowers/research/2026-07-17-memory-split/` — literature
   dossier (architectures, prior evidence incl. LMLM verification, eval
   methodology with seed-variance numbers).

## Headline results so far (160M pilots, 0.8B tokens)

- Mechanism proven: split arm stores 0.0 measured fact-bits in weights;
  recall 99.8% with organizer / 0.0% without; 100% correct lookups on
  entities never seen in training text (500/500); 2x dense on fact-use QA.
- Gate A (are knowledge-free reasoning tasks learnable at pilot budget?):
  failed twice at chance level (iGSM ~4%, deduction ~46%) — including
  after doubling the reasoning share. Decision pending between full-budget
  sweep bet, difficulty flooring, or endpoint re-anchor (report section
  6.3). This cost ~8 GPU-hours instead of the ~550-GPU-hour battery: the
  gating system working as designed.

## Results files (`results/`)

`<run>.summary.json` — eval battery output per gate run; `<run>.log.jsonl`
— training curves (the split arm's `loss_masked_values` staying ~9.5 while
train loss falls is the mechanism signature). n200k files are the round-2
(remediated-mixture) runs; n50k/n800k are round-1.

## Run it yourself

```bash
uv venv .venv --python 3.12 && uv pip install -r requirements.txt --python .venv/bin/python
.venv/bin/python -m pytest tests -q        # 124 tests, ~30 s, offline
PYTHONPATH=. .venv/bin/python scripts/smoke_test.py   # end-to-end toy pilot (~10 min CPU)
```

The smoke test builds a toy corpus, trains both arms, and asserts the
mechanism (masked-fact CE stays high while language loss falls), then runs
the eval pipeline with live organizer interception.

Cluster battery: `cluster/RUNBOOK.md` (FarmShare; one Duo-auth step, then
scripted).

## Contact

Stephen Zhang. Repo state = git HEAD at packaging time; battery status
changes daily — ask before citing schedule numbers.
