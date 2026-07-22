# MemorySplit — Provenance, Findings & Guide (single source of truth)

*This one file consolidates what used to be spread across `README.md`, `docs/README-SHARE.md`,
`HANDOFF-AGENT.md`, and `STATUS.md`. Deep design docs (preregistration, interim report, design
spec) stay where they are and are linked at the bottom. Newcomers can read this top-to-bottom.*

---

## 1. What this project is (plain English)

**The idea.** Small language models spend a lot of their limited capacity **memorizing facts**.
The "memory-split" bet: if a model **looks facts up in an external database** instead of
memorizing them, it can spend that freed capacity on **reasoning** — and its knowledge becomes
editable/auditable as a bonus.

**The experiment.** Two identical **twin** models (160M params) train on the exact same made-up
biographies. They differ in one way — how they handle **fact values** (birth city, employer, …):
- **Dense twin** — graded on every word, so it must **memorize facts in its weights**
  (answering with no help = "closed-book").
- **Split twin** — each fact value is wrapped in a **look-it-up call** and **loss-masked**, so
  the model only learns *when to ask and how to phrase the query*; at inference an external
  **organizer** (a plain `(entity, relation) → value` table) supplies the value.

Everything else is **paired**: same init seed, same data order, same content; the only
difference between arms is the loss mask + lookup wrapping.

**The dial ("fact load").** Three sizes — **50k / 200k / 800k people** (n50k/n200k/n800k). The
training budget is fixed, so more people = each seen fewer times (**196 / 49 / 12** exposures,
verified) = more memorization pressure on the dense twin.

**The three hypotheses (referenced as H1 / H2 / H3 everywhere below):**
- **H1 — the payoff (the primary claim):** at the *same* parameter and token budget, the
  **split** model **reasons better** than its **dense** twin, because not spending capacity on
  memorizing facts frees it up for reasoning.
- **H2 — no capability tax (a guardrail):** off-loading facts does **not hurt** the model's
  general/language ability. (H1 only matters if H2 holds — you can't "win" by breaking the model.)
- **H3 — the mechanism (the precondition):** in the split model, facts genuinely live
  **outside** the weights — in the external organizer — not memorized inside them. This is what
  *makes* H1 possible, and it's the thing the mech-interp probes mostly test.

Primary preregistered endpoint = held-out **fact-use QA** (split scored with its organizer,
dense closed-book). Guardrails/margins are frozen in
`docs/superpowers/specs/2026-07-20-preregistration.md`.

---

## 2. Findings scoreboard (one seed ⇒ directions, not proofs)

- **H3 — strongly established.** Split recall = **≈100% with its table, 0% without**; split
  weights hold **≈0** fact-value information; ablating dense's fact-neurons **halves** its
  recall; the split model's knowledge is **editable** (update/add/delete) with no retraining.
- **H2 — supported.** Knowledge-free reasoning likelihood is identical across arms; nothing
  shows splitting *hurts*.
- **Safety bonus.** On a store miss the split model **does not fabricate** a fact (0%) — it was
  never trained to generate values, so it can't confabulate them (dense *does* confabulate).
- **H1 — NOT established (the crux).** On the one learnable reasoning task (**deduction**, ≈65%,
  above its 50% floor) there is **no consistent split advantage at any fact load** — the gap
  even flips sign (split +3pp at n200k, dense +5.5pp at n800k), and with one seed per arm those
  are within run-to-run noise. **iGSM** (mod-23 arithmetic, ≈12%) is **task-blocked** — hard for
  this scale, so it can't test H1. This matches Stephen's 0.8B/20% null.

**In one line:** the split model genuinely externalizes facts, generalizes addressing in-domain,
is editable, and doesn't hallucinate; the dense twin hits a memorization wall — but **splitting
has not (yet) been shown to buy better reasoning.**

---

## 3. What each probe does & found (results live in `outputs/mechinterp/<probe>/RESULTS.md`)

**Storage — "where do the facts live?"**
- `mechanism` — split=100%/0% recall with/without table; dense memorizes 62% @n50k → ≈1% higher;
  dense's fact-neurons are 3–6× quieter in split (alignment-free confirmed; split builds its own
  *different* selective units). **Non-null.**
- `double_dissociation` — **causal:** ablating dense's 64 fact-neurons drops recall 0.66→0.35 vs
  ≈0 for random. **Non-null (strongest causal).**
- `ppl_slices` — split store-OFF fact-value likelihood pinned at the ceiling (≈+7 nats vs dense);
  reasoning likelihood equal across arms. **Non-null.**
- `extractability` — dense @n50k is stored **and** producible (recall .64, recognize .90);
  @200k/800k genuinely *absent* (recognition control passes); split absent. **Non-null.**
- `fact_ledger` — memorization wall in bits/person: **33 → 0.2** (n50k → n200k/800k). *(derived)*
- `weight_spectral` — **null** (no dense-vs-split difference at whole-matrix effective rank).
- `fact_weight_attribution` — cross-arm claim **collapsed** on trained-people data; inconclusive.

**Using the store**
- `keyguess` — split writes the correct lookup address ≈100% for **unseen** synthetic people →
  a general copy+select procedure. **Non-null.**
- `popqa_keyguess` — that skill **does not transfer** to real-world entities (fires <3%, never
  copies a real name) → distribution-bound. **Non-null (a limitation).**
- `editability` (incl. add/delete + hallucination guardrail) — update/add/delete all 99–100%,
  no collateral; 0% hallucination on a miss. **Non-null.**
- `value_injection` — retrieved value's effect builds across depth *(peak-at-end partly a
  norm artifact — caveated).*

**Internal geometry**
- `superposition` — dense fact-activations **collapse** (PR ≈1.8) at high load; split stays roomy
  (≈6). **Non-null.**
- `geometry_cka` — arms differ on fact processing, **converge with load** (0.36→0.76 vs a 0.95
  generic ceiling); a follow-up test showed the divergence is *not* per-memorized-fact. **Non-null.**
- `attention_heads` — fact production leans on mid/late attention (peak L8–9); *not yet shown to
  be fact-specific — caveated.*

**Reasoning (H1)** — `h1_deduction` (the null above); iGSM task-blocked; deeper causal reasoning
probes (`step_patch`, `intermediate_value`, `faithfulness`, …) are gated (need working reasoning
and/or gold chain-of-thought).

---

## 4. Provenance & caveats (READ before citing or reusing)

**The trainer that produced everything is `train/trainer.py` (v1) — UNCHANGED.** Every
checkpoint and result in this repo was produced by v1 (equal-weighted gradient accumulation).
Verified: `train/trainer.py` still has the old `(loss/accum).backward()` / mean-of-means logic.

**`train/trainer_v2.py` is NEW and PROPOSED — it trained NOTHING here.** The `evals_and_edits`
bundle proposed a token-weighted gradient-accumulation fix; it is **vendored alongside v1** as
`train/trainer_v2.py` (with a big status banner) purely so the proposed change is reviewable.
It was **deliberately not applied** to any run (frozen preregistered code; owner sign-off
required — `docs/deferred/ESCALATIONS-frozen-code.md` §1). It's a no-op for the dense arm but
would **change split-arm training**, so adopting it requires **retraining the split runs** — do
not treat v2 as tested/validated. (A second deferred edit — adding gold chain-of-thought to
eval metadata, `corpusgen §2` — is also unapplied.)

**Other additions are analysis-only (new, not part of the original tested battery).** The
`probe/` toolkit, the added `evals/*.py` instruments (`keyguess`, `frozen`, `extractability`,
`mechanism`, `interp`, `perplexity`, `editability`, …), and new `scripts/run_*.py` are **new**;
they produced the `outputs/mechinterp/` analysis but were **not** part of the original
training/eval battery and change no training. The original battery evals are the pre-existing
`evals/` scorers + `scripts/run_evals.py`.

**The mech-interp was run against this ("old") code — with one wrinkle you must know.** The
repo's live entity generator (`corpusgen/bios.py`) has **drifted** from the generator that built
the training corpus, so regenerating entities at probe time yields people the model was **not**
trained on. Storage/memorization probes are therefore run on the **frozen trained entities**
(reconstructed via `evals/frozen.py` from the frozen `eval/recall.jsonl`), not by regenerating.
Some early probe passes used the drifted ("unseen") entities and were re-run on the frozen
("seen") data. **Every `outputs/mechinterp/<probe>/RESULTS.md` carries a seen/unseen banner**,
and `outputs/mechinterp/ENTITY-POPULATIONS.md` explains the whole issue.

**Standing caveats for all results:**
- **One seed per arm** ⇒ read findings as directions; a few-pp task difference is within
  training-seed noise (this is exactly why the H1 deduction result is a null, not a claim).
- **Most probes are correlational** (they show what's present/decodable); the causal ones are
  `double_dissociation` (recall) and `editability` — flagged as such.
- **Deferred/held changes** live in `docs/deferred/` and are NOT applied.

---

## 5. Run status (as of 2026-07-21)

- **160M battery — done:** all 6 seed-0 runs (`d160m_{dense,split}_{n50k,n200k,n800k}_s0`)
  trained to 3.2B tokens, with `ckpt.pt` + `evals/summary.json` + snapshots.
- **1B confirmation — still training:** `d1b_dense_n800k_s0_gate`, `d1b_dense_n4m_s0_gate`
  (dense-only, 2 Slurm jobs running, latest snapshot ≈step 1716, no eval yet).
- Account A's remit was the corpora + seed-0 sweep + calib; the 1B confirmation is
  Stephen's/AWS (see the deep docs). Seed-1 sweep is Account B (yuenkai).

---

## 6. Next steps (to actually test H1)

1. **Multiple seeds** on the split-vs-dense deduction comparison — the single most important fix;
   converts "±5pp could be noise" into a real effect-size ± CI.
2. **Make reasoning learnable / give it headroom:** replace mod-23 iGSM with a learnable task;
   deepen deduction beyond depth 1–2 (65% may be near-ceiling).
3. **Isolate reasoning from memory:** give the dense arm open-book facts so any gap is capacity,
   not fact-access.
4. **Test at the capacity-saturated load** (n800k / n4m), where the freed-capacity effect, if
   real, should be largest.
5. Once reasoning clears chance robustly, the gated causal reasoning probes unlock.

---

## 7. How to run

**Local (macOS), offline:**
```bash
uv venv .venv --python 3.12 && uv pip install -r requirements.txt --python .venv/bin/python
export PYTHONPATH=.
.venv/bin/python -m pytest tests -q          # unit tests (offline)
.venv/bin/python scripts/smoke_test.py       # end-to-end toy pilot (CPU/MPS)
```
**Re-run a probe on the finished checkpoints** (need the checkpoints pulled to
`outputs/_local_probe/` and frozen eval data in `outputs/_frozen_data/`):
```bash
PYTHONPATH=. .venv/bin/python scripts/run_probe_suite.py --dense outputs/_local_probe/d160m_dense_n200k_s0 \
    --split outputs/_local_probe/d160m_split_n200k_s0 --load n200k --records-jsonl outputs/_frozen_data/n200k_recall.jsonl --tag _seen
```
**Cluster battery:** `cluster/RUNBOOK.md` and `HANDOFF-AGENT.md` §3–§8 (FarmShare/Slurm; one
Duo-auth step, then scripted).

---

## 8. Repo layout & where to read more
```
corpusgen/  seeded generators (bios = the fact dose, iGSM math, deduction, fact-use QA)
organizer/  exact-match (entity, relation) -> value store
train/      tokenizer, GPT model, masked dataloader; trainer.py = v1 [UNCHANGED, trained everything];
            trainer_v2.py = PROPOSED token-weighted fix [NEW, trained nothing — see §4]
evals/      generative scorers w/ lookup interception, recall/bits accounting  (+ added probe instruments)
probe/      mech-interp toolkit (geometry, weights, attention, splice, ledger, causal_steps, …)  [ADDED]
scripts/    build_corpus, run_train, run_evals, + run_* probe drivers  [probe drivers ADDED]
cluster/    FarmShare (Slurm) scaffolding + RUNBOOK.md
outputs/mechinterp/   probe results (START-HERE.md, PROBES-INDEX.md, ENTITY-POPULATIONS.md, <probe>/RESULTS.md)
docs/       design spec, preregistration, interim report, research dossier, deferred/ (unapplied edits)
```
**Deep docs (unchanged, authoritative for design):**
- Preregistration (frozen): `docs/superpowers/specs/2026-07-20-preregistration.md`
- Interim report (full story + pilot results): `docs/superpowers/2026-07-20-interim-report.md`
- Design spec: `docs/superpowers/specs/2026-07-17-memory-split-design.md`
- Probe details: `outputs/mechinterp/START-HERE.md` → `PROBES-INDEX.md` → each `RESULTS.md`
- Operational log of this analysis work: `OPS-LOG-ROLE-A.md`

