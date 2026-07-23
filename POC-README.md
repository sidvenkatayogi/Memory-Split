# PoC: Facts in Context (Memory-Split, no DB retrieval)

**Question.** At a fixed parameter budget, does a model that *offloads facts to
context* — never memorizing them — reason better than a dense twin that stores
them in weights?

No database, no retrieval interface, no `<|db_*|>` tokens. Every fact simply
appears in a `Context:` block in the training text. The two arms train on the
**same corpus**; the **only difference is one loss mask**:

- **DENSE** — loss ON everywhere → trained to predict the fact value, so it
  **memorizes** `(subject, relation) → value` into its weights.
- **SPLIT** — loss OFF on the fact **value** inside the context → never trained
  to produce it, so it **doesn't memorize** it; it must READ it from context to
  answer (the Answer line keeps loss ON). Facts offloaded, weight capacity freed.

```
Context: The director of Film A is Xavier Dolan.     <- SPLIT masks "Xavier Dolan" here
Question: Who directed Film A?
Answer: Xavier Dolan                                  <- loss ON for both arms
```

Facts are real Wikidata triples (PopQA, from the team's `corpusgen/realfact.py`).
Three open-book tasks:
- **fact-QA** (single-hop): recall/copy one fact.
- **reason-over-facts** (yes/no comparison over two facts — the answer isn't any
  single fact, so it requires combining them).
- **pure reasoning** (`puremath`): compute a named operation (e.g. `a mod b`);
  the operation's **definition** is the "relevant fact" that can be given in
  context ("mod means the remainder…") — a closer starting point, not the answer.
  Split masks the definition; dense memorizes it.

## Fair evaluation — four conditions

Closed-book **and** +context for **both** arms, so "does context help?" is a
within-arm comparison (which is the point of the pure-reasoning task):

| Condition | context in prompt? | note |
|---|---|---|
| **DENSE @ closed-book** | no | parametric recall / no scaffold |
| **DENSE + context** | yes | dense with the fact/definition available |
| **SPLIT @ closed-book** | no | split without context (facts ≈ 0 by construction; meaningful for pure reasoning) |
| **SPLIT + context** | yes | split's trained mode |

Both arms trained **with** context blocks, so `+context` is in-distribution for
both — no format asymmetry; the only difference is whether the fact also lives
in the weights. Held-out facts (never in training context) test generalization.

## How it works

- `corpusgen/poc_build.py` — builds the corpus: fact-QA + comparison docs, each
  rendered for both arms (split masks the context value via the `(text, masked)`
  segment API — no DB tokens). Writes `{dense,split}/train.bin` (+ mask) and
  `eval/{factqa,reason}.jsonl`.
- `evals/context_eval.py` — generates answers for each condition (plain greedy,
  no retrieval) and grades: fact-QA by string-match or an LLM judge; reasoning by
  a yes/no check.
- `evals/gpt_oracle.py` — `GatewayClient` + `judge_answer` (GPT-5.6-sol), used
  **only to grade** free-form fact-QA answers (credits paraphrases/aliases).
- `scripts/poc_run.py` — `build → train (both arms) → eval → report`.

## Run it (Colab)

Open `notebooks/poc_optimal_retriever.ipynb` on a GPU runtime, run top to bottom.
Paste a TrueFoundry token (only for grading). Checkpoints persist to Drive under
`runs/<model>_incontext/` and resume on a disconnect.

## Run it (local)

```bash
uv venv .venv --python 3.12 && uv pip install -r requirements.txt --python .venv/bin/python
export PYTHONPATH=.
.venv/bin/python scripts/poc_run.py --stage build
.venv/bin/python scripts/poc_run.py --stage train --device auto --steps 4000
set -a && . ./.env && set +a          # TrueFoundry creds, for --judge
.venv/bin/python scripts/poc_run.py --stage eval --judge
.venv/bin/python scripts/poc_run.py --stage report
.venv/bin/python -m pytest tests/test_poc.py -q     # offline wiring tests
```

## Picking the model size (the pilot)

Capacity ≈ params, and the real wall is the **token budget**, so at a fixed
overnight budget a *large* model is barely stressed (d160m sits at ~3% capacity
utilization) — you want a **small** model so a feasible fact dose actually
crowds it. But too small and it can't reason at all (below width ~256 the vocab
embeddings dominate the params and the transformer gets thin). So:

```bash
python scripts/poc_run.py --stage pilot --model mini   # 13.7M
python scripts/poc_run.py --stage pilot --model toy    # 29M
```

`--stage pilot` trains reasoning-only (no fact dose) and reports whether the
model beats chance. **Pick the smallest size that's ABOVE floor** for the full
run — that maximizes capacity pressure while keeping the reasoning signal
measurable. Presets: `micro` 6.7M · `mini` 13.7M · `toy` 29M · `d160m` · `d360m`.

## Knobs (`scripts/poc_run.py`)

`--stage {pilot,build,train,eval,report,all}`,
`--model {micro,mini,toy,d160m,d360m}` (default d160m — but smaller is usually
right for the capacity test; see the pilot), `--steps`, `--fresh` (retrain),
`--judge` (LLM-graded fact-QA), `--max-facts` (fact dose), `--exposures`,
`--reason-train`, `--puremath-train`, `--bed-docs`,
`--heldout`/`--seen`/`--reason-eval`/`--puremath-eval` (eval sizes), `--limit`.

## Reading the results

- **Fact-QA, held-out:** DENSE @ closed-book ≈ 0 (never saw them); +context
  conditions can answer. Partly definitional — it shows offloading generalizes.
- **Fact-QA, seen:** the fair capacity comparison — SPLIT + context vs DENSE
  + context on facts both trained on.
- **Reason-over-facts:** the reasoning-capacity test (answer ≠ any single fact),
  shown against the **majority-class baseline** — near-baseline for both arms
  means the signal is weak at this scale.
- **Pure reasoning (`puremath`):** does the definition-in-context help?
  Compare **closed-book vs +context within each arm** — if +context > closed-book,
  the relevant fact (definition) gave the model a closer starting point. Compare
  **dense vs split** to see whether offloading freed capacity for the computation.
- **DENSE + context vs SPLIT + context** is the headline across tasks: both get the
  same context; if SPLIT does better, that's freed capacity, not context access.

**Caveats (honest).** The PopQA fact dose is small (~3k facts), which may not
stress a 162M model's capacity — the capacity effect can be weak or null here;
raise `--max-facts` / scale entities for a real dose (PopQA itself caps at ~3k).
The token budget is far smaller than the team's cluster runs, so treat the
reasoning numbers as indicative, not final.
