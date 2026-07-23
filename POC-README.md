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
Two open-book tasks: **fact-QA** (single-hop) and **reason-over-facts** (yes/no
comparison over two facts — the answer isn't any single fact, so it requires
combining them).

## Fair evaluation — three conditions

| Condition | facts in prompt? | what it shows |
|---|---|---|
| **DENSE @ closed-book** | no | parametric recall from weights |
| **DENSE + context** | yes | dense with facts available |
| **SPLIT + context** | yes | split's trained mode |

Both arms trained **with** context blocks, so `+context` is in-distribution for
both — the earlier "split was never trained to read context" asymmetry is gone;
the only difference is whether facts also live in the weights. (SPLIT @
closed-book is omitted — split never memorized, so it can't answer without
context.) Held-out facts (never in training context) test generalization.

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

## Knobs (`scripts/poc_run.py`)

`--stage {build,train,eval,report,all}`, `--model {toy,d160m,d360m}` (default
d160m), `--steps`, `--fresh` (retrain), `--judge` (LLM-graded fact-QA),
`--max-facts` (fact dose), `--exposures`, `--reason-train`, `--bed-docs`,
`--heldout`/`--seen`/`--reason-eval` (eval sizes), `--limit`.

## Reading the results

- **Fact-QA, held-out:** DENSE @ closed-book ≈ 0 (never saw them); +context
  conditions can answer. Partly definitional — it shows offloading generalizes.
- **Fact-QA, seen:** the fair capacity comparison — SPLIT + context vs DENSE
  + context on facts both trained on.
- **Reason-over-facts:** the reasoning-capacity test (answer ≠ any single fact),
  shown against the **majority-class baseline** — near-baseline for both arms
  means the signal is weak at this scale.
- **DENSE + context vs SPLIT + context** is the headline: both get the same facts
  in context; if SPLIT reasons better, that's freed capacity, not retrieval access.

**Caveats (honest).** The PopQA fact dose is small (~3k facts), which may not
stress a 162M model's capacity — the capacity effect can be weak or null here;
raise `--max-facts` / scale entities for a real dose (PopQA itself caps at ~3k).
The token budget is far smaller than the team's cluster runs, so treat the
reasoning numbers as indicative, not final.
