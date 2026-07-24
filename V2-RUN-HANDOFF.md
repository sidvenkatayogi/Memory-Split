# Memory-Split v2 — Run Handoff & Readiness Review

**Branch:** `v2/wikidata5m-mh` (fork `sidvenkatayogi/Memory-Split`), latest commit `9bca240`.
**Purpose of this doc:** everything a reviewer needs to confirm the v2 pipeline is ready to launch the full run, plus the exact commands. This is **my own work** (sid's), built on top of the team base; it deliberately does **not** depend on teammates' experiment design.

> **Status:** pipeline BUILT + component-verified. **Not yet launched.** One paid run (~$300) is the only remaining step, gated on this review. See §7 (verified vs not) — that's the important section for a reviewer.

---

## 1. The experiment

**Hypothesis (capacity crowding).** At a fixed parameter budget, a model that **offloads facts to context (SPLIT)** — never memorizing fact values — should have more weight capacity left for **reasoning** than a **dense twin (DENSE)** that memorizes the same facts. The effect should appear only when (a) the reasoning is hard enough to need the freed capacity and (b) the fact dose is large enough to actually pressure DENSE's memory.

**Why v2 differs from the earlier PoC.** The earlier single-hop PoC showed SPLIT ≥ DENSE on fact recall but **no reasoning gain** — because single-hop reasoning is too easy to reveal the effect and ~3k PopQA facts don't pressure a 162M model (≈4 orders of magnitude too few). v2 fixes both:
- **Multi-hop reasoning** over facts (chains + aggregation), not single-hop recall.
- **Wikidata5M** fact source at a large **dose** (up to ~20M triples) to create real capacity pressure.

**Arms (matched twins).** Identical architecture, corpus, tokens, init, seed — the **only** difference is the loss mask on fact **values**:
- **DENSE** — loss ON everywhere → memorizes `(subject, relation) → value` in weights.
- **SPLIT** — loss OFF on retrieved fact-value spans → never memorizes them; must read them from context. Still trained on the reasoning plan, queries, and final answer.

---

## 2. Task design (decisions already made)

- **Multi-hop chains**, depth ≤ 3, over **functional** entity→entity relations (single unique object per hop) so each chain has one gold answer. Question phrased nested: *"What is the country of the place of birth of X?"*.
- **Aggregation over a branch** for one-to-many relations: *"How many cast members of X have country of citizenship France?"* → a **count** (compute over retrieved values = genuine reasoning, and how one-to-many hops are handled without ambiguity).
- **Grading: Claude Sonnet 5** (`claude-group/claude-sonnet-5`) as LLM judge (credits paraphrases/aliases); aggregation graded numerically.
- **Fixed-exposure dose ladder**: train each dose to the same number of exposures (default 30). Ladder rungs (e.g. 1M / 5M / 20M entities) trace the SPLIT−DENSE gap vs fact load. The 20M rung is this run.

---

## 3. What's on the branch (all new/own unless noted)

| File | Role |
|---|---|
| `corpusgen/wikidata5m.py` | Wikidata5M offline loader: alias index, triple graph, functional-relation detection, **stratified dose subsetting**, single-hop fact records. |
| `corpusgen/multihop.py` | Multi-hop generator: depth-≤3 chains + aggregation; renders **retrieve-then-reason** traces (values masked for SPLIT, answer loss-on). |
| `corpusgen/mh_build.py` | Builds one corpus at a dose: atomic facts + multi-hop QA + FineWeb-style bed → `{dense,split}/train.bin` (+ `.mask.bin`), `eval/{multihop,factqa}.jsonl`, `build_report.json`. Reuses PoC helpers (`_factqa_doc`, `_toy_bed_docs`, `_encode_arm`). |
| `scripts/build_mh_corpus.py` | Dose → corpus → emits per-arm Trainer config YAMLs (with `s3_ckpt`). |
| `train/trainer.py` (edited) | Added **multi-GPU DDP** (rank-sharded data, `no_sync` on non-final micro-steps, rank-0 logging/ckpt, global-batch accounting) + **S3 checkpoint push/pull-to-resume**. Single-GPU path preserved. |
| `train/data.py` (edited) | Rank-shards the packed token stream into disjoint per-rank regions. |
| `scripts/train_ddp.py` | torchrun entrypoint; asserts exact global batch; S3 resume then load. |
| `cluster/torchrun_poc.sh` | torchrun launcher (defaults NPROC to visible GPUs). |
| `evals/context_eval.py` (edited) | `build_prompt`/`_grade` now handle `mh_chain`/`mh_agg`; lazy torch imports so prompt/grade logic is unit-testable. |
| `evals/gpt_oracle.py` (edited) | Grader = `claude-group/claude-sonnet-5`; **omits `temperature`** (Sonnet-5 rejects it); larger `max_tokens`. |
| `scripts/eval_mh.py` | Eval runner: DENSE/SPLIT × {closed-book, +context} over multihop+factqa, Claude-judged, headline table. |
| `tests/test_{wikidata5m,multihop,mh_eval}.py` | Offline unit tests (pass). |
| `cluster/{smoke_gpu,s3_test}.sh` | The GPU smoke + S3 durability test scripts (already run green). |

---

## 4. How training works (masking detail)

Each corpus doc is rendered twice — `dense_segments` / `split_segments` — as `(text, masked)` spans; `masked=True` → label `-100` (no loss). A 2-hop chain (⟦…⟧ = masked for SPLIT only):
```
Context: The place of birth of Douglas Adams is ⟦Cambridge⟧. The country of Cambridge is ⟦United Kingdom⟧.
Question: What is the country of the place of birth of Douglas Adams?
Reasoning: The place of birth of Douglas Adams is ⟦Cambridge⟧. The country of Cambridge is ⟦United Kingdom⟧. So the answer is United Kingdom.
Answer: United Kingdom
```
DENSE: nothing masked (memorizes). SPLIT: retrieved values masked in Context **and** Reasoning; plan text, relation phrases, and the final `Answer:` stay loss-on (so SPLIT learns to reason + copy, not to store values). Aggregation docs mask the branch objects + their attribute values; the computed **count** is loss-on for both.

Training: `train_ddp.py` → `trainer.py`, DDP over 8 GPUs, AdamW + cosine + bf16, `tokens_per_step` = global batch (524288), checkpoint every `ckpt_minutes` (15) pushed to S3.

---

## 5. How eval works (how facts are supplied)

**No DB/retriever** — the "optimal retriever" is abstracted as: the relevant **gold atomic facts are placed in the `Context:` block** of the prompt. Four conditions per item:

| Condition | Context block | tests |
|---|---|---|
| DENSE @ closed-book | no | parametric recall + reason from memory |
| DENSE + context | yes | dense with facts available |
| SPLIT @ closed-book | no | ~0 by construction (manipulation check) |
| SPLIT + context | yes | SPLIT's trained mode |

**Headline = DENSE+context vs SPLIT+context** (same facts in front of both → a gap is freed capacity, not access). Prompt ends at `Reasoning:` to elicit CoT; `parse_answer` reads after the last `Answer:`. Grading: Claude Sonnet 5 judge for entity answers, numeric match for aggregation counts.

---

## 6. Runbook (exact commands)

**Provision a p4d** (sbsandbox; see §8 for the infra facts):
```bash
aws ec2 run-instances --image-id ami-06dfbba2881736cd8 --instance-type p4d.24xlarge \
  --iam-instance-profile Name=EswManagedInstance --security-group-ids sg-048178b0708d33ff2 \
  --subnet-id <public subnet in vpc-08fdb401672d10c74, an AZ WITH p4d capacity> \
  --instance-initiated-shutdown-behavior terminate \
  --user-data $'#!/bin/bash\nshutdown -h +780' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=memorysplit-d160m-sid},{Key=Owner,Value=sid},{Key=Purpose,Value=memorysplit-training}]'
```
Drive it via SSM (`aws ssm send-command … AWS-RunShellScript`). Bucket `memorysplit-sid-056956104102` already has a policy granting the `EswManagedInstance` role Get/Put/Delete/List (see §8 — this is required and non-obvious).

**On the box** (bootstrap):
```bash
git clone --branch v2/wikidata5m-mh --single-branch https://github.com/sidvenkatayogi/Memory-Split.git
cd Memory-Split && export PYTHONPATH=$PWD
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip && pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
pip install numpy pyyaml tqdm tiktoken openai huggingface_hub

# Wikidata5M (verify the extracted filenames match corpusgen/wikidata5m.py:Wikidata5MPaths — see §7)
hf download intfloat/wikidata5m --repo-type dataset --local-dir data/wikidata5m
#   → need: wikidata5m_transductive_train.txt, wikidata5m_inductive_test.txt,
#           wikidata5m_alias/wikidata5m_entity.txt, wikidata5m_alias/wikidata5m_relation.txt
#     (the HF distribution ships these as .tar.gz / .txt.gz — extract + place accordingly)

# TrueFoundry / Claude judge creds (token NOT in repo; supply it):
export OPENAI_API_KEY=<PROMPTLENS token>
export OPENAI_BASE_URL=https://tfy.promptlens.trilogy.com/v1
export POC_GPT_MODEL=claude-group/claude-sonnet-5
```

**Build the 20M corpus + configs:**
```bash
python scripts/build_mh_corpus.py --wikidata-root data/wikidata5m \
  --dose 20000000 --exposures 30 --model d160m --micro-bs 8 \
  --s3-ckpt-prefix s3://memorysplit-sid-056956104102/runs --out data/mh/n20m
# writes configs/mh/d160m_n20000000_{dense,split}.yaml  and  data/mh/n20m/**
# INSPECT data/mh/n20m/build_report.json: n_atomic_facts (the real dose), component_docs, masked_frac
```

**Train both arms** (sequential = each on 8 GPUs; or 4+4 concurrent):
```bash
NPROC=8 bash cluster/torchrun_poc.sh configs/mh/d160m_n20000000_dense.yaml
NPROC=8 bash cluster/torchrun_poc.sh configs/mh/d160m_n20000000_split.yaml
# concurrent alternative:
# CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash cluster/torchrun_poc.sh ..._dense.yaml &
# CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 bash cluster/torchrun_poc.sh ..._split.yaml &
```

**Eval:**
```bash
python scripts/eval_mh.py --corpus data/mh/n20m \
  --dense-run runs/d160m_n20000000_dense --split-run runs/d160m_n20000000_split \
  --judge --device auto
# writes data/mh/n20m/eval_mh_results.json + headline table
```

---

## 7. VERIFIED vs NOT — the review checklist

**Verified (green):**
- ✅ Loader logic, multi-hop generator logic — offline unit tests.
- ✅ Corpus build end-to-end — offline fixture **and** on a real GPU (g5.12xlarge): dense masked_frac 0.0 vs split 0.073.
- ✅ Multi-GPU **DDP training** — 4×A10G smoke: exact global batch (`8192 = mbs4×ctx256×ws4×accum2`), checkpoint written.
- ✅ **S3 checkpoint durability** — push → delete local → `--resume auto` pulled from S3 and **resumed at the right step** (log showed steps 4,5,6 after a step-3 checkpoint).
- ✅ **Claude Sonnet 5 grader** — gateway HTTP 200; client omits `temperature`.
- ✅ Multi-hop eval **prompt/grade logic** — offline unit tests.

**NOT yet verified — a reviewer/first-run should confirm these before/at launch:**
1. **Wikidata5M filenames** — `Wikidata5MPaths` defaults must match what `hf download` extracts. The HF repo ships tarballs; confirm the four expected files exist at those paths (or pass overrides). *Most likely first failure point.*
2. **20M corpus build at scale** — only fixture-scale built. Check: build wall-time (tokenizing ~hundreds of M tokens is single-threaded and may take 30–60+ min), host RAM, and that `generate()` actually yields enough chains/aggregations at scale. **Inspect `build_report.json`** — `n_atomic_facts` is the *true* dose (only functional-relation triples become atomic facts; it may be < 20M).
3. **S3 sync under 8-GPU DDP on p4d** — verified only on single-node CPU (2-proc) + a separate 4-GPU smoke. Watch the **first checkpoint cycle** land in S3 on the real box.
4. **`eval_mh.py` end-to-end on GPU** — only the string logic is unit-tested; model-loading + generation are unexercised. Watch the **first eval batch**. Check `--max-new 192` is enough for depth-3 CoT (bump if answers get truncated before `Answer:`).
5. **Throughput / wall-time** — my ~25–40B tokens/10h (160M, 30–45% MFU) is an estimate. Measure tokens/s on ~200 steps and extrapolate before trusting the ~10h budget.
6. **p4d capacity** — scarce; may need a specific AZ or the p4de (80 GB) pool. (g5 was already capacity-out in one AZ during testing.)
7. **Exposures/dose calibration (science, not code)** — is 30 exposures over the 20M dose enough for multi-hop reasoning to *grok*? Full dose maximizes capacity pressure but minimizes exposures for a fixed budget (facts × exposures = fixed tokens). Consider starting a smaller rung (1M/5M) too.
8. **The scientific result is not guaranteed** — this validates the *machinery*; whether SPLIT beats DENSE on reasoning at this scale is the open question the run answers.

---

## 8. sb-aws infra playbook (account 056956104102 / sbsandbox)

- **AMI** `ami-06dfbba2881736cd8` (NVIDIA drivers baked; torch pip-installed at runtime). **IAM profile** `EswManagedInstance` (SSM-managed — drive via `ssm send-command`, no SSH key). **SG** `sg-048178b0708d33ff2`, **VPC** `vpc-08fdb401672d10c74`.
- **No default VPC** → must pass a public `--subnet-id` (e.g. 1b `subnet-0d88e99ba10258ee0`, 1d `subnet-04dd46c921c40074f`). GPU capacity varies by AZ.
- **Deliver code via `git clone`** of the fork — the instance role cannot read arbitrary S3 (it 403'd fetching a bootstrap script).
- **S3 checkpoints work only because the bucket policy names the role ARN explicitly**: `Principal: arn:aws:iam::056956104102:role/EswManagedInstance`. Granting `:root` is **not** enough (S3 treats root-principal as "defer to identity policies," which this instance role lacks for the bucket). Already configured on `memorysplit-sid-056956104102`.
- **Cost/safety:** p4d ≈ $32/hr (~$300 for ~10h). Always set `--instance-initiated-shutdown-behavior terminate` + a `shutdown -h +<minutes>` dead-man switch in user-data.
- **Do not touch** teammate `adarsh`'s two 8×A100 boxes (Owner=adarsh, `memorysplit-d360m-adarsh`).

---

## 9. Notes for the reviewer
- Everything here is intended to stand alone from the team's design; if something references teammate code, it's only the shared trainer/infra plumbing, not the experiment.
- The single most valuable checks are **§7 items 1–4** (Wikidata5M paths, the real build at scale + `build_report.json`, first S3 checkpoint on 8-GPU, first eval batch). If those pass on the live box, the run is ready.
