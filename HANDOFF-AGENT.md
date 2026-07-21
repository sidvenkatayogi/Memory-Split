# AGENT HANDOFF — run the Memory Split battery in a fresh FarmShare account

You are an agent taking over execution of a preregistered ML experiment.
This document is self-contained: read it top to bottom before running
anything. The previous account (syz) hit its scratch quota mid-battery on
2026-07-20; your job is to run the battery in a new account, exactly as
preregistered, and hand the results back.

## 0. What this project is (60 seconds)

Hypothesis: a small LM trained so that facts are never rewarded into its
weights (they are wrapped in database lookup calls, values loss-masked,
served at inference by an exact-match key-value store called the
organizer) learns better reasoning-over-facts than a dense twin trained
identically otherwise. Everything is paired: same init seed, same data
order, same content; the ONLY difference between arms is the loss mask
plus lookup wrapping. The fact load (number of synthetic entities) is a
swept dose. Primary endpoint (H1): held-out fact-use QA, split arm scored
with its organizer, dense closed-book. Guardrails H2/H3 and all margins
are FROZEN in `docs/superpowers/specs/2026-07-20-preregistration.md`.
Read that file before anything else, and do not deviate from it: if a
step seems to require changing an endpoint, margin, seed, or exclusion
rule, STOP and escalate to Stephen instead.

Background reading order (all in this package): the preregistration, then
`docs/superpowers/2026-07-20-interim-report.md` (full story + pilot
results), then `docs/superpowers/specs/2026-07-17-memory-split-design.md`
(design rationale), then `cluster/RUNBOOK.md` (operational lore).

## 1. What is already done (do not redo)

- All code is built and tested (124 unit tests + slow pipeline test).
- Gate phase is complete on the old account: mechanism proven 3x
  (results in `results/` and in the interim report); gate A failed 3x at
  chance, which is WHY the primary endpoint is fact-use QA; the
  preregistration is frozen and committed.
- The battery was submitted on the old account and then cancelled when
  its 261GB scratch usage hit quota. Four of five corpus builds had
  completed there; ONE sweep run (d160m_dense_n50k_s0) trained for ~80
  minutes before cancellation. None of that state is needed: corpora
  rebuild byte-identically from code (deterministic seeds), and the one
  partial run is simply rerun. You start clean.

## 2. What you will run (the whole remaining battery)

1. 5 corpus builds (3x 160M loads at 3.2B tokens, 2x 1B loads at 10B).
2. 160M sweep: 12 runs (3 loads x 2 arms x seeds 0,1), ~8-10 L40S-hours
   each.
3. calib1b: 2 short dense 1B runs (1.5B tokens, ~20 h each).
4. Apply the preregistered calib rule (mechanical, section 5 below).
5. 1B confirmation: 4 runs (2 arms x seeds 0,1) at ~130-160 h each,
   submitted as dependency chains.
6. Eval battery for every finished run; analysis; hand back results.

Steps 1-3 are one command (`cluster/run_battery.sh`), fully
dependency-chained. Only steps 4-5 need a decision, and it is mechanical.

## 3. Bring-up in the new account (~30 min + queue time)

Everything runs ON FarmShare; no laptop-side tooling is needed.

```bash
# on a FarmShare login node (rice-XX), as the new user:
cd /scratch/users/$USER
unzip ~/MemorySplit-battery-handoff-2026-07-20.zip   # wherever the zip landed
mv MemorySplit memorysplit && cd memorysplit

bash cluster/setup_env.sh        # venv + torch cu130 + deps + tokenizer cache (~10 min)
                                 # (SUNET_ID defaults to $USER; nothing to edit)

# validation gate before any submission — all three must pass:
VENV=/scratch/users/$USER/venvs/memorysplit
PYTHONPATH=$PWD $VENV/bin/python -m pytest tests -q          # expect: 124 passed
sbatch cluster/slurm/smoke_gpu.sbatch                        # expect: SMOKE PASS in slurm-<id>.out (~15 min)
```

Do not proceed past a failing validation step. If torch cannot see CUDA
inside the smoke job (login nodes correctly report cuda False), you are
on a GPU node problem: resubmit with `--exclude=<node>`.

## 4. Launch (one command)

```bash
bash cluster/run_battery.sh
```

This submits the 5 corpus builds and all 14 chained training runs
(sweep + calib1b). Expected timeline at 4 concurrent GPUs: corpora finish
in 0.5-2.5 h (160M loads ~35 min, 1B loads ~2 h); sweep completes over
~2-3 days of queue-shared time; calib1b overlaps.

Sanity checks as corpora land (per `/scratch/users/$USER/memorysplit_data/<load>/report.json`):
- every value under "checks" is true;
- split arm masked_token_frac is between 0.02 and 0.10;
- bio_cross_arm_rel_diff < 0.02.
If any check is false, stop and escalate; do not train on a corpus whose
report fails.

As training runs finish (each writes outputs/<run_id>/ with ckpt.pt):

```bash
bash cluster/run_evals_pending.sh     # idempotent; run it whenever you check in
```

Each eval writes `outputs/<run_id>/evals/summary.json`. Spot-check the
first sweep pair: the split run's summary must show recall "on" > 0.9 and
"off" < 0.05 with bits_in_weights == 0.0, and its dense twin must show
closed-book recall > 0 and bits_in_weights > 1000. That is the mechanism
working; if it does not hold, stop and escalate.

## 5. The one decision: calib rule (preregistered, mechanical)

When both calib1b runs have evals:

```bash
grep -H "closed" outputs/d1b_dense_n800k_s0_gate/evals/summary.json \
                 outputs/d1b_dense_n4m_s0_gate/evals/summary.json
```

Top load = the one with the LOWER dense closed-book recall (the dose that
binds at 1B). Tie goes to n4m. Then:

```bash
bash cluster/run_confirm.sh <n800k|n4m>
```

This submits the 4 confirmation runs as 4-link dependency chains (each
link is a 47 h job that resumes from checkpoint; TIMEOUT does not requeue
on this cluster, the chain handles it). They occupy all 4 GPU slots for
roughly 6 days. Run `run_evals_pending.sh` when links complete; the run
is finished when log.jsonl's last step reaches max_steps (19073 for 1B).

## 6. Analysis and handback

After confirmation evals exist:

```bash
PYTHONPATH=$PWD $VENV/bin/python scripts/analyze.py --runs-root outputs --out outputs/analysis
```

This writes `analysis.json`, `summary.md`, and `dose_response.png`
(fact-use is in per-task contrasts; the composite figure only populates
if the emergence-watch trigger fired). Hand back to Stephen: the entire
`outputs/*/evals/` trees, all `log.jsonl` files, `outputs/analysis/`, and
`outputs/manifests/` (tiny; tar them together — no checkpoints needed
unless asked). The preregistered decision is then applied by the analysis
owner; you do not apply the H1 verdict yourself.

## 7. Disk budget (the reason you exist; do not repeat it)

Scratch has no small per-user quota but ~260GB sank the old account.
Budget: corpora ~92 GB + sweep outputs ~96 GB + calib ~12 GB +
confirmation ~90-220 GB depending on snapshots. Manage it:

- After a run's eval summary exists and is sane, delete its snapshots
  except the first, middle, and last:
  `ls outputs/<run>/snapshots | head -n -1 | sed "1d;\$d"` style cleanup,
  or simply `rm outputs/<run>/snapshots/step000{2,3,5,6,8,9}*.pt` for
  160M runs. Keep ckpt.pt until the final analysis is handed back.
- Keep all evals/ and log.jsonl forever (they are tiny and are the
  science).
- The gate-stage corpora (n*_gate) are not used by the battery; if
  run_battery's data jobs were given a clean data root you will not have
  them at all.

## 8. Known gotchas (each cost us real time; all are already mitigated)

- **wheat-01 is a bad node**: every submission script already passes
  --exclude=wheat-01. Keep doing so.
- **Submit from the repo root on scratch**: templates resolve
  cluster/config.env relative to $SLURM_SUBMIT_DIR.
- **Zombie data jobs**: if a data job shows RUNNING but its load's
  report.json already exists, the interpreter hung at exit (HF datasets
  bug; scripts force os._exit now, so this should not recur). scancel it;
  the corpus is fine.
- **afterok chains die with their parent**: if you resubmit a data job,
  resubmit its dependent training jobs against the new job id (dependents
  of a cancelled/failed parent pend forever with DependencyNeverSatisfied).
- **OOM on the n4m corpus build**: worker processes each hold the 4M
  entity records; the build script now caps workers at 6 for loads >= 2M
  entities. If an OOM recurs, resubmit that one build with --mem=100G
  --cpus-per-task=25 (mem scales with cpus on this cluster: 4000M/cpu).
- **HF access must be anonymous**: do not export HF_TOKEN; a stale token
  turns public datasets into 401s. Scripts already pass token=False.
- **Other-user queue contention is normal**: sweep runs at 4-GPU QOS cap;
  pending with reason Priority just means wait.

## 9. Two-collaborator mode (optional, recommended if two accounts exist)

The battery splits across two accounts, A and B. Rules first: a seed pair
(dense + split, same load, same seed) is ATOMIC and runs entirely inside
one account; both accounts run the same zip and the same setup_env (same
torch wheels); results merge at analysis time by run name.

Assignment:

- **Account A (corpus host).** Runs `bash cluster/run_battery.sh` as
  written EXCEPT edit the sweep loop to skip seed-1 configs (or simply
  scancel the six `*_s1` jobs it submits). Hosts all corpora; runs both
  calib1b runs; runs the three seed-0 sweep pairs; later runs the
  confirmation seed-0 pair (`run_confirm.sh` submits both pairs; scancel
  the `*_s1` chains, account B submits those).
- **Account B.** Does NOT build corpora. After A's builds finish (every
  report.json check true), set the data root to A's scratch and submit
  only seed-1 configs:

```bash
# in B's repo clone, after A's corpora exist:
DATA_A=/scratch/users/<accountA>/memorysplit_data
PYTHONPATH=$PWD $VENV/bin/python scripts/make_manifest.py --stage sweep --data-root $DATA_A
grep _s1 outputs/manifests/sweep.tsv > outputs/manifests/sweep_b.tsv
while read cfg; do sbatch --exclude=wheat-01 --export=ALL,CONFIG="$cfg" \
    cluster/slurm/train_single.sbatch; done < outputs/manifests/sweep_b.tsv
# confirmation, after the calib rule (top load decided in A):
PYTHONPATH=$PWD $VENV/bin/python scripts/make_manifest.py --stage confirm --top-load <winner> --data-root $DATA_A
grep _s1 outputs/manifests/confirm.tsv | while read cfg; do bash cluster/submit_chain.sh "$cfg" 4; done
```

  Verify read access first (`head -c 100 $DATA_A/n50k/dense/train.bin`);
  if scratch permissions block cross-account reads, B rebuilds its needed
  loads with `data_prep.sbatch` and MUST verify byte-identity against A
  before training: every value and digest in B's report.json must equal
  A's (deterministic builds make this exact, not approximate).
- **Account A must not delete corpora until B's runs are all finished.**
- Both accounts run `run_evals_pending.sh` for their own runs. Handback:
  B rsyncs its `outputs/` into A's tree (run names are globally unique),
  then A runs the analysis step from section 6 over the merged tree.

What this buys: the sweep+calib phase compresses to under a day, storage
splits across accounts, and each account carries one whole confirmation
pair (halving blast radius). It does NOT shorten the ~6-day 1B
confirmation wall (4 single-GPU runs fit in one account's 4 slots
already); shortening that requires multi-GPU training or paid compute,
which is an escalation to Stephen, not an agent decision.

## 10. AWS / bare-metal mode (for the 1B confirmation on 8xH100)

The confirmation tier runs dramatically faster on H100s (~1.5 days for
all four runs vs ~6 on FarmShare) and keeps both seed pairs on one
platform, which the preregistration prefers. There is no Slurm on the
box; use the bundled launcher.

```bash
# on the AWS node (Ubuntu + CUDA assumed), from the unzipped repo:
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export PATH=$PWD/.venv/bin:$PATH
PYTHONPATH=$PWD python -m pytest tests -q          # expect 124 passed
PYTHONPATH=$PWD python scripts/smoke_test.py --device cuda   # expect SMOKE PASS

# corpus: rebuild the calib-winning 1B load locally (deterministic).
# ~2h on a box this size; verify report.json checks are all true and, if
# a FarmShare copy exists, that bed_hash_digest matches it exactly.
PYTHONPATH=$PWD python scripts/build_corpus.py --out-root data_root \
    --stage full1b --loads <n800k|n4m> --workers 16

# configs + launch (4 runs pinned to GPUs 0-3, auto-resume, idempotent):
PYTHONPATH=$PWD python scripts/make_manifest.py --stage confirm \
    --top-load <winner> --data-root $PWD/data_root
PYTHONPATH=$PWD nohup python scripts/run_local_gpus.py \
    --manifest outputs/manifests/confirm.tsv --gpus 0,1,2,3 > launcher.out 2>&1 &

# optional 410M add-back tier on the spare GPUs (preregistered optional;
# needs the 160M sweep corpora too — rebuild with --stage full first):
PYTHONPATH=$PWD python scripts/make_manifest.py --stage mid410 \
    --top-load <winner> --data-root $PWD/data_root
PYTHONPATH=$PWD nohup python scripts/run_local_gpus.py \
    --manifest outputs/manifests/mid410.tsv --gpus 4,5,6,7 > launcher410.out 2>&1 &
```

Evals on the same box once runs finish (no sbatch; direct):

```bash
for run in outputs/d1b_*; do
  PYTHONPATH=$PWD python scripts/run_evals.py --run "$run" --limit 1500
done
```

Notes: if the instance is spot/preemptible, the launcher plus checkpoint
resume already handles interruption (rerun the same launcher command).
Keep the machine's clock and disk in mind: each 1B run writes ~20-40 GB
of checkpoints/snapshots; prune snapshots after evals as in section 7.
Mixed-platform caveat: sweep pairs (FarmShare L40S) and confirmation
pairs (H100) sit on different hardware, which the preregistration
permits (pairs are platform-atomic; cross-TIER hardware may differ) but
report it in the final writeup.

## 11. Integrity constraints (non-negotiable)

- No endpoint, margin, seed, mixture, difficulty, or exclusion-rule
  changes. The preregistration is frozen; violations void the experiment.
- Never train on a corpus with a failing report check; never eval against
  a different load's organizer (run_evals reads the run's own config).
- Report every anomaly (excluded run, requeue, node failure) in the
  handback notes; exclusions are infrastructure-only, never results-based.
- If anything here contradicts observed reality, stop and escalate to
  Stephen rather than improvising.
