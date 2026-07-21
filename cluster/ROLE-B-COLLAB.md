# YOUR ROLE: Collaborator account — read this first

You are running two independent workloads in your FarmShare account while
the corpus-host account (Stephen's, "Account A") runs its half. This card
supersedes ROLE-B.md (same Account-B battery duties, plus a CPU-only
experiment you can start IMMEDIATELY). Total active attention: ~30 min;
wall-clock: ~1-2 days, mostly queue time.

## Workload 1 — keyguess replication seeds (START NOW; no dependency)

CPU-only, does not touch the GPU QOS. Four Slurm jobs, one per replication
seed of the held-out key-generalization experiment (EXP-COPY-01, Tier-S
half). The PopQA snapshot is SHIPPED in this zip at
`data/realfacts/popqa_clean.jsonl` — never re-fetch it; identical facts
across accounts is the point.

```bash
cd /scratch/users/$USER/memorysplit          # after bring-up (below)
for S in 1 2 3 4; do
  sbatch --export=ALL,KEYGUESS_SEED=$S cluster/slurm/keyguess_cpu.sbatch
done
```

Each job runs data -> train(2 models, 800 steps, ~29M params) -> eval
(4 arms x 806 items) for its seed, ~3-4 h on 8 CPUs. Artifacts land in
`data/keyguess_local/` with `_s<S>` suffixes. Sanity per finished seed:
`data/keyguess_local/summary_s<S>.json` exists and its arm "A" heldout
`relation_half` is > 0.9 (the schema transfers; if not, the run is broken —
escalate, don't rerun blindly).

## Workload 2 — battery Account B (WAIT for Stephen's green light)

Exactly ROLE-B.md: bring-up validation, then after Stephen confirms his
corpora are built,

```bash
bash cluster/run_battery.sh --role b --data-root /scratch/users/<stephens-sunetid>/memorysplit_data
```

runs your three seed-1 sweep pairs (6 GPU runs, ~8-15 L40S-h each, 4-slot
QOS). Fallback if cross-account reads are blocked: ROLE-B.md step 3.
Run `bash cluster/run_evals_pending.sh` whenever you check in; mechanism
check per ROLE-B.md step 4 (first split run: recall on > 0.9, off < 0.05,
bits_in_weights == 0.0).

## Bring-up (once, ~30 min + queue)

Per HANDOFF-AGENT.md section 3, with ONE difference — the test count:

```bash
cd /scratch/users/$USER
unzip <the zip you were sent> && mv MemorySplit* memorysplit && cd memorysplit
bash cluster/setup_env.sh
VENV=/scratch/users/$USER/venvs/memorysplit
PYTHONPATH=$PWD $VENV/bin/python -m pytest tests -q   # expect: 204 passed
sbatch cluster/slurm/smoke_gpu.sbatch                 # expect: SMOKE PASS (~15 min)
```

(The handoff doc says "124 passed" — that predates the keyguess harness in
this package; 204 is correct here.) Workload 1 needs only the pytest gate;
you can submit its jobs before the GPU smoke returns.

## What to return (two tarballs, both small — no checkpoints)

```bash
# when all four keyguess seeds have summary_s<S>.json:
tar czf handback_keyguess.tgz \
    data/keyguess_local/summary_s*.json \
    data/keyguess_local/results_*_s*.json \
    data/keyguess_local/records_*_s*.jsonl \
    data/keyguess_local/data_manifest.json \
    data/keyguess_local/emittability.json \
    data/keyguess_local/runs/*_s*/log.jsonl

# when your six battery runs all have evals (per ROLE-B.md):
tar czf handback_B.tgz outputs/*/evals outputs/*/log.jsonl outputs/manifests
```

Send both to Stephen; tell him you are done so Account A can release the
corpora. Prune snapshots per HANDOFF-AGENT.md section 7 as evals finish.

## Rules that protect the science (do not skip)

- Everything in HANDOFF-AGENT.md sections 7 (disk), 8 (gotchas — wheat-01
  exclusion, afterok chains, zombie data jobs), and 11 (integrity) applies.
- The preregistration is frozen; the keyguess seeds change NOTHING about
  the battery. If a step seems to require deviating, stop and message
  Stephen instead.
- Do not modify code in this package. If something crashes, send the
  slurm-*.out and stop.
