# YOUR ROLE: Account A (corpus host) — read this first

You are one of two agents running a preregistered ML battery in parallel
FarmShare accounts. You are **Account A**. Your scope ends at the 1B
calibration evals. You do NOT run any 1B confirmation training; that
happens on AWS and is handled by Stephen. Ignore HANDOFF-AGENT.md
sections 5 (the confirm submission part) and 10 entirely.

Your responsibilities:

1. Bring-up and validation exactly per HANDOFF-AGENT.md section 3
   (setup_env, 124 tests, GPU smoke). Do not proceed past a failure.
2. Launch your share of the battery:

```bash
bash cluster/run_battery.sh --role a
```

   This builds ALL corpora (both accounts depend on yours; ~92 GB), runs
   the three seed-0 sweep pairs, and both calib1b runs, all chained.
3. When the corpora finish (report.json present for n50k, n200k, n800k
   with every "checks" value true), tell Account B they can launch, and
   give them this path: `/scratch/users/<your-sunetid>/memorysplit_data`.
   Do not delete or modify anything under that path until B confirms all
   their runs are complete.
4. Run `bash cluster/run_evals_pending.sh` whenever you check in; verify
   the first finished split run's summary shows recall on > 0.9, off
   < 0.05, bits_in_weights == 0.0 (mechanism check, HANDOFF section 4).
5. **The moment both calib runs have evals**, report these two numbers to
   Stephen and stop there (no confirmation submission):

```bash
grep -H "closed" outputs/d1b_dense_n800k_s0_gate/evals/summary.json \
                 outputs/d1b_dense_n4m_s0_gate/evals/summary.json
```

6. Handback when your sweep+calib runs all have evals:

```bash
tar czf handback_A.tgz outputs/*/evals outputs/*/log.jsonl outputs/manifests
```

   Send handback_A.tgz to Stephen. Keep checkpoints until told otherwise;
   prune snapshots per HANDOFF section 7 as evals complete.

Everything else you need (gotchas, disk budget, integrity rules) is in
HANDOFF-AGENT.md sections 7, 8, and 11. The preregistration in
docs/superpowers/specs/2026-07-20-preregistration.md is frozen; escalate
rather than deviate.
