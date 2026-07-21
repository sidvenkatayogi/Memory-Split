# YOUR ROLE: Account B (reader) — read this first

You are one of two agents running a preregistered ML battery in parallel
FarmShare accounts. You are **Account B**. Your scope: the three seed-1
sweep pairs, evaluated, handed back. You build NO corpora, run NO
calibration, and run NO 1B confirmation (that is AWS, handled by
Stephen). Ignore HANDOFF-AGENT.md sections 5 and 10 entirely.

Your responsibilities:

1. Bring-up and validation exactly per HANDOFF-AGENT.md section 3
   (setup_env, 124 tests, GPU smoke). Do not proceed past a failure.
2. WAIT for Account A to confirm their corpora are built, and get their
   data path from them (`/scratch/users/<accountA>/memorysplit_data`).
   Do not submit anything before that confirmation.
3. Launch your share (the script verifies readability and refuses to
   start if A's corpora are missing):

```bash
bash cluster/run_battery.sh --role b --data-root /scratch/users/<accountA>/memorysplit_data
```

   Fallback if cross-account reads are blocked by permissions: build your
   own copies (`--stage full`, loads n50k,n200k,n800k, one data_prep job
   each per HANDOFF section 3 submission style), then verify BYTE
   IDENTITY before training: every digest and count in your report.json
   files must equal Account A's. If they differ, stop and escalate.
4. Run `bash cluster/run_evals_pending.sh` whenever you check in; verify
   the first finished split run's summary shows recall on > 0.9, off
   < 0.05, bits_in_weights == 0.0 (mechanism check, HANDOFF section 4).
5. Handback when your six runs all have evals:

```bash
tar czf handback_B.tgz outputs/*/evals outputs/*/log.jsonl outputs/manifests
```

   Send handback_B.tgz to Stephen, and tell Account A you are done (they
   are holding the corpora for you). Prune snapshots per HANDOFF
   section 7 as evals complete; keep checkpoints until told otherwise.

Everything else (gotchas, disk budget, integrity rules) is in
HANDOFF-AGENT.md sections 7, 8, and 11. The preregistration in
docs/superpowers/specs/2026-07-20-preregistration.md is frozen; escalate
rather than deviate.
