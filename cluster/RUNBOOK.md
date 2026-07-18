# FarmShare runbook — memory-split battery

One human step is required whenever the SSH control socket has expired
(FarmShare is password + Duo only):

```bash
cd ~/Documents/MemorySplit
bash cluster/connect.sh syz          # password + Duo push; persists 8h
```

Everything below is scriptable once the socket is warm.

## Day 1 — bring-up (after connect)

```bash
bash cluster/sync_push.sh                          # rsync repo -> scratch
ssh -o ControlPath="$HOME/.ssh/cm-%r@%h:%p" syz@rice-04.farmshare.stanford.edu bash -s <<'EOF'
cd /scratch/users/syz/memorysplit
bash cluster/setup_env.sh                          # idempotent (done 2026-07-17)
sbatch cluster/slurm/smoke_gpu.sbatch              # pytest + toy pipeline on L40S (~15 min)
sbatch --export=ALL,BUILD_ARGS="--stage gates" cluster/slurm/data_prep.sbatch
EOF
```

`data_prep --stage gates` builds all three loads (n50k/n200k/n800k) at the
800M-token gate budget into `/scratch/users/syz/memorysplit_data/`.

## Day 2 — gates A-C (after data_prep finishes)

```bash
ssh ... 'cd /scratch/users/syz/memorysplit && \
  /scratch/users/syz/venvs/memorysplit/bin/python scripts/make_manifest.py \
    --stage gates --data-root /scratch/users/syz/memorysplit_data && \
  bash cluster/submit_manifest.sh outputs/manifests/gates.tsv'
```

Gate criteria (spec §7): A — dense pilot iGSM held-out > 90%;
B — dense recall degrades across N (pick top load; escalate once if flat);
C — split pilot lookup parse rate > 95% and recall ON-OFF gap > 30 pts.
Evaluate finished runs with:

```bash
ssh ... '.../python scripts/run_evals.py --run outputs/<run_id> --limit 2000'
```

Then write and commit `docs/superpowers/specs/2026-07-22-preregistration.md`
(margins = max(2 x pooled pilot seed-sigma, 0.5 pt); freeze before any
confirmation run).

## Days 3-9 — battery

```bash
# full-budget corpora (3.2B dense tokens per load; several hours, CPU node)
sbatch --export=ALL,BUILD_ARGS="--stage full" cluster/slurm/data_prep.sbatch
# 12 sweep runs (~10-15 L40S-h each), 4 run concurrently (QOS cap)
python scripts/make_manifest.py --stage sweep --data-root ...
bash cluster/submit_manifest.sh outputs/manifests/sweep.tsv
# then 6 confirmation runs at the gate-B top load
python scripts/make_manifest.py --stage confirm --top-load nXXXk --data-root ...
bash cluster/submit_manifest.sh outputs/manifests/confirm.tsv
# stretch pair only if sweep+confirm are analyzed and the calendar allows
python scripts/make_manifest.py --stage stretch --top-load nXXXk --data-root ...
```

Evals per finished run: `scripts/run_evals.py --run outputs/<run_id>`
(add `--natural` on the final checkpoint). Pull results home and analyze:

```bash
bash cluster/sync_pull.sh
.venv/bin/python scripts/analyze.py --runs-root outputs/cluster --out outputs/analysis
```

## Kill order (schedule pressure)

1. drop the 1B stretch; 2. drop one fact level from the sweep;
3. confirmation 3 seeds -> 2. The 410M top-load multi-seed contrast is
protected last; the <= $300 RunPod burst is its contingency.

## Known facts (recon 2026-07-12/13 + this bring-up)

- QOS gpu: 4 concurrent GPU jobs, 32 submitted max, MaxWall 2 days,
  L40S 48GB (oat-01..06, 4 per node). Login rice-04; scratch
  /scratch/users/syz; egress OK from login and compute nodes.
- venv at /scratch/users/syz/venvs/memorysplit — torch 2.13.0+cu130
  installed 2026-07-17 (login node reports cuda False; GPU nodes have
  driver 595.71.05 / CUDA 13.2).
- train_single.sbatch requeues and `--resume auto` continues from ckpt.pt
  (checkpoint every 30 min), so the 2-day wall is safe for all presets.
