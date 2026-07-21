#!/usr/bin/env bash
# Submit the ENTIRE preregistered battery from any FarmShare account.
# Run ON a FarmShare login node, from the repo root on scratch:
#
#   cd /scratch/users/$USER/memorysplit && bash cluster/run_battery.sh
#
# Submits, fully dependency-chained (nothing else to do until calib1b
# verdicts are in):
#   1. 5 corpus builds on the frozen recipe (3x 160M loads, 2x 1B loads)
#   2. 12 sweep runs (d160m: 3 loads x 2 arms x seeds 0,1), each chained
#      on its own load's corpus
#   3. 2 calib1b runs (dense d1b at n800k_1b and n4m_1b, 1.5B tokens),
#      chained on their corpora
# The 1B confirmation is NOT submitted here: it needs the preregistered
# calib rule applied first (see cluster/run_confirm.sh).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=./config.env
source "$SCRIPT_DIR/config.env"
require_sunet
DATA=$(expand_path "$FS_DATA")
VENV=$(expand_path "$FS_VENV")
mkdir -p "$DATA"

EXC="--exclude=wheat-01"

echo "== corpus builds (frozen recipe; report.json checks must all be true)"
D1=$(sbatch --parsable --time=04:00:00 $EXC --export=ALL,BUILD_ARGS="--stage full --loads n50k" cluster/slurm/data_prep.sbatch)
D2=$(sbatch --parsable --time=04:00:00 $EXC --export=ALL,BUILD_ARGS="--stage full --loads n200k" cluster/slurm/data_prep.sbatch)
D3=$(sbatch --parsable --time=04:00:00 $EXC --export=ALL,BUILD_ARGS="--stage full --loads n800k" cluster/slurm/data_prep.sbatch)
D4=$(sbatch --parsable --time=12:00:00 --mem=60G --cpus-per-task=16 $EXC --export=ALL,BUILD_ARGS="--stage full1b --loads n800k" cluster/slurm/data_prep.sbatch)
D5=$(sbatch --parsable --time=12:00:00 --mem=60G --cpus-per-task=16 $EXC --export=ALL,BUILD_ARGS="--stage full1b --loads n4m" cluster/slurm/data_prep.sbatch)
echo "  data jobs: n50k=$D1 n200k=$D2 n800k=$D3 n800k_1b=$D4 n4m_1b=$D5"

echo "== sweep (12 runs, chained per load)"
PYTHONPATH="$PWD" "$VENV/bin/python" scripts/make_manifest.py --stage sweep --data-root "$DATA"
while IFS= read -r cfg; do
    [ -z "$cfg" ] && continue
    case "$cfg" in
        *n50k*) DEP=$D1 ;;
        *n200k*) DEP=$D2 ;;
        *n800k*) DEP=$D3 ;;
        *) echo "unmatched config $cfg" >&2; exit 1 ;;
    esac
    jid=$(sbatch --parsable $EXC --dependency=afterok:"$DEP" \
        --export=ALL,CONFIG="$cfg" cluster/slurm/train_single.sbatch)
    echo "  sweep $cfg: job $jid (afterok:$DEP)"
done < outputs/manifests/sweep.tsv

echo "== calib1b (2 short dense-1B runs, chained)"
PYTHONPATH="$PWD" "$VENV/bin/python" scripts/make_manifest.py --stage calib1b --data-root "$DATA"
while IFS= read -r cfg; do
    [ -z "$cfg" ] && continue
    case "$cfg" in
        *n800k*) DEP=$D4 ;;
        *n4m*) DEP=$D5 ;;
        *) echo "unmatched config $cfg" >&2; exit 1 ;;
    esac
    jid=$(sbatch --parsable $EXC --dependency=afterok:"$DEP" \
        --export=ALL,CONFIG="$cfg" cluster/slurm/train_single.sbatch)
    echo "  calib $cfg: job $jid (afterok:$DEP)"
done < outputs/manifests/calib1b.tsv

echo
squeue -u "$SUNET_ID" -o "%.9i %.13j %.2t %.20E"
echo "run_battery: all submitted. Next human/agent step: evals as runs finish"
echo "(cluster/run_evals_pending.sh), then the calib rule -> run_confirm.sh."
