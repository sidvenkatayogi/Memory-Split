#!/usr/bin/env bash
#SBATCH --job-name=rel-gate-eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=rel-gate-eval-%j.out
set -euo pipefail

: "${RELATIONAL_SOURCE_ROOT:?set RELATIONAL_SOURCE_ROOT}"
: "${RELATIONAL_VENV:?set RELATIONAL_VENV}"
: "${OUT_ROOT:?set OUT_ROOT}"
: "${RUN_ID:?set RUN_ID}"

case "$RUN_ID" in
    *[!A-Za-z0-9_-]*)
        echo "RUN_ID must be a safe identifier" >&2
        exit 2
        ;;
esac

cd "$RELATIONAL_SOURCE_ROOT"
export PYTHONPATH="$RELATIONAL_SOURCE_ROOT"
exec "$RELATIONAL_VENV/bin/python" \
    scripts/run_relational_evals.py \
    --run "$OUT_ROOT/$RUN_ID" \
    --device cuda
