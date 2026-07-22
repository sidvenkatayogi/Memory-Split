#!/usr/bin/env bash
#SBATCH --job-name=rel-bed
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=rel-bed-%j.out
set -euo pipefail

: "${RELATIONAL_KIT_ROOT:?set RELATIONAL_KIT_ROOT to the unzipped kit}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${RELATIONAL_VENV:?set RELATIONAL_VENV}"

mkdir -p "$DATA_ROOT"
exec "$RELATIONAL_VENV/bin/python" \
    "$RELATIONAL_KIT_ROOT/stage_fineweb.py" \
    --out "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl" \
    --cache-dir "$DATA_ROOT/hf-cache" \
    --execute
