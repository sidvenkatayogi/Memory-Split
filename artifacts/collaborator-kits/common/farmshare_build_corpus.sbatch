#!/usr/bin/env bash
#SBATCH --job-name=rel-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=rel-data-%j.out
set -euo pipefail

: "${RELATIONAL_KIT_ROOT:?set RELATIONAL_KIT_ROOT to the unzipped kit}"
: "${RELATIONAL_SOURCE_ROOT:?set RELATIONAL_SOURCE_ROOT to extracted source}"
: "${RELATIONAL_VENV:?set RELATIONAL_VENV}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${DATA_REL:?set DATA_REL to one owned corpus}"

BED_JSONL="$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl"
exec "$RELATIONAL_VENV/bin/python" \
    "$RELATIONAL_KIT_ROOT/build_owned_corpora.py" \
    --assignment "$RELATIONAL_KIT_ROOT/assignment.json" \
    --source-root "$RELATIONAL_SOURCE_ROOT" \
    --data-root "$DATA_ROOT" \
    --bed-jsonl "$BED_JSONL" \
    --data-rel "$DATA_REL" \
    --execute
