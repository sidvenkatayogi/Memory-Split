#!/usr/bin/env bash
#SBATCH --job-name=rel-gate-bind
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=rel-gate-bind-%j.out
set -euo pipefail

: "${RELATIONAL_SOURCE_ROOT:?set RELATIONAL_SOURCE_ROOT}"
: "${RELATIONAL_SOURCE_ARCHIVE:?set RELATIONAL_SOURCE_ARCHIVE}"
: "${RELATIONAL_KIT_ROOT:?set RELATIONAL_KIT_ROOT}"
: "${RELATIONAL_VENV:?set RELATIONAL_VENV}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${RELATIONAL_BUNDLE:?set RELATIONAL_BUNDLE}"

exec "$RELATIONAL_VENV/bin/python" \
    "$RELATIONAL_KIT_ROOT/bind_farmshare_gate.py" \
    --source-root "$RELATIONAL_SOURCE_ROOT" \
    --source-archive "$RELATIONAL_SOURCE_ARCHIVE" \
    --bundle "$RELATIONAL_BUNDLE" \
    --data-root "$DATA_ROOT" \
    --bed-jsonl \
        "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl" \
    --execute
