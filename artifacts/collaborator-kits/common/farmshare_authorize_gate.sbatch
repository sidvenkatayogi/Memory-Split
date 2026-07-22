#!/usr/bin/env bash
#SBATCH --job-name=rel-gate-auth
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=rel-gate-auth-%j.out
set -euo pipefail

: "${RELATIONAL_SOURCE_ROOT:?set RELATIONAL_SOURCE_ROOT}"
: "${RELATIONAL_KIT_ROOT:?set RELATIONAL_KIT_ROOT}"
: "${RELATIONAL_VENV:?set RELATIONAL_VENV}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${OUT_ROOT:?set OUT_ROOT}"
: "${RELATIONAL_BUNDLE:?set RELATIONAL_BUNDLE}"
RELATIONAL_SOURCE_ARCHIVE="${RELATIONAL_SOURCE_ARCHIVE:-$RELATIONAL_KIT_ROOT/relational-source-0045398.tar.gz}"

exec "$RELATIONAL_VENV/bin/python" \
    "$RELATIONAL_KIT_ROOT/authorize_farmshare_gate.py" \
    --source-root "$RELATIONAL_SOURCE_ROOT" \
    --data-root "$DATA_ROOT" \
    --out-root "$OUT_ROOT" \
    --bundle "$RELATIONAL_BUNDLE" \
    --source-archive "$RELATIONAL_SOURCE_ARCHIVE" \
    --bed-jsonl \
        "$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl" \
    --evidence-out "$RELATIONAL_KIT_ROOT/farmshare-29m-gate-evidence.json" \
    --out "$RELATIONAL_KIT_ROOT/farmshare-29m-gate-authorization.json"
