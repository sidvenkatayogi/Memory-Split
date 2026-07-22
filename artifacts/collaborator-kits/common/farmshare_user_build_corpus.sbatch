#!/usr/bin/env bash
#SBATCH --job-name=rel-user-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=rel-user-data-%j.out
set -euo pipefail

: "${RELATIONAL_SOURCE_ROOT:?set RELATIONAL_SOURCE_ROOT}"
: "${RELATIONAL_VENV:?set RELATIONAL_VENV}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${DATA_REL:?set DATA_REL}"

case "$DATA_REL" in
    n50k_gate_ds10000)
        entities=50000
        tokens=299892736
        data_seed=10000
        ;;
    n50k_ds10000)
        entities=50000
        tokens=1599602688
        data_seed=10000
        ;;
    n800k_ds10000)
        entities=800000
        tokens=1599602688
        data_seed=10000
        ;;
    *)
        echo "DATA_REL is not owned by the coordinator: $DATA_REL" >&2
        exit 2
        ;;
esac

BED_JSONL="$DATA_ROOT/fineweb-edu-sample10bt-r87f091-first3.jsonl"
cd "$RELATIONAL_SOURCE_ROOT"
exec "$RELATIONAL_VENV/bin/python" \
    scripts/build_relational_corpus.py \
    --out "$DATA_ROOT/$DATA_REL" \
    --entities "$entities" \
    --tokens "$tokens" \
    --data-seed "$data_seed" \
    --bed-jsonl "$BED_JSONL" \
    --route-policy configs/route-policy.json \
    --route-policy-sha256 \
    0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058
