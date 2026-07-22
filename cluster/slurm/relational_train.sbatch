#!/usr/bin/env bash
#SBATCH --job-name=relational-train
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j.out
#
# Submit only through:
#   python scripts/make_relational_manifest.py farmshare configs/160m.tsv
# Add --execute only after cluster preflight passes.
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?must run under Slurm}"
: "${SLURM_JOB_ID:?must run under Slurm}"
: "${CONFIG_REL:?CONFIG_REL is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"

case "$CONFIG_REL" in
    /*|*\\*|*".."*)
        echo "CONFIG_REL must be a portable relative path: $CONFIG_REL" >&2
        exit 2
        ;;
esac

cd "$SLURM_SUBMIT_DIR"
# shellcheck source=../config.env
source cluster/config.env
require_sunet
init_modules
module load "$PYTHON_MODULE"
VENV="${RELATIONAL_VENV:-$(expand_path "$FS_VENV")}"
PYTHON="$VENV/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Python environment is missing: $PYTHON" >&2
    exit 2
fi

gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
gpu_count="$(printf '%s\n' "$gpu_names" | awk 'NF {count++} END {print count+0}')"
if [ "$gpu_count" -ne 1 ] || ! printf '%s\n' "$gpu_names" | grep -q 'L40S'; then
    echo "relational_train requires exactly one visible L40S" >&2
    exit 2
fi

export PYTHONPATH="$SLURM_SUBMIT_DIR"
runtime_dir="$OUT_ROOT/.launch-configs"
mkdir -p "$runtime_dir"
config_name="$(basename "$CONFIG_REL" .yaml)"
runtime_config="$runtime_dir/${config_name}-${SLURM_JOB_ID}.yaml"

"$PYTHON" scripts/make_relational_manifest.py resolve \
    --config "$CONFIG_REL" \
    --runtime-config "$runtime_config" \
    --data-root "$DATA_ROOT" \
    --out-root "$OUT_ROOT" \
    --execute

echo "node=$(hostname) gpu=$gpu_names config=$CONFIG_REL runtime=$runtime_config"
exec "$PYTHON" scripts/run_train.py \
    --config "$runtime_config" \
    --resume auto
