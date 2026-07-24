#!/usr/bin/env bash
# torchrun launcher for the user's PoC DDP path (own; not the relational harness).
#   NPROC=8 cluster/torchrun_poc.sh CONFIG.yaml [--max-steps N] [--resume none]
# Defaults NPROC to the visible GPU count. Single GPU / CPU also works.
set -euo pipefail
CFG="${1:?usage: torchrun_poc.sh CONFIG.yaml [extra args]}"; shift || true
NPROC="${NPROC:-$(python -c 'import torch;print(torch.cuda.device_count() or 1)' 2>/dev/null || echo 1)}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[torchrun_poc] nproc=$NPROC config=$CFG"
exec torchrun --standalone --nproc_per_node="$NPROC" scripts/train_ddp.py --config "$CFG" "$@"
