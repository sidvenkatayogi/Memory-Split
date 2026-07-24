#!/bin/bash
# GPU smoke: build pipeline + multi-GPU DDP. Run from repo root: bash cluster/smoke_gpu.sh
# Quiet pip + concise stdout markers so SSM captures the result (no S3 needed).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apt-get update -y >/dev/null 2>&1 || true
apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true
python3 -m venv .venv
source .venv/bin/activate
pip install -q -U pip
pip install -q "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
pip install -q numpy pyyaml tqdm tiktoken

python -c "import torch;print('TORCH',torch.__version__,'NGPU',torch.cuda.device_count())"
NG=$(python -c "import torch;print(torch.cuda.device_count())")

echo "=== SMOKE A: build pipeline (loader -> generator -> tokenizer -> bin) ==="
python - <<'PY'
import tempfile, pathlib, json
from tests.test_wikidata5m import _fixture
from corpusgen.mh_build import MHBuildCfg, build_mh_corpus
from train.tokenizer import get_tok
root = pathlib.Path(tempfile.mkdtemp()); _fixture(root); out = root / "c"
cfg = MHBuildCfg(wikidata_root=str(root), n_entities=5, n_mh_train=40, n_mh_eval=10,
                 max_depth=3, atomic_exposures=2, n_bed_docs=8, func_min_support=1)
rep = build_mh_corpus(cfg, get_tok(), out)
assert (out / "split" / "train.bin").exists()
assert rep["arms"]["split"]["masked_frac"] > rep["arms"]["dense"]["masked_frac"]
print("SMOKE-A-PASS", json.dumps(rep["arms"]))
PY

echo "=== SMOKE B: multi-GPU DDP training (toy model, 5 steps) ==="
python - <<'PY'
import numpy as np, yaml
n = 4_000_000
np.random.randint(0, 50257, size=n, dtype=np.uint16).tofile("/tmp/tb.bin")
np.ones(n, dtype=np.uint8).tofile("/tmp/tb.mask.bin")
cfg = dict(model="toy", seed=0, ctx=256, micro_batch_size=4, tokens_per_step=8192,
           total_tokens=40960, lr=1e-3, warmup_steps=2, weight_decay=0.1, compile=False,
           device="cuda", log_every=1, eval_every=100, snap_frac=0.5, ckpt_minutes=999,
           train_bin="/tmp/tb.bin", train_mask="/tmp/tb.mask.bin", out_dir="/tmp/ddp_out")
yaml.safe_dump(cfg, open("/tmp/smoke_ddp.yaml", "w"))
print("wrote /tmp/smoke_ddp.yaml")
PY
NPROC=$NG bash cluster/torchrun_poc.sh /tmp/smoke_ddp.yaml --max-steps 5 --resume none
test -f /tmp/ddp_out/ckpt.pt && echo "SMOKE-B-PASS ckpt-written" || { echo "SMOKE-B-FAIL no-ckpt"; exit 24; }
echo "ALL-SMOKES-PASS ngpu=$NG"
