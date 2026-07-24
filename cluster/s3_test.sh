#!/bin/bash
# Validate S3 checkpoint durability from the instance role (no full run).
#   (1) probe raw read/write to the user's bucket
#   (2) real push -> delete local -> pull -> resume cycle (cpu, 2-proc gloo)
set -uo pipefail
B=memorysplit-sid-056956104102
REGION=us-east-1

echo "=== S3 PROBE (instance role -> s3://$B) ==="
echo "probe $(date -u +%FT%TZ)" | aws s3 cp - "s3://$B/s3test/probe.txt" --region $REGION \
  && echo "S3-WRITE-OK" || echo "S3-WRITE-FAIL"
aws s3 cp "s3://$B/s3test/probe.txt" - --region $REGION >/dev/null 2>&1 \
  && echo "S3-READ-OK" || echo "S3-READ-FAIL"
if ! aws s3 ls "s3://$B/s3test/probe.txt" --region $REGION >/dev/null 2>&1; then
  echo "S3-PROBE-BLOCKED: instance role cannot use this bucket; durability needs another mechanism"
  exit 0
fi

echo "=== TRAINER S3 RESUME TEST (cpu, 2-proc gloo, toy) ==="
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
apt-get update -y >/dev/null 2>&1 || true
apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true
python3 -m venv .venv && source .venv/bin/activate
pip install -q -U pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy pyyaml tqdm tiktoken

python - <<'PY'
import numpy as np, yaml
n = 500_000
np.random.randint(0, 50257, size=n, dtype=np.uint16).tofile("/tmp/tb.bin")
np.ones(n, dtype=np.uint8).tofile("/tmp/tb.mask.bin")
cfg = dict(model="toy", seed=0, ctx=128, micro_batch_size=2, tokens_per_step=512,
           total_tokens=999999, lr=1e-3, warmup_steps=1, weight_decay=0.1, compile=False,
           device="cpu", log_every=1, eval_every=999, snap_frac=0.5, ckpt_minutes=0,
           train_bin="/tmp/tb.bin", train_mask="/tmp/tb.mask.bin", out_dir="/tmp/run1",
           s3_ckpt="s3://memorysplit-sid-056956104102/s3test/run1", s3_region="us-east-1")
yaml.safe_dump(cfg, open("/tmp/cfg.yaml", "w"))
print("wrote cfg")
PY

echo "--- RUN 1: train to step 3 (should write + push ckpt to S3) ---"
NPROC=2 bash cluster/torchrun_poc.sh /tmp/cfg.yaml --max-steps 3 --resume none
aws s3 ls "s3://$B/s3test/run1/ckpt.pt" --region $REGION && echo "S3-CKPT-PRESENT" || echo "S3-CKPT-MISSING"

echo "--- simulate instance loss: rm local out_dir ---"
rm -rf /tmp/run1
echo "--- RUN 2: --resume auto (must pull from S3, resume@3, train to 6) ---"
NPROC=2 bash cluster/torchrun_poc.sh /tmp/cfg.yaml --max-steps 6 --resume auto
echo "S3-RESUME-TEST-DONE"
