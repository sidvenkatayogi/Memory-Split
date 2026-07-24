#!/bin/bash
# Full v2 run: env -> Wikidata5M -> build 20M-dose corpus -> launch DENSE+SPLIT
# concurrently (4 GPUs each) with S3 checkpointing. Training is detached (setsid)
# so it survives this SSM command returning. Logs + build report pushed to S3 so
# progress is monitorable off-box. The 9:30 hard stop is handled by the box's
# own shutdown dead-man + external monitoring — NOT here.
set -uxo pipefail
exec > /home/ubuntu/run_all.log 2>&1
B=memorysplit-sid-056956104102
REGION=us-east-1
S3RUN="s3://$B/runs"
push() { aws s3 cp /home/ubuntu/run_all.log "$S3RUN/bootstrap.log" --region $REGION >/dev/null 2>&1 || true; }

cd /home/ubuntu/MemorySplit || exit 1
export PYTHONPATH="$PWD" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

apt-get update -y >/dev/null 2>&1 || true
apt-get install -y python3-venv python3-pip >/dev/null 2>&1 || true
python3 -m venv .venv && source .venv/bin/activate
pip install -q -U pip
pip install -q "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124
pip install -q numpy pyyaml tqdm tiktoken huggingface_hub
python -c "import torch;print('TORCH',torch.__version__,'NGPU',torch.cuda.device_count())"
push

echo "=== download Wikidata5M ==="
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('intfloat/wikidata5m', repo_type='dataset', local_dir='data/wd5m_raw')
print("downloaded")
PY
mkdir -p data/wikidata5m
( cd data/wd5m_raw
  for f in *.tar.gz;  do [ -e "$f" ] && tar xzf "$f" -C ../wikidata5m || true; done
  for f in *.txt.gz;  do [ -e "$f" ] && gunzip -c "$f" > "../wikidata5m/${f%.gz}" || true; done
  find . -maxdepth 2 -name '*.txt' -exec cp -n {} ../wikidata5m/ \; 2>/dev/null || true )
echo "wikidata5m dir:"; ls -la data/wikidata5m | head -20
push

echo "=== build 20M-dose corpus + configs ==="
python scripts/build_mh_corpus.py --wikidata-root data/wikidata5m \
  --dose 20000000 --exposures 30 --model d160m --micro-bs 8 \
  --n-mh-train 300000 --n-bed-docs 50000 \
  --s3-ckpt-prefix "$S3RUN" --out data/mh/n20m --tag d160m_n20m
aws s3 cp data/mh/n20m/build_report.json "$S3RUN/build_report.json" --region $REGION || true
push

echo "=== launch DENSE + SPLIT (4 GPUs each, detached) ==="
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 RDZV_PORT=29500 setsid nohup \
  bash cluster/torchrun_poc.sh configs/mh/d160m_n20m_dense.yaml \
  > /home/ubuntu/train_dense.log 2>&1 &
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC=4 RDZV_PORT=29501 setsid nohup \
  bash cluster/torchrun_poc.sh configs/mh/d160m_n20m_split.yaml \
  > /home/ubuntu/train_split.log 2>&1 &
sleep 20
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv || true
aws s3 cp - "$S3RUN/LAUNCHED.txt" --region $REGION <<<"launched $(date -u +%FT%TZ)" || true
echo "RUN_ALL_DONE"
push
