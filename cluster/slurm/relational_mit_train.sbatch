#!/usr/bin/env bash
#SBATCH --job-name=relational-mit
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm-%j.out
#
# Partition, account, QoS, GRES, CPU, memory, and wall time are supplied as
# argv by cluster/mit/run_relational_manifest.py.
set -euo pipefail

: "${SLURM_SUBMIT_DIR:?must run under Slurm}"
: "${SLURM_JOB_ID:?must run under Slurm}"
: "${CONFIG_REL:?CONFIG_REL is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"
: "${RELATIONAL_VENV:?RELATIONAL_VENV is required}"
: "${GPU_NAME_REGEX:?GPU_NAME_REGEX is required}"
: "${PROFILE_SHA256:?PROFILE_SHA256 is required}"
: "${BUNDLE_SHA256:?BUNDLE_SHA256 is required}"
: "${PROFILE_PYTHON:?PROFILE_PYTHON is required}"

case "$CONFIG_REL" in
    configs/360m/*.yaml) ;;
    *)
        echo "CONFIG_REL must name a frozen configs/360m YAML" >&2
        exit 2
        ;;
esac
case "$CONFIG_REL" in
    *\\*|*".."*|*$'\n'*|*$'\r'*)
        echo "CONFIG_REL must be a portable relative path" >&2
        exit 2
        ;;
esac
for root_name in DATA_ROOT OUT_ROOT RELATIONAL_VENV; do
    root_value="${!root_name}"
    case "$root_value" in
        /*) ;;
        *)
            echo "$root_name must be absolute" >&2
            exit 2
            ;;
    esac
    case "$root_value" in
        *$'\n'*|*$'\r'*)
            echo "$root_name contains a newline" >&2
            exit 2
            ;;
    esac
done
if [ "$PROFILE_PYTHON" != '${RELATIONAL_VENV}/bin/python' ]; then
    echo "profile Python template is not frozen" >&2
    exit 2
fi
if [ -n "${STEP_LIMIT:-}" ]; then
    case "$STEP_LIMIT" in
        *[!0-9]*|"")
            echo "STEP_LIMIT must be a positive integer" >&2
            exit 2
            ;;
    esac
    if [ "$STEP_LIMIT" -le 0 ]; then
        echo "STEP_LIMIT must be positive" >&2
        exit 2
    fi
fi

cd "$SLURM_SUBMIT_DIR"
PYTHON="$RELATIONAL_VENV/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "Python environment is missing: $PYTHON" >&2
    exit 2
fi
if [ ! -d "$DATA_ROOT" ]; then
    echo "DATA_ROOT does not exist: $DATA_ROOT" >&2
    exit 2
fi
mkdir -p "$OUT_ROOT"

gpu_csv="$(
    nvidia-smi \
        --query-gpu=name,memory.total,memory.free \
        --format=csv,noheader,nounits
)"
gpu_json="$(
    "$PYTHON" - "$GPU_NAME_REGEX" "$gpu_csv" <<'PY'
import json
import re
import sys

pattern, raw = sys.argv[1:]
try:
    matcher = re.compile(pattern)
except re.error as error:
    raise SystemExit(f"invalid GPU_NAME_REGEX: {error}")
rows = [line.strip() for line in raw.splitlines() if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"expected exactly one visible GPU, found {len(rows)}")
fields = [field.strip() for field in rows[0].rsplit(",", 2)]
if len(fields) != 3:
    raise SystemExit("nvidia-smi returned malformed GPU evidence")
name, total, free = fields
if matcher.search(name) is None:
    raise SystemExit(f"visible GPU does not match frozen profile: {name}")
print(
    json.dumps(
        {
            "count": 1,
            "name": name,
            "total_memory_mib": int(total),
            "free_memory_mib": int(free),
        },
        sort_keys=True,
    )
)
PY
)"

runtime_dir="$OUT_ROOT/.launch-configs"
mkdir -p "$runtime_dir"
config_name="$(basename "$CONFIG_REL" .yaml)"
runtime_config="$runtime_dir/${config_name}-${SLURM_JOB_ID}.yaml"
resolve_args=(
    scripts/make_relational_manifest.py
    resolve
    --config "$CONFIG_REL"
    --runtime-config "$runtime_config"
    --data-root "$DATA_ROOT"
    --out-root "$OUT_ROOT"
    --execute
)
if [ -n "${STEP_LIMIT:-}" ]; then
    resolve_args+=(--max-steps "$STEP_LIMIT")
fi
"$PYTHON" "${resolve_args[@]}"

mapfile -t runtime_metadata < <(
    "$PYTHON" - "$runtime_config" "$OUT_ROOT" <<'PY'
from pathlib import Path
import sys

import yaml

config_path = Path(sys.argv[1])
out_root = Path(sys.argv[2]).resolve()
config = yaml.safe_load(config_path.read_text())
if not isinstance(config, dict) or config.get("model") != "d360m":
    raise SystemExit("runtime config must use the frozen standard d360m GPT")
run_id = config.get("run_id")
out_dir = Path(config.get("out_dir", "")).resolve()
if not isinstance(run_id, str) or not run_id:
    raise SystemExit("runtime config has no run_id")
try:
    out_dir.relative_to(out_root)
except ValueError as error:
    raise SystemExit("runtime out_dir escapes OUT_ROOT") from error
maximum = config.get(
    "max_steps",
    int(config["total_tokens"]) // int(config["tokens_per_step"]),
)
print(out_dir)
print(run_id)
print(int(maximum))
PY
)
if [ "${#runtime_metadata[@]}" -ne 3 ]; then
    echo "runtime metadata is incomplete" >&2
    exit 2
fi
run_dir="${runtime_metadata[0]}"
run_id="${runtime_metadata[1]}"
max_steps="${runtime_metadata[2]}"
mkdir -p "$run_dir"
evidence_path="$run_dir/mit-job-evidence.json"
launcher_log="$run_dir/mit-launcher.log"
slurm_version="$(sinfo --version)"

write_evidence() {
    local status="$1"
    local steps_completed="$2"
    local returncode="$3"
    local oom_detected="$4"
    local checkpoint_present="$5"
    local peak_memory_mib="$6"
    "$PYTHON" - \
        "$evidence_path" \
        "$CONFIG_REL" \
        "$run_id" \
        "$runtime_config" \
        "$PROFILE_SHA256" \
        "$BUNDLE_SHA256" \
        "$SLURM_JOB_ID" \
        "$slurm_version" \
        "$gpu_json" \
        "$status" \
        "$max_steps" \
        "$steps_completed" \
        "$returncode" \
        "$oom_detected" \
        "$checkpoint_present" \
        "$peak_memory_mib" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

(
    evidence_path,
    config_rel,
    run_id,
    runtime_config,
    profile_sha256,
    bundle_sha256,
    job_id,
    slurm_version,
    gpu_json,
    status,
    max_steps,
    steps_completed,
    returncode,
    oom_detected,
    checkpoint_present,
    peak_memory_mib,
) = sys.argv[1:]
config_sha256 = hashlib.sha256(Path(config_rel).read_bytes()).hexdigest()
evidence = {
    "schema_version": 1,
    "platform": "mit",
    "status": status,
    "config_rel": config_rel,
    "config_sha256": config_sha256,
    "run_id": run_id,
    "profile_sha256": profile_sha256,
    "bundle_sha256": bundle_sha256,
    "slurm": {"job_id": job_id, "version": slurm_version},
    "gpu": json.loads(gpu_json),
    "max_steps": int(max_steps),
    "steps_completed": int(steps_completed),
    "returncode": int(returncode),
    "oom_detected": oom_detected == "true",
    "checkpoint_present": checkpoint_present == "true",
    "peak_memory_mib": int(peak_memory_mib),
    "runtime_config": runtime_config,
}
path = Path(evidence_path)
temporary = path.with_name(f".{path.name}.tmp")
try:
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
finally:
    temporary.unlink(missing_ok=True)
PY
}

write_evidence "running" 0 -1 false false 0
: > "$launcher_log"
peak_file="$runtime_dir/.${config_name}-${SLURM_JOB_ID}.peak"
trap 'rm -f "$peak_file"' EXIT

monitor_peak_memory() {
    local process_id="$1"
    local output="$2"
    local peak=0
    local used
    while kill -0 "$process_id" 2>/dev/null; do
        used="$(
            nvidia-smi \
                --query-gpu=memory.used \
                --format=csv,noheader,nounits 2>/dev/null || true
        )"
        case "$used" in
            *[!0-9]*|"") ;;
            *)
                if [ "$used" -gt "$peak" ]; then
                    peak="$used"
                fi
                ;;
        esac
        sleep 1
    done
    printf '%s\n' "$peak" > "$output"
}

set +e
"$PYTHON" -u scripts/run_train.py \
    --config "$runtime_config" \
    --resume auto >>"$launcher_log" 2>&1 &
train_pid=$!
set -e
monitor_peak_memory "$train_pid" "$peak_file" &
monitor_pid=$!
set +e
wait "$train_pid"
returncode=$?
set -e
wait "$monitor_pid"
peak_memory_mib="$(<"$peak_file")"
rm -f "$peak_file"

readarray -t final_facts < <(
    "$PYTHON" - "$run_dir/log.jsonl" "$launcher_log" "$run_dir/ckpt.pt" <<'PY'
import json
from pathlib import Path
import sys

log_path, launcher_path, checkpoint_path = map(Path, sys.argv[1:])
steps = 0
if log_path.is_file():
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            steps = max(steps, int(row.get("step", 0)))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
launcher = launcher_path.read_text(errors="replace").lower()
oom = any(
    marker in launcher
    for marker in ("out of memory", "cuda oom", "cuda error: out of memory")
)
checkpoint = checkpoint_path.is_file() and not checkpoint_path.is_symlink()
print(steps)
print("true" if oom else "false")
print("true" if checkpoint else "false")
PY
)
steps_completed="${final_facts[0]:-0}"
oom_detected="${final_facts[1]:-true}"
checkpoint_present="${final_facts[2]:-false}"
status="completed"
if [ "$returncode" -ne 0 ] || [ "$oom_detected" != false ]; then
    status="failed"
fi
write_evidence \
    "$status" \
    "$steps_completed" \
    "$returncode" \
    "$oom_detected" \
    "$checkpoint_present" \
    "$peak_memory_mib"

echo "node=$(hostname) gpu=$gpu_json config=$CONFIG_REL evidence=$evidence_path"
exit "$returncode"
