#!/usr/bin/env python
"""Run the frozen 360M manifest on an eight-worker CUDA queue."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_relational_manifest import (  # noqa: E402
    read_manifest,
    resolve_job,
    write_runtime_config,
)


# Frozen p5.48xlarge topology; do not resize this to the six-job queue.
AWS_WORKERS = 8
STATUS_NAME = "aws-launch-status.json"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _query_memory_used(gpu: str) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"nvidia-smi memory probe failed: {detail}")
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError("nvidia-smi memory probe returned an invalid row count")
    return int(rows[0])


def _run_training(
    command: Sequence[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> dict:
    gpu = env["CUDA_VISIBLE_DEVICES"]
    peak = 0
    with log_path.open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            peak = max(peak, _query_memory_used(gpu))
            time.sleep(1)
        peak = max(peak, _query_memory_used(gpu))
    return {
        "returncode": int(process.returncode),
        "peak_memory_mib": peak,
    }


def _normalize_result(result, *, gpu: str, memory_probe) -> tuple[int, int]:
    if isinstance(result, int):
        return result, int(memory_probe(gpu))
    if hasattr(result, "returncode"):
        return int(result.returncode), int(memory_probe(gpu))
    if isinstance(result, dict):
        if "returncode" not in result or "peak_memory_mib" not in result:
            raise ValueError("launcher result requires returncode and peak memory")
        return int(result["returncode"]), int(result["peak_memory_mib"])
    raise TypeError("unsupported launcher command result")


def run_manifest(
    manifest: Path | str,
    *,
    data_root: Path | str,
    out_root: Path | str,
    gpu_ids: Sequence[int | str],
    steps: int | None = None,
    execute: bool = False,
    run_command: Callable = _run_training,
    memory_probe: Callable[[str], int] = _query_memory_used,
) -> dict:
    """Run every job, retaining failures while independent workers finish."""

    gpu_ids = tuple(str(gpu) for gpu in gpu_ids)
    if len(gpu_ids) != AWS_WORKERS or len(set(gpu_ids)) != AWS_WORKERS:
        raise ValueError("AWS relational launches require eight unique workers")
    if steps is not None and steps <= 0:
        raise ValueError("steps must be positive")
    entries = read_manifest(manifest)
    if len(entries) != 6 or any(job["model"] != "d360m" for _, job in entries):
        raise ValueError("AWS launcher requires the exact six-run 360M manifest")

    if not execute:
        return {
            "schema_version": 1,
            "launcher": "aws",
            "dry_run": True,
            "planned": len(entries),
            "gpu_ids": list(gpu_ids),
            "step_limit": steps,
            "commands": [
                {
                    "config_rel": relative,
                    "run_id": job["run_id"],
                    "resume": "auto",
                }
                for relative, job in entries
            ],
        }

    data_root = Path(data_root)
    out_root = Path(out_root)
    if not data_root.is_absolute() or not out_root.is_absolute():
        raise ValueError("DATA_ROOT and OUT_ROOT must be absolute")
    work: queue.Queue = queue.Queue()
    for entry in entries:
        work.put(entry)
    results = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                relative, job = work.get_nowait()
            except queue.Empty:
                return
            record = {
                "config_rel": relative,
                "run_id": job["run_id"],
                "gpu": gpu,
                "step_limit": steps,
            }
            try:
                resolved = resolve_job(
                    job,
                    data_root=data_root,
                    out_root=out_root,
                    max_steps=steps,
                )
                run_dir = Path(resolved["out_dir"])
                run_dir.mkdir(parents=True, exist_ok=True)
                runtime_config = run_dir / "launch-config.yaml"
                write_runtime_config(
                    job,
                    runtime_config,
                    data_root=data_root,
                    out_root=out_root,
                    max_steps=steps,
                )
                record["runtime_config"] = str(runtime_config)
                command = [
                    sys.executable,
                    "-u",
                    "scripts/run_train.py",
                    "--config",
                    str(runtime_config),
                    "--resume",
                    "auto",
                ]
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=gpu,
                    PYTHONPATH=str(REPO_ROOT),
                )
                raw_result = run_command(
                    command,
                    env=env,
                    log_path=run_dir / "launcher.log",
                )
                returncode, peak = _normalize_result(
                    raw_result,
                    gpu=gpu,
                    memory_probe=memory_probe,
                )
                record.update(
                    {
                        "returncode": returncode,
                        "peak_memory_mib": peak,
                    }
                )
            except Exception as error:
                record.update(
                    {
                        "returncode": 255,
                        "peak_memory_mib": 0,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            with lock:
                results.append(record)
            work.task_done()

    workers = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in gpu_ids
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    ordered = sorted(results, key=lambda item: item["run_id"])
    failed = [
        item["run_id"] for item in ordered if item["returncode"] != 0
    ]
    status = {
        "schema_version": 1,
        "launcher": "aws",
        "dry_run": False,
        "planned": len(entries),
        "completed": len(entries) - len(failed),
        "failed": failed,
        "exit_code": int(bool(failed)),
        "gpu_ids": list(gpu_ids),
        "step_limit": steps,
        "jobs": ordered,
    }
    _atomic_json(out_root / STATUS_NAME, status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run six 360M configs on an eight-worker CUDA queue. "
            "Dry-run unless --execute is supplied."
        )
    )
    parser.add_argument("manifest")
    parser.add_argument("--gpus", type=int, default=AWS_WORKERS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--out-root", default=os.environ.get("OUT_ROOT"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.gpus != AWS_WORKERS:
        parser.error("the protected AWS queue requires --gpus 8")
    if args.data_root is None or args.out_root is None:
        parser.error("DATA_ROOT and OUT_ROOT are required")
    result = run_manifest(
        args.manifest,
        data_root=args.data_root,
        out_root=args.out_root,
        gpu_ids=tuple(range(args.gpus)),
        steps=args.steps,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not args.execute:
        print("dry-run; pass --execute to launch jobs", file=sys.stderr)
    return int(result.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
