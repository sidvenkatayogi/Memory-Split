#!/usr/bin/env python
"""Safely plan or submit the frozen six-run 360M MIT Slurm manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cluster.mit.profile import MITProfile, load_profile  # noqa: E402
from scripts.make_relational_manifest import read_manifest  # noqa: E402


SLURM_SCRIPT = "cluster/slurm/relational_mit_train.sbatch"
EXPECTED_RUNS = 6
MAX_CAPTURE_CHARS = 16_384


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(path: Path | str) -> str:
    bundle = Path(path)
    if not bundle.is_file() or bundle.is_symlink():
        raise ValueError(f"bundle is missing or unsafe: {bundle}")
    return _sha256_file(bundle)


def _wall_time(minutes: int) -> str:
    days, remainder = divmod(minutes, 24 * 60)
    hours, minute = divmod(remainder, 60)
    clock = f"{hours:02d}:{minute:02d}:00"
    return f"{days}-{clock}" if days else clock


def _validate_manifest(manifest: Path | str):
    entries = read_manifest(manifest)
    if (
        len(entries) != EXPECTED_RUNS
        or any(job.get("model") != "d360m" for _, job in entries)
    ):
        raise ValueError("MIT launcher requires the exact six-run 360M manifest")
    return entries


def _validate_steps(entries, steps: int | None) -> None:
    if steps is None:
        return
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    maximum = min(
        int(job["total_tokens"]) // int(job["tokens_per_step"])
        for _, job in entries
    )
    if steps > maximum:
        raise ValueError("steps exceeds the frozen run length")


def _command_for(
    relative: str,
    job: dict,
    *,
    profile: MITProfile,
    bundle_sha256: str,
    steps: int | None,
) -> list[str]:
    resources = [
        "sbatch",
        f"--partition={profile.partition}",
    ]
    if profile.account is not None:
        resources.append(f"--account={profile.account}")
    if profile.qos is not None:
        resources.append(f"--qos={profile.qos}")
    resources.extend(
        [
            f"--gres={profile.gres}",
            f"--cpus-per-task={profile.cpus}",
            f"--mem={profile.memory_gb}G",
            f"--time={_wall_time(profile.wall_minutes)}",
            f"--job-name=rel-{job['run_id']}",
        ]
    )
    exports = [
        "ALL",
        f"CONFIG_REL={relative}",
        f"GPU_NAME_REGEX={profile.gpu_name_regex}",
        f"PROFILE_SHA256={profile.sha256}",
        f"BUNDLE_SHA256={bundle_sha256}",
        f"PROFILE_PYTHON={profile.python}",
    ]
    if steps is not None:
        exports.append(f"STEP_LIMIT={steps}")
    resources.extend(
        [
            f"--export={','.join(exports)}",
            SLURM_SCRIPT,
        ]
    )
    return resources


def plan_manifest(
    manifest: Path | str,
    *,
    profile: Path | str,
    bundle: Path | str,
    steps: int | None = None,
) -> list[list[str]]:
    """Render deterministic argv arrays; no command is executed."""

    frozen_profile = load_profile(profile)
    entries = _validate_manifest(manifest)
    _validate_steps(entries, steps)
    bundle_digest = _bundle_sha256(bundle)
    return [
        _command_for(
            relative,
            job,
            profile=frozen_profile,
            bundle_sha256=bundle_digest,
            steps=steps,
        )
        for relative, job in entries
    ]


def _subprocess_command(command: Sequence[str]):
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )


def _result_fields(result: object) -> tuple[int, str, str]:
    if isinstance(result, bool):
        raise TypeError("boolean is not a command result")
    if isinstance(result, int):
        return result, "", ""
    returncode = getattr(result, "returncode", None)
    if returncode is None:
        raise TypeError("command result has no returncode")
    stdout = str(getattr(result, "stdout", "") or "")[:MAX_CAPTURE_CHARS]
    stderr = str(getattr(result, "stderr", "") or "")[:MAX_CAPTURE_CHARS]
    return int(returncode), stdout, stderr


def run_manifest(
    manifest: Path | str,
    *,
    profile: Path | str,
    bundle: Path | str,
    steps: int | None = None,
    execute: bool = False,
    run_command: Callable[[Sequence[str]], object] = _subprocess_command,
) -> dict:
    """Plan by default; with execute, attempt all six independent submissions."""

    frozen_profile = load_profile(profile)
    entries = _validate_manifest(manifest)
    _validate_steps(entries, steps)
    bundle_digest = _bundle_sha256(bundle)
    commands = [
        _command_for(
            relative,
            job,
            profile=frozen_profile,
            bundle_sha256=bundle_digest,
            steps=steps,
        )
        for relative, job in entries
    ]
    report = {
        "schema_version": 1,
        "launcher": "mit-slurm",
        "dry_run": not execute,
        "planned": len(commands),
        "profile_sha256": frozen_profile.sha256,
        "bundle_sha256": bundle_digest,
        "steps": steps,
        "commands": commands,
        "exit_code": 0,
    }
    if not execute:
        return report

    failures = []
    jobs = []
    submitted = 0
    for index, ((relative, job), command) in enumerate(zip(entries, commands)):
        record = {
            "index": index,
            "config_rel": relative,
            "run_id": job["run_id"],
            "command": command,
        }
        try:
            result = run_command(command)
            returncode, stdout, stderr = _result_fields(result)
            record.update(
                {
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                }
            )
            if returncode == 0:
                submitted += 1
            else:
                failures.append(dict(record))
        except Exception as error:
            record.update(
                {
                    "returncode": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            failures.append(dict(record))
        jobs.append(record)

    report.update(
        {
            "attempted": len(jobs),
            "submitted": submitted,
            "failures": failures,
            "jobs": jobs,
            "exit_code": int(bool(failures)),
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan six one-GPU MIT Slurm jobs. "
            "Submission requires explicit --execute."
        )
    )
    parser.add_argument("manifest")
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--bundle",
        default="artifacts/relational-run.tar.gz",
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report = run_manifest(
        args.manifest,
        profile=args.profile,
        bundle=args.bundle,
        steps=args.steps,
        execute=args.execute,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        print("dry-run; pass --execute to submit jobs", file=sys.stderr)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
