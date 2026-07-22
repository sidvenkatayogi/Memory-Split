#!/usr/bin/env python
"""Collect bounded, read-only facts from an unknown MIT Slurm cluster."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path


COMMANDS = (
    ("sinfo", "--version"),
    ("sinfo", "-h", "-o", "%P|%G|%l|%D|%m"),
    ("scontrol", "show", "config"),
)
COMMAND_NAMES = ("slurm_version", "partitions", "config")
COMMAND_TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 1_048_576
SCRATCH_ENVIRONMENT = ("SCRATCH", "SLURM_TMPDIR", "TMPDIR", "HOME")
_CAPACITY_RE = re.compile(r"^(?P<value>[0-9]+)(?P<suffix>[+*]?)$")


def _bounded_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[:MAX_OUTPUT_CHARS]


def _run_command(
    command: Sequence[str],
    *,
    command_runner: Callable,
) -> dict:
    try:
        completed = command_runner(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        returncode = int(completed.returncode)
        stdout = _bounded_text(completed.stdout)
        stderr = _bounded_text(completed.stderr)
        result = {
            "argv": list(command),
            "ok": returncode == 0,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        if returncode != 0:
            result["error"] = (
                stderr.strip()
                or stdout.strip()
                or f"command exited {returncode}"
            )
        return result
    except Exception as error:
        return {
            "argv": list(command),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"{type(error).__name__}: {error}",
        }


def _parse_capacity(
    raw: str,
    *,
    field: str,
    line_number: int,
) -> tuple[int, str | None]:
    match = _CAPACITY_RE.fullmatch(raw)
    if match is None:
        prefix = re.match(r"^[0-9]+", raw)
        reason = (
            "has no numeric prefix"
            if prefix is None
            else f"has unexpected trailing text {raw[prefix.end():]!r}"
        )
        raise ValueError(
            f"partition row {line_number} {field} capacity {raw!r} {reason}"
        )
    suffix = match.group("suffix") or None
    return int(match.group("value")), suffix


def parse_partitions(text: str) -> list[dict]:
    """Parse the exact delimiter format requested by ``COMMANDS``."""

    partitions = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 5:
            raise ValueError(
                f"partition row {line_number} has {len(fields)} fields"
            )
        raw_partition, gres, time_limit, raw_nodes, raw_memory = fields
        default = raw_partition.endswith("*")
        partition = raw_partition[:-1] if default else raw_partition
        if not partition:
            raise ValueError(f"partition row {line_number} has no name")
        nodes, nodes_suffix = _parse_capacity(
            raw_nodes,
            field="node",
            line_number=line_number,
        )
        memory_mb, memory_suffix = _parse_capacity(
            raw_memory,
            field="memory",
            line_number=line_number,
        )
        partitions.append(
            {
                "partition": partition,
                "default": default,
                "gres": gres,
                "time_limit": time_limit,
                "nodes": nodes,
                "nodes_approximate": nodes_suffix is not None,
                "nodes_suffix": nodes_suffix,
                "memory_mb": memory_mb,
                "memory_mb_approximate": memory_suffix is not None,
                "memory_mb_suffix": memory_suffix,
            }
        )
    return partitions


def parse_slurm_config(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in values:
            values[key] = value.strip()
    return values


def _read_small_text(path: Path) -> str | None:
    try:
        if not path.is_file() or path.is_symlink() or path.stat().st_size > 65_536:
            return None
        return path.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def _git_revision_from_files(source_root: Path | str) -> str | None:
    """Read HEAD without starting another process or contacting a remote."""

    root = Path(source_root)
    dot_git = root / ".git"
    if dot_git.is_file() and not dot_git.is_symlink():
        marker = _read_small_text(dot_git)
        if not marker or not marker.startswith("gitdir: "):
            return None
        git_dir = Path(marker.removeprefix("gitdir: ").strip())
        if not git_dir.is_absolute():
            git_dir = (root / git_dir).resolve()
    elif dot_git.is_dir() and not dot_git.is_symlink():
        git_dir = dot_git
    else:
        return None

    head = _read_small_text(git_dir / "HEAD")
    if not head:
        return None
    if len(head) == 40 and all(character in "0123456789abcdef" for character in head):
        return head
    if not head.startswith("ref: "):
        return None
    reference = head.removeprefix("ref: ").strip()
    if (
        not reference.startswith("refs/")
        or ".." in reference.split("/")
        or "\\" in reference
    ):
        return None

    common_dir = git_dir
    common_marker = _read_small_text(git_dir / "commondir")
    if common_marker:
        candidate = Path(common_marker)
        common_dir = (
            candidate
            if candidate.is_absolute()
            else (git_dir / candidate).resolve()
        )
    for candidate in (git_dir / reference, common_dir / reference):
        revision = _read_small_text(candidate)
        if (
            revision
            and len(revision) == 40
            and all(
                character in "0123456789abcdef" for character in revision
            )
        ):
            return revision
    packed = _read_small_text(common_dir / "packed-refs")
    if packed:
        suffix = f" {reference}"
        for line in packed.splitlines():
            if line.endswith(suffix):
                revision = line.split(" ", 1)[0]
                if len(revision) == 40:
                    return revision
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_fact(bundle: Path | str | None) -> dict | None:
    if bundle is None:
        return None
    path = Path(bundle)
    if not path.is_file() or path.is_symlink():
        return {
            "path": str(path),
            "error": "bundle is missing or unsafe",
        }
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _python_facts(
    *,
    which: Callable[[str], str | None],
    is_executable: Callable[[str], bool],
) -> list[str]:
    candidates = [sys.executable, which("python3"), which("python")]
    result = []
    for candidate in candidates:
        if (
            candidate
            and candidate not in result
            and is_executable(candidate)
        ):
            result.append(candidate)
    return result


def _scratch_facts(
    environ: Mapping[str, str],
    *,
    disk_usage: Callable,
    path_exists: Callable[[str], bool],
) -> list[dict]:
    result = []
    seen = set()
    for name in SCRATCH_ENVIRONMENT:
        value = environ.get(name)
        if not value or value in seen or not path_exists(value):
            continue
        seen.add(value)
        try:
            usage = disk_usage(value)
            result.append(
                {
                    "environment": name,
                    "path": value,
                    "free_bytes": int(usage.free),
                    "total_bytes": int(usage.total),
                }
            )
        except Exception as error:
            result.append(
                {
                    "environment": name,
                    "path": value,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return result


def _filesystem_fact(source_root: Path | str, *, disk_usage: Callable) -> dict:
    path = Path(source_root)
    result = {"path": str(path)}
    try:
        usage = disk_usage(path)
        result.update(
            {
                "free_bytes": int(usage.free),
                "total_bytes": int(usage.total),
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def collect_probe(
    *,
    command_runner: Callable = subprocess.run,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
    disk_usage: Callable = shutil.disk_usage,
    path_exists: Callable[[str], bool] = os.path.isdir,
    is_executable: Callable[[str], bool] | None = None,
    git_revision_reader: Callable[[Path | str], str | None] = (
        _git_revision_from_files
    ),
    source_root: Path | str = ".",
    bundle: Path | str | None = None,
) -> dict:
    """Attempt every bounded probe and return evidence without side effects."""

    environment = os.environ if environ is None else environ
    executable_probe = is_executable or (
        lambda path: os.path.isfile(path) and os.access(path, os.X_OK)
    )
    commands = {
        name: _run_command(command, command_runner=command_runner)
        for name, command in zip(COMMAND_NAMES, COMMANDS)
    }
    partition_rows = []
    partition_error = None
    partition_command = commands["partitions"]
    if partition_command["ok"]:
        try:
            partition_rows = parse_partitions(partition_command["stdout"])
        except ValueError as error:
            partition_error = f"{type(error).__name__}: {error}"
            partition_command["ok"] = False
            partition_command["error"] = partition_error

    version = (
        commands["slurm_version"]["stdout"].strip()
        if commands["slurm_version"]["ok"]
        else None
    )
    config_text = (
        commands["config"]["stdout"] if commands["config"]["ok"] else ""
    )
    module_command = which("module")
    module_home = environment.get("MODULESHOME")
    lmod_command = environment.get("LMOD_CMD")
    return {
        "schema_version": 1,
        "commands": commands,
        "slurm": {
            "version": version,
            "partitions": partition_rows,
            "config": parse_slurm_config(config_text),
            "partition_parse_error": partition_error,
        },
        "module": {
            "command": module_command,
            "module_home": module_home,
            "lmod_command": lmod_command,
            "available": bool(module_command or module_home or lmod_command),
        },
        "python_executables": _python_facts(
            which=which,
            is_executable=executable_probe,
        ),
        "scratch_candidates": _scratch_facts(
            environment,
            disk_usage=disk_usage,
            path_exists=path_exists,
        ),
        "filesystem": _filesystem_fact(
            source_root,
            disk_usage=disk_usage,
        ),
        "source_revision": git_revision_reader(source_root),
        "bundle": _bundle_fact(bundle),
    }


def write_probe(path: Path | str, evidence: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record bounded read-only Slurm, filesystem, module, and Python facts."
        ),
        epilog=(
            "A trailing '+' or '*' on sinfo %D/%m is retained and marks "
            "the capacity approximate."
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--bundle")
    args = parser.parse_args(argv)
    evidence = collect_probe(
        source_root=args.source_root,
        bundle=args.bundle,
    )
    output = write_probe(args.out, evidence)
    print(output)
    return 0 if all(
        item["ok"] for item in evidence["commands"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
