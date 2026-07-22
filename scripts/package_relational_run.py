#!/usr/bin/env python
"""Build a deterministic, source-complete relational run bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.relational_smoke_test import SMOKE_FIXTURE, SMOKE_STEPS


EXPECTED_RUN_COUNTS = {"160m": 15, "360m": 6}
REQUIRED_ENVIRONMENT = ["DATA_ROOT", "OUT_ROOT"]
_SOURCE_PREFIXES = (
    "corpusgen/",
    "evals/",
    "organizer/",
    "scripts/",
    "tests/",
    "train/",
)
_TASK5_SOURCE = {
    "scripts/package_relational_run.py",
    "scripts/relational_smoke_test.py",
    "tests/test_relational_bundle.py",
    "tests/test_relational_smoke.py",
}
_SMOKE_BUNDLE_FIXTURE = {
    "data_seed": SMOKE_FIXTURE["data_seed"],
    "eval_pairs_per_task": SMOKE_FIXTURE["eval_pairs_per_task"],
    "n_entities": SMOKE_FIXTURE["n_entities"],
    "steps": SMOKE_STEPS,
    "total_tokens": SMOKE_FIXTURE["total_tokens"],
}


def _run_git(source_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), *args],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _source_revision(source_root: Path, *, require_clean: bool) -> str:
    revision = _run_git(source_root, "rev-parse", "--verify", "HEAD").strip()
    if require_clean:
        status = _run_git(
            source_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        )
        if status:
            raise ValueError("production bundles require a clean source tree")
    return revision


def _portable_relative(value: str | Path, *, label: str) -> str:
    text = str(value)
    if (
        not text
        or "\\" in text
        or text.startswith("/")
        or (len(text) >= 2 and text[1] == ":")
    ):
        raise ValueError(f"{label} must be a portable relative path: {text!r}")
    raw_parts = text.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"{label} must not contain traversal: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a portable relative path: {text!r}")
    return path.as_posix()


def _read_input(input_root: Path, relative: str, *, label: str) -> bytes:
    root = input_root.resolve()
    path = input_root
    for part in PurePosixPath(relative).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {relative}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(
            f"{label} must remain inside the input root: {relative}"
        ) from error
    if not path.is_file():
        raise ValueError(f"{label} must name a regular file: {relative}")
    return path.read_bytes()


def _require_suffix(
    relative: str,
    suffixes: tuple[str, ...],
    *,
    label: str,
) -> None:
    if PurePosixPath(relative).suffix.lower() not in suffixes:
        expected = " or ".join(suffix.upper() for suffix in suffixes)
        raise ValueError(f"{label} must be a {expected} file")


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _iter_strings(item)


def _validate_portable_string(value: str, *, label: str) -> None:
    text = value.strip()
    if any(marker in value for marker in ("/Users/", "/scratch/", "s3://")):
        raise ValueError(f"{label} contains a nonportable path")
    if "\\" in value:
        raise ValueError(f"{label} contains a nonportable path separator")
    if (
        text.startswith(("/", "~/"))
        or re.match(r"^[A-Za-z]:[/\\]", text)
        or ".." in PurePosixPath(text).parts
    ):
        raise ValueError(f"{label} contains a nonportable path")


def _validate_input_content(data: bytes, *, label: str, kind: str) -> None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} must be portable UTF-8 text") from error
    if "\x00" in text:
        raise ValueError(f"{label} must be portable UTF-8 text")
    if kind == "yaml":
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ValueError(f"{label} must contain valid YAML") from error
        values = _iter_strings(parsed)
    elif kind == "json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label} must contain valid JSON") from error
        values = _iter_strings(parsed)
    elif kind == "tsv":
        values = (
            field
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
            for field in line.split("\t")
        )
    else:
        raise ValueError(f"unknown portable input kind: {kind}")
    for value in values:
        _validate_portable_string(value, label=label)


def _tracked_source_paths(source_root: Path) -> list[str]:
    tracked = {
        line
        for line in _run_git(source_root, "ls-files", "*.py", "requirements.txt")
        .splitlines()
        if line
    }
    selected = {
        path
        for path in tracked | _TASK5_SOURCE
        if path == "requirements.txt"
        or path.endswith(".py")
        and path.startswith(_SOURCE_PREFIXES)
    }
    missing = [
        path
        for path in sorted(selected)
        if not (source_root / path).is_file()
        or (source_root / path).is_symlink()
    ]
    if missing:
        raise ValueError(f"source members are missing or not regular: {missing}")
    return sorted(selected)


def _add_payload(payload: dict[str, bytes], name: str, data: bytes) -> None:
    if name in payload:
        raise ValueError(f"duplicate bundle member: {name}")
    payload[name] = data


def _manifest_member(name: str, data: bytes) -> dict:
    return {
        "path": name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _write_archive(out: Path, files: Mapping[str, bytes]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw,
                compresslevel=9,
                mtime=0,
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed,
                    mode="w",
                    format=tarfile.PAX_FORMAT,
                ) as archive:
                    for name in sorted(files):
                        data = files[name]
                        info = tarfile.TarInfo(name)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.pax_headers = {
                            "SHA256": hashlib.sha256(data).hexdigest()
                        }
                        archive.addfile(info, io.BytesIO(data))
        os.replace(temporary, out)
    finally:
        if temporary.exists():
            temporary.unlink()


def package_run(
    out: Path | str,
    *,
    config_inputs: Mapping[str, Sequence[str | Path]],
    manifest_inputs: Mapping[str, str | Path],
    route_policy: str | Path,
    source_root: Path | str | None = None,
    input_root: Path | str | None = None,
    require_clean: bool = False,
) -> Path:
    """Package supplied run inputs plus the exact executable source surface."""

    source = (
        Path(source_root)
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    inputs = Path(input_root) if input_root is not None else source
    revision = _source_revision(source, require_clean=require_clean)
    if set(config_inputs) != set(EXPECTED_RUN_COUNTS):
        raise ValueError("config input scales must be exactly 160m and 360m")
    if set(manifest_inputs) != set(EXPECTED_RUN_COUNTS):
        raise ValueError("manifest input scales must be exactly 160m and 360m")

    payload: dict[str, bytes] = {}
    for path in _tracked_source_paths(source):
        _add_payload(payload, path, (source / path).read_bytes())
    _add_payload(payload, "source-revision.txt", f"{revision}\n".encode())
    _add_payload(
        payload,
        "fixtures/relational-smoke.json",
        (
            json.dumps(_SMOKE_BUNDLE_FIXTURE, indent=2, sort_keys=True) + "\n"
        ).encode(),
    )

    policy_path = _portable_relative(route_policy, label="route policy")
    _require_suffix(policy_path, (".json",), label="route policy")
    policy_data = _read_input(inputs, policy_path, label="route policy")
    _validate_input_content(
        policy_data,
        label="route policy",
        kind="json",
    )
    _add_payload(
        payload,
        "route-policy.json",
        policy_data,
    )
    for scale, expected_count in EXPECTED_RUN_COUNTS.items():
        paths = [
            _portable_relative(path, label=f"{scale} config")
            for path in config_inputs[scale]
        ]
        if len(paths) != expected_count or len(set(paths)) != expected_count:
            raise ValueError(
                f"{scale} requires exactly {expected_count} unique configs"
            )
        for path in paths:
            _require_suffix(
                path,
                (".yaml", ".yml"),
                label=f"{scale} config",
            )
            config_data = _read_input(inputs, path, label=f"{scale} config")
            _validate_input_content(
                config_data,
                label=f"{scale} config",
                kind="yaml",
            )
            _add_payload(
                payload,
                path,
                config_data,
            )
        manifest_path = _portable_relative(
            manifest_inputs[scale],
            label=f"{scale} manifest",
        )
        _require_suffix(
            manifest_path,
            (".tsv",),
            label=f"{scale} manifest",
        )
        manifest_data = _read_input(
            inputs,
            manifest_path,
            label=f"{scale} manifest",
        )
        _validate_input_content(
            manifest_data,
            label=f"{scale} manifest",
            kind="tsv",
        )
        _add_payload(payload, manifest_path, manifest_data)

    manifest = {
        "schema_version": 1,
        "source_revision": revision,
        "expected_run_counts": EXPECTED_RUN_COUNTS,
        "required_environment": REQUIRED_ENVIRONMENT,
        "members": [
            _manifest_member(name, payload[name]) for name in sorted(payload)
        ],
    }
    files = dict(payload)
    files["manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    output = Path(out)
    _write_archive(output, files)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package a deterministic portable relational run."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--input-root")
    parser.add_argument("--route-policy", required=True)
    parser.add_argument("--manifest-160m", required=True)
    parser.add_argument("--manifest-360m", required=True)
    parser.add_argument("--config-160m", action="append", required=True)
    parser.add_argument("--config-360m", action="append", required=True)
    args = parser.parse_args(argv)
    archive = package_run(
        args.out,
        source_root=args.source_root,
        input_root=args.input_root,
        route_policy=args.route_policy,
        config_inputs={
            "160m": args.config_160m,
            "360m": args.config_360m,
        },
        manifest_inputs={
            "160m": args.manifest_160m,
            "360m": args.manifest_360m,
        },
        require_clean=True,
    )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(
        json.dumps(
            {"archive": str(archive), "sha256": digest},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
