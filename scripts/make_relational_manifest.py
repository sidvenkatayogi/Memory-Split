#!/usr/bin/env python
"""Create and safely launch the frozen relational run manifests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml


SEEDS = (0, 1, 2)
DATA_SEED_BASE = 10_000
TOKENS_PER_STEP = 524_288
AWS_PROBE_STEPS = 200
ROUTE_POLICY_SHA256 = (
    "0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058"
)
SCALE_SETTINGS = {
    "160m": {
        "model": "d160m",
        "total_tokens": 1_599_602_688,
        "micro_batch_size": 16,
        "lr": 1.5e-3,
    },
    "360m": {
        "model": "d360m",
        "total_tokens": 3_599_761_408,
        "micro_batch_size": 8,
        "lr": 1.0e-3,
    },
}
LOAD_ENTITIES = {
    "n50k": 50_000,
    "n800k": 800_000,
    "n1p8m": 1_800_000,
}
_RUNTIME_PATH_KEYS = {
    "data_dir",
    "train_bin",
    "train_mask",
    "train_weights",
    "out_dir",
}


class FarmshareSubmissionError(RuntimeError):
    """All failed sbatch calls after every independent config was attempted."""

    def __init__(self, failures: Sequence[Mapping]):
        self.failures = tuple(dict(failure) for failure in failures)
        super().__init__(
            json.dumps(
                {"failed_submissions": list(self.failures)},
                sort_keys=True,
            )
        )


def _portable_relative(value: str | os.PathLike[str], *, label: str) -> str:
    text = os.fspath(value)
    if (
        not text
        or text.startswith("/")
        or "\\" in text
        or (len(text) >= 2 and text[1] == ":")
    ):
        raise ValueError(f"{label} must be a portable relative path: {text!r}")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{label} contains traversal: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a portable relative path: {text!r}")
    return path.as_posix()


def _job(scale: str, condition: str, load: str, seed: int) -> dict:
    settings = SCALE_SETTINGS[scale]
    data_seed = DATA_SEED_BASE + seed
    run_id = f"{settings['model']}_{condition}_{load}_s{seed}"
    return {
        "schema_version": 1,
        "run_id": run_id,
        "model": settings["model"],
        "condition": condition,
        "load": load,
        "n_entities": LOAD_ENTITIES[load],
        "data_seed": data_seed,
        "seed": seed,
        "data_rel": f"{load}_ds{data_seed}",
        "out_rel": run_id,
        "route_policy_sha256": ROUTE_POLICY_SHA256,
        "total_tokens": settings["total_tokens"],
        "tokens_per_step": TOKENS_PER_STEP,
        "micro_batch_size": settings["micro_batch_size"],
        "ctx": 1024,
        "lr": settings["lr"],
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": 0.10,
        "ckpt_minutes": 30,
    }


def make_jobs(scale: str) -> list[dict]:
    """Return the exact frozen jobs for one protected model scale."""

    if scale == "160m":
        jobs = [
            _job(scale, condition, load, seed)
            for load in ("n50k", "n800k")
            for condition in ("dense", "split")
            for seed in SEEDS
        ]
        jobs.extend(_job(scale, "random", "n800k", seed) for seed in SEEDS)
        return jobs
    if scale == "360m":
        return [
            _job(scale, condition, "n1p8m", seed)
            for condition in ("dense", "split")
            for seed in SEEDS
        ]
    raise ValueError(f"unknown relational scale: {scale!r}")


def _config_relative(scale: str, job: Mapping) -> str:
    run_id = _portable_relative(str(job["run_id"]), label="run id")
    if "/" in run_id:
        raise ValueError("run id must not contain directories")
    return f"configs/{scale}/{run_id}.yaml"


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifests(root: Path | str = ".") -> dict[str, dict]:
    """Write deterministic production YAML and relative TSV manifests."""

    root = Path(root)
    results = {}
    for scale in SCALE_SETTINGS:
        config_paths = []
        manifest_rows = []
        for job in make_jobs(scale):
            relative = _config_relative(scale, job)
            path = root / relative
            rendered = yaml.safe_dump(
                job,
                sort_keys=False,
                default_flow_style=False,
            ).encode()
            _atomic_write(path, rendered)
            config_paths.append(path)
            manifest_rows.append(relative)
        manifest = root / "configs" / f"{scale}.tsv"
        _atomic_write(
            manifest,
            ("".join(f"{row}\n" for row in manifest_rows)).encode(),
        )
        results[scale] = {
            "manifest": manifest,
            "configs": tuple(config_paths),
        }
    return results


def _manifest_root(manifest: Path) -> Path:
    if manifest.parent.name != "configs":
        raise ValueError("manifest must be stored directly under configs/")
    return manifest.parent.parent


def read_manifest(manifest: Path | str) -> list[tuple[str, dict]]:
    """Read a generated manifest and validate its complete frozen matrix."""

    manifest = Path(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    root = _manifest_root(manifest)
    scale = manifest.stem
    expected = make_jobs(scale)
    rows = [
        line.strip()
        for line in manifest.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) != len(expected) or len(set(rows)) != len(expected):
        raise ValueError(
            f"{scale} manifest requires exactly {len(expected)} unique configs"
        )
    loaded = []
    for row in rows:
        relative = _portable_relative(row, label="manifest config")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        job = yaml.safe_load(path.read_text())
        if not isinstance(job, dict):
            raise ValueError(f"config must contain a mapping: {relative}")
        loaded.append((relative, job))
    if [job for _, job in loaded] != expected:
        raise ValueError(f"{scale} manifest differs from the frozen run matrix")
    return loaded


def _absolute_root(value: Path | str, *, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path.resolve()


def _join_under(root: Path, relative: str, *, label: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its environment root") from error
    return resolved


def resolve_job(
    job: Mapping,
    *,
    data_root: Path | str,
    out_root: Path | str,
    max_steps: int | None = None,
) -> dict:
    """Resolve only data_rel/out_rel into the legacy trainer path fields."""

    if any(key in job for key in _RUNTIME_PATH_KEYS):
        present = sorted(_RUNTIME_PATH_KEYS.intersection(job))
        raise ValueError(f"relative config contains runtime paths: {present}")
    data_rel = _portable_relative(job["data_rel"], label="data_rel")
    out_rel = _portable_relative(job["out_rel"], label="out_rel")
    data_base = _absolute_root(data_root, label="DATA_ROOT")
    out_base = _absolute_root(out_root, label="OUT_ROOT")
    data_dir = _join_under(data_base, data_rel, label="data_rel")
    out_dir = _join_under(out_base, out_rel, label="out_rel")
    condition = str(job["condition"])
    if condition not in ("dense", "split", "random"):
        raise ValueError(f"unexpected condition: {condition!r}")

    resolved = dict(job)
    resolved.update(
        {
            "data_dir": str(data_dir),
            "train_bin": str(data_dir / "train.bin"),
            "train_weights": str(
                data_dir / f"{condition}.weights.bin"
            ),
            "out_dir": str(out_dir),
        }
    )
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps override must be positive")
        frozen_steps = (
            int(job["total_tokens"]) // int(job["tokens_per_step"])
        )
        if max_steps > frozen_steps:
            raise ValueError("max_steps override exceeds the frozen run")
        resolved["max_steps"] = max_steps
    return resolved


def write_runtime_config(
    job: Mapping,
    path: Path | str,
    *,
    data_root: Path | str,
    out_root: Path | str,
    max_steps: int | None = None,
) -> Path:
    resolved = resolve_job(
        job,
        data_root=data_root,
        out_root=out_root,
        max_steps=max_steps,
    )
    run_dir = Path(resolved["out_dir"])
    checkpoint = run_dir / "ckpt.pt"
    if checkpoint.exists():
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError("resume checkpoint must be a regular file")
        saved_config_path = run_dir / "config.yaml"
        if not saved_config_path.is_file() or saved_config_path.is_symlink():
            raise ValueError("resume checkpoint requires its saved config")
        saved = yaml.safe_load(saved_config_path.read_text())
        if not isinstance(saved, dict):
            raise ValueError("resume config must contain a mapping")
        intended = dict(resolved)
        if (
            "max_steps" not in intended
            and saved.get("max_steps") == AWS_PROBE_STEPS
        ):
            saved = dict(saved)
            saved.pop("max_steps")
        if saved != intended:
            raise ValueError("resume config does not match the requested job")
    output = Path(path)
    _atomic_write(
        output,
        yaml.safe_dump(
            resolved,
            sort_keys=False,
            default_flow_style=False,
        ).encode(),
    )
    return output


def farmshare_plan(manifest: Path | str) -> list[list[str]]:
    return [
        [
            "sbatch",
            f"--export=ALL,CONFIG_REL={relative}",
            "cluster/slurm/relational_train.sbatch",
        ]
        for relative, _ in read_manifest(manifest)
    ]


def _subprocess_command(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def submit_farmshare(
    manifest: Path | str,
    *,
    execute: bool = False,
    run_command: Callable[[Sequence[str]], object] = _subprocess_command,
) -> list[list[str]]:
    """Plan Slurm jobs; call sbatch only when execute is explicitly true."""

    plan = farmshare_plan(manifest)
    if not execute:
        return plan
    failures = []
    for command in plan:
        config_rel = command[1].split("CONFIG_REL=", 1)[1]
        try:
            result = run_command(command)
            returncode = (
                result
                if isinstance(result, int)
                else getattr(result, "returncode", None)
            )
        except Exception as error:
            failures.append(
                {
                    "command": list(command),
                    "config_rel": config_rel,
                    "error": f"{type(error).__name__}: {error}",
                    "returncode": None,
                }
            )
            continue
        if returncode != 0:
            failures.append(
                {
                    "command": list(command),
                    "config_rel": config_rel,
                    "returncode": returncode,
                }
            )
    if failures:
        raise FarmshareSubmissionError(failures)
    return plan


def _load_config(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("config must contain a mapping")
    return value


def _print_plan(commands: Sequence[Sequence[str]]) -> None:
    for command in commands:
        print(json.dumps({"command": list(command)}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate or safely launch frozen relational manifests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--root", default=".")
    generate.add_argument("--execute", action="store_true")

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--config", required=True)
    resolve.add_argument("--runtime-config", required=True)
    resolve.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    resolve.add_argument("--out-root", default=os.environ.get("OUT_ROOT"))
    resolve.add_argument("--max-steps", type=int)
    resolve.add_argument("--execute", action="store_true")

    farmshare = subparsers.add_parser("farmshare")
    farmshare.add_argument("manifest")
    farmshare.add_argument("--execute", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "generate":
        if not args.execute:
            for scale in SCALE_SETTINGS:
                for job in make_jobs(scale):
                    print(_config_relative(scale, job))
                print(f"configs/{scale}.tsv")
            print("dry-run; pass --execute to write manifests", file=sys.stderr)
            return 0
        results = write_manifests(args.root)
        print(
            json.dumps(
                {
                    scale: {
                        "manifest": str(result["manifest"]),
                        "configs": len(result["configs"]),
                    }
                    for scale, result in results.items()
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "resolve":
        if args.data_root is None or args.out_root is None:
            parser.error("DATA_ROOT and OUT_ROOT are required")
        job = _load_config(Path(args.config))
        if not args.execute:
            resolved = resolve_job(
                job,
                data_root=args.data_root,
                out_root=args.out_root,
                max_steps=args.max_steps,
            )
            print(yaml.safe_dump(resolved, sort_keys=False), end="")
            print(
                "dry-run; pass --execute to write the runtime config",
                file=sys.stderr,
            )
            return 0
        output = write_runtime_config(
            job,
            args.runtime_config,
            data_root=args.data_root,
            out_root=args.out_root,
            max_steps=args.max_steps,
        )
        print(output)
        return 0

    try:
        plan = submit_farmshare(args.manifest, execute=args.execute)
    except FarmshareSubmissionError as error:
        print(error, file=sys.stderr)
        return 1
    _print_plan(plan)
    if not args.execute:
        print("dry-run; pass --execute to submit jobs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
