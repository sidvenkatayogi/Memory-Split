#!/usr/bin/env python
"""Fail-closed local, FarmShare, and AWS launch preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster.mit.profile import MITProfile, load_profile  # noqa: E402
from scripts.make_relational_manifest import (  # noqa: E402
    ROUTE_POLICY_SHA256,
    make_jobs,
    resolve_job,
)
from scripts.relational_smoke_test import (  # noqa: E402
    SMOKE_FIXTURE,
    SMOKE_STEPS,
    run_smoke,
)


REQUIRED_MODULES = ("torch", "numpy", "tiktoken", "yaml")
EXPECTED_RUN_COUNTS = {"160m": 15, "360m": 6}
REQUIRED_ENVIRONMENT = ["DATA_ROOT", "OUT_ROOT"]
FARMSHARE_FREE_BYTES = 500_000_000_000
AWS_FREE_BYTES = 1_000_000_000_000
L40S_MIN_MEMORY_MIB = 44 * 1024
# nvidia-smi reports MiB; this threshold is 72 GiB = 72 * 1024 MiB.
H100_USABLE_MEMORY_MIB = 72 * 1024
AWS_THROUGHPUT_MIN = 60_000.0
AWS_WARMUP_STEPS = 50
AWS_PROBE_STEPS = 200
MIT_WARMUP_STEPS = 50
MIT_PROBE_STEPS = 200
MIT_EVIDENCE_NAME = "mit-job-evidence.json"
RESUME_TOLERANCE = 1e-5
LEARNABILITY_THRESHOLD = 0.75
LEARNABILITY_PAIRS_PER_TASK = 10_000
LEARNABILITY_TASKS = (
    "path_composition",
    "date_ordering",
    "balanced_equality",
)
_SMOKE_FIXTURE = {
    "data_seed": SMOKE_FIXTURE["data_seed"],
    "eval_pairs_per_task": SMOKE_FIXTURE["eval_pairs_per_task"],
    "n_entities": SMOKE_FIXTURE["n_entities"],
    "steps": SMOKE_STEPS,
    "total_tokens": SMOKE_FIXTURE["total_tokens"],
}
_GREEN_SMOKE_REPORT = {
    "shared_stream": True,
    "dense_steps": SMOKE_STEPS,
    "split_steps": SMOKE_STEPS,
    "resume_exact": True,
    "memory_modes": ["off", "on"],
    "pairs_complete": True,
}
_SOURCE_PREFIXES = (
    "cluster/",
    "configs/",
    "corpusgen/",
    "evals/",
    "organizer/",
    "schemas/",
    "scripts/",
    "tests/",
    "train/",
    "vendor/",
)
_TOKENIZER_ASSETS = {
    "vendor/tiktoken/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
    "vendor/tiktoken/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
}


@dataclass(frozen=True)
class GPUInfo:
    name: str
    total_memory_mib: int
    free_memory_mib: int


@dataclass(frozen=True)
class DiskInfo:
    free_bytes: int
    writable: bool


@dataclass(frozen=True)
class ResumeInfo:
    steps: int
    exact: bool
    next_loss_delta: float


@dataclass(frozen=True)
class FixtureProbe:
    """Injected high-level command/probe fixture for GPU-free tests."""

    modules_ok: bool
    commands: frozenset[str]
    gpus: tuple[GPUInfo, ...]
    disk: DiskInfo
    resume: ResumeInfo

    def python_modules_available(self, modules) -> bool:
        return self.modules_ok

    def has_command(self, command: str) -> bool:
        return command in self.commands

    def gpu_info(self) -> tuple[GPUInfo, ...]:
        return self.gpus

    def disk_info(self, root: Path) -> DiskInfo:
        return self.disk

    def resume_info(
        self,
        *,
        platform: str,
        runs_root: Path | str | None = None,
    ) -> ResumeInfo:
        return self.resume

    @classmethod
    def from_json(cls, path: Path | str) -> "FixtureProbe":
        raw = json.loads(Path(path).read_text())
        required = {"modules_ok", "commands", "gpus", "disk", "resume"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("probe fixture fields do not match the schema")
        return cls(
            modules_ok=bool(raw["modules_ok"]),
            commands=frozenset(str(item) for item in raw["commands"]),
            gpus=tuple(GPUInfo(**item) for item in raw["gpus"]),
            disk=DiskInfo(**raw["disk"]),
            resume=ResumeInfo(**raw["resume"]),
        )


class LiveProbe:
    """Local-only probes; never contacts AWS, Slurm, or another service."""

    def python_modules_available(self, modules) -> bool:
        imports = ";".join(f"import {module}" for module in modules)
        completed = subprocess.run(
            [sys.executable, "-c", imports],
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.returncode == 0

    def has_command(self, command: str) -> bool:
        return shutil.which(command) is not None

    def gpu_info(self) -> tuple[GPUInfo, ...]:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"nvidia-smi failed: {detail}")
        devices = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                raise ValueError("nvidia-smi returned malformed GPU data")
            devices.append(
                GPUInfo(
                    name=fields[0],
                    total_memory_mib=int(fields[1]),
                    free_memory_mib=int(fields[2]),
                )
            )
        if not devices:
            raise ValueError("nvidia-smi returned no GPUs")
        return tuple(devices)

    def disk_info(self, root: Path) -> DiskInfo:
        if not root.is_dir():
            raise FileNotFoundError(root)
        writable = False
        try:
            with tempfile.NamedTemporaryFile(dir=root):
                writable = True
        except OSError:
            writable = False
        return DiskInfo(
            free_bytes=shutil.disk_usage(root).free,
            writable=writable,
        )

    def resume_info(
        self,
        *,
        platform: str,
        runs_root: Path | str | None = None,
    ) -> ResumeInfo:
        if platform == "aws":
            return _aws_checkpoint_resume_info(runs_root)
        if platform == "mit":
            return _mit_checkpoint_resume_info(runs_root)
        with tempfile.TemporaryDirectory(prefix="relational-resume-") as raw:
            report = run_smoke(Path(raw) / "smoke", device="cpu")
        exact = report == _GREEN_SMOKE_REPORT
        return ResumeInfo(
            steps=SMOKE_STEPS,
            exact=exact,
            next_loss_delta=0.0 if exact else math.inf,
        )


def _portable_relative(value: str, *, label: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or (len(value) >= 2 and value[1] == ":")
    ):
        raise ValueError(f"{label} is not a portable relative path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{label} contains traversal")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} is not a portable relative path")
    return path.as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_digest(policy: dict) -> str:
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _bundle_files(bundle: Path) -> tuple[dict[str, bytes], dict]:
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("bundle contains duplicate members")
        files = {}
        for member in members:
            relative = _portable_relative(member.name, label="bundle member")
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"bundle member is not a regular file: {relative}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"bundle member cannot be read: {relative}")
            data = extracted.read()
            pax_digest = member.pax_headers.get("SHA256")
            if pax_digest != _sha256_bytes(data):
                raise ValueError(f"bundle pax hash mismatch: {relative}")
            files[relative] = data
    if "manifest.json" not in files:
        raise ValueError("bundle is missing manifest.json")
    manifest = json.loads(files["manifest.json"])
    if not isinstance(manifest, dict):
        raise ValueError("bundle manifest must contain an object")
    return files, manifest


def verify_bundle(
    bundle: Path | str,
    *,
    source_root: Path | str | None = None,
) -> dict:
    """Verify archive safety, every member hash, smoke, and frozen YAML."""

    files, manifest = _bundle_files(Path(bundle))
    if manifest.get("expected_run_counts") != EXPECTED_RUN_COUNTS:
        raise ValueError("bundle run counts differ from the frozen matrix")
    if manifest.get("required_environment") != REQUIRED_ENVIRONMENT:
        raise ValueError("bundle environment contract is incorrect")
    members = manifest.get("members")
    if not isinstance(members, list):
        raise ValueError("bundle manifest members must be a list")
    indexed = {}
    for item in members:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes"}
        ):
            raise ValueError("bundle member index has an invalid schema")
        relative = _portable_relative(item["path"], label="indexed member")
        if relative in indexed:
            raise ValueError("bundle member index contains duplicates")
        indexed[relative] = item
    if set(indexed) != set(files) - {"manifest.json"}:
        raise ValueError("bundle member index is incomplete")
    for relative, item in indexed.items():
        data = files[relative]
        if item["bytes"] != len(data):
            raise ValueError(f"bundle byte count mismatch: {relative}")
        if item["sha256"] != _sha256_bytes(data):
            raise ValueError(f"bundle member hash mismatch: {relative}")
    missing_tokenizer_assets = sorted(_TOKENIZER_ASSETS - set(indexed))
    if missing_tokenizer_assets:
        raise ValueError(
            "bundle is missing offline tokenizer assets: "
            f"{missing_tokenizer_assets}"
        )

    fixture = json.loads(files["fixtures/relational-smoke.json"])
    if fixture != _SMOKE_FIXTURE:
        raise ValueError("bundle smoke fixture differs from Task 5")
    smoke_report = json.loads(
        files["fixtures/relational-smoke-report.json"]
    )
    if smoke_report != _GREEN_SMOKE_REPORT:
        raise ValueError("bundle smoke report is not entirely green")

    policy_document = json.loads(files["route-policy.json"])
    policy = policy_document.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("bundle route policy is missing policy fields")
    policy_sha256 = _policy_digest(policy)
    declared_policy = policy_document.get("policy_sha256", policy_sha256)
    if (
        policy_sha256 != ROUTE_POLICY_SHA256
        or declared_policy != policy_sha256
    ):
        raise ValueError("bundle route-policy hash is not frozen")

    run_counts = {}
    for scale, expected_count in EXPECTED_RUN_COUNTS.items():
        manifest_name = f"configs/{scale}.tsv"
        rows = [
            line.strip()
            for line in files[manifest_name].decode().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        expected_jobs = make_jobs(scale)
        expected_rows = [
            f"configs/{scale}/{job['run_id']}.yaml"
            for job in expected_jobs
        ]
        if rows != expected_rows or len(set(rows)) != expected_count:
            raise ValueError(f"{scale} bundle manifest is not exact")
        for relative, expected_job in zip(rows, expected_jobs):
            _portable_relative(relative, label=f"{scale} config")
            job = yaml.safe_load(files[relative])
            if job != expected_job:
                raise ValueError(f"bundle config differs from frozen job: {relative}")
            if set(
                key
                for key in job
                if key.endswith(("_rel", "_dir", "_bin", "_path"))
            ) != {"data_rel", "out_rel"}:
                raise ValueError(f"bundle config contains runtime paths: {relative}")
        run_counts[scale] = len(rows)

    if source_root is not None:
        root = Path(source_root)
        for relative, item in indexed.items():
            if not relative.startswith(_SOURCE_PREFIXES):
                continue
            source = root / relative
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"source member is missing: {relative}")
            if (
                source.stat().st_size != item["bytes"]
                or _sha256_file(source) != item["sha256"]
            ):
                raise ValueError(f"source member differs from bundle: {relative}")

    return {
        "run_counts": run_counts,
        "policy": policy,
        "policy_sha256": policy_sha256,
        "smoke_report": smoke_report,
        "member_count": len(indexed),
        "tokenizer_assets": len(_TOKENIZER_ASSETS),
        "archive_sha256": _sha256_file(Path(bundle)),
    }


def _verify_corpus(root: Path, job: dict, policy: dict) -> None:
    relative = _portable_relative(job["data_rel"], label="data_rel")
    corpus = root.joinpath(*PurePosixPath(relative).parts)
    if not corpus.is_dir() or corpus.is_symlink():
        raise ValueError(f"corpus directory is missing or unsafe: {relative}")
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"corpus manifest has no artifacts: {relative}")
    seen = set()
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes"}
        ):
            raise ValueError(f"corpus artifact schema is invalid: {relative}")
        artifact_rel = _portable_relative(
            item["path"],
            label="corpus artifact",
        )
        if artifact_rel in seen:
            raise ValueError(f"duplicate corpus artifact: {artifact_rel}")
        seen.add(artifact_rel)
        path = corpus.joinpath(*PurePosixPath(artifact_rel).parts)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"corpus artifact is missing: {artifact_rel}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"corpus byte count mismatch: {artifact_rel}")
        if _sha256_file(path) != item["sha256"]:
            raise ValueError(f"corpus hash mismatch: {artifact_rel}")
    required = {
        "train.bin",
        "dense.weights.bin",
        "split.weights.bin",
        "random.weights.bin",
        "route-policy.json",
        "report.json",
        "eval/route-audit.json",
    }
    if not required <= seen:
        raise ValueError(
            f"corpus manifest omits required artifacts: {sorted(required - seen)}"
        )
    route = json.loads((corpus / "eval" / "route-audit.json").read_text())
    route_rate = float(route["route_rate"])
    tail_rate = float(route["low_use_high_entropy_external_rate"])
    structure_rate = float(route["rules_top_centrality_internal_rate"])
    if (
        not all(
            math.isfinite(value)
            for value in (route_rate, tail_rate, structure_rate)
        )
        or not 0.40 <= route_rate <= 0.60
        or tail_rate < 0.80
        or structure_rate < 0.80
        or int(route["route_total"]) <= 0
        or int(route["low_use_high_entropy_total"]) <= 0
        or int(route["rules_top_centrality_total"]) <= 0
    ):
        raise ValueError(f"corpus route guardrails failed: {relative}")
    report = json.loads((corpus / "report.json").read_text())
    config = report.get("config")
    expected_config = {
        "n_entities": job["n_entities"],
        "total_tokens": job["total_tokens"],
        "data_seed": job["data_seed"],
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected_config.items()
    ):
        raise ValueError(f"corpus config does not match {relative}")
    checks = report.get("checks")
    if (
        not isinstance(checks, dict)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError(f"corpus report is not entirely green: {relative}")
    corpus_policy = json.loads((corpus / "route-policy.json").read_text())
    if (
        corpus_policy.get("policy") != policy
        or corpus_policy.get(
            "policy_sha256",
            _policy_digest(corpus_policy.get("policy", {})),
        )
        != _policy_digest(policy)
    ):
        raise ValueError(f"corpus route policy mismatch: {relative}")


def verify_corpora(
    data_root: Path | str,
    policy: dict,
    *,
    scales: tuple[str, ...] = ("160m", "360m"),
) -> dict:
    root = Path(data_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("DATA_ROOT must be an existing absolute directory")
    jobs = {
        job["data_rel"]: job
        for scale in scales
        for job in make_jobs(scale)
    }
    for job in jobs.values():
        _verify_corpus(root, job, policy)
    return {"corpora": len(jobs)}


def _aws_checkpoint_resume_info(
    runs_root: Path | str | None,
) -> ResumeInfo:
    if runs_root is None:
        raise ValueError("AWS runs root is required for resume verification")
    root = Path(runs_root)
    job = make_jobs("360m")[0]
    run_dir = root / job["out_rel"]
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "ckpt.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise ValueError("AWS 200-step checkpoint fixture is incomplete")
    cfg = yaml.safe_load(config_path.read_text())
    if (
        not isinstance(cfg, dict)
        or cfg.get("model") != "d360m"
        or cfg.get("max_steps") != AWS_PROBE_STEPS
    ):
        raise ValueError("AWS resume fixture is not the 200-step 360M config")

    import torch

    from train.trainer import Trainer

    def resumed_next_loss() -> float:
        with tempfile.TemporaryDirectory(
            prefix=".resume-probe-",
            dir=root,
        ) as raw:
            probe_cfg = dict(
                cfg,
                out_dir=str(Path(raw) / "run"),
                compile=False,
            )
            trainer = Trainer(probe_cfg)
            trainer.load_ckpt(checkpoint_path)
            if trainer.step != AWS_PROBE_STEPS:
                raise ValueError("AWS checkpoint is not at step 200")
            with torch.no_grad():
                if trainer.cfg.get("train_weights"):
                    x, targets, weights = trainer.data.next_weighted_batch()
                    with trainer._autocast():
                        _, loss = trainer.model(
                            x,
                            targets,
                            target_weights=weights,
                        )
                else:
                    x, targets = trainer.data.next_batch()
                    with trainer._autocast():
                        _, loss = trainer.model(x, targets)
            if loss is None or not torch.isfinite(loss):
                raise ValueError("AWS resumed next loss is not finite")
            value = float(loss.item())
            del trainer, x, targets, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return value

    first = resumed_next_loss()
    second = resumed_next_loss()
    return ResumeInfo(
        steps=AWS_PROBE_STEPS,
        exact=first == second,
        next_loss_delta=abs(first - second),
    )


def _mit_checkpoint_resume_info(
    runs_root: Path | str | None,
) -> ResumeInfo:
    """Use the same deterministic 200-step checkpoint probe on MIT."""

    try:
        return _aws_checkpoint_resume_info(runs_root)
    except ValueError as error:
        raise ValueError(str(error).replace("AWS", "MIT")) from error


def _resume_detail(
    probe,
    *,
    platform: str,
    runs_root: Path | str | None,
) -> dict:
    resume = probe.resume_info(platform=platform, runs_root=runs_root)
    expected_steps = (
        AWS_PROBE_STEPS
        if platform == "aws"
        else MIT_PROBE_STEPS
        if platform == "mit"
        else SMOKE_STEPS
    )
    if (
        resume.steps != expected_steps
        or not math.isfinite(resume.next_loss_delta)
        or resume.next_loss_delta > RESUME_TOLERANCE
        or (
            platform not in ("aws", "mit")
            and resume.exact is not True
        )
    ):
        message = (
            "requires finite next_loss_delta <= 1e-5"
            if platform in ("aws", "mit")
            else "must be exact with finite next_loss_delta <= 1e-5"
        )
        raise ValueError(f"{expected_steps}-step checkpoint resume {message}")
    return asdict(resume)


def _farm_gpu_detail(probe) -> dict:
    devices = probe.gpu_info()
    qualifying = [
        device
        for device in devices
        if "L40S" in device.name
        and device.total_memory_mib >= L40S_MIN_MEMORY_MIB
    ]
    if not qualifying:
        raise ValueError("FarmShare requires at least one 44GiB+ L40S")
    return {"devices": [asdict(device) for device in devices]}


def _aws_gpu_detail(probe) -> dict:
    devices = probe.gpu_info()
    # Frozen p5.48xlarge topology, not a minimum based on the six-job queue.
    if len(devices) != 8:
        raise ValueError("AWS requires exactly eight visible GPUs")
    if any("H100" not in device.name for device in devices):
        raise ValueError("every AWS GPU must be an H100")
    if any(
        device.total_memory_mib < H100_USABLE_MEMORY_MIB
        or device.free_memory_mib < H100_USABLE_MEMORY_MIB
        for device in devices
    ):
        raise ValueError("every H100 requires 72GiB usable memory")
    return {"devices": [asdict(device) for device in devices]}


def _mit_gpu_detail(probe, profile: MITProfile) -> dict:
    devices = probe.gpu_info()
    if len(devices) != 1:
        raise ValueError("MIT requires exactly one visible GPU")
    device = devices[0]
    if re.search(profile.gpu_name_regex, device.name) is None:
        raise ValueError(
            "visible MIT GPU does not match the frozen profile regex"
        )
    if device.total_memory_mib <= 0 or device.free_memory_mib < 0:
        raise ValueError("MIT GPU memory evidence is invalid")
    return {
        "gpu_name_regex": profile.gpu_name_regex,
        "devices": [asdict(device)],
    }


def _disk_detail(probe, root: Path | str | None, minimum: int) -> dict:
    if root is None:
        raise ValueError("writable storage root is required")
    path = Path(root)
    if not path.is_absolute():
        raise ValueError("storage root must be absolute")
    disk = probe.disk_info(path)
    if not disk.writable or disk.free_bytes < minimum:
        raise ValueError(
            f"storage requires writable={True} and at least {minimum} bytes free"
        )
    return asdict(disk)


def _read_unique_json_object(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"summary is missing or unsafe: {path}")

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"JSON object contains duplicate key: {key}")
            value[key] = item
        return value

    value = json.loads(
        path.read_text(),
        object_pairs_hook=unique_object,
        parse_constant=lambda constant: (_ for _ in ()).throw(
            ValueError(f"JSON contains non-finite value: {constant}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"summary must contain a JSON object: {path}")
    return value


def _existing_root(value: Path | str | None, *, label: str) -> Path:
    if value is None:
        raise ValueError(f"{label} is required")
    path = Path(value)
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{label} must be an existing absolute directory")
    return path.resolve(strict=True)


def _safe_path_identity(
    root: Path,
    path: Path,
    *,
    label: str,
    kind: str,
) -> tuple[Path, tuple[int, int]]:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its declared root") from error
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink: {current}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f"{label} is missing or escapes its declared root") from error
    info = resolved.stat()
    valid_kind = (
        stat.S_ISDIR(info.st_mode)
        if kind == "directory"
        else stat.S_ISREG(info.st_mode)
    )
    if not valid_kind:
        raise ValueError(f"{label} must be a {kind}")
    return resolved, (info.st_dev, info.st_ino)


def _exact_runtime_config(job: dict, config: dict, expected: dict) -> None:
    expected_keys = set(expected)
    actual_keys = set(config)
    if actual_keys != expected_keys:
        missing = sorted(str(key) for key in expected_keys - actual_keys)
        unexpected = sorted(str(key) for key in actual_keys - expected_keys)
        raise ValueError(
            f"{job['run_id']}: runtime config keys mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    mismatched = [
        key
        for key, value in expected.items()
        if type(config[key]) is not type(value) or config[key] != value
    ]
    if mismatched:
        raise ValueError(
            f"{job['run_id']}: runtime config values mismatch: "
            f"{sorted(mismatched)}"
        )


def _learnability_scores(summary: dict, job: dict) -> dict[str, float]:
    expected_mode_fields = {
        "memory",
        "tasks",
        "primary_composite",
        "n_rows",
        "n_pairs_per_task",
    }
    if (
        not isinstance(summary, Mapping)
        or set(summary) != expected_mode_fields
        or summary["memory"] != "on"
        or summary["n_pairs_per_task"] != LEARNABILITY_PAIRS_PER_TASK
        or summary["n_rows"]
        != 2 * LEARNABILITY_PAIRS_PER_TASK * len(LEARNABILITY_TASKS)
    ):
        raise ValueError(
            f"{job['run_id']}: memory-on summary is incomplete or mismatched"
        )
    tasks = summary["tasks"]
    if not isinstance(tasks, Mapping) or set(tasks) != set(LEARNABILITY_TASKS):
        raise ValueError(
            f"{job['run_id']}: summary task set is incomplete or mismatched"
        )

    scores = {}
    for task in LEARNABILITY_TASKS:
        measurement = tasks[task]
        if (
            not isinstance(measurement, Mapping)
            or measurement.get("n_pairs") != LEARNABILITY_PAIRS_PER_TASK
            or measurement.get("n_rows")
            != 2 * LEARNABILITY_PAIRS_PER_TASK
        ):
            raise ValueError(
                f"{job['run_id']}:{task}: evaluation is incomplete"
            )
        raw_score = measurement.get("counterfactual_pair_accuracy")
        if isinstance(raw_score, bool) or not isinstance(
            raw_score, (int, float)
        ):
            raise ValueError(
                f"{job['run_id']}:{task}: pair accuracy is not numeric"
            )
        score = float(raw_score)
        if (
            not math.isfinite(score)
            or score <= LEARNABILITY_THRESHOLD
            or score > 1.0
        ):
            raise ValueError(
                f"{job['run_id']}:{task}: counterfactual pair accuracy "
                "must be finite, at most 1.0, and greater than 0.75"
            )
        scores[task] = score

    composite = summary["primary_composite"]
    expected_composite = sum(scores.values()) / len(scores)
    if (
        isinstance(composite, bool)
        or not isinstance(composite, (int, float))
        or not math.isfinite(float(composite))
        or not math.isclose(
            float(composite),
            expected_composite,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            f"{job['run_id']}: primary composite is inconsistent"
        )
    return scores


def _farmshare_learnability_detail(
    data_root: Path | str | None,
    out_root: Path | str | None,
) -> dict:
    data_base = _existing_root(data_root, label="FarmShare data root")
    out_base = _existing_root(out_root, label="FarmShare output root")
    expected_jobs = make_jobs("29m")
    expected_by_id = {job["run_id"]: job for job in expected_jobs}
    candidates = {run_id: [] for run_id in expected_by_id}
    canonical = {
        job["run_id"]: out_base / job["out_rel"] for job in expected_jobs
    }

    for directory in out_base.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        config_path = directory / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = yaml.safe_load(config_path.read_text())
        except Exception:
            if directory in canonical.values():
                raise ValueError(
                    f"{directory.name}: gate run config cannot be read"
                )
            continue
        if not isinstance(config, dict):
            if directory in canonical.values():
                raise ValueError(
                    f"{directory.name}: gate run config is not a mapping"
                )
            continue
        run_id = config.get("run_id")
        if run_id in candidates:
            if config_path.is_symlink():
                raise ValueError(f"{run_id}: gate run config is unsafe")
            candidates[run_id].append((directory, config))

    runs = []
    config_files = set()
    summary_files = set()
    for job in expected_jobs:
        run_id = job["run_id"]
        matches = candidates[run_id]
        if len(matches) != 1:
            raise ValueError(
                f"{run_id}: requires exactly one completed run, "
                f"found {len(matches)}"
            )
        run_dir, config = matches[0]
        if run_dir != canonical[run_id]:
            raise ValueError(f"{run_id}: run directory is mismatched")
        run_dir, run_identity = _safe_path_identity(
            out_base,
            run_dir,
            label=f"{run_id} run directory",
            kind="directory",
        )
        config_path = run_dir / "config.yaml"
        _, config_identity = _safe_path_identity(
            out_base,
            config_path,
            label=f"{run_id} runtime config",
            kind="regular file",
        )
        if config_identity in config_files:
            raise ValueError("29M gate runtime configs must be distinct files")
        config_files.add(config_identity)

        data_dir, data_identity = _safe_path_identity(
            data_base,
            data_base / job["data_rel"],
            label=f"{run_id} data directory",
            kind="directory",
        )
        train_bin, train_identity = _safe_path_identity(
            data_base,
            data_dir / "train.bin",
            label=f"{run_id} train_bin",
            kind="regular file",
        )
        train_weights, weights_identity = _safe_path_identity(
            data_base,
            data_dir / f"{job['condition']}.weights.bin",
            label=f"{run_id} train_weights",
            kind="regular file",
        )
        expected = resolve_job(
            job,
            data_root=data_base,
            out_root=out_base,
        )
        _exact_runtime_config(job, config, expected)
        if (
            config["data_dir"] != str(data_dir)
            or config["train_bin"] != str(train_bin)
            or config["train_weights"] != str(train_weights)
            or config["out_dir"] != str(run_dir)
        ):
            raise ValueError(f"{run_id}: resolved runtime paths are mismatched")

        summary_path = run_dir / "evals" / "memory_on" / "summary.json"
        summary_path, summary_identity = _safe_path_identity(
            out_base,
            summary_path,
            label=f"{run_id} memory-on summary",
            kind="regular file",
        )
        summary = _read_unique_json_object(summary_path)
        if summary_identity in summary_files:
            raise ValueError("29M gate summaries must be distinct files")
        summary_files.add(summary_identity)
        runs.append(
            {
                "run_id": run_id,
                "condition": job["condition"],
                "scores": _learnability_scores(summary, job),
                "runtime_config": str(config_path),
                "summary": str(summary_path),
                "run_identity": run_identity,
                "data_identity": data_identity,
                "train_identity": train_identity,
                "weights_identity": weights_identity,
                "runtime": config,
            }
        )

    if len({run["run_identity"] for run in runs}) != len(expected_jobs):
        raise ValueError("29M gate run directories must be distinct")
    if len({run["data_identity"] for run in runs}) != 1:
        raise ValueError("29M Dense/Split must share one gate corpus")
    if len({run["train_identity"] for run in runs}) != 1:
        raise ValueError("29M Dense/Split must share one train_bin")
    if len({run["weights_identity"] for run in runs}) != len(expected_jobs):
        raise ValueError("29M Dense/Split require distinct condition sidecars")
    runtime_by_condition = {run["condition"]: run["runtime"] for run in runs}
    if (
        runtime_by_condition["dense"]["train_bin"]
        != runtime_by_condition["split"]["train_bin"]
        or not runtime_by_condition["dense"]["train_weights"].endswith(
            "/dense.weights.bin"
        )
        or not runtime_by_condition["split"]["train_weights"].endswith(
            "/split.weights.bin"
        )
    ):
        raise ValueError("29M Dense/Split runtime pairing is mismatched")

    return {
        "threshold": LEARNABILITY_THRESHOLD,
        "comparison": "strictly_greater",
        "runs": [
            {
                "run_id": run["run_id"],
                "condition": run["condition"],
                "scores": run["scores"],
                "runtime_config": run["runtime_config"],
                "summary": run["summary"],
            }
            for run in runs
        ],
    }


def _aws_run_evidence(runs_root: Path | str | None) -> dict:
    if runs_root is None:
        raise ValueError("AWS runs root is required")
    root = Path(runs_root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("AWS runs root must be an existing absolute directory")
    status = json.loads((root / "aws-launch-status.json").read_text())
    expected_jobs = make_jobs("360m")
    expected_ids = {job["run_id"] for job in expected_jobs}
    status_jobs = status.get("jobs")
    if (
        status.get("launcher") != "aws"
        or status.get("dry_run") is not False
        or status.get("planned") != 6
        or status.get("completed") != 6
        or status.get("failed") != []
        or status.get("exit_code") != 0
        or status.get("step_limit") != AWS_PROBE_STEPS
        or len(status.get("gpu_ids", [])) != 8
        or not isinstance(status_jobs, list)
        or {item.get("run_id") for item in status_jobs} != expected_ids
    ):
        raise ValueError("AWS launcher status is not a complete 200-step probe")
    peaks = {}
    for item in status_jobs:
        if (
            item.get("returncode") != 0
            or item.get("step_limit") != AWS_PROBE_STEPS
        ):
            raise ValueError("AWS status contains a failed or partial run")
        peak = item.get("peak_memory_mib")
        if not isinstance(peak, (int, float)) or not math.isfinite(peak):
            raise ValueError("AWS status is missing peak memory")
        peaks[item["run_id"]] = float(peak)

    throughputs = {}
    for job in expected_jobs:
        log_path = root / job["out_rel"] / "log.jsonl"
        if not log_path.is_file():
            raise ValueError(f"AWS probe log is missing: {job['run_id']}")
        rows = [
            json.loads(line)
            for line in log_path.read_text().splitlines()
            if line.strip()
        ]
        if not rows or max(int(row.get("step", -1)) for row in rows) < 200:
            raise ValueError(f"AWS probe did not reach step 200: {job['run_id']}")
        values = [
            float(row["tok_s"])
            for row in rows
            if AWS_WARMUP_STEPS < int(row.get("step", -1)) <= AWS_PROBE_STEPS
            and "tok_s" in row
        ]
        # Trainer timing windows include checkpoint/snapshot work between
        # log rows, so this intentionally keeps the 60k gate conservative.
        if (
            not values
            or any(not math.isfinite(value) or value <= 0 for value in values)
        ):
            raise ValueError(
                f"AWS probe lacks post-warmup throughput: {job['run_id']}"
            )
        throughputs[job["run_id"]] = values
    all_values = [
        value for values in throughputs.values() for value in values
    ]
    return {
        "mean_throughput": sum(all_values) / len(all_values),
        "throughput_samples": len(all_values),
        "peak_memory_mib": peaks,
    }


def _mit_run_evidence(
    runs_root: Path | str | None,
    *,
    data_root: Path | str | None,
    profile: MITProfile,
    bundle_sha256: str,
) -> dict:
    if runs_root is None:
        raise ValueError("MIT runs root is required")
    root = _existing_root(runs_root, label="MIT runs root")
    data_base = _existing_root(data_root, label="MIT data root")
    runtime_root = root / ".launch-configs"
    if not runtime_root.is_dir() or runtime_root.is_symlink():
        raise ValueError("MIT runtime-config directory is missing or unsafe")
    if profile.wall_minutes <= max(
        int(job["ckpt_minutes"]) for job in make_jobs("360m")
    ):
        raise ValueError(
            "MIT wall time must exceed the checkpoint interval"
        )

    expected_fields = {
        "schema_version",
        "platform",
        "status",
        "config_rel",
        "config_sha256",
        "run_id",
        "profile_sha256",
        "bundle_sha256",
        "slurm",
        "gpu",
        "max_steps",
        "steps_completed",
        "returncode",
        "oom_detected",
        "checkpoint_present",
        "peak_memory_mib",
        "runtime_config",
    }
    expected_gpu_fields = {
        "count",
        "name",
        "total_memory_mib",
        "free_memory_mib",
    }
    expected_slurm_fields = {"job_id", "version"}
    run_details = {}
    throughput_runs = {}
    all_throughputs = []
    checkpoint_identities = set()
    evidence_identities = set()
    slurm_versions = set()
    repository = Path(__file__).resolve().parents[1]

    for job in make_jobs("360m"):
        run_id = job["run_id"]
        relative = f"configs/360m/{run_id}.yaml"
        run_dir, _ = _safe_path_identity(
            root,
            root / job["out_rel"],
            label=f"{run_id} run directory",
            kind="directory",
        )
        evidence_path, evidence_identity = _safe_path_identity(
            root,
            run_dir / MIT_EVIDENCE_NAME,
            label=f"{run_id} MIT evidence",
            kind="regular file",
        )
        if evidence_identity in evidence_identities:
            raise ValueError("MIT job evidence files must be distinct")
        evidence_identities.add(evidence_identity)
        evidence = _read_unique_json_object(evidence_path)
        if set(evidence) != expected_fields:
            raise ValueError(f"{run_id}: MIT evidence fields are not exact")
        if (
            evidence["schema_version"] != 1
            or isinstance(evidence["schema_version"], bool)
            or evidence["platform"] != "mit"
            or evidence["status"] != "completed"
            or evidence["config_rel"] != relative
            or evidence["run_id"] != run_id
            or evidence["profile_sha256"] != profile.sha256
            or evidence["bundle_sha256"] != bundle_sha256
            or evidence["max_steps"] != MIT_PROBE_STEPS
            or isinstance(evidence["max_steps"], bool)
            or evidence["steps_completed"] != MIT_PROBE_STEPS
            or isinstance(evidence["steps_completed"], bool)
            or evidence["returncode"] != 0
            or isinstance(evidence["returncode"], bool)
            or evidence["oom_detected"] is not False
            or evidence["checkpoint_present"] is not True
        ):
            raise ValueError(
                f"{run_id}: MIT evidence is incomplete or mismatched"
            )

        config_path = repository / relative
        if not config_path.is_file() or config_path.is_symlink():
            raise ValueError(f"{run_id}: frozen config is missing or unsafe")
        if evidence["config_sha256"] != _sha256_file(config_path):
            raise ValueError(f"{run_id}: frozen config hash is mismatched")

        slurm = evidence["slurm"]
        if (
            not isinstance(slurm, Mapping)
            or set(slurm) != expected_slurm_fields
            or not isinstance(slurm["job_id"], str)
            or not slurm["job_id"]
            or not isinstance(slurm["version"], str)
            or not slurm["version"].strip()
        ):
            raise ValueError(f"{run_id}: Slurm evidence is incomplete")
        slurm_versions.add(slurm["version"])

        gpu = evidence["gpu"]
        if (
            not isinstance(gpu, Mapping)
            or set(gpu) != expected_gpu_fields
            or gpu["count"] != 1
            or isinstance(gpu["count"], bool)
            or not isinstance(gpu["name"], str)
            or re.search(profile.gpu_name_regex, gpu["name"]) is None
        ):
            raise ValueError(
                f"{run_id}: GPU evidence does not match the profile"
            )
        for field, allow_zero in (
            ("total_memory_mib", False),
            ("free_memory_mib", True),
        ):
            value = gpu[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < int(not allow_zero)
            ):
                raise ValueError(f"{run_id}: GPU memory evidence is invalid")

        peak = evidence["peak_memory_mib"]
        if (
            isinstance(peak, bool)
            or not isinstance(peak, (int, float))
            or not math.isfinite(float(peak))
            or float(peak) <= 0
        ):
            raise ValueError(f"{run_id}: peak memory evidence is missing")

        runtime_value = evidence["runtime_config"]
        if not isinstance(runtime_value, str):
            raise ValueError(f"{run_id}: runtime config path is invalid")
        runtime_config, _ = _safe_path_identity(
            root,
            Path(runtime_value),
            label=f"{run_id} runtime config",
            kind="regular file",
        )
        try:
            runtime_config.relative_to(runtime_root)
        except ValueError as error:
            raise ValueError(
                f"{run_id}: runtime config is outside .launch-configs"
            ) from error
        expected_runtime = resolve_job(
            job,
            data_root=data_base,
            out_root=root,
            max_steps=MIT_PROBE_STEPS,
        )
        runtime = yaml.safe_load(runtime_config.read_text())
        if not isinstance(runtime, dict):
            raise ValueError(f"{run_id}: runtime config is not a mapping")
        _exact_runtime_config(job, runtime, expected_runtime)

        saved_config, _ = _safe_path_identity(
            root,
            run_dir / "config.yaml",
            label=f"{run_id} saved config",
            kind="regular file",
        )
        saved = yaml.safe_load(saved_config.read_text())
        if not isinstance(saved, dict):
            raise ValueError(f"{run_id}: saved config is not a mapping")
        _exact_runtime_config(job, saved, expected_runtime)

        checkpoint, checkpoint_identity = _safe_path_identity(
            root,
            run_dir / "ckpt.pt",
            label=f"{run_id} checkpoint",
            kind="regular file",
        )
        if checkpoint_identity in checkpoint_identities:
            raise ValueError("MIT probe checkpoints must be distinct")
        checkpoint_identities.add(checkpoint_identity)
        if not checkpoint.is_file():
            raise ValueError(f"{run_id}: checkpoint is missing")

        log_path, _ = _safe_path_identity(
            root,
            run_dir / "log.jsonl",
            label=f"{run_id} training log",
            kind="regular file",
        )
        rows = []
        for line_number, line in enumerate(
            log_path.read_text().splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{run_id}: malformed log row {line_number}"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(f"{run_id}: log row is not an object")
            rows.append(row)
        steps = [
            row.get("step")
            for row in rows
            if isinstance(row.get("step"), int)
            and not isinstance(row.get("step"), bool)
        ]
        if not steps or max(steps) != MIT_PROBE_STEPS:
            raise ValueError(f"{run_id}: probe did not complete 200 steps")
        throughputs = []
        for row in rows:
            step = row.get("step")
            if (
                not isinstance(step, int)
                or isinstance(step, bool)
                or not MIT_WARMUP_STEPS < step <= MIT_PROBE_STEPS
                or "tok_s" not in row
            ):
                continue
            value = row["tok_s"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{run_id}: throughput is not numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{run_id}: throughput is not positive")
            throughputs.append(numeric)
        if not throughputs:
            raise ValueError(
                f"{run_id}: post-warmup throughput evidence is missing"
            )
        mean = sum(throughputs) / len(throughputs)
        projected_hours = float(job["total_tokens"]) / mean / 3600.0
        projected_resubmissions = math.ceil(
            projected_hours * 60.0 / profile.wall_minutes
        )
        all_throughputs.extend(throughputs)
        throughput_runs[run_id] = {
            "mean_raw_tokens_per_second_per_gpu": mean,
            "samples": len(throughputs),
            "projected_full_hours": projected_hours,
            "projected_resubmissions": projected_resubmissions,
        }
        run_details[run_id] = {
            "evidence": str(evidence_path),
            "checkpoint": str(checkpoint),
            "steps_completed": MIT_PROBE_STEPS,
            "peak_memory_mib": float(peak),
            "gpu": dict(gpu),
        }

    if len(slurm_versions) != 1:
        raise ValueError("MIT probe Slurm versions are inconsistent")
    return {
        "runs": run_details,
        "slurm_version": next(iter(slurm_versions)),
        "profile_sha256": profile.sha256,
        "bundle_sha256": bundle_sha256,
        "throughput": {
            "mean_raw_tokens_per_second_per_gpu": (
                sum(all_throughputs) / len(all_throughputs)
            ),
            "samples": len(all_throughputs),
            "measurement": (
                "checkpoint-inclusive post-warmup logged raw throughput"
            ),
            "threshold": None,
            "runs": throughput_runs,
        },
    }


def run_preflight(
    platform: str,
    *,
    bundle: Path | str,
    profile: Path | str | None = None,
    probe=None,
    source_root: Path | str | None = None,
    data_root: Path | str | None = None,
    out_root: Path | str | None = None,
    runs_root: Path | str | None = None,
    capacity_type: str | None = None,
) -> dict:
    """Run every required check and return all failures without exceptions."""

    if platform not in ("local", "farmshare", "aws", "mit"):
        raise ValueError(f"unknown platform: {platform}")
    probe = probe or LiveProbe()
    checks = {}

    def record(name, operation) -> object | None:
        try:
            detail = operation()
            checks[name] = {"passed": True, "detail": detail}
            return detail
        except Exception as error:
            checks[name] = {
                "passed": False,
                "detail": f"{type(error).__name__}: {error}",
            }
            return None

    record(
        "dependencies",
        lambda: (
            {"modules": list(REQUIRED_MODULES)}
            if probe.python_modules_available(REQUIRED_MODULES)
            else (_ for _ in ()).throw(
                ValueError("required Python dependencies are unavailable")
            )
        ),
    )
    bundle_evidence = record(
        "bundle_hashes",
        lambda: verify_bundle(bundle, source_root=source_root),
    )
    record(
        "smoke",
        lambda: (
            bundle_evidence["smoke_report"]
            if bundle_evidence is not None
            and bundle_evidence["smoke_report"] == _GREEN_SMOKE_REPORT
            else (_ for _ in ()).throw(
                ValueError("verified green smoke report is unavailable")
            )
        ),
    )

    mit_profile_cache = {}
    profile_evidence = None
    if platform == "mit":
        def mit_profile_detail():
            if profile is None:
                raise ValueError("MIT profile is required")
            loaded = load_profile(profile)
            mit_profile_cache["value"] = loaded
            return {
                "sha256": loaded.sha256,
                "profile": loaded.as_dict(),
            }

        profile_evidence = record("profile", mit_profile_detail)

    def mit_profile() -> MITProfile:
        value = mit_profile_cache.get("value")
        if value is None:
            raise ValueError("validated MIT profile is unavailable")
        return value

    if platform in ("farmshare", "aws", "mit"):
        record(
            "gpu_command",
            lambda: (
                {"command": "nvidia-smi"}
                if probe.has_command("nvidia-smi")
                else (_ for _ in ()).throw(
                    ValueError("nvidia-smi is unavailable")
                )
            ),
        )
        record(
            "corpus_hashes",
            lambda: (
                verify_corpora(
                    data_root,
                    bundle_evidence["policy"],
                    scales=(
                        ("29m", "160m")
                        if platform == "farmshare"
                        else ("360m",)
                    ),
                )
                if bundle_evidence is not None
                else (_ for _ in ()).throw(
                    ValueError("bundle policy is unavailable")
                )
            ),
        )
        record(
            "resume",
            lambda: _resume_detail(
                probe,
                platform=platform,
                runs_root=runs_root,
            ),
        )

    if platform == "farmshare":
        record(
            "learnability",
            lambda: _farmshare_learnability_detail(data_root, out_root),
        )
        required_slurm = ("sbatch", "scontrol", "sinfo")
        record(
            "slurm",
            lambda: (
                {"commands": list(required_slurm)}
                if all(probe.has_command(command) for command in required_slurm)
                else (_ for _ in ()).throw(
                    ValueError("required Slurm commands are unavailable")
                )
            ),
        )
        record("gpu", lambda: _farm_gpu_detail(probe))
        record(
            "free_space",
            lambda: _disk_detail(
                probe,
                out_root,
                FARMSHARE_FREE_BYTES,
            ),
        )

    if platform == "aws":
        record("gpu", lambda: _aws_gpu_detail(probe))
        record(
            "free_space",
            lambda: _disk_detail(probe, out_root, AWS_FREE_BYTES),
        )
        record(
            "capacity",
            lambda: (
                {"capacity_type": capacity_type}
                if capacity_type in ("on-demand", "capacity-block")
                else (_ for _ in ()).throw(
                    ValueError(
                        "AWS capacity must be declared on-demand or capacity-block"
                    )
                )
            ),
        )
        evidence_cache = {}

        def aws_evidence():
            if "value" not in evidence_cache:
                evidence_cache["value"] = _aws_run_evidence(runs_root)
            return evidence_cache["value"]

        def throughput_detail():
            evidence = aws_evidence()
            if evidence["mean_throughput"] < AWS_THROUGHPUT_MIN:
                raise ValueError("post-warmup throughput is below 60k tokens/s/GPU")
            return {
                "mean_raw_tokens_per_second_per_gpu": evidence[
                    "mean_throughput"
                ],
                "measurement": (
                    "checkpoint/snapshot-inclusive logged throughput"
                ),
                "samples": evidence["throughput_samples"],
            }

        def peak_detail():
            evidence = aws_evidence()
            peaks = evidence["peak_memory_mib"]
            if any(
                peak <= 0 or peak > H100_USABLE_MEMORY_MIB
                for peak in peaks.values()
            ):
                raise ValueError("360M peak memory exceeds 72GiB or is missing")
            return {"peak_memory_mib": peaks}

        record("throughput", throughput_detail)
        record("peak_memory", peak_detail)

    if platform == "mit":
        required_slurm = ("sbatch", "scontrol", "sinfo")
        record(
            "slurm",
            lambda: (
                {"commands": list(required_slurm)}
                if all(probe.has_command(command) for command in required_slurm)
                else (_ for _ in ()).throw(
                    ValueError("required Slurm commands are unavailable")
                )
            ),
        )
        record("gpu", lambda: _mit_gpu_detail(probe, mit_profile()))
        mit_evidence_cache = {}

        def mit_evidence():
            if "value" not in mit_evidence_cache:
                if bundle_evidence is None:
                    raise ValueError("verified bundle hash is unavailable")
                mit_evidence_cache["value"] = _mit_run_evidence(
                    runs_root,
                    data_root=data_root,
                    profile=mit_profile(),
                    bundle_sha256=bundle_evidence["archive_sha256"],
                )
            return mit_evidence_cache["value"]

        record(
            "mit_runs",
            lambda: {
                key: value
                for key, value in mit_evidence().items()
                if key != "throughput"
            },
        )
        record("throughput", lambda: mit_evidence()["throughput"])

    report = {
        "schema_version": 1,
        "platform": platform,
        "ok": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "bundle": bundle_evidence,
        "profile": profile_evidence,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed relational launch preflight."
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=("local", "farmshare", "aws", "mit"),
    )
    parser.add_argument(
        "--bundle",
        default="artifacts/relational-run.tar.gz",
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--profile")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--out-root", default=os.environ.get("OUT_ROOT"))
    parser.add_argument("--runs-root")
    parser.add_argument(
        "--capacity-type",
        default=os.environ.get("AWS_CAPACITY_TYPE"),
    )
    parser.add_argument(
        "--probe-fixture",
        help="offline JSON fixture for command, GPU, disk, and resume probes",
    )
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    probe = (
        FixtureProbe.from_json(args.probe_fixture)
        if args.probe_fixture
        else LiveProbe()
    )
    runs_root = args.runs_root or (
        args.out_root if args.platform in ("aws", "mit") else None
    )
    report = run_preflight(
        args.platform,
        bundle=args.bundle,
        profile=args.profile,
        probe=probe,
        source_root=args.source_root,
        data_root=args.data_root,
        out_root=args.out_root,
        runs_root=runs_root,
        capacity_type=args.capacity_type,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    print(rendered, end="")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
