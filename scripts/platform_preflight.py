#!/usr/bin/env python
"""Fail-closed local, FarmShare, and AWS launch preflight."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_relational_manifest import (  # noqa: E402
    ROUTE_POLICY_SHA256,
    make_jobs,
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
H100_USABLE_MEMORY_MIB = 72 * 1024
AWS_THROUGHPUT_MIN = 60_000.0
AWS_WARMUP_STEPS = 50
AWS_PROBE_STEPS = 200
RESUME_TOLERANCE = 1e-5
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
    "scripts/",
    "tests/",
    "train/",
)


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


def _resume_detail(
    probe,
    *,
    platform: str,
    runs_root: Path | str | None,
) -> dict:
    resume = probe.resume_info(platform=platform, runs_root=runs_root)
    expected_steps = (
        AWS_PROBE_STEPS if platform == "aws" else SMOKE_STEPS
    )
    if (
        resume.steps != expected_steps
        or resume.exact is not True
        or not math.isfinite(resume.next_loss_delta)
        or resume.next_loss_delta > RESUME_TOLERANCE
    ):
        raise ValueError(
            f"{expected_steps}-step checkpoint/resume is not exact within 1e-5"
        )
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


def run_preflight(
    platform: str,
    *,
    bundle: Path | str,
    probe=None,
    source_root: Path | str | None = None,
    data_root: Path | str | None = None,
    out_root: Path | str | None = None,
    runs_root: Path | str | None = None,
    capacity_type: str | None = None,
) -> dict:
    """Run every required check and return all failures without exceptions."""

    if platform not in ("local", "farmshare", "aws"):
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

    if platform in ("farmshare", "aws"):
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
                        ("160m",)
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

    report = {
        "schema_version": 1,
        "platform": platform,
        "ok": all(check["passed"] for check in checks.values()),
        "checks": checks,
        "bundle": bundle_evidence,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed relational launch preflight."
    )
    parser.add_argument(
        "--platform",
        required=True,
        choices=("local", "farmshare", "aws"),
    )
    parser.add_argument(
        "--bundle",
        default="artifacts/relational-run.tar.gz",
    )
    parser.add_argument(
        "--source-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
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
        args.out_root if args.platform == "aws" else None
    )
    report = run_preflight(
        args.platform,
        bundle=args.bundle,
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
