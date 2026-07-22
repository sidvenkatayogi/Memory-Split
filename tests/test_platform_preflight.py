from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from dataclasses import replace
from pathlib import Path

from scripts.make_relational_manifest import (
    ROUTE_POLICY_SHA256,
    make_jobs,
    write_manifests,
)
from scripts.package_relational_run import package_run


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_MODULE = REPO_ROOT / "scripts" / "platform_preflight.py"
GIB = 1 << 30
POLICY = {
    "schema_version": 1,
    "policy": {
        "write_cost": 1.0,
        "read_cost": 0.25,
        "hop_cost": 0.25,
    },
    "policy_sha256": ROUTE_POLICY_SHA256,
}
SMOKE_REPORT = {
    "shared_stream": True,
    "dense_steps": 2,
    "split_steps": 2,
    "resume_exact": True,
    "memory_modes": ["off", "on"],
    "pairs_complete": True,
}


def _preflight_module():
    assert PREFLIGHT_MODULE.is_file(), (
        f"missing Task 6 module: {PREFLIGHT_MODULE}"
    )
    spec = importlib.util.spec_from_file_location(
        "task6_platform_preflight",
        PREFLIGHT_MODULE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_bundle(tmp_path: Path) -> Path:
    inputs = tmp_path / "bundle-inputs"
    written = write_manifests(inputs)
    (inputs / "route-policy.json").write_text(
        json.dumps(POLICY, indent=2, sort_keys=True) + "\n"
    )
    (inputs / "smoke-report.json").write_text(
        json.dumps(SMOKE_REPORT, indent=2, sort_keys=True) + "\n"
    )
    return package_run(
        tmp_path / "relational-run.tar.gz",
        source_root=REPO_ROOT,
        input_root=inputs,
        config_inputs={
            scale: [
                path.relative_to(inputs).as_posix()
                for path in result["configs"]
            ]
            for scale, result in written.items()
        },
        manifest_inputs={
            scale: result["manifest"].relative_to(inputs).as_posix()
            for scale, result in written.items()
        },
        route_policy="route-policy.json",
        smoke_report="smoke-report.json",
    )


def _write_corpus(root: Path, job: dict) -> None:
    corpus = root / job["data_rel"]
    corpus.mkdir(parents=True, exist_ok=True)
    files = {
        "train.bin": b"tokens",
        "dense.weights.bin": b"dense",
        "split.weights.bin": b"split",
        "random.weights.bin": b"random",
        "route-policy.json": (
            json.dumps(POLICY, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "eval/route-audit.json": (
            json.dumps(
                {
                    "route_rate": 0.5,
                    "route_total": 100,
                    "low_use_high_entropy_external_rate": 0.9,
                    "low_use_high_entropy_total": 50,
                    "rules_top_centrality_internal_rate": 0.9,
                    "rules_top_centrality_total": 50,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        "report.json": (
            json.dumps(
                {
                    "config": {
                        "n_entities": job["n_entities"],
                        "total_tokens": job["total_tokens"],
                        "data_seed": job["data_seed"],
                    },
                    "checks": {"fixture": True},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    }
    for relative, data in files.items():
        path = corpus / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    artifacts = [
        {
            "path": relative,
            "sha256": _sha256(corpus / relative),
            "bytes": (corpus / relative).stat().st_size,
        }
        for relative in sorted(files)
    ]
    (corpus / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _stage_corpora(
    root: Path,
    scales: tuple[str, ...] = ("160m", "360m"),
) -> None:
    by_relative = {
        job["data_rel"]: job
        for scale in scales
        for job in make_jobs(scale)
    }
    for job in by_relative.values():
        _write_corpus(root, job)


def _write_aws_runs(root: Path, *, throughput: float = 61_000.0) -> None:
    jobs = make_jobs("360m")
    for job in jobs:
        run = root / job["out_rel"]
        run.mkdir(parents=True, exist_ok=True)
        rows = [
            {"step": step, "loss": 1.0, "tok_s": throughput}
            for step in range(20, 201, 20)
        ]
        (run / "log.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    status = {
        "schema_version": 1,
        "launcher": "aws",
        "dry_run": False,
        "planned": 6,
        "completed": 6,
        "failed": [],
        "exit_code": 0,
        "gpu_ids": [str(index) for index in range(8)],
        "step_limit": 200,
        "jobs": [
            {
                "config_rel": f"configs/360m/{job['run_id']}.yaml",
                "run_id": job["run_id"],
                "gpu": str(index),
                "runtime_config": str(
                    root / job["out_rel"] / "launch-config.yaml"
                ),
                "step_limit": 200,
                "returncode": 0,
                "peak_memory_mib": 70_000,
            }
            for index, job in enumerate(jobs)
        ],
    }
    (root / "aws-launch-status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )


def _local_probe(module, **changes):
    values = {
        "modules_ok": True,
        "commands": frozenset(),
        "gpus": (),
        "disk": module.DiskInfo(free_bytes=2_000_000_000_000, writable=True),
        "resume": module.ResumeInfo(
            steps=2,
            exact=True,
            next_loss_delta=0.0,
        ),
    }
    values.update(changes)
    return module.FixtureProbe(**values)


def _farm_probe(module, **changes):
    values = {
        "modules_ok": True,
        "commands": frozenset(
            {"nvidia-smi", "sbatch", "scontrol", "sinfo"}
        ),
        "gpus": (
            module.GPUInfo(
                name="NVIDIA L40S",
                total_memory_mib=46_068,
                free_memory_mib=45_000,
            ),
        ),
        "disk": module.DiskInfo(
            free_bytes=600_000_000_000,
            writable=True,
        ),
        "resume": module.ResumeInfo(
            steps=2,
            exact=True,
            next_loss_delta=0.0,
        ),
    }
    values.update(changes)
    return module.FixtureProbe(**values)


def _aws_probe(module, **changes):
    values = {
        "modules_ok": True,
        "commands": frozenset({"nvidia-smi"}),
        "gpus": tuple(
            module.GPUInfo(
                name="NVIDIA H100 80GB HBM3",
                total_memory_mib=81_559,
                free_memory_mib=80_000,
            )
            for _ in range(8)
        ),
        "disk": module.DiskInfo(
            free_bytes=1_100_000_000_000,
            writable=True,
        ),
        "resume": module.ResumeInfo(
            steps=200,
            exact=True,
            next_loss_delta=0.0,
        ),
    }
    values.update(changes)
    return module.FixtureProbe(**values)


def test_local_preflight_validates_a_real_task5_bundle(tmp_path):
    module = _preflight_module()
    bundle = _make_bundle(tmp_path)

    report = module.run_preflight(
        "local",
        bundle=bundle,
        probe=_local_probe(module),
    )

    assert report["ok"] is True
    assert report["checks"]["dependencies"]["passed"] is True
    assert report["checks"]["bundle_hashes"]["passed"] is True
    assert report["checks"]["smoke"]["passed"] is True
    assert report["bundle"]["run_counts"] == {"160m": 15, "360m": 6}


def test_local_preflight_fails_closed_on_dependencies(tmp_path):
    module = _preflight_module()
    report = module.run_preflight(
        "local",
        bundle=_make_bundle(tmp_path),
        probe=_local_probe(module, modules_ok=False),
    )

    assert report["ok"] is False
    assert report["checks"]["dependencies"]["passed"] is False


def test_local_preflight_fails_closed_on_bundle_hashes(tmp_path):
    module = _preflight_module()
    bundle = _make_bundle(tmp_path)
    corrupted = tmp_path / "corrupted.tar.gz"
    with (
        tarfile.open(bundle, "r:gz") as source,
        tarfile.open(corrupted, "w:gz") as destination,
    ):
        for member in source.getmembers():
            data = source.extractfile(member).read()
            if member.name.startswith("configs/160m/"):
                data += b"# corrupt\n"
            replacement = tarfile.TarInfo(member.name)
            replacement.size = len(data)
            replacement.mode = member.mode
            replacement.pax_headers = dict(member.pax_headers)
            destination.addfile(replacement, io.BytesIO(data))

    report = module.run_preflight(
        "local",
        bundle=corrupted,
        probe=_local_probe(module),
    )

    assert report["ok"] is False
    assert report["checks"]["bundle_hashes"]["passed"] is False


def test_farmshare_preflight_accepts_injected_l40s_fixture(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    _stage_corpora(data_root)

    report = module.run_preflight(
        "farmshare",
        bundle=_make_bundle(tmp_path),
        probe=_farm_probe(module),
        data_root=data_root,
        out_root=out_root,
    )

    assert report["ok"] is True
    assert report["checks"]["slurm"]["passed"] is True
    assert report["checks"]["gpu"]["passed"] is True
    assert report["checks"]["free_space"]["passed"] is True
    assert report["checks"]["resume"]["passed"] is True
    assert report["checks"]["corpus_hashes"]["passed"] is True


def test_farmshare_preflight_requires_only_its_160m_corpora(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    _stage_corpora(data_root, ("160m",))

    report = module.run_preflight(
        "farmshare",
        bundle=_make_bundle(tmp_path),
        probe=_farm_probe(module),
        data_root=data_root,
        out_root=out_root,
    )

    assert report["ok"] is True
    assert report["checks"]["corpus_hashes"]["detail"] == {"corpora": 6}


def test_farmshare_preflight_fails_closed_on_slurm(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    _stage_corpora(data_root)
    probe = _farm_probe(
        module,
        commands=frozenset({"nvidia-smi", "sbatch", "scontrol"}),
    )

    report = module.run_preflight(
        "farmshare",
        bundle=_make_bundle(tmp_path),
        probe=probe,
        data_root=data_root,
        out_root=out_root,
    )

    assert report["ok"] is False
    assert report["checks"]["slurm"]["passed"] is False


def test_farmshare_preflight_fails_closed_on_space_gpu_and_resume(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    _stage_corpora(data_root)
    probe = _farm_probe(
        module,
        gpus=(
            module.GPUInfo(
                name="NVIDIA A100",
                total_memory_mib=40_000,
                free_memory_mib=40_000,
            ),
        ),
        disk=module.DiskInfo(
            free_bytes=499_999_999_999,
            writable=True,
        ),
        resume=module.ResumeInfo(
            steps=2,
            exact=False,
            next_loss_delta=1.0,
        ),
    )

    report = module.run_preflight(
        "farmshare",
        bundle=_make_bundle(tmp_path),
        probe=probe,
        data_root=data_root,
        out_root=out_root,
    )

    assert report["ok"] is False
    assert report["checks"]["gpu"]["passed"] is False
    assert report["checks"]["free_space"]["passed"] is False
    assert report["checks"]["resume"]["passed"] is False


def test_farmshare_preflight_fails_closed_on_corpus_hashes(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    _stage_corpora(data_root)
    first = make_jobs("160m")[0]
    (data_root / first["data_rel"] / "train.bin").write_bytes(b"tampered")

    report = module.run_preflight(
        "farmshare",
        bundle=_make_bundle(tmp_path),
        probe=_farm_probe(module),
        data_root=data_root,
        out_root=out_root,
    )

    assert report["ok"] is False
    assert report["checks"]["corpus_hashes"]["passed"] is False


def test_farmshare_preflight_fails_closed_on_route_guardrails(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    _stage_corpora(data_root, ("160m",))
    first = make_jobs("160m")[0]
    corpus = data_root / first["data_rel"]
    audit_path = corpus / "eval" / "route-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["low_use_high_entropy_external_rate"] = 0.79
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["artifacts"]:
        if item["path"] == "eval/route-audit.json":
            item["bytes"] = audit_path.stat().st_size
            item["sha256"] = _sha256(audit_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    report = module.run_preflight(
        "farmshare",
        bundle=_make_bundle(tmp_path),
        probe=_farm_probe(module),
        data_root=data_root,
        out_root=out_root,
    )

    assert report["ok"] is False
    assert report["checks"]["corpus_hashes"]["passed"] is False


def test_aws_preflight_accepts_eight_h100_and_200_step_fixture(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    runs_root.mkdir()
    _stage_corpora(data_root)
    _write_aws_runs(runs_root)

    report = module.run_preflight(
        "aws",
        bundle=_make_bundle(tmp_path),
        probe=_aws_probe(module),
        data_root=data_root,
        out_root=runs_root,
        runs_root=runs_root,
        capacity_type="on-demand",
    )

    assert report["ok"] is True
    assert report["checks"]["gpu"]["passed"] is True
    assert report["checks"]["throughput"]["passed"] is True
    assert report["checks"]["peak_memory"]["passed"] is True
    assert report["checks"]["resume"]["passed"] is True
    assert report["checks"]["capacity"]["passed"] is True


def test_aws_preflight_requires_only_its_360m_corpora(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    runs_root.mkdir()
    _stage_corpora(data_root, ("360m",))
    _write_aws_runs(runs_root)

    report = module.run_preflight(
        "aws",
        bundle=_make_bundle(tmp_path),
        probe=_aws_probe(module),
        data_root=data_root,
        out_root=runs_root,
        runs_root=runs_root,
        capacity_type="on-demand",
    )

    assert report["ok"] is True
    assert report["checks"]["corpus_hashes"]["detail"] == {"corpora": 3}


def test_aws_preflight_fails_closed_on_gpu_count_model_and_memory(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    runs_root.mkdir()
    _stage_corpora(data_root)
    _write_aws_runs(runs_root)
    bad = (
        module.GPUInfo(
            name="NVIDIA A100",
            total_memory_mib=81_559,
            free_memory_mib=70_000,
        ),
    ) * 7

    report = module.run_preflight(
        "aws",
        bundle=_make_bundle(tmp_path),
        probe=_aws_probe(module, gpus=bad),
        data_root=data_root,
        out_root=runs_root,
        runs_root=runs_root,
        capacity_type="capacity-block",
    )

    assert report["ok"] is False
    assert report["checks"]["gpu"]["passed"] is False


def test_aws_preflight_fails_closed_on_throughput_peak_and_resume(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    runs_root.mkdir()
    _stage_corpora(data_root)
    _write_aws_runs(runs_root, throughput=59_999.0)
    status_path = runs_root / "aws-launch-status.json"
    status = json.loads(status_path.read_text())
    status["jobs"][0]["peak_memory_mib"] = 72 * 1024 + 1
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    resume = replace(
        _aws_probe(module).resume,
        next_loss_delta=1.1e-5,
    )

    report = module.run_preflight(
        "aws",
        bundle=_make_bundle(tmp_path),
        probe=_aws_probe(module, resume=resume),
        data_root=data_root,
        out_root=runs_root,
        runs_root=runs_root,
        capacity_type="on-demand",
    )

    assert report["ok"] is False
    assert report["checks"]["throughput"]["passed"] is False
    assert report["checks"]["peak_memory"]["passed"] is False
    assert report["checks"]["resume"]["passed"] is False


def test_aws_preflight_rejects_spot_or_missing_capacity_declaration(tmp_path):
    module = _preflight_module()
    data_root = tmp_path / "data"
    runs_root = tmp_path / "runs"
    data_root.mkdir()
    runs_root.mkdir()
    _stage_corpora(data_root)
    _write_aws_runs(runs_root)
    bundle = _make_bundle(tmp_path)

    for capacity in (None, "spot"):
        report = module.run_preflight(
            "aws",
            bundle=bundle,
            probe=_aws_probe(module),
            data_root=data_root,
            out_root=runs_root,
            runs_root=runs_root,
            capacity_type=capacity,
        )
        assert report["ok"] is False
        assert report["checks"]["capacity"]["passed"] is False
