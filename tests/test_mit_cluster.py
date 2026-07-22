from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts.make_relational_manifest import make_jobs, resolve_job


REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "cluster" / "mit" / "profile.example.json"
SCHEMA_PATH = REPO_ROOT / "schemas" / "mit-cluster-profile-v1.schema.json"
MANIFEST_PATH = REPO_ROOT / "configs" / "360m.tsv"
SLURM_SCRIPT = REPO_ROOT / "cluster" / "slurm" / "relational_mit_train.sbatch"
TASK_FILES = {
    "cluster/mit/profile.py",
    "cluster/mit/probe_cluster.py",
    "cluster/mit/run_relational_manifest.py",
    "cluster/mit/profile.example.json",
    "cluster/slurm/relational_mit_train.sbatch",
    "schemas/mit-cluster-profile-v1.schema.json",
}
VALID_PROFILE = {
    "schema_version": 1,
    "partition": "mit-gpu",
    "account": "research",
    "qos": "normal",
    "gres": "gpu:h100:1",
    "gpu_name_regex": "H100|H200|L40S|A100",
    "cpus": 16,
    "memory_gb": 96,
    "wall_minutes": 360,
    "python": "${RELATIONAL_VENV}/bin/python",
}


def _import(name: str):
    importlib.invalidate_caches()
    return importlib.import_module(name)


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_profile(tmp_path: Path, value: dict | None = None) -> Path:
    path = tmp_path / "mit-profile.json"
    path.write_text(
        json.dumps(value if value is not None else VALID_PROFILE, indent=2) + "\n"
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _platform_test_helpers():
    return _load_file(
        "mit_platform_test_helpers",
        REPO_ROOT / "tests" / "test_platform_preflight.py",
    )


def _preflight_module():
    return _load_file(
        "mit_platform_preflight",
        REPO_ROOT / "scripts" / "platform_preflight.py",
    )


def test_required_task_files_exist():
    missing = sorted(
        relative for relative in TASK_FILES if not (REPO_ROOT / relative).is_file()
    )
    assert missing == []


def test_discovery_uses_only_bounded_read_only_slurm_commands(tmp_path):
    probe = _import("cluster.mit.probe_cluster")
    calls = []
    responses = {
        ("sinfo", "--version"): SimpleNamespace(
            returncode=0,
            stdout="slurm 23.11.7\n",
            stderr="",
        ),
        ("sinfo", "-h", "-o", "%P|%G|%l|%D|%m"): SimpleNamespace(
            returncode=0,
            stdout=(
                "mit-gpu*|gpu:h100:4|06:00:00|3|512000\n"
                "mit-long|gpu:a100:8|2-00:00:00|2|256000\n"
            ),
            stderr="",
        ),
        ("scontrol", "show", "config"): SimpleNamespace(
            returncode=0,
            stdout="ClusterName=student\nSlurmctldTimeout=120\n",
            stderr="",
        ),
    }

    def runner(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return responses[tuple(command)]

    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"frozen bundle")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = probe.collect_probe(
        command_runner=runner,
        environ={"SCRATCH": str(scratch), "MODULESHOME": "/opt/modules"},
        which=lambda command: {
            "module": "/usr/bin/module",
            "python3": "/usr/bin/python3",
            "python": None,
        }.get(command),
        disk_usage=lambda path: SimpleNamespace(
            total=1_000,
            used=250,
            free=750,
        ),
        path_exists=lambda path: Path(path) == scratch,
        is_executable=lambda path: path in {"/usr/bin/python3", sys.executable},
        git_revision_reader=lambda root: "a" * 40,
        source_root=REPO_ROOT,
        bundle=bundle,
    )

    assert [command for command, _ in calls] == list(probe.COMMANDS)
    for _, kwargs in calls:
        assert kwargs == {
            "capture_output": True,
            "text": True,
            "check": False,
            "timeout": probe.COMMAND_TIMEOUT_SECONDS,
        }
    assert result["slurm"]["version"] == "slurm 23.11.7"
    assert result["slurm"]["partitions"] == [
        {
            "partition": "mit-gpu",
            "default": True,
            "gres": "gpu:h100:4",
            "time_limit": "06:00:00",
            "nodes": 3,
            "memory_mb": 512000,
        },
        {
            "partition": "mit-long",
            "default": False,
            "gres": "gpu:a100:8",
            "time_limit": "2-00:00:00",
            "nodes": 2,
            "memory_mb": 256000,
        },
    ]
    assert result["module"]["command"] == "/usr/bin/module"
    assert result["python_executables"] == [
        sys.executable,
        "/usr/bin/python3",
    ]
    assert result["scratch_candidates"] == [
        {
            "environment": "SCRATCH",
            "path": str(scratch),
            "free_bytes": 750,
            "total_bytes": 1000,
        }
    ]
    assert result["filesystem"] == {
        "path": str(REPO_ROOT),
        "free_bytes": 750,
        "total_bytes": 1000,
    }
    assert result["source_revision"] == "a" * 40
    assert result["bundle"] == {
        "path": str(bundle),
        "bytes": len(b"frozen bundle"),
        "sha256": hashlib.sha256(b"frozen bundle").hexdigest(),
    }


def test_discovery_attempts_every_command_and_records_bounded_failures(tmp_path):
    probe = _import("cluster.mit.probe_cluster")
    calls = []

    def runner(command, **kwargs):
        calls.append(tuple(command))
        if tuple(command) == probe.COMMANDS[0]:
            raise TimeoutError("fixture timeout")
        return SimpleNamespace(returncode=1, stdout="x" * 2_000_000, stderr="denied")

    result = probe.collect_probe(
        command_runner=runner,
        environ={},
        which=lambda command: None,
        disk_usage=lambda path: None,
        path_exists=lambda path: False,
        is_executable=lambda path: False,
        git_revision_reader=lambda root: None,
        source_root=tmp_path,
        bundle=None,
    )

    assert calls == list(probe.COMMANDS)
    assert result["commands"]["slurm_version"]["ok"] is False
    assert "TimeoutError" in result["commands"]["slurm_version"]["error"]
    assert result["commands"]["partitions"]["ok"] is False
    assert result["commands"]["config"]["ok"] is False
    assert len(result["commands"]["partitions"]["stdout"]) == probe.MAX_OUTPUT_CHARS


def test_probe_json_is_deterministic(tmp_path):
    probe = _import("cluster.mit.probe_cluster")
    output = tmp_path / "probe.json"
    evidence = {"z": 1, "a": {"b": True}}

    probe.write_probe(output, evidence)

    assert output.read_text() == (
        '{\n  "a": {\n    "b": true\n  },\n  "z": 1\n}\n'
    )


def test_profile_schema_is_closed_and_requires_resource_contract():
    schema = json.loads(SCHEMA_PATH.read_text())

    assert schema["$id"].endswith("mit-cluster-profile-v1.schema.json")
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "partition",
        "gres",
        "gpu_name_regex",
        "cpus",
        "memory_gb",
        "wall_minutes",
        "python",
    }
    assert set(schema["properties"]) == set(schema["required"]) | {"account", "qos"}
    assert schema["properties"]["schema_version"] == {"const": 1}


def test_profile_loader_accepts_example_and_returns_byte_hash():
    profile_module = _import("cluster.mit.profile")

    loaded = profile_module.load_profile(PROFILE_PATH)

    assert loaded.partition
    assert loaded.gres.endswith(":1")
    assert loaded.python == "${RELATIONAL_VENV}/bin/python"
    assert loaded.sha256 == _sha256(PROFILE_PATH)
    assert loaded.as_dict() == json.loads(PROFILE_PATH.read_text())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("partition", "gpu;touch-pwned"),
        ("partition", "gpu\nother"),
        ("account", "../../account"),
        ("qos", "normal && false"),
        ("gres", "gpu:1 --wrap=whoami"),
        ("gres", "gpu:h100:2"),
        ("gpu_name_regex", "H100;touch /tmp/pwned"),
        ("gpu_name_regex", "$(whoami)"),
        ("gpu_name_regex", "H100\nA100"),
        ("gpu_name_regex", "../H100"),
        ("python", "/opt/venv/bin/python"),
        ("python", "${DATA_ROOT}/bin/python"),
        ("python", "${OUT_ROOT}/bin/python"),
        ("python", "../venv/bin/python"),
        ("python", "${RELATIONAL_VENV}/bin/python;whoami"),
    ],
)
def test_profile_loader_rejects_shell_paths_roots_and_non_one_gpu(
    tmp_path,
    field,
    value,
):
    profile_module = _import("cluster.mit.profile")
    raw = dict(VALID_PROFILE)
    raw[field] = value

    with pytest.raises(ValueError):
        profile_module.load_profile(_write_profile(tmp_path, raw))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update(extra="unknown"),
        lambda raw: raw.pop("cpus"),
        lambda raw: raw.update(schema_version=2),
        lambda raw: raw.update(cpus=True),
        lambda raw: raw.update(cpus=0),
        lambda raw: raw.update(memory_gb=-1),
        lambda raw: raw.update(wall_minutes=0),
        lambda raw: raw.update(account=3),
        lambda raw: raw.update(qos=""),
    ],
)
def test_profile_loader_rejects_unknown_missing_or_invalid_fields(
    tmp_path,
    mutation,
):
    profile_module = _import("cluster.mit.profile")
    raw = dict(VALID_PROFILE)
    mutation(raw)

    with pytest.raises(ValueError):
        profile_module.load_profile(_write_profile(tmp_path, raw))


def test_launcher_renders_deterministic_one_gpu_sbatch_argv(tmp_path):
    launcher = _import("cluster.mit.run_relational_manifest")
    profile_path = _write_profile(tmp_path)
    bundle = tmp_path / "relational-run.tar.gz"
    bundle.write_bytes(b"bundle bytes")
    before = {
        path: path.read_bytes()
        for path in sorted((REPO_ROOT / "configs" / "360m").glob("*.yaml"))
    }

    first = launcher.plan_manifest(
        MANIFEST_PATH,
        profile=profile_path,
        bundle=bundle,
        steps=200,
    )
    second = launcher.plan_manifest(
        MANIFEST_PATH,
        profile=profile_path,
        bundle=bundle,
        steps=200,
    )

    assert first == second
    assert len(first) == 6
    export = (
        "ALL,"
        "CONFIG_REL=configs/360m/d360m_dense_n1p8m_s0.yaml,"
        "GPU_NAME_REGEX=H100|H200|L40S|A100,"
        f"PROFILE_SHA256={_sha256(profile_path)},"
        f"BUNDLE_SHA256={_sha256(bundle)},"
        "PROFILE_PYTHON=${RELATIONAL_VENV}/bin/python,"
        "STEP_LIMIT=200"
    )
    assert first[0] == [
        "sbatch",
        "--partition=mit-gpu",
        "--account=research",
        "--qos=normal",
        "--gres=gpu:h100:1",
        "--cpus-per-task=16",
        "--mem=96G",
        "--time=06:00:00",
        "--job-name=rel-d360m_dense_n1p8m_s0",
        f"--export={export}",
        "cluster/slurm/relational_mit_train.sbatch",
    ]
    assert all(command[0] == "sbatch" for command in first)
    assert all("--gres=gpu:h100:1" in command for command in first)
    assert before == {path: path.read_bytes() for path in before}


def test_launcher_defaults_to_dry_run_without_calling_sbatch(tmp_path):
    launcher = _import("cluster.mit.run_relational_manifest")
    profile = _write_profile(tmp_path)
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    calls = []

    report = launcher.run_manifest(
        MANIFEST_PATH,
        profile=profile,
        bundle=bundle,
        run_command=lambda command: calls.append(command),
    )

    assert calls == []
    assert report["dry_run"] is True
    assert report["planned"] == 6
    assert report["exit_code"] == 0
    assert len(report["commands"]) == 6


def test_execute_attempts_all_six_and_aggregates_returncodes_and_exceptions(
    tmp_path,
):
    launcher = _import("cluster.mit.run_relational_manifest")
    profile = _write_profile(tmp_path)
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    calls = []

    def runner(command):
        index = len(calls)
        calls.append(list(command))
        if index == 1:
            raise OSError("fixture sbatch failure")
        return SimpleNamespace(
            returncode=7 if index == 4 else 0,
            stdout=f"submitted {index}",
            stderr="rejected" if index == 4 else "",
        )

    report = launcher.run_manifest(
        MANIFEST_PATH,
        profile=profile,
        bundle=bundle,
        steps=200,
        execute=True,
        run_command=runner,
    )

    assert len(calls) == 6
    assert report["dry_run"] is False
    assert report["attempted"] == 6
    assert report["submitted"] == 4
    assert report["exit_code"] == 1
    assert [failure["index"] for failure in report["failures"]] == [1, 4]
    assert "OSError" in report["failures"][0]["error"]
    assert report["failures"][1]["returncode"] == 7


def test_default_sbatch_runner_never_uses_a_shell(monkeypatch):
    launcher = _import("cluster.mit.run_relational_manifest")
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    launcher._subprocess_command(["sbatch", "--version"])

    assert observed["command"] == ["sbatch", "--version"]
    assert observed["kwargs"] == {
        "capture_output": True,
        "text": True,
        "check": False,
    }


def test_generic_slurm_script_has_no_fixed_cluster_resources_or_gpu_model():
    text = SLURM_SCRIPT.read_text()

    for forbidden in (
        "#SBATCH --partition=",
        "#SBATCH --qos=",
        "#SBATCH --account=",
        "#SBATCH --gres=",
        "H100",
        "A100",
        "L40S",
        "/Users/",
        "/scratch/",
    ):
        assert forbidden not in text
    for required in (
        "DATA_ROOT",
        "OUT_ROOT",
        "RELATIONAL_VENV",
        "GPU_NAME_REGEX",
        "nvidia-smi",
        "mit-job-evidence.json",
        "scripts/make_relational_manifest.py",
        "scripts/run_train.py",
        "--resume",
        "auto",
    ):
        assert required in text


def test_slurm_script_reaps_training_before_finishing_memory_monitor():
    text = SLURM_SCRIPT.read_text()

    assert "monitor_peak_memory" in text
    assert 'wait "$train_pid"' in text
    assert 'wait "$monitor_pid"' in text
    assert text.index('wait "$train_pid"') < text.index('wait "$monitor_pid"')


def _write_mit_runs(
    out_root: Path,
    *,
    data_root: Path,
    profile_sha256: str,
    bundle_sha256: str,
    throughput: float = 25_000.0,
) -> None:
    rows = MANIFEST_PATH.read_text().splitlines()
    jobs = make_jobs("360m")
    assert len(rows) == len(jobs) == 6
    runtime_root = out_root / ".launch-configs"
    runtime_root.mkdir()
    for index, (relative, job) in enumerate(zip(rows, jobs)):
        run = out_root / job["out_rel"]
        run.mkdir(parents=True)
        runtime = resolve_job(
            job,
            data_root=data_root,
            out_root=out_root,
            max_steps=200,
        )
        runtime_config = runtime_root / f"{job['run_id']}-{1000 + index}.yaml"
        runtime_config.write_text(yaml.safe_dump(runtime, sort_keys=False))
        (run / "config.yaml").write_text(yaml.safe_dump(runtime, sort_keys=False))
        (run / "ckpt.pt").write_bytes(b"checkpoint")
        (run / "log.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "step": step,
                        "loss": 1.0,
                        "tok_s": throughput,
                    }
                )
                + "\n"
                for step in range(20, 201, 20)
            )
        )
        config_path = REPO_ROOT / relative
        evidence = {
            "schema_version": 1,
            "platform": "mit",
            "status": "completed",
            "config_rel": relative,
            "config_sha256": _sha256(config_path),
            "run_id": job["run_id"],
            "profile_sha256": profile_sha256,
            "bundle_sha256": bundle_sha256,
            "slurm": {
                "job_id": str(1000 + index),
                "version": "slurm 23.11.7",
            },
            "gpu": {
                "count": 1,
                "name": "NVIDIA H100 80GB HBM3",
                "total_memory_mib": 81559,
                "free_memory_mib": 80000,
            },
            "max_steps": 200,
            "steps_completed": 200,
            "returncode": 0,
            "oom_detected": False,
            "checkpoint_present": True,
            "peak_memory_mib": 70000,
            "runtime_config": str(runtime_config),
        }
        (run / "mit-job-evidence.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n"
        )


def _mit_probe(module, **changes):
    values = {
        "modules_ok": True,
        "commands": frozenset({"nvidia-smi", "sbatch", "scontrol", "sinfo"}),
        "gpus": (
            module.GPUInfo(
                name="NVIDIA H100 80GB HBM3",
                total_memory_mib=81559,
                free_memory_mib=80000,
            ),
        ),
        "disk": module.DiskInfo(
            free_bytes=1_000_000_000_000,
            writable=True,
        ),
        "resume": module.ResumeInfo(
            steps=200,
            exact=False,
            next_loss_delta=1e-7,
        ),
    }
    values.update(changes)
    return module.FixtureProbe(**values)


def _valid_mit_preflight(tmp_path):
    helpers = _platform_test_helpers()
    module = _preflight_module()
    profile = _write_profile(tmp_path)
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    data_root.mkdir()
    out_root.mkdir()
    helpers._stage_corpora(data_root, ("360m",))
    bundle = helpers._make_bundle(tmp_path)
    _write_mit_runs(
        out_root,
        data_root=data_root,
        profile_sha256=_sha256(profile),
        bundle_sha256=_sha256(bundle),
    )
    return module, profile, bundle, data_root, out_root


def test_mit_preflight_accepts_profile_hashes_gpu_and_complete_probe(tmp_path):
    module, profile, bundle, data_root, out_root = _valid_mit_preflight(tmp_path)

    report = module.run_preflight(
        "mit",
        bundle=bundle,
        profile=profile,
        probe=_mit_probe(module),
        data_root=data_root,
        out_root=out_root,
        runs_root=out_root,
    )

    assert report["ok"] is True
    assert report["checks"]["profile"]["passed"] is True
    assert report["checks"]["slurm"]["passed"] is True
    assert report["checks"]["gpu"]["passed"] is True
    assert report["checks"]["bundle_hashes"]["passed"] is True
    assert report["checks"]["corpus_hashes"]["passed"] is True
    assert report["checks"]["mit_runs"]["passed"] is True
    assert report["checks"]["resume"]["passed"] is True
    throughput = report["checks"]["throughput"]["detail"]
    assert throughput["mean_raw_tokens_per_second_per_gpu"] == 25_000.0
    assert throughput["threshold"] is None
    assert set(throughput["runs"]) == {job["run_id"] for job in make_jobs("360m")}
    expected_hours = make_jobs("360m")[0]["total_tokens"] / 25_000.0 / 3600.0
    first = throughput["runs"][make_jobs("360m")[0]["run_id"]]
    assert first["projected_full_hours"] == pytest.approx(expected_hours)
    assert first["projected_resubmissions"] == math.ceil(
        expected_hours * 60 / VALID_PROFILE["wall_minutes"]
    )


def test_mit_preflight_has_no_fixed_hardware_throughput_gate(tmp_path):
    module, profile, bundle, data_root, out_root = _valid_mit_preflight(tmp_path)
    for run in out_root.iterdir():
        log = run / "log.jsonl"
        if not log.is_file():
            continue
        records = [json.loads(line) for line in log.read_text().splitlines()]
        for record in records:
            record["tok_s"] = 1.0
        log.write_text("".join(json.dumps(row) + "\n" for row in records))

    report = module.run_preflight(
        "mit",
        bundle=bundle,
        profile=profile,
        probe=_mit_probe(module),
        data_root=data_root,
        out_root=out_root,
        runs_root=out_root,
    )

    assert report["ok"] is True
    assert report["checks"]["throughput"]["passed"] is True
    assert report["checks"]["throughput"]["detail"]["threshold"] is None


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("profile_hash", "mit_runs"),
        ("bundle_hash", "mit_runs"),
        ("config_hash", "mit_runs"),
        ("gpu_count", "mit_runs"),
        ("gpu_name", "mit_runs"),
        ("oom", "mit_runs"),
        ("steps", "mit_runs"),
        ("returncode", "mit_runs"),
        ("checkpoint", "mit_runs"),
        ("peak_memory", "mit_runs"),
        ("resume", "resume"),
    ],
)
def test_mit_preflight_fails_closed_on_incomplete_or_mismatched_evidence(
    tmp_path,
    mutation,
    failed_check,
):
    module, profile, bundle, data_root, out_root = _valid_mit_preflight(tmp_path)
    first_job = make_jobs("360m")[0]
    evidence_path = out_root / first_job["out_rel"] / "mit-job-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    probe = _mit_probe(module)
    if mutation == "profile_hash":
        evidence["profile_sha256"] = "0" * 64
    elif mutation == "bundle_hash":
        evidence["bundle_sha256"] = "0" * 64
    elif mutation == "config_hash":
        evidence["config_sha256"] = "0" * 64
    elif mutation == "gpu_count":
        evidence["gpu"]["count"] = 2
    elif mutation == "gpu_name":
        evidence["gpu"]["name"] = "Unsupported GPU"
    elif mutation == "oom":
        evidence["oom_detected"] = True
    elif mutation == "steps":
        evidence["steps_completed"] = 199
    elif mutation == "returncode":
        evidence["returncode"] = 1
    elif mutation == "checkpoint":
        (out_root / first_job["out_rel"] / "ckpt.pt").unlink()
    elif mutation == "peak_memory":
        evidence["peak_memory_mib"] = 0
    elif mutation == "resume":
        probe = _mit_probe(
            module,
            resume=module.ResumeInfo(
                steps=200,
                exact=False,
                next_loss_delta=1.0001e-5,
            ),
        )
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    report = module.run_preflight(
        "mit",
        bundle=bundle,
        profile=profile,
        probe=probe,
        data_root=data_root,
        out_root=out_root,
        runs_root=out_root,
    )

    assert report["ok"] is False
    assert report["checks"][failed_check]["passed"] is False


def test_mit_preflight_requires_one_current_profile_matching_gpu(tmp_path):
    module, profile, bundle, data_root, out_root = _valid_mit_preflight(tmp_path)
    bad_probe = _mit_probe(
        module,
        gpus=(
            module.GPUInfo(
                name="Unsupported GPU",
                total_memory_mib=81559,
                free_memory_mib=80000,
            ),
        ),
    )

    report = module.run_preflight(
        "mit",
        bundle=bundle,
        profile=profile,
        probe=bad_probe,
        data_root=data_root,
        out_root=out_root,
        runs_root=out_root,
    )

    assert report["ok"] is False
    assert report["checks"]["gpu"]["passed"] is False


def test_portable_bundle_contains_mit_launcher_profile_schema_and_script(tmp_path):
    helpers = _platform_test_helpers()
    bundle = helpers._make_bundle(tmp_path)

    with tarfile.open(bundle, "r:gz") as archive:
        members = set(archive.getnames())

    assert TASK_FILES <= members


def test_preflight_compares_bundled_schema_to_source_checkout():
    module = _preflight_module()

    assert "schemas/" in module._SOURCE_PREFIXES
