from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_MODULE = REPO_ROOT / "scripts" / "make_relational_manifest.py"
AWS_MODULE = REPO_ROOT / "cluster" / "aws" / "run_relational_manifest.py"
SEEDS = (0, 1, 2)
POLICY_SHA256 = (
    "0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058"
)


def _load_module(path: Path, name: str):
    assert path.is_file(), f"missing Task 6 module: {path}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_module():
    return _load_module(MANIFEST_MODULE, "task6_manifest")


def _aws_module():
    return _load_module(AWS_MODULE, "task6_aws_launcher")


def test_160m_manifest_is_the_exact_frozen_15_run_matrix():
    jobs = _manifest_module().make_jobs("160m")

    expected = {
        ("dense", load, seed)
        for load in ("n50k", "n800k")
        for seed in SEEDS
    }
    expected.update(
        ("split", load, seed)
        for load in ("n50k", "n800k")
        for seed in SEEDS
    )
    expected.update(("random", "n800k", seed) for seed in SEEDS)

    assert len(jobs) == 15
    assert {
        (job["condition"], job["load"], job["seed"]) for job in jobs
    } == expected
    assert all(job["model"] == "d160m" for job in jobs)
    assert all(job["total_tokens"] == 1_599_602_688 for job in jobs)


def test_360m_manifest_is_the_exact_frozen_6_run_matrix():
    jobs = _manifest_module().make_jobs("360m")

    assert len(jobs) == 6
    assert {
        (job["condition"], job["load"], job["seed"]) for job in jobs
    } == {
        (condition, "n1p8m", seed)
        for condition in ("dense", "split")
        for seed in SEEDS
    }
    assert all(job["model"] == "d360m" for job in jobs)
    assert all(job["total_tokens"] == 3_599_761_408 for job in jobs)


def test_configs_contain_only_relative_runtime_roots():
    jobs = _manifest_module().make_jobs("160m")
    jobs += _manifest_module().make_jobs("360m")

    for job in jobs:
        path_keys = {
            key
            for key in job
            if key.endswith(("_rel", "_dir", "_bin", "_path"))
        }
        assert path_keys == {"data_rel", "out_rel"}
        assert job["data_seed"] == 10_000 + job["seed"]
        assert job["data_rel"] == (
            f"{job['load']}_ds{job['data_seed']}"
        )
        assert job["out_rel"] == job["run_id"]
        assert job["route_policy_sha256"] == POLICY_SHA256
        rendered = yaml.safe_dump(job)
        assert "/Users/" not in rendered
        assert "/scratch/" not in rendered
        assert "s3://" not in rendered
        assert not Path(job["data_rel"]).is_absolute()
        assert not Path(job["out_rel"]).is_absolute()


def test_manifest_rendering_is_deterministic_and_relative(tmp_path):
    module = _manifest_module()

    first = module.write_manifests(tmp_path)
    first_bytes = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for result in first.values()
        for path in (result["manifest"], *result["configs"])
    }
    second = module.write_manifests(tmp_path)
    second_bytes = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for result in second.values()
        for path in (result["manifest"], *result["configs"])
    }

    assert first_bytes == second_bytes
    assert len(first["160m"]["configs"]) == 15
    assert len(first["360m"]["configs"]) == 6
    for scale, count in (("160m", 15), ("360m", 6)):
        manifest = first[scale]["manifest"]
        rows = manifest.read_text().splitlines()
        assert len(rows) == count
        assert all(
            row.startswith(f"configs/{scale}/")
            and row.endswith(".yaml")
            and not Path(row).is_absolute()
            and ".." not in Path(row).parts
            for row in rows
        )


def test_checked_in_manifests_match_generator_byte_for_byte(tmp_path):
    generated = _manifest_module().write_manifests(tmp_path)

    for result in generated.values():
        for generated_path in (result["manifest"], *result["configs"]):
            relative = generated_path.relative_to(tmp_path)
            committed = REPO_ROOT / relative
            assert committed.is_file(), f"missing production file: {relative}"
            assert committed.read_bytes() == generated_path.read_bytes()


def test_launch_resolution_uses_only_environment_roots(tmp_path):
    module = _manifest_module()
    job = module.make_jobs("160m")[0]
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"

    resolved = module.resolve_job(
        job,
        data_root=data_root,
        out_root=out_root,
    )

    data_dir = data_root / job["data_rel"]
    assert resolved["data_dir"] == str(data_dir)
    assert resolved["train_bin"] == str(data_dir / "train.bin")
    assert resolved["train_weights"] == str(
        data_dir / f"{job['condition']}.weights.bin"
    )
    assert resolved["out_dir"] == str(out_root / job["out_rel"])
    assert resolved["data_rel"] == job["data_rel"]
    assert resolved["out_rel"] == job["out_rel"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_rel", "../escape"),
        ("data_rel", "/absolute"),
        ("out_rel", "nested/../../escape"),
        ("out_rel", r"windows\escape"),
    ],
)
def test_launch_resolution_rejects_nonportable_relative_paths(
    tmp_path,
    field,
    value,
):
    module = _manifest_module()
    job = dict(module.make_jobs("160m")[0], **{field: value})

    with pytest.raises(ValueError, match="portable|traversal"):
        module.resolve_job(
            job,
            data_root=tmp_path / "data",
            out_root=tmp_path / "out",
        )


def test_launch_resolution_rejects_existing_symlink_escape(tmp_path):
    module = _manifest_module()
    job = module.make_jobs("160m")[0]
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    outside = tmp_path / "outside"
    data_root.mkdir()
    out_root.mkdir()
    outside.mkdir()
    (out_root / job["out_rel"]).symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="root|symlink|escape"):
        module.resolve_job(
            job,
            data_root=data_root,
            out_root=out_root,
        )


def test_runtime_config_refuses_mismatched_existing_checkpoint(tmp_path):
    module = _manifest_module()
    job = module.make_jobs("360m")[0]
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    resolved = module.resolve_job(
        job,
        data_root=data_root,
        out_root=out_root,
        max_steps=200,
    )
    run_dir = Path(resolved["out_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "ckpt.pt").write_bytes(b"checkpoint")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(resolved, seed=99), sort_keys=False)
    )

    with pytest.raises(ValueError, match="resume|config"):
        module.write_runtime_config(
            job,
            run_dir / "launch-config.yaml",
            data_root=data_root,
            out_root=out_root,
            max_steps=200,
        )


def test_runtime_config_allows_200_step_checkpoint_to_resume_full_run(tmp_path):
    module = _manifest_module()
    job = module.make_jobs("360m")[0]
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    probe = module.resolve_job(
        job,
        data_root=data_root,
        out_root=out_root,
        max_steps=200,
    )
    run_dir = Path(probe["out_dir"])
    run_dir.mkdir(parents=True)
    (run_dir / "ckpt.pt").write_bytes(b"checkpoint")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(probe, sort_keys=False)
    )

    path = module.write_runtime_config(
        job,
        run_dir / "launch-config.yaml",
        data_root=data_root,
        out_root=out_root,
    )

    resumed = yaml.safe_load(path.read_text())
    assert "max_steps" not in resumed
    assert resumed["out_dir"] == str(run_dir)


def test_farmshare_submission_is_dry_run_unless_explicit(tmp_path):
    module = _manifest_module()
    written = module.write_manifests(tmp_path)
    calls = []

    plan = module.submit_farmshare(
        written["160m"]["manifest"],
        execute=False,
        run_command=lambda command: calls.append(command),
    )

    assert calls == []
    assert len(plan) == 15
    assert plan[0][0] == "sbatch"
    assert plan[0][1].startswith("--export=ALL,CONFIG_REL=configs/160m/")
    assert plan[0][-1] == "cluster/slurm/relational_train.sbatch"


def test_farmshare_execute_submits_one_job_per_config(tmp_path):
    module = _manifest_module()
    written = module.write_manifests(tmp_path)
    calls = []

    module.submit_farmshare(
        written["160m"]["manifest"],
        execute=True,
        run_command=lambda command: calls.append(command) or 0,
    )

    assert len(calls) == 15
    assert len({call[1] for call in calls}) == 15


def test_farmshare_submitter_attempts_all_after_injected_failures(tmp_path):
    module = _manifest_module()
    written = module.write_manifests(tmp_path)
    calls = []
    failure_codes = {1: 17, 12: 23}

    def fake_submit(command):
        index = len(calls)
        calls.append(command)
        return failure_codes.get(index, 0)

    with pytest.raises(module.FarmshareSubmissionError) as caught:
        module.submit_farmshare(
            written["160m"]["manifest"],
            execute=True,
            run_command=fake_submit,
        )

    assert len(calls) == 15
    assert calls[-1][1].endswith("d160m_random_n800k_s2.yaml")
    assert [
        (failure["config_rel"], failure["returncode"])
        for failure in caught.value.failures
    ] == [
        (
            calls[index][1].split("CONFIG_REL=", 1)[1],
            returncode,
        )
        for index, returncode in failure_codes.items()
    ]


def test_aws_dry_run_does_not_start_processes_or_write_outputs(tmp_path):
    manifest_module = _manifest_module()
    launcher = _aws_module()
    written = manifest_module.write_manifests(tmp_path)
    calls = []
    out_root = tmp_path / "runs"

    result = launcher.run_manifest(
        written["360m"]["manifest"],
        data_root=tmp_path / "data",
        out_root=out_root,
        gpu_ids=tuple(range(8)),
        steps=200,
        execute=False,
        run_command=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert result["dry_run"] is True
    assert result["planned"] == 6
    assert calls == []
    assert not out_root.exists()


def test_aws_failure_is_recorded_after_all_independent_jobs_run(tmp_path):
    manifest_module = _manifest_module()
    launcher = _aws_module()
    written = manifest_module.write_manifests(tmp_path)
    calls = []

    def fake_run(command, *, env, log_path):
        calls.append((command, env["CUDA_VISIBLE_DEVICES"], log_path))
        return 7 if len(calls) == 1 else 0

    result = launcher.run_manifest(
        written["360m"]["manifest"],
        data_root=tmp_path / "data",
        out_root=tmp_path / "runs",
        gpu_ids=tuple(range(8)),
        steps=200,
        execute=True,
        run_command=fake_run,
        memory_probe=lambda gpu: 1024,
    )

    assert len(calls) == 6
    assert result["planned"] == 6
    assert result["completed"] == 5
    assert len(result["failed"]) == 1
    assert result["exit_code"] == 1
    assert (tmp_path / "runs" / "aws-launch-status.json").is_file()


def test_aws_setup_failure_is_recorded_without_dropping_other_jobs(
    tmp_path,
    monkeypatch,
):
    manifest_module = _manifest_module()
    launcher = _aws_module()
    written = manifest_module.write_manifests(tmp_path)
    real_write = launcher.write_runtime_config
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(args[0]["run_id"])
        if len(attempts) == 1:
            raise ValueError("fixture setup failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(launcher, "write_runtime_config", fail_once)
    result = launcher.run_manifest(
        written["360m"]["manifest"],
        data_root=tmp_path / "data",
        out_root=tmp_path / "runs",
        gpu_ids=tuple(range(8)),
        steps=200,
        execute=True,
        run_command=lambda *args, **kwargs: 0,
        memory_probe=lambda gpu: 1024,
    )

    assert len(attempts) == 6
    assert result["planned"] == 6
    assert result["completed"] == 5
    assert len(result["failed"]) == 1
    assert len(result["jobs"]) == 6
    assert result["exit_code"] == 1
