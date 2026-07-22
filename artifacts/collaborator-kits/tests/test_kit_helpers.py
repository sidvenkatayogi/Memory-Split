from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / "collaborator-kits" / "common"
KITS = ROOT / "collaborator-kits"
BUNDLE = ROOT / "relational-run.tar.gz"
SOURCE_ARCHIVE = ROOT / "relational-source-0045398.tar.gz"
sys.path.insert(0, str(COMMON))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, COMMON / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


launcher = load_module("launch_assignment", "launch_assignment.py")
builder = load_module("build_owned_corpora", "build_owned_corpora.py")
verifier = load_module("verify_assignment", "verify_assignment.py")
mit_authorizer = load_module(
    "authorize_mit_full_run",
    "authorize_mit_full_run.py",
)


def assignment_path(label: str) -> Path:
    return KITS / label / "assignment.json"


@pytest.mark.parametrize(
    "label",
    [
        "farmshare-collaborator-seed1",
        "farmshare-collaborator-seed2",
        "mit-collaborator-a",
        "mit-collaborator-b",
    ],
)
def test_assignment_contract_is_exact(label, tmp_path):
    original = json.loads(assignment_path(label).read_text())
    assert launcher.load_assignment(assignment_path(label)) == original

    for mutation in (
        lambda value: value["configs"].pop(),
        lambda value: value.__setitem__("bed_owner", not value["bed_owner"]),
        lambda value: value["corpora_owned"].clear(),
    ):
        changed = json.loads(json.dumps(original))
        mutation(changed)
        path = tmp_path / "assignment.json"
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="frozen assignment"):
            launcher.load_assignment(path)


@pytest.mark.parametrize("steps", [1, 199, 201, 6866])
def test_mit_steps_cannot_bypass_probe_authorization(steps):
    with pytest.raises(ValueError, match="exactly 200"):
        launcher.validate_mit_steps(steps)


def test_mit_probe_and_full_step_modes_are_the_only_modes():
    assert launcher.validate_mit_steps(200) == "probe"
    assert launcher.validate_mit_steps(None) == "full"


def test_mit_import_context_cleans_namespace_packages(tmp_path):
    import tarfile

    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE) as archive:
        archive.extractall(source, filter="data")
    assignment = launcher.load_assignment(
        assignment_path("mit-collaborator-b")
    )
    profile = source / "cluster" / "mit" / "profile.example.json"
    for _ in range(2):
        commands = launcher.mit_commands(
            source,
            assignment["configs"],
            profile=profile,
            bundle=BUNDLE,
            steps=200,
        )
        assert len(commands) == 3


def test_mit_import_cleanup_is_hash_order_independent(tmp_path):
    import tarfile

    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE) as archive:
        archive.extractall(source, filter="data")
    for seed in range(16):
        completed = subprocess.run(
            [
                sys.executable,
                str(COMMON / "launch_assignment.py"),
                "--assignment",
                str(assignment_path("mit-collaborator-b")),
                "--source-root",
                str(source),
                "--bundle",
                str(BUNDLE),
                "--profile",
                str(source / "cluster" / "mit" / "profile.example.json"),
                "--steps",
                "200",
            ],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (seed, completed.stderr)


def test_source_import_cleanup_handles_dynamic_module_paths(tmp_path):
    import tarfile

    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE) as archive:
        archive.extractall(source, filter="data")
    with launcher.source_import_path(source):
        from scripts.platform_preflight import verify_bundle

        assert callable(verify_bundle)


def test_direct_source_verifier_rejects_tampering(tmp_path):
    import tarfile

    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE) as archive:
        archive.extractall(source, filter="data")
    launcher.verify_extracted_source(source, BUNDLE)
    target = source / "scripts" / "run_train.py"
    target.write_text(target.read_text() + "\n# tampered\n")
    with pytest.raises(ValueError, match="source member differs"):
        launcher.verify_extracted_source(source, BUNDLE)


def test_direct_source_verifier_rejects_unindexed_import_code(tmp_path):
    import tarfile

    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE) as archive:
        archive.extractall(source, filter="data")
    (source / "scripts" / "__init__.py").write_text(
        "raise RuntimeError('unindexed code executed')\n"
    )
    with pytest.raises(ValueError, match="unindexed source file"):
        launcher.verify_extracted_source(source, BUNDLE)


def test_coordinator_source_archive_is_closed(tmp_path):
    import tarfile

    with tarfile.open(SOURCE_ARCHIVE) as archive:
        archive.extractall(tmp_path, filter="data")
    source = tmp_path / "relational-core"
    (source / "source-revision.txt").write_text(
        launcher.SOURCE_REVISION + "\n"
    )
    farm_authorizer = load_module(
        "authorize_farmshare_gate_test",
        "authorize_farmshare_gate.py",
    )
    farm_authorizer.verify_source_archive(source, SOURCE_ARCHIVE)
    (source / "scripts" / "__init__.py").write_text("raise RuntimeError\n")
    with pytest.raises(ValueError, match="unindexed coordinator source"):
        farm_authorizer.verify_source_archive(source, SOURCE_ARCHIVE)


def test_farmshare_requires_exact_portable_bundle_bytes(tmp_path):
    farm_authorizer = load_module(
        "authorize_farmshare_gate_bundle_test",
        "authorize_farmshare_gate.py",
    )
    farm_authorizer.verify_exact_bundle(BUNDLE)
    changed = tmp_path / "relational-run.tar.gz"
    changed.write_bytes(BUNDLE.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="exact frozen bytes"):
        farm_authorizer.verify_exact_bundle(changed)


def test_authorization_binds_roots_and_scope(tmp_path):
    assignment = launcher.load_assignment(
        assignment_path("mit-collaborator-a")
    )
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    data_root = tmp_path / "data"
    out_root = tmp_path / "out"
    evidence = tmp_path / "report.json"
    evidence.write_text("{}\n")
    evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
    expected = launcher.authorization_document(
        assignment=assignment,
        kind="mit-six-probe-preflight",
        profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        data_root=data_root,
        out_root=out_root,
        evidence_sha256=evidence_sha256,
    )
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(expected))
    launcher.verify_authorization(
        path,
        evidence,
        assignment=assignment,
        kind="mit-six-probe-preflight",
        profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        data_root=data_root,
        out_root=out_root,
    )
    changed = dict(expected)
    changed["out_root"] = str(tmp_path / "other")
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="does not match"):
        launcher.verify_authorization(
            path,
            evidence,
            assignment=assignment,
            kind="mit-six-probe-preflight",
            profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
            data_root=data_root,
            out_root=out_root,
        )


def test_fineweb_staging_is_dry_run_by_default(tmp_path):
    output = tmp_path / "bed.jsonl"
    cache = tmp_path / "cache"
    completed = subprocess.run(
        [
            sys.executable,
            str(COMMON / "stage_fineweb.py"),
            "--out",
            str(output),
            "--cache-dir",
            str(cache),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["dry_run"] is True
    assert not output.exists()
    assert not cache.exists()


def test_mit_a_verifies_cross_owned_seed2_corpus(tmp_path):
    import tarfile

    source = tmp_path / "source"
    source.mkdir()
    with tarfile.open(BUNDLE) as archive:
        archive.extractall(source, filter="data")
    assignment = launcher.load_assignment(
        assignment_path("mit-collaborator-a")
    )
    canonical = launcher.load_canonical_entries(source, "360m")
    required = verifier.required_corpora(assignment, canonical)
    assert [item["data_rel"] for item in required] == [
        "n1p8m_ds10000",
        "n1p8m_ds10002",
    ]


def test_bed_binding_is_frozen_to_source_snapshot_and_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest = corpus / "manifest.json"
    manifest.write_text('{"artifacts":[],"schema_version":1}\n')
    binding = builder.bed_binding_document(corpus)
    assert binding["source_revision"] == launcher.SOURCE_REVISION
    assert binding["bundle_sha256"] == launcher.BUNDLE_SHA256
    assert binding["bed_output"]["sha256"] == (
        "f89e844723887daa9714d906bf148bfe6931c2f063e94f791a1e861fefc668ca"
    )
    (corpus / "bed-source.json").write_text(json.dumps(binding))
    verifier.verify_bed_binding(corpus)

    manifest.write_text('{"artifacts":["changed"],"schema_version":1}\n')
    with pytest.raises(ValueError, match="bed binding"):
        verifier.verify_bed_binding(corpus)


def test_pinned_bed_link_survives_path_replacement(tmp_path):
    bed = tmp_path / "bed.jsonl"
    bed.write_bytes(b"original bytes")
    with builder.pinned_bed_link(bed) as locked:
        bed.rename(tmp_path / "old.jsonl")
        bed.write_bytes(b"replacement bytes")
        assert locked.read_bytes() == b"original bytes"


def test_mit_authorization_requires_exact_six_probe_report(tmp_path):
    assignment = launcher.load_assignment(
        assignment_path("mit-collaborator-a")
    )
    run_ids = mit_authorizer.EXPECTED_RUN_IDS
    checks = {
        name: {"passed": True, "detail": {}}
        for name in mit_authorizer.EXPECTED_MIT_CHECKS
    }
    checks["mit_runs"]["detail"] = {
        "runs": {
            run_id: {"steps_completed": 200}
            for run_id in run_ids
        },
        "slurm_version": "slurm 24",
        "profile_sha256": "p" * 64,
        "bundle_sha256": launcher.BUNDLE_SHA256,
    }
    checks["throughput"]["detail"] = {
        "runs": {run_id: {} for run_id in run_ids}
    }
    report = {
        "schema_version": 1,
        "platform": "mit",
        "ok": True,
        "checks": checks,
        "bundle": {"archive_sha256": launcher.BUNDLE_SHA256},
        "profile": {"sha256": "p" * 64},
    }
    mit_authorizer.validate_mit_report(
        report,
        assignment=assignment,
        profile_sha256="p" * 64,
    )
    del report["checks"]["mit_runs"]["detail"]["runs"][run_ids[0]]
    with pytest.raises(ValueError, match="six frozen runs"):
        mit_authorizer.validate_mit_report(
            report,
            assignment=assignment,
            profile_sha256="p" * 64,
        )


def test_farmshare_eval_rejects_run_id_traversal(tmp_path):
    environment = {
        **os.environ,
        "RELATIONAL_SOURCE_ROOT": str(tmp_path / "source"),
        "RELATIONAL_VENV": str(tmp_path / "venv"),
        "OUT_ROOT": str(tmp_path / "out"),
        "RUN_ID": "../escape",
    }
    completed = subprocess.run(
        ["bash", str(COMMON / "farmshare_relational_eval.sbatch")],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "safe identifier" in completed.stderr
