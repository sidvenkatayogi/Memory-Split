#!/usr/bin/env python
"""Validate, preview, and explicitly submit one collaborator assignment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


SOURCE_REVISION = "0045398a8a8c4cfde13c24192efdf42f361e1913"
BUNDLE_SHA256 = "e6b1e64d6c984ee3f04141d08e870b8f2aa7c2a72e130ce32e08da8dd15cc4dc"
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


def _corpus(data_rel: str, entities: int, tokens: int, data_seed: int) -> dict:
    return {
        "data_rel": data_rel,
        "entities": entities,
        "tokens": tokens,
        "data_seed": data_seed,
    }


EXPECTED_ASSIGNMENTS = {
    "farmshare-collaborator-seed1": {
        "schema_version": 1,
        "label": "farmshare-collaborator-seed1",
        "platform": "farmshare",
        "scale": "160m",
        "source_revision": SOURCE_REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "configs": [
            "configs/160m/d160m_dense_n50k_s1.yaml",
            "configs/160m/d160m_split_n50k_s1.yaml",
            "configs/160m/d160m_dense_n800k_s1.yaml",
            "configs/160m/d160m_split_n800k_s1.yaml",
            "configs/160m/d160m_random_n800k_s1.yaml",
        ],
        "corpora_owned": [
            _corpus("n50k_ds10001", 50_000, 1_599_602_688, 10_001),
            _corpus("n800k_ds10001", 800_000, 1_599_602_688, 10_001),
        ],
        "bed_owner": True,
    },
    "farmshare-collaborator-seed2": {
        "schema_version": 1,
        "label": "farmshare-collaborator-seed2",
        "platform": "farmshare",
        "scale": "160m",
        "source_revision": SOURCE_REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "configs": [
            "configs/160m/d160m_dense_n50k_s2.yaml",
            "configs/160m/d160m_split_n50k_s2.yaml",
            "configs/160m/d160m_dense_n800k_s2.yaml",
            "configs/160m/d160m_split_n800k_s2.yaml",
            "configs/160m/d160m_random_n800k_s2.yaml",
        ],
        "corpora_owned": [
            _corpus("n50k_ds10002", 50_000, 1_599_602_688, 10_002),
            _corpus("n800k_ds10002", 800_000, 1_599_602_688, 10_002),
        ],
        "bed_owner": True,
    },
    "mit-collaborator-a": {
        "schema_version": 1,
        "label": "mit-collaborator-a",
        "platform": "mit",
        "scale": "360m",
        "source_revision": SOURCE_REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "configs": [
            "configs/360m/d360m_dense_n1p8m_s0.yaml",
            "configs/360m/d360m_split_n1p8m_s0.yaml",
            "configs/360m/d360m_dense_n1p8m_s2.yaml",
        ],
        "corpora_owned": [
            _corpus("n1p8m_ds10000", 1_800_000, 3_599_761_408, 10_000),
        ],
        "bed_owner": True,
    },
    "mit-collaborator-b": {
        "schema_version": 1,
        "label": "mit-collaborator-b",
        "platform": "mit",
        "scale": "360m",
        "source_revision": SOURCE_REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "configs": [
            "configs/360m/d360m_dense_n1p8m_s1.yaml",
            "configs/360m/d360m_split_n1p8m_s1.yaml",
            "configs/360m/d360m_split_n1p8m_s2.yaml",
        ],
        "corpora_owned": [
            _corpus("n1p8m_ds10001", 1_800_000, 3_599_761_408, 10_001),
            _corpus("n1p8m_ds10002", 1_800_000, 3_599_761_408, 10_002),
        ],
        "bed_owner": False,
    },
}


def distribution_sha256() -> str:
    encoded = json.dumps(
        EXPECTED_ASSIGNMENTS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _module_uses_root(module, root: Path) -> bool:
    locations = []
    filename = getattr(module, "__file__", None)
    if filename:
        locations.append(filename)
    package_paths = getattr(module, "__path__", ())
    try:
        locations.extend(list(package_paths))
    except (AttributeError, KeyError, TypeError):
        pass
    for location in locations:
        try:
            Path(location).resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        return True
    return False


@contextmanager
def source_import_path(source_root: Path):
    root = source_root.resolve(strict=True)
    old_prefix = sys.pycache_prefix
    before = set(sys.modules)
    with tempfile.TemporaryDirectory(prefix="relational-pycache-") as pycache:
        sys.pycache_prefix = pycache
        sys.path.insert(0, str(root))
        for key in list(sys.path_importer_cache):
            try:
                Path(key).resolve().relative_to(root)
            except (OSError, ValueError):
                continue
            sys.path_importer_cache.pop(key, None)
        try:
            yield
        finally:
            sys.path.remove(str(root))
            sys.pycache_prefix = old_prefix
            imported_from_root = []
            for name in set(sys.modules) - before:
                module = sys.modules.get(name)
                if module is not None and _module_uses_root(module, root):
                    imported_from_root.append(name)
            for name in sorted(
                imported_from_root,
                key=lambda value: value.count("."),
                reverse=True,
            ):
                sys.modules.pop(name, None)
            for key in list(sys.path_importer_cache):
                try:
                    Path(key).resolve().relative_to(root)
                except (OSError, ValueError):
                    continue
                sys.path_importer_cache.pop(key, None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assignment(path: Path) -> dict:
    assignment = json.loads(path.read_text())
    if not isinstance(assignment, dict):
        raise ValueError("assignment must be a JSON object")
    expected = EXPECTED_ASSIGNMENTS.get(assignment.get("label"))
    if expected is None or assignment != expected:
        raise ValueError("assignment differs from the frozen assignment matrix")
    return assignment


def load_canonical_entries(source_root: Path, scale: str) -> dict[str, dict]:
    with source_import_path(source_root):
        from scripts.make_relational_manifest import read_manifest

        entries = read_manifest(source_root / "configs" / f"{scale}.tsv")
    return {relative: job for relative, job in entries}


def verify_extracted_source(source_root: Path, bundle: Path) -> None:
    if (
        not bundle.is_file()
        or bundle.is_symlink()
        or sha256_file(bundle) != BUNDLE_SHA256
    ):
        raise ValueError("portable bundle is missing, unsafe, or mismatched")
    with tarfile.open(bundle, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("portable bundle contains duplicate members")
        by_name = {member.name: member for member in members}
        manifest_member = by_name.get("manifest.json")
        if manifest_member is None or not manifest_member.isfile():
            raise ValueError("portable bundle manifest is missing")
        extracted = archive.extractfile(manifest_member)
        if extracted is None:
            raise ValueError("portable bundle manifest cannot be read")
        manifest = json.load(extracted)

    indexed = {}
    for item in manifest.get("members", []):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256", "bytes"}
            or not isinstance(item["path"], str)
        ):
            raise ValueError("portable bundle member index is malformed")
        relative = PurePosixPath(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in indexed
        ):
            raise ValueError("portable bundle member path is unsafe")
        indexed[relative.as_posix()] = item
    revision = indexed.get("source-revision.txt")
    if (
        manifest.get("source_revision") != SOURCE_REVISION
        or revision is None
    ):
        raise ValueError("portable bundle source revision is mismatched")

    root = source_root.resolve(strict=True)
    selected = indexed
    local_manifest = root / "manifest.json"
    if (
        not local_manifest.is_file()
        or local_manifest.is_symlink()
        or json.loads(local_manifest.read_text()) != manifest
    ):
        raise ValueError("extracted bundle manifest is missing or mismatched")
    allowed_files = set(selected) | {"manifest.json"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"extracted source contains a symlink: {relative}")
        if path.is_dir():
            continue
        relative_text = relative.as_posix()
        if relative_text not in allowed_files:
            raise ValueError(f"unindexed source file is present: {relative_text}")
    for relative, item in selected.items():
        path = source_root.joinpath(*PurePosixPath(relative).parts)
        current = source_root
        for part in PurePosixPath(relative).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"source member traverses a symlink: {relative}")
        try:
            path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"source member is missing: {relative}") from error
        if (
            not path.is_file()
            or path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise ValueError(f"source member differs from bundle: {relative}")


def require_runtime_environment() -> dict[str, Path]:
    roots = {}
    for name in ("DATA_ROOT", "OUT_ROOT", "RELATIONAL_VENV"):
        value = os.environ.get(name)
        if (
            not value
            or not Path(value).is_absolute()
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError(f"{name} must be an absolute launch-time path")
        roots[name] = Path(value).resolve()
    return roots


def validate_mit_steps(steps: int | None) -> str:
    if steps is None:
        return "full"
    if isinstance(steps, bool) or steps != 200:
        raise ValueError("MIT assignment probes must use exactly 200 steps")
    return "probe"


def farmshare_commands(configs: list[str]) -> list[list[str]]:
    return [
        [
            "sbatch",
            f"--export=ALL,CONFIG_REL={relative}",
            "cluster/slurm/relational_train.sbatch",
        ]
        for relative in configs
    ]


def mit_commands(
    source_root: Path,
    configs: list[str],
    *,
    profile: Path,
    bundle: Path,
    steps: int | None,
) -> list[list[str]]:
    with source_import_path(source_root):
        from cluster.mit.run_relational_manifest import plan_manifest

        commands = plan_manifest(
            source_root / "configs" / "360m.tsv",
            profile=profile,
            bundle=bundle,
            steps=steps,
        )
    indexed = {}
    for command in commands:
        export = next(item for item in command if item.startswith("--export="))
        config_field = next(
            item for item in export.removeprefix("--export=").split(",")
            if item.startswith("CONFIG_REL=")
        )
        indexed[config_field.split("=", 1)[1]] = command
    return [indexed[relative] for relative in configs]


def authorization_document(
    *,
    assignment: dict,
    kind: str,
    profile_sha256: str | None = None,
    data_root: Path | None = None,
    out_root: Path | None = None,
    evidence_sha256: str | None = None,
) -> dict:
    if (
        not isinstance(evidence_sha256, str)
        or len(evidence_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in evidence_sha256
        )
    ):
        raise ValueError("authorization requires a frozen evidence digest")
    document = {
        "schema_version": 1,
        "kind": kind,
        "passed": True,
        "source_revision": assignment["source_revision"],
        "bundle_sha256": assignment["bundle_sha256"],
        "distribution_sha256": distribution_sha256(),
        "evidence_sha256": evidence_sha256,
    }
    if kind == "mit-six-probe-preflight":
        if profile_sha256 is None or data_root is None or out_root is None:
            raise ValueError("MIT authorization requires profile and runtime roots")
        document.update(
            {
                "profile_sha256": profile_sha256,
                "data_root": str(Path(data_root).resolve()),
                "out_root": str(Path(out_root).resolve()),
            }
        )
    elif kind == "farmshare-29m-gate":
        pass
    else:
        raise ValueError("unknown authorization kind")
    return document


def verify_authorization(
    path: Path,
    evidence_path: Path,
    *,
    assignment: dict,
    kind: str,
    profile_sha256: str | None = None,
    data_root: Path | None = None,
    out_root: Path | None = None,
) -> None:
    authorization = json.loads(path.read_text())
    evidence_sha256 = (
        authorization.get("evidence_sha256")
        if isinstance(authorization, dict)
        else None
    )
    expected = authorization_document(
        assignment=assignment,
        kind=kind,
        profile_sha256=profile_sha256,
        data_root=data_root,
        out_root=out_root,
        evidence_sha256=evidence_sha256,
    )
    if authorization != expected:
        raise ValueError(f"{kind} authorization does not match this assignment")
    if (
        not evidence_path.is_file()
        or evidence_path.is_symlink()
        or sha256_file(evidence_path) != evidence_sha256
    ):
        raise ValueError(f"{kind} evidence does not match its authorization")


def execute_commands(commands: list[list[str]], *, cwd: Path) -> dict:
    jobs = []
    failures = []
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            record = {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout[:16_384],
                "stderr": result.stderr[:16_384],
            }
        except Exception as error:
            record = {
                "command": command,
                "returncode": None,
                "error": f"{type(error).__name__}: {error}",
            }
        jobs.append(record)
        if record["returncode"] != 0:
            failures.append(record)
    return {
        "attempted": len(jobs),
        "submitted": len(jobs) - len(failures),
        "failures": failures,
        "jobs": jobs,
        "exit_code": int(bool(failures)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", default="assignment.json")
    parser.add_argument("--source-root", default="source")
    parser.add_argument("--bundle", default="relational-run.tar.gz")
    parser.add_argument("--profile")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--authorization")
    parser.add_argument("--authorization-evidence")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status-out", default="launch-status.json")
    args = parser.parse_args()

    assignment_path = Path(args.assignment).resolve()
    assignment = load_assignment(assignment_path)
    source_root = Path(args.source_root).resolve()
    bundle = Path(args.bundle).resolve()
    if sha256_file(bundle) != assignment["bundle_sha256"]:
        raise ValueError("portable bundle SHA-256 does not match assignment")
    revision = (source_root / "source-revision.txt").read_text().strip()
    if revision != assignment["source_revision"]:
        raise ValueError("extracted source revision does not match assignment")
    verify_extracted_source(source_root, bundle)

    canonical = load_canonical_entries(source_root, assignment["scale"])
    if any(relative not in canonical for relative in assignment["configs"]):
        raise ValueError("assignment contains a noncanonical config")

    profile_sha256 = None
    mit_mode = None
    if assignment["platform"] == "farmshare":
        if args.profile is not None or args.steps is not None:
            raise ValueError("FarmShare assignments do not accept MIT options")
        commands = farmshare_commands(assignment["configs"])
        authorization_kind = "farmshare-29m-gate"
    else:
        if args.profile is None:
            parser.error("--profile is required for MIT assignments")
        mit_mode = validate_mit_steps(args.steps)
        profile = Path(args.profile).resolve()
        commands = mit_commands(
            source_root,
            assignment["configs"],
            profile=profile,
            bundle=bundle,
            steps=args.steps,
        )
        profile_sha256 = sha256_file(profile)
        authorization_kind = "mit-six-probe-preflight"

    if args.execute:
        runtime_roots = require_runtime_environment()
        requires_authorization = (
            assignment["platform"] == "farmshare"
            or assignment["platform"] == "mit" and mit_mode == "full"
        )
        if requires_authorization:
            if args.authorization is None or args.authorization_evidence is None:
                parser.error(
                    "--authorization and --authorization-evidence are required "
                    "for this submission"
                )
            verify_authorization(
                Path(args.authorization),
                Path(args.authorization_evidence),
                assignment=assignment,
                kind=authorization_kind,
                profile_sha256=profile_sha256,
                data_root=runtime_roots["DATA_ROOT"],
                out_root=runtime_roots["OUT_ROOT"],
            )
        from verify_assignment import verify_runtime

        verify_runtime(
            assignment_path=assignment_path,
            source_root=source_root,
            bundle=bundle,
            data_root=runtime_roots["DATA_ROOT"],
            bed_jsonl=(
                runtime_roots["DATA_ROOT"]
                / "fineweb-edu-sample10bt-r87f091-first3.jsonl"
            ),
        )
        report = execute_commands(commands, cwd=source_root)
    else:
        report = {
            "attempted": 0,
            "submitted": 0,
            "failures": [],
            "jobs": [],
            "exit_code": 0,
        }

    report.update(
        {
            "schema_version": 1,
            "label": assignment["label"],
            "platform": assignment["platform"],
            "dry_run": not args.execute,
            "planned": len(commands),
            "commands": commands,
        }
    )
    if args.execute:
        status = Path(args.status_out)
        temporary = status.with_name(f".{status.name}.partial")
        try:
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, status)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        print("dry-run; pass --execute to submit", file=sys.stderr)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
