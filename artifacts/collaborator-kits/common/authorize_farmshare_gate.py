#!/usr/bin/env python
"""Write the coordinator attestation only after the frozen 29M gate passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

from launch_assignment import (
    BUNDLE_SHA256,
    EXPECTED_ASSIGNMENTS,
    authorization_document,
    sha256_file,
    source_import_path,
)
from verify_assignment import verify_bed_binding
from stage_fineweb import verify_snapshot

SOURCE_ARCHIVE_SHA256 = (
    "5642b3919951b719958d40d256cb9dc08d65bf187884056975db2e9514b6cc1a"
)


def verify_exact_bundle(bundle: Path) -> None:
    if (
        not bundle.is_file()
        or bundle.is_symlink()
        or sha256_file(bundle) != BUNDLE_SHA256
    ):
        raise ValueError("portable bundle does not match exact frozen bytes")


def atomic_json(path: Path, value: dict) -> bytes:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return encoded


def verify_source_archive(source_root: Path, archive_path: Path) -> None:
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or sha256_file(archive_path) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("coordinator source archive is missing or mismatched")
    expected = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "relational-core"
            ):
                raise ValueError("coordinator source archive path is unsafe")
            stripped = PurePosixPath(*relative.parts[1:])
            if not stripped.parts:
                if not member.isdir():
                    raise ValueError("coordinator source archive root is invalid")
                continue
            if member.isdir():
                continue
            if not member.isfile() or stripped.as_posix() in expected:
                raise ValueError("coordinator source archive member is unsafe")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError("coordinator source member cannot be read")
            payload = handle.read()
            expected[stripped.as_posix()] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

    root = source_root.resolve(strict=True)
    revision = root / "source-revision.txt"
    if (
        not revision.is_file()
        or revision.is_symlink()
        or revision.read_text().strip()
        != EXPECTED_ASSIGNMENTS["farmshare-collaborator-seed1"][
            "source_revision"
        ]
    ):
        raise ValueError("coordinator source revision is mismatched")
    allowed = set(expected) | {"source-revision.txt"}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"coordinator source symlink is unsafe: {relative}")
        if path.is_dir():
            continue
        relative_text = relative.as_posix()
        ignored_cache = (
            "__pycache__" in relative.parts and path.suffix == ".pyc"
        )
        if relative_text not in allowed and not ignored_cache:
            raise ValueError(
                f"unindexed coordinator source file is present: {relative_text}"
            )
    for relative, record in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"coordinator source member differs: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--source-archive")
    parser.add_argument("--source-revision")
    parser.add_argument("--bed-jsonl")
    parser.add_argument("--evidence-out")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    bundle = Path(args.bundle).resolve()
    verify_exact_bundle(bundle)
    source_archive = (
        Path(args.source_archive).resolve()
        if args.source_archive is not None
        else bundle.with_name("relational-source-0045398.tar.gz")
    )
    expected_revision = EXPECTED_ASSIGNMENTS[
        "farmshare-collaborator-seed1"
    ]["source_revision"]
    if args.source_revision not in (None, expected_revision):
        raise ValueError("caller-supplied source revision is mismatched")
    verify_source_archive(source_root, source_archive)
    data_root = Path(args.data_root).resolve()
    bed_jsonl = (
        Path(args.bed_jsonl).resolve()
        if args.bed_jsonl is not None
        else data_root / "fineweb-edu-sample10bt-r87f091-first3.jsonl"
    )
    bed = verify_snapshot(bed_jsonl)
    gate_corpus = data_root / "n50k_gate_ds10000"
    verify_bed_binding(gate_corpus)
    with source_import_path(source_root):
        from scripts.make_relational_manifest import make_jobs
        from scripts.platform_preflight import (
            _farmshare_learnability_detail,
            _verify_corpus,
            verify_bundle,
        )

        bundle_evidence = verify_bundle(bundle, source_root=source_root)
        gate_job = make_jobs("29m")[0]
        _verify_corpus(data_root, gate_job, bundle_evidence["policy"])
        detail = _farmshare_learnability_detail(
            data_root,
            Path(args.out_root).resolve(),
        )
    evidence = {
        "schema_version": 1,
        "source_revision": expected_revision,
        "bundle_sha256": sha256_file(bundle),
        "bed": bed["output"],
        "gate_corpus": json.loads(
            (gate_corpus / "bed-source.json").read_text()
        ),
        "gate": detail,
    }
    output = Path(args.out)
    evidence_output = (
        Path(args.evidence_out)
        if args.evidence_out is not None
        else output.with_name("farmshare-29m-gate-evidence.json")
    )
    evidence_bytes = atomic_json(evidence_output, evidence)
    authorization = authorization_document(
        assignment=EXPECTED_ASSIGNMENTS["farmshare-collaborator-seed1"],
        kind="farmshare-29m-gate",
        evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
    )
    atomic_json(output, authorization)
    print(json.dumps({"authorization": authorization, "gate": detail}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
