#!/usr/bin/env python
"""Validate and build only the corpora owned by one assignment."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

from launch_assignment import (
    BUNDLE_SHA256,
    SOURCE_REVISION,
    load_assignment,
    load_canonical_entries,
    verify_extracted_source,
)
from stage_fineweb import (
    EXPECTED_OUTPUT,
    REPO_ID,
    REVISION,
    sha256_file,
    verify_snapshot,
)


def validate_corpora(assignment: dict, canonical: dict[str, dict]) -> list[dict]:
    records = assignment["corpora_owned"]
    if not isinstance(records, list) or not records:
        raise ValueError("assignment must own at least one corpus")
    expected_by_data_rel = {}
    for job in canonical.values():
        expected_by_data_rel[job["data_rel"]] = {
            "data_rel": job["data_rel"],
            "entities": job["n_entities"],
            "tokens": job["total_tokens"],
            "data_seed": job["data_seed"],
        }
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "data_rel",
            "entities",
            "tokens",
            "data_seed",
        }:
            raise ValueError("corpus record fields do not match the closed schema")
        if record["data_rel"] in seen:
            raise ValueError("assignment contains duplicate corpus ownership")
        seen.add(record["data_rel"])
        if expected_by_data_rel.get(record["data_rel"]) != record:
            raise ValueError("owned corpus differs from the frozen run matrix")
    return records


def command_for(
    record: dict,
    *,
    source_root: Path,
    data_root: Path,
    bed_jsonl: Path,
) -> list[str]:
    return [
        sys.executable,
        str(source_root / "scripts" / "build_relational_corpus.py"),
        "--out",
        str(data_root / record["data_rel"]),
        "--entities",
        str(record["entities"]),
        "--tokens",
        str(record["tokens"]),
        "--data-seed",
        str(record["data_seed"]),
        "--bed-jsonl",
        str(bed_jsonl),
        "--route-policy",
        str(source_root / "configs" / "route-policy.json"),
        "--route-policy-sha256",
        "0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058",
    ]


def bed_binding_document(corpus: Path) -> dict:
    manifest = corpus / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("corpus manifest is missing or unsafe")
    return {
        "schema_version": 1,
        "dataset": REPO_ID,
        "revision": REVISION,
        "bed_output": EXPECTED_OUTPUT,
        "source_revision": SOURCE_REVISION,
        "bundle_sha256": BUNDLE_SHA256,
        "corpus_manifest": {
            "path": "manifest.json",
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
        },
    }


def write_bed_binding(corpus: Path) -> None:
    output = corpus / "bed-source.json"
    temporary = output.with_name(f".{output.name}.partial")
    try:
        temporary.write_text(
            json.dumps(
                bed_binding_document(corpus),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def pinned_bed_link(bed_jsonl: Path):
    locked = bed_jsonl.with_name(
        f".{bed_jsonl.name}.build-{os.getpid()}-{secrets.token_hex(8)}"
    )
    try:
        os.link(bed_jsonl, locked, follow_symlinks=False)
        yield locked
    finally:
        locked.unlink(missing_ok=True)


def verify_locked_bed(path: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != EXPECTED_OUTPUT["bytes"]
        or sha256_file(path) != EXPECTED_OUTPUT["sha256"]
    ):
        raise ValueError("locked FineWeb snapshot changed during corpus build")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", default="assignment.json")
    parser.add_argument("--source-root", default="source")
    parser.add_argument("--bundle", default="relational-run.tar.gz")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--bed-jsonl", required=True)
    parser.add_argument("--data-rel")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    assignment = load_assignment(Path(args.assignment))
    source_root = Path(args.source_root).resolve()
    bundle = Path(args.bundle).resolve()
    data_root = Path(args.data_root).resolve()
    bed_jsonl = Path(args.bed_jsonl).resolve()
    verify_extracted_source(source_root, bundle)
    canonical = load_canonical_entries(source_root, assignment["scale"])
    records = validate_corpora(assignment, canonical)
    if args.data_rel is not None:
        records = [
            record for record in records if record["data_rel"] == args.data_rel
        ]
        if not records:
            raise ValueError("--data-rel is not owned by this assignment")
    commands = [
        command_for(
            record,
            source_root=source_root,
            data_root=data_root,
            bed_jsonl=bed_jsonl,
        )
        for record in records
    ]
    print(json.dumps({"commands": commands, "dry_run": not args.execute}, indent=2))
    if not args.execute:
        print("dry-run; pass --execute to build", file=sys.stderr)
        return 0
    verify_snapshot(bed_jsonl)
    data_root.mkdir(parents=True, exist_ok=True)
    built = []
    with pinned_bed_link(bed_jsonl) as locked_bed:
        verify_locked_bed(locked_bed)
        for record in records:
            command = command_for(
                record,
                source_root=source_root,
                data_root=data_root,
                bed_jsonl=locked_bed,
            )
            completed = subprocess.run(command, cwd=source_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
            built.append(data_root / record["data_rel"])
        verify_locked_bed(locked_bed)
    for corpus in built:
        write_bed_binding(corpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
