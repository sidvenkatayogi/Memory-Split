#!/usr/bin/env python
"""Verify bundle bytes, the pinned bed, and every required corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_owned_corpora import bed_binding_document
from launch_assignment import (
    BUNDLE_SHA256,
    SOURCE_REVISION,
    load_assignment,
    load_canonical_entries,
    source_import_path,
    verify_extracted_source,
)
from stage_fineweb import verify_snapshot


def required_corpora(
    assignment: dict,
    canonical: dict[str, dict],
) -> list[dict]:
    records = {}
    for relative in assignment["configs"]:
        job = canonical[relative]
        records[job["data_rel"]] = {
            "data_rel": job["data_rel"],
            "entities": job["n_entities"],
            "tokens": job["total_tokens"],
            "data_seed": job["data_seed"],
        }
    return [records[name] for name in sorted(records)]


def verify_bed_binding(corpus: Path) -> None:
    binding = corpus / "bed-source.json"
    if (
        not binding.is_file()
        or binding.is_symlink()
        or json.loads(binding.read_text()) != bed_binding_document(corpus)
    ):
        raise ValueError(f"{corpus.name}: frozen bed binding is missing")


def verify_runtime(
    *,
    assignment_path: Path,
    source_root: Path,
    bundle: Path,
    data_root: Path,
    bed_jsonl: Path,
) -> dict:
    assignment = load_assignment(assignment_path)
    source_root = source_root.resolve()
    bundle = bundle.resolve()
    data_root = data_root.resolve()
    if assignment["bundle_sha256"] != BUNDLE_SHA256:
        raise ValueError("assignment bundle digest is not frozen")
    if assignment["source_revision"] != SOURCE_REVISION:
        raise ValueError("assignment source revision is not frozen")
    verify_extracted_source(source_root, bundle)
    bed = verify_snapshot(bed_jsonl.resolve())
    canonical = load_canonical_entries(source_root, assignment["scale"])
    records = required_corpora(assignment, canonical)

    with source_import_path(source_root):
        from scripts.platform_preflight import _verify_corpus, verify_bundle

        bundle_evidence = verify_bundle(bundle, source_root=source_root)
        verified_corpora = []
        for record in records:
            job = next(
                item
                for item in canonical.values()
                if item["data_rel"] == record["data_rel"]
            )
            _verify_corpus(data_root, job, bundle_evidence["policy"])
            verify_bed_binding(data_root / record["data_rel"])
            verified_corpora.append(record["data_rel"])

    return {
        "schema_version": 1,
        "label": assignment["label"],
        "bundle_sha256": bundle_evidence["archive_sha256"],
        "source_revision": SOURCE_REVISION,
        "bed": bed["output"],
        "verified_corpora": verified_corpora,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", default="assignment.json")
    parser.add_argument("--source-root", default="source")
    parser.add_argument("--bundle", default="relational-run.tar.gz")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--bed-jsonl", required=True)
    args = parser.parse_args()
    report = verify_runtime(
        assignment_path=Path(args.assignment),
        source_root=Path(args.source_root),
        bundle=Path(args.bundle),
        data_root=Path(args.data_root),
        bed_jsonl=Path(args.bed_jsonl),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
