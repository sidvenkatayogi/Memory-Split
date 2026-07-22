#!/usr/bin/env python
"""Bind the coordinator's verified 29M corpus to the frozen FineWeb bed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from authorize_farmshare_gate import verify_exact_bundle, verify_source_archive
from build_owned_corpora import write_bed_binding
from launch_assignment import source_import_path
from stage_fineweb import verify_snapshot
from verify_assignment import verify_bed_binding


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--bed-jsonl", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    source_archive = Path(args.source_archive).resolve()
    bundle = Path(args.bundle).resolve()
    data_root = Path(args.data_root).resolve()
    bed_jsonl = Path(args.bed_jsonl).resolve()
    gate_corpus = data_root / "n50k_gate_ds10000"
    plan = {
        "dry_run": not args.execute,
        "gate_corpus": str(gate_corpus),
        "bed_jsonl": str(bed_jsonl),
    }
    if not args.execute:
        print(json.dumps(plan, sort_keys=True))
        return 0

    verify_exact_bundle(bundle)
    verify_source_archive(source_root, source_archive)
    bed = verify_snapshot(bed_jsonl)
    with source_import_path(source_root):
        from scripts.make_relational_manifest import make_jobs
        from scripts.platform_preflight import _verify_corpus, verify_bundle

        bundle_evidence = verify_bundle(bundle, source_root=source_root)
        _verify_corpus(
            data_root,
            make_jobs("29m")[0],
            bundle_evidence["policy"],
        )
    write_bed_binding(gate_corpus)
    verify_bed_binding(gate_corpus)
    print(
        json.dumps(
            {
                **plan,
                "bed": bed["output"],
                "receipt": json.loads(
                    (gate_corpus / "bed-source.json").read_text()
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
