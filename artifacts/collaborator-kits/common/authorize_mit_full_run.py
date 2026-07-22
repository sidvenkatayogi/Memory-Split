#!/usr/bin/env python
"""Re-run the six-probe MIT gate and write a root-bound attestation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from launch_assignment import (
    BUNDLE_SHA256,
    EXPECTED_ASSIGNMENTS,
    authorization_document,
    load_assignment,
    sha256_file,
    source_import_path,
    verify_extracted_source,
)


EXPECTED_MIT_CHECKS = frozenset(
    {
        "dependencies",
        "bundle_hashes",
        "smoke",
        "profile",
        "gpu_command",
        "corpus_hashes",
        "resume",
        "slurm",
        "gpu",
        "mit_runs",
        "throughput",
    }
)
EXPECTED_RUN_IDS = tuple(
    sorted(
        {
            Path(relative).stem
            for assignment in EXPECTED_ASSIGNMENTS.values()
            if assignment["platform"] == "mit"
            for relative in assignment["configs"]
        }
    )
)


def validate_mit_report(
    report: dict,
    *,
    assignment: dict,
    profile_sha256: str,
) -> None:
    if (
        not isinstance(report, dict)
        or set(report) != {
            "schema_version",
            "platform",
            "ok",
            "checks",
            "bundle",
            "profile",
        }
        or report["schema_version"] != 1
        or report["platform"] != "mit"
        or report["ok"] is not True
    ):
        raise ValueError("MIT preflight report header is not exact")
    checks = report["checks"]
    if (
        not isinstance(checks, dict)
        or set(checks) != EXPECTED_MIT_CHECKS
        or any(
            not isinstance(item, dict)
            or set(item) != {"passed", "detail"}
            or item["passed"] is not True
            for item in checks.values()
        )
    ):
        raise ValueError("MIT preflight checks are not exactly green")
    if (
        report["bundle"].get("archive_sha256")
        != assignment["bundle_sha256"]
        or assignment["bundle_sha256"] != BUNDLE_SHA256
        or report["profile"].get("sha256") != profile_sha256
    ):
        raise ValueError("MIT report bundle or profile is mismatched")

    mit_runs = checks["mit_runs"]["detail"]
    runs = mit_runs.get("runs") if isinstance(mit_runs, dict) else None
    throughput = checks["throughput"]["detail"]
    throughput_runs = (
        throughput.get("runs") if isinstance(throughput, dict) else None
    )
    if (
        not isinstance(runs, dict)
        or set(runs) != set(EXPECTED_RUN_IDS)
        or any(
            not isinstance(item, dict) or item.get("steps_completed") != 200
            for item in runs.values()
        )
        or mit_runs.get("profile_sha256") != profile_sha256
        or mit_runs.get("bundle_sha256") != BUNDLE_SHA256
        or not isinstance(throughput_runs, dict)
        or set(throughput_runs) != set(EXPECTED_RUN_IDS)
    ):
        raise ValueError("MIT report does not contain all six frozen runs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", default="assignment.json")
    parser.add_argument("--source-root", default="source")
    parser.add_argument("--bundle", default="relational-run.tar.gz")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--report-out",
        default="mit-six-probe-preflight.json",
    )
    parser.add_argument(
        "--authorization-out",
        default="mit-six-probe-authorization.json",
    )
    args = parser.parse_args()

    assignment = load_assignment(Path(args.assignment))
    if assignment["platform"] != "mit":
        raise ValueError("MIT authorization requires an MIT assignment")
    source_root = Path(args.source_root).resolve()
    bundle = Path(args.bundle).resolve()
    profile = Path(args.profile).resolve()
    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    verify_extracted_source(source_root, bundle)
    profile_sha256 = sha256_file(profile)

    with source_import_path(source_root):
        from scripts.platform_preflight import run_preflight

        report = run_preflight(
            "mit",
            bundle=bundle,
            profile=profile,
            source_root=source_root,
            data_root=data_root,
            out_root=out_root,
            runs_root=out_root,
        )
    validate_mit_report(
        report,
        assignment=assignment,
        profile_sha256=profile_sha256,
    )
    report_path = Path(args.report_out)
    report_temporary = report_path.with_name(f".{report_path.name}.partial")
    try:
        report_temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        os.replace(report_temporary, report_path)
    finally:
        report_temporary.unlink(missing_ok=True)

    authorization = authorization_document(
        assignment=assignment,
        kind="mit-six-probe-preflight",
        profile_sha256=profile_sha256,
        data_root=data_root,
        out_root=out_root,
        evidence_sha256=sha256_file(report_path),
    )
    output = Path(args.authorization_out)
    temporary = output.with_name(f".{output.name}.partial")
    try:
        temporary.write_text(
            json.dumps(authorization, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"authorization": str(output), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
