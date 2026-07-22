#!/usr/bin/env python
"""Materialize the frozen three-shard FineWeb-Edu natural-text snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


REPO_ID = "HuggingFaceFW/fineweb-edu"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SHARDS = (
    (
        "sample/10BT/000_00000.parquet",
        2_152_819_114,
        "b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871",
    ),
    (
        "sample/10BT/001_00000.parquet",
        2_152_222_432,
        "3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693",
    ),
    (
        "sample/10BT/002_00000.parquet",
        2_151_796_315,
        "547ae182d132c9f06b6ce63149567208ea9f57630bfd9b1a2938e504f0c9ebd7",
    ),
)
EXPECTED_OUTPUT = {
    "rows": 2_182_000,
    "bytes": 10_498_726_596,
    "sha256": "f89e844723887daa9714d906bf148bfe6931c2f063e94f791a1e861fefc668ca",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_shards(paths: list[Path], output: Path) -> tuple[int, int, str]:
    """Write only ``text`` fields in stable shard/row order."""

    import pyarrow.parquet as parquet

    temporary = output.with_name(f".{output.name}.partial")
    temporary.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows = 0
    bytes_written = 0
    try:
        with temporary.open("wb") as destination:
            for path in paths:
                source = parquet.ParquetFile(path)
                for batch in source.iter_batches(
                    batch_size=4096,
                    columns=["text"],
                    use_threads=False,
                ):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str) or not text:
                            raise ValueError(f"{path} contains an empty text row")
                        line = (
                            json.dumps(
                                {"text": text},
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ).encode("utf-8")
                            + b"\n"
                        )
                        destination.write(line)
                        digest.update(line)
                        rows += 1
                        bytes_written += len(line)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return rows, bytes_written, digest.hexdigest()


def expected_manifest(output: Path) -> dict:
    return {
        "schema_version": 1,
        "dataset": REPO_ID,
        "revision": REVISION,
        "sources": [
            {"path": name, "bytes": size, "sha256": digest}
            for name, size, digest in SHARDS
        ],
        "output": {"path": output.name, **EXPECTED_OUTPUT},
    }


def verify_snapshot(output: Path) -> dict:
    manifest_path = output.with_name(f"{output.name}.manifest.json")
    if (
        not output.is_file()
        or output.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
    ):
        raise ValueError("FineWeb snapshot and manifest must be regular files")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest != expected_manifest(output)
        or output.stat().st_size != EXPECTED_OUTPUT["bytes"]
        or sha256_file(output) != EXPECTED_OUTPUT["sha256"]
    ):
        raise ValueError("existing FineWeb snapshot does not match frozen bytes")
    return manifest


def stage_snapshot(output: Path, cache_dir: Path) -> dict:
    from huggingface_hub import hf_hub_download

    manifest_path = output.with_name(f"{output.name}.manifest.json")
    if output.is_symlink() or manifest_path.is_symlink():
        raise ValueError("snapshot outputs must not be symlinks")
    if output.exists() or manifest_path.exists():
        return verify_snapshot(output)

    resolved = []
    source_records = []
    for filename, expected_bytes, expected_sha256 in SHARDS:
        path = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                cache_dir=cache_dir,
                token=False,
            )
        )
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise ValueError(f"pinned shard failed verification: {filename}")
        resolved.append(path)
        source_records.append(
            {
                "path": filename,
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )

    rows, output_bytes, output_sha256 = convert_shards(resolved, output)
    actual_output = {
        "rows": rows,
        "bytes": output_bytes,
        "sha256": output_sha256,
    }
    if actual_output != EXPECTED_OUTPUT:
        raise ValueError("materialized FineWeb snapshot differs from frozen bytes")
    manifest = expected_manifest(output)
    if manifest["sources"] != source_records:
        raise ValueError("downloaded FineWeb source manifest differs from frozen bytes")
    temporary = manifest_path.with_name(f".{manifest_path.name}.partial")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "dataset": REPO_ID,
                    "revision": REVISION,
                    "sources": [name for name, _, _ in SHARDS],
                    "expected_output": EXPECTED_OUTPUT,
                    "out": args.out,
                    "cache_dir": args.cache_dir,
                },
                sort_keys=True,
            )
        )
        return 0
    manifest = stage_snapshot(Path(args.out), Path(args.cache_dir))
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
