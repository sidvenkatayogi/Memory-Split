#!/usr/bin/env python
"""Build per-load corpora (both arms) + organizer + eval sets.

Usage:
  python scripts/build_corpus.py --out-root DATA_ROOT --stage gates|full \
      [--loads n50k,n200k,n800k] [--bed-file local_text.txt] [--total-tokens N]

Bed text: FineWeb-Edu sample-10BT streamed via HF datasets (deterministic
shard order), or --bed-file (one doc per paragraph split on blank lines)
for offline/smoke use.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from corpusgen.build import BuildCfg, build_corpus
from train.tokenizer import get_tok

LOADS = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}
STAGE_TOKENS = {"gates": 800_000_000, "full": 3_200_000_000}


def bed_iter_hf():
    from datasets import load_dataset

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    for row in ds:
        yield row["text"]


def bed_iter_file(path: str):
    text = Path(path).read_text()
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    # cycle forever; builder stops at its token budget
    yield from itertools.cycle(docs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stage", default="full", choices=list(STAGE_TOKENS))
    ap.add_argument("--loads", default="n50k,n200k,n800k")
    ap.add_argument("--bed-file", default=None)
    ap.add_argument("--total-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    tok = get_tok()
    total = args.total_tokens or STAGE_TOKENS[args.stage]
    for load in args.loads.split(","):
        out_dir = Path(args.out_root) / load
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = BuildCfg(
            n_entities=LOADS[load],
            total_tokens=total,
            seed=args.seed,
        )
        bed = bed_iter_file(args.bed_file) if args.bed_file else bed_iter_hf()
        report = build_corpus(cfg, tok, bed, out_dir)
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"{load}: done -> {out_dir}")
        print(json.dumps({k: v for k, v in report.items() if k != "bed_hashes"}, indent=2)[:800])


if __name__ == "__main__":
    main()
