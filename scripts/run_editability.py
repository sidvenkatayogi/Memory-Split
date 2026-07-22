#!/usr/bin/env python
"""E2 — editability / unlearning eval for the split arm.

Edit organizer entries and show the split arm's answers follow the store with no
weight update (H3 backing, independent of ON/OFF recall). Needs a split-arm
checkpoint (cluster / outputs/pulled/).

Usage:
  python scripts/run_editability.py --run outputs/d160m_split_n200k_s0 \
      [--ckpt snapshots/stepNNNN.pt] [--n-edits 200]

Writes <run>/evals/editability.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from evals.editability import editability_eval
from organizer.store import Organizer
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-edits", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--records-jsonl", default=None,
                    help="load FROZEN trained entities (SEEN) + build the organizer from them "
                         "(use when the cluster organizer.jsonl isn't reachable locally).")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    if cfg.get("arm") != "split":
        print("warning: editability is a split-arm eval (dense has no store)")
    n_entities = cfg["n_entities"]
    seed = cfg.get("seed", 0)
    device = pick_device("auto")
    tok = get_tok()

    model_cfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    model = GPT(model_cfg)
    state = torch.load(run_dir / (args.ckpt or "ckpt.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    if args.records_jsonl:
        from evals.frozen import records_from_recall_jsonl
        from corpusgen.records import ATTRIBUTES
        records = records_from_recall_jsonl(args.records_jsonl)
        organizer = Organizer()
        for rec in records:
            for attr in ATTRIBUTES:
                organizer.add(rec.name, attr, rec.attrs[attr])
    else:
        records = bios.generate_records(n_entities + cfg.get("n_fresh_entities", 200), seed)[:n_entities]
        organizer = Organizer.load(Path(cfg["data_dir"]) / "organizer.jsonl")

    result = editability_eval(model, tok, organizer, records, device,
                              n_edits=args.n_edits, seed=seed * 1000 + 71,
                              batch_size=args.batch_size)
    result = {"run": run_dir.name, "arm": cfg.get("arm"), **result}
    out = run_dir / "evals"
    out.mkdir(exist_ok=True)
    (out / "editability.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
