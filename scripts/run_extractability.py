#!/usr/bin/env python
"""NR-4 — extractability probe: generative recall vs MC recognition on one run.

Reveals whether low generative recall means "not stored" (MC ~ chance) or
"stored-but-not-extractable" (MC >> chance) — the L11 confound in bits-in-weights.

Usage:
  python scripts/run_extractability.py --run outputs/d160m_dense_n200k_s0 \
      [--ckpt snapshots/stepXXXX.pt] [--n-items 2000] [--n-choices 4]

Needs a checkpoint (runs on cluster / outputs/pulled/, not the results-only repo
copy). Writes <run>/evals/extractability.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from evals.extractability import make_mc_recall_items, mc_recall_accuracy
from evals.recall import recall_accuracy
from organizer.store import Organizer
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-items", type=int, default=2000)
    ap.add_argument("--n-choices", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--records-jsonl", default=None,
                    help="load FROZEN trained entities from this recall.jsonl "
                         "(SEEN). If omitted, regenerate locally (UNSEEN — note "
                         "the repo generator has drifted from the trained corpus).")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    arm = cfg.get("arm", "dense")
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
        records = records_from_recall_jsonl(args.records_jsonl)
        entity_source = "frozen_seen"
    else:
        # NOTE: repo generator drifted from the trained corpus → these are
        # effectively UNSEEN in-distribution entities, not the trained ones.
        records = bios.generate_records(n_entities + cfg.get("n_fresh_entities", 200), seed)[:n_entities]
        entity_source = "regenerated_unseen"

    # generative recall (existing instrument): dense=closed, split=off (weights only)
    probes = bios.recall_probes(records, min(2000, len(records)), seed * 1000 + 88)
    gen_mode = "closed" if arm == "dense" else "off"
    gen = recall_accuracy(model, tok, probes, gen_mode, None, device)

    # MC recognition (new instrument)
    mc_items = make_mc_recall_items(records, args.n_items, seed * 1000 + 91,
                                    n_choices=args.n_choices)
    mc = mc_recall_accuracy(model, tok, mc_items, device)

    result = {
        "run": run_dir.name,
        "arm": arm,
        "entity_source": entity_source,
        "n_entities_probed": len(records),
        "generative_recall": {"mode": gen_mode, "overall": gen["overall"],
                              "per_attribute": gen["per_attribute"]},
        "mc_recognition": mc,
        "interpretation": (
            "stored_but_not_extractable"
            if gen["overall"] < 0.05 and mc["overall"] > 2 * mc["chance"]
            else "consistent_with_absence"
            if gen["overall"] < 0.05 and mc["overall"] <= 1.5 * mc["chance"]
            else "extractable"
        ),
    }
    out = run_dir / "evals"
    out.mkdir(exist_ok=True)
    (out / "extractability.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
