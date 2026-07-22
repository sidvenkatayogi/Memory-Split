#!/usr/bin/env python
"""In-schema OOD-entity key-generation eval for a split-arm run.

Zero-shot on an existing checkpoint (no training). Generates FRESH held-out
synthetic entities (never trained on) plus a SEEN control (trained entities),
builds an organizer over exactly the probed facts, and scores how often the
model emits the exactly-correct organizer key "{name}, {relation}" — decomposed
into name-half (copy) and relation-half (6-way) and an outcome taxonomy.

The held-out set is `generate_records(n + K)[n:]`, i.e. the same fresh entities
the pipeline reserves (`n_fresh_entities`) and never trains on — deterministic
and prefix-stable, so this is a clean OOD-entity generalization measurement.

Usage:
  PYTHONPATH=. python scripts/run_keyguess_eval.py \
      --run outputs/_local_probe/d160m_split_n200k_s0 \
      --ckpt snapshots/step0006100.pt --n-eval 150 --out outputs/mechinterp/keyguess

Writes <out>/<load>_keyguess.json. SPLIT arms only (dense has no lookup machinery
→ everything would be no_lookup by construction).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from corpusgen.records import ATTRIBUTES
from evals.keyguess import keyguess_eval
from organizer.store import Organizer
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _build_organizer(records) -> Organizer:
    org = Organizer()
    for rec in records:
        for attr in ATTRIBUTES:
            org.add(rec.name, attr, rec.attrs[attr])
    return org


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-eval", type=int, default=150,
                    help="entities scored per split (× 6 relations = probes)")
    ap.add_argument("--n-heldout", type=int, default=None,
                    help="size of the fresh held-out pool (default: max(n-eval, "
                         "cfg.n_fresh_entities))")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="outputs/mechinterp/keyguess")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    arm = cfg.get("arm", "dense")
    load = cfg.get("load", run_dir.name)
    n = cfg["n_entities"]
    seed = cfg.get("seed", 0)
    if arm != "split":
        print(f"WARNING: arm={arm!r} has no lookup machinery; key-gen eval is "
              f"only meaningful for split arms. Proceeding anyway.")

    device = pick_device("auto")
    tok = get_tok()
    model_cfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    model = GPT(model_cfg)
    state = torch.load(run_dir / (args.ckpt or "ckpt.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    k_held = args.n_heldout or max(args.n_eval, cfg.get("n_fresh_entities", 200))
    all_records = bios.generate_records(n + k_held, seed)
    seen_pool = all_records[:n]
    heldout_pool = all_records[n : n + k_held]

    rng = random.Random(seed * 1000 + 4242)
    seen_eval = rng.sample(seen_pool, min(args.n_eval, len(seen_pool)))
    heldout_eval = heldout_pool[: args.n_eval]

    # organizer holds exactly the probed facts (seen + heldout) so a correct key
    # always retrieves — keeps memory bounded even at n800k.
    org = _build_organizer(seen_eval + heldout_eval)

    common = dict(attributes=ATTRIBUTES, relation_phrases=bios.RELATION_PHRASES,
                  organizer=org, device=device, max_new=args.max_new,
                  batch_size=args.batch_size)
    print(f"[{load}/{arm}] scoring heldout ({len(heldout_eval)} entities)…")
    res_heldout = keyguess_eval(model, tok, heldout_eval, **common)
    print(f"[{load}/{arm}] scoring seen control ({len(seen_eval)} entities)…")
    res_seen = keyguess_eval(model, tok, seen_eval, **common)

    result = {
        "run": run_dir.name,
        "arm": arm,
        "load": load,
        "n_entities_trained": n,
        "ckpt": args.ckpt or "ckpt.pt",
        "n_eval_entities": args.n_eval,
        "heldout": res_heldout,
        "seen": res_seen,
        "key_accuracy_gap": res_seen["key_accuracy"] - res_heldout["key_accuracy"],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{load}_keyguess.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "load": load,
        "heldout_key_acc": round(res_heldout["key_accuracy"], 4),
        "seen_key_acc": round(res_seen["key_accuracy"], 4),
        "heldout_name_half": round(res_heldout["name_half_accuracy"], 4),
        "heldout_rel_half": round(res_heldout["relation_half_accuracy"], 4),
        "heldout_outcomes": res_heldout["outcomes_frac"],
    }, indent=2))


if __name__ == "__main__":
    main()
