#!/usr/bin/env python
"""Add / delete extension of the editability probe (split arm, no retraining).

Editability (update) showed the split model follows an overwritten store value. This
tests the other two store operations, by the same live-lookup mechanism:

  ADD    — insert facts for BRAND-NEW (never-trained) in-distribution people into the
           store, then check the model retrieves them (it must first emit the right key,
           which is the keyguess skill), and that existing facts are undisturbed.
  DELETE — remove people's keys from the store, then check the model can NO LONGER
           produce their true value (instant unlearning), whether it misses cleanly
           (emits a lookup that finds nothing) vs. guesses, and that other facts survive.

Split arm only; weights frozen throughout. Writes <out>/<load>_add_delete.json.
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
from evals.frozen import records_from_recall_jsonl
from evals.recall import recall_accuracy
from organizer.store import Organizer, normalize
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def org_from(records) -> Organizer:
    o = Organizer()
    for r in records:
        for a in ATTRIBUTES:
            o.add(r.name, a, r.attrs[a])
    return o


def copy_org(o: Organizer) -> Organizer:
    n = Organizer(); n._table = dict(o._table); return n


def score(model, tok, records, organizer, device, n, seed):
    probes = bios.recall_probes(records, min(n, len(records)), seed)
    r = recall_accuracy(model, tok, probes, "on", organizer, device, max_new=32)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="snapshots/step0006100.pt")
    ap.add_argument("--records-jsonl", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="outputs/mechinterp/editability")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    load = cfg.get("load", run_dir.name)
    device = pick_device("auto"); tok = get_tok()
    mcfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    model = GPT(mcfg)
    model.load_state_dict(torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=False)["model"])
    model.to(device).eval()

    trained = records_from_recall_jsonl(args.records_jsonl)
    base_org = org_from(trained)

    # ---------- ADD: brand-new in-distribution people, facts only in the store ----------
    new_records = bios.generate_records(4000, seed=987)  # a fresh synthetic draw
    # keep only names not already keyed in the base store (avoid accidental collisions)
    new_records = [r for r in new_records
                   if normalize(f"{r.name}, birth_city") not in base_org._table][: args.n]
    add_org = copy_org(base_org)
    for r in new_records:
        for a in ATTRIBUTES:
            add_org.add(r.name, a, r.attrs[a])
    add = score(model, tok, new_records, add_org, device, args.n, seed=11)
    # locality of ADD: existing trained facts still fine under the enlarged store
    add_loc = score(model, tok, trained, add_org, device, args.n, seed=12)

    # ---------- DELETE: remove trained people's keys → unlearning ----------
    rng = random.Random(7)
    pool = list(trained)
    rng.shuffle(pool)
    del_recs = pool[: args.n]
    keep_recs = pool[args.n: 2 * args.n]
    del_org = copy_org(base_org)
    for r in del_recs:
        for a in ATTRIBUTES:
            del_org._table.pop(normalize(f"{r.name}, {a}"), None)
    dele = score(model, tok, del_recs, del_org, device, args.n, seed=13)
    del_loc = score(model, tok, keep_recs, del_org, device, args.n, seed=14)

    result = {
        "run": run_dir.name, "arm": cfg.get("arm"), "load": load, "n": args.n,
        "ADD": {
            "add_success_rate": add["overall"],
            "add_locality_rate": add_loc["overall"],
            "note": "new in-distribution people, facts only in the store; success = model "
                    "emits the right key AND the store returns the added value.",
        },
        "DELETE": {
            "true_value_still_produced_rate": dele["overall"],
            "delete_success_rate": 1.0 - dele["overall"],
            "delete_locality_rate": del_loc["overall"],
            "lookup_miss_stats_on_deleted": dele["stats"],
            "note": "keys removed from the store; delete_success = fraction where the true "
                    "value is NO LONGER produced. misses (n_misses) = model emitted a lookup "
                    "that found nothing (clean unlearning) vs. it may still guess a wrong value.",
        },
    }
    d = Path(args.out); d.mkdir(parents=True, exist_ok=True)
    (d / f"{load}_add_delete.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "load": load,
        "add_success": round(add["overall"], 4),
        "add_locality": round(add_loc["overall"], 4),
        "delete_success": round(1 - dele["overall"], 4),
        "true_value_leak_after_delete": round(dele["overall"], 4),
        "delete_locality": round(del_loc["overall"], 4),
        "deleted_lookup_stats": dele["stats"],
    }, indent=2))


if __name__ == "__main__":
    main()
