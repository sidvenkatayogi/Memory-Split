#!/usr/bin/env python
"""Guardrail check: when a lookup MISSES (fact not in the store), does the split model
fabricate a plausible-looking value, or does it decline?

Setup: build the store from the trained people, then query the model about people whose
facts are NOT in the store (fresh in-distribution people → the lookup is guaranteed to
miss). With the store ON, a miss means the harness force-feeds nothing, so whatever
appears in the answer slot is the MODEL's own continuation. We classify that slot:

  fabricated_pool_value : the model emitted a valid value from the attribute's pool
                          (e.g. a real city name for birth_city, a real date for
                          birth_date) — a confident, plausible-but-baseless hallucination.
  nonvalue / empty      : it emitted no pool-valid value (closed the lookup, produced
                          filler, or nothing usable) — i.e. it did NOT assert a fact.
  no_lookup             : it never emitted a lookup at all.

Also runs a positive control on people whose facts ARE in the store (should retrieve the
correct value). Split arm only. Writes <out>/<load>_hallucination.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from corpusgen.bios import RELATION_PHRASES, VALUE_POOLS, BIRTH_DATE_MIN, BIRTH_DATE_MAX
from corpusgen.records import ATTRIBUTES
from evals.frozen import records_from_recall_jsonl
from evals.generate import generate_batch_with_stats
from organizer.store import Organizer, normalize
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device

DB_RET, DB_END = "<|db_retrieve|>", "<|db_end|>"
_DATE = re.compile(r"[A-Z][a-z]+ \d{1,2}, \d{4}")


def answer_slot(text: str) -> str | None:
    """Text the model produced after the (first) <|db_retrieve|>, up to <|db_end|>."""
    i = text.find(DB_RET)
    if i == -1:
        return None
    rest = text[i + len(DB_RET):]
    j = rest.find(DB_END)
    return (rest if j == -1 else rest[:j]).strip()


def is_pool_value(attr: str, span: str) -> bool:
    s = span.lower()
    if attr == "birth_date":
        return bool(_DATE.search(span))
    return any(v.lower() in s for v in VALUE_POOLS[attr])


def org_from(records) -> Organizer:
    o = Organizer()
    for r in records:
        for a in ATTRIBUTES:
            o.add(r.name, a, r.attrs[a])
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="snapshots/step0006100.pt")
    ap.add_argument("--records-jsonl", required=True)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="outputs/mechinterp/editability/hallucination")
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
    store = org_from(trained)  # contains ONLY trained people

    # unknown people (fresh, in-distribution) whose facts are NOT in the store -> misses
    fresh = [r for r in bios.generate_records(4000, seed=555)
             if normalize(f"{r.name}, birth_city") not in store._table][: args.n]

    def run(records, tag):
        rng = __import__("random").Random(3)
        probes = [(r, rng.choice(ATTRIBUTES)) for r in records]
        prompts = [f"{r.name}'s {RELATION_PHRASES[a]} is" for r, a in probes]
        texts, stats = [], {"n_lookups": 0, "n_hits": 0, "n_misses": 0, "n_malformed": 0}
        for lo in range(0, len(prompts), 64):
            t, s = generate_batch_with_stats(model, tok, prompts[lo:lo+64], 24, store, device)
            texts += t
            for k in stats:
                stats[k] += s[k]
        counts = {"fabricated_pool_value": 0, "nonvalue_or_empty": 0, "no_lookup": 0}
        examples = []
        for (r, a), text in zip(probes, texts):
            slot = answer_slot(text)
            if slot is None:
                counts["no_lookup"] += 1
                continue
            if is_pool_value(a, slot):
                counts["fabricated_pool_value"] += 1
                if len(examples) < 15:
                    examples.append({"prompt": f"{r.name}'s {RELATION_PHRASES[a]} is",
                                     "attr": a, "answer_slot": slot[:40]})
            else:
                counts["nonvalue_or_empty"] += 1
        n = len(probes)
        return {"n": n, "lookup_stats": stats,
                "counts": counts,
                "fabrication_rate": round(counts["fabricated_pool_value"] / n, 4),
                "nonvalue_or_empty_rate": round(counts["nonvalue_or_empty"] / n, 4),
                "no_lookup_rate": round(counts["no_lookup"] / n, 4),
                "examples": examples}, tag

    unknown_res, _ = run(fresh, "unknown_miss")
    control_res, _ = run(trained[: args.n], "known_control")

    result = {"run": run_dir.name, "arm": cfg.get("arm"), "load": load,
              "unknown_not_in_store": unknown_res,
              "known_in_store_control": control_res}
    d = Path(args.out); d.mkdir(parents=True, exist_ok=True)
    (d / f"{load}_hallucination.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "load": load,
        "UNKNOWN(miss) fabrication_rate": unknown_res["fabrication_rate"],
        "UNKNOWN nonvalue/empty_rate": unknown_res["nonvalue_or_empty_rate"],
        "UNKNOWN lookup_stats": unknown_res["lookup_stats"],
        "KNOWN(control) correct-ish via store": control_res["fabrication_rate"],
        "KNOWN lookup_stats": control_res["lookup_stats"],
    }, indent=2))


if __name__ == "__main__":
    main()
