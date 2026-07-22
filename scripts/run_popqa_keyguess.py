#!/usr/bin/env python
"""PopQA key-generation stress test (the collaborator's real-world OOD probe).

Adapts the key-gen idea to real Wikidata entities/relations from PopQA. Our 160M
model only knows 6 SYNTHETIC relations, so only PopQA "place of birth" loosely
maps (-> birth_city); the other 15 relations are outside its vocabulary. The
informative, transferable signals are therefore:
  - lookup-fired rate: does the <db_start>..<db_retrieve> machinery trigger at all
    on real-world prompts?
  - name-half copy: given a REAL entity name in the prompt, does the model copy it
    correctly into the key? (the one skill that could transfer)
  - relation-half / exact key: only meaningful on the mapped 'place of birth'
    subset; near-floor by construction elsewhere.

Split arm only. Zero-shot on an existing checkpoint. Writes popqa_keyguess.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
import yaml

from evals.generate import generate_batch_with_stats
from evals.keyguess import extract_key
from corpusgen.bios import RELATION_PHRASES
from organizer.store import Organizer, normalize
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device

# PopQA relation -> our model's known relation (only one loose match exists).
RELATION_MAP = {"place of birth": "birth_city"}


def load_popqa(path, per_relation, seed):
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    by_rel: dict[str, list] = {}
    for r in rows:
        by_rel.setdefault(r["prop"], []).append(r)
    rng = random.Random(seed)
    out = []
    for rel, items in by_rel.items():
        rng.shuffle(items)
        out.extend(items[:per_relation])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="snapshots/step0006100.pt")
    ap.add_argument("--popqa", default="outputs/_popqa/popqa_test.tsv")
    ap.add_argument("--per-relation", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out", default="outputs/mechinterp/popqa_keyguess")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    arm = cfg.get("arm", "dense")
    load = cfg.get("load", run_dir.name)
    device = pick_device("auto")
    tok = get_tok()
    mcfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    model = GPT(mcfg)
    state = torch.load(run_dir / args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    facts = load_popqa(args.popqa, args.per_relation, seed=7)

    # prompt in our training-consistent stub; use our phrase for the mapped
    # relation, else PopQA's own relation words (natural English).
    def phrase_for(prop):
        our = RELATION_MAP.get(prop)
        return RELATION_PHRASES[our] if our else prop

    def gold_rel(prop):
        return RELATION_MAP.get(prop, prop)

    prompts = [f"{f['subj']}'s {phrase_for(f['prop'])} is" for f in facts]

    # organizer so a correct key can retrieve (enables answer-half)
    org = Organizer()
    for f in facts:
        org.add(f["subj"], gold_rel(f["prop"]), f["obj"])

    # decode
    texts = []
    for lo in range(0, len(prompts), args.batch_size):
        t, _ = generate_batch_with_stats(
            model, tok, prompts[lo:lo + args.batch_size], args.max_new, org, device)
        texts.extend(t)

    # score
    per_rel: dict[str, dict] = {}
    tot = Counter()
    emitted_rel_counter = Counter()
    examples = []  # (prompt, subj, emitted_key) for a handful of fired lookups
    for f, gen in zip(facts, texts):
        prop = f["prop"]
        subj = f["subj"]
        mapped = prop in RELATION_MAP
        pr = per_rel.setdefault(prop, Counter())
        pr["n"] += 1; tot["n"] += 1
        key, status = extract_key(gen)
        if status != "lookup":
            pr[status] += 1; tot[status] += 1
            continue
        pr["lookup_fired"] += 1; tot["lookup_fired"] += 1
        if ", " in key:
            e_name, e_rel = key.rsplit(", ", 1)
        else:
            e_name, e_rel = key, ""
        emitted_rel_counter[normalize(e_rel)] += 1
        n_subj, n_emit = normalize(subj), normalize(e_name)
        name_exact = n_emit == n_subj
        name_loose = (n_subj in n_emit) or (len(n_emit) > 0 and n_emit in n_subj)
        # surname copy: does the emitted name contain the real subject's last token?
        subj_toks = n_subj.split()
        surname_ok = bool(subj_toks) and subj_toks[-1] in n_emit.split()
        pr["name_exact"] += name_exact; tot["name_exact"] += name_exact
        pr["name_loose"] += name_loose; tot["name_loose"] += name_loose
        pr["surname_ok"] += surname_ok; tot["surname_ok"] += surname_ok
        if mapped:
            rel_ok = normalize(e_rel) == gold_rel(prop)
            pr["rel_ok"] += rel_ok; tot["rel_ok"] += rel_ok
            pr["exact_key"] += (name_exact and rel_ok); tot["exact_key"] += (name_exact and rel_ok)
        if normalize(f["obj"]) in normalize(gen):
            pr["answer_ok"] += 1; tot["answer_ok"] += 1
        if len(examples) < 20:
            examples.append({"prompt": f"{subj}'s {phrase_for(prop)} is",
                             "real_subject": subj, "emitted_key": key})

    def frac(c, num, den="n"):
        return round(c[num] / c[den], 4) if c[den] else 0.0

    def rel_summary(c):
        fired = c.get("lookup_fired", 0)
        return {
            "n": c["n"],
            "lookup_fired_rate": frac(c, "lookup_fired"),
            "no_lookup_rate": frac(c, "no_lookup"),
            "name_exact_rate_overall": frac(c, "name_exact"),
            "name_exact_rate_given_fired": round(c["name_exact"] / fired, 4) if fired else 0.0,
            "name_loose_rate_given_fired": round(c["name_loose"] / fired, 4) if fired else 0.0,
            "surname_copy_rate_given_fired": round(c["surname_ok"] / fired, 4) if fired else 0.0,
            "rel_ok_rate": frac(c, "rel_ok"),
            "exact_key_rate": frac(c, "exact_key"),
            "answer_rate": frac(c, "answer_ok"),
        }

    result = {
        "run": run_dir.name, "arm": arm, "load": load, "ckpt": args.ckpt,
        "n_facts": tot["n"], "per_relation_cap": args.per_relation,
        "overall": rel_summary(tot),
        "mapped_relation": "place of birth -> birth_city (only clean mapping)",
        "place_of_birth": rel_summary(per_rel.get("place of birth", Counter(n=0))),
        "emitted_relation_distribution": dict(emitted_rel_counter.most_common()),
        "examples_fired": examples,
        "per_relation": {k: rel_summary(v) for k, v in sorted(per_rel.items())},
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{load}_popqa.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "load": load,
        "n": tot["n"],
        "lookup_fired": result["overall"]["lookup_fired_rate"],
        "no_lookup": result["overall"]["no_lookup_rate"],
        "name_exact_given_fired": result["overall"]["name_exact_rate_given_fired"],
        "surname_copy_given_fired": result["overall"]["surname_copy_rate_given_fired"],
        "PoB_exact_key": result["place_of_birth"]["exact_key_rate"],
        "PoB_rel_ok": result["place_of_birth"]["rel_ok_rate"],
        "top_emitted_relations": dict(emitted_rel_counter.most_common(6)),
    }, indent=2))


if __name__ == "__main__":
    main()
