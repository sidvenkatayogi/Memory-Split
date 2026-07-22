#!/usr/bin/env python
"""Test the 'CKA convergence = dense stops storing facts' hypothesis with OUR two models.

Two alignment-independent tests (no 0.8B needed, no extra training):

Test A (within a single load, n50k where dense actually memorized): split the fact prompts
  into ones dense RECALLED correctly vs ones it FAILED, and compute cross-arm last-layer CKA
  on each subset. If the convergence is really about memorization, dense and split should be
  MORE different (lower CKA) on facts dense recalled, and MORE similar (higher CKA) on facts
  it failed — using the exact same two models, so no baseline is needed.

Test B (per load): cross-arm last-layer CKA on FACT prompts vs on KNOWLEDGE-FREE reasoning
  text (which neither model stores). Reasoning-CKA is a "generic, nothing-stored" baseline.
  If at high load fact-CKA rises to meet reasoning-CKA, dense is treating facts as generically
  as non-facts (i.e. it stopped doing anything fact-special) — and low-load fact-CKA being far
  below shows it does something special (memorize) only when it can.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from corpusgen.records import ATTRIBUTES
from evals.frozen import records_from_recall_jsonl
from evals.generate import generate_batch
from evals.mechanism import capture_last_token
from evals.scorers import normalize_answer
from probe.geometry import linear_cka
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def load_model(run_dir, ckpt, device):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    m = GPT(mcfg)
    m.load_state_dict(torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)["model"])
    return m.to(device).eval(), cfg


def reasoning_texts(cfg, n):
    seed = cfg.get("seed", 0) * 1000 + 909
    igsm = importlib.import_module("corpusgen.igsm_lite")
    ded = importlib.import_module("corpusgen.deduction")
    op = tuple(cfg.get("igsm_op", (1, 4))); depth = tuple(cfg.get("deduction_depth", (1, 2)))
    half = max(1, n // 2)
    docs = igsm.generate_igsm_docs(half, op[0], op[1], seed)
    docs += ded.generate_deduction_docs(half, depth[0], depth[1], seed + 1)
    return [d.dense_text() for d in docs]


def last_layer(model, tok, prompts, device):
    return capture_last_token(model, tok, prompts, device)["resid"][:, -1, :]  # [N, D]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--ckpt", default="snapshots/step0006100.pt")
    ap.add_argument("--records-jsonl", required=True)
    ap.add_argument("--n-fact-entities", type=int, default=60)
    ap.add_argument("--out", default="outputs/mechinterp/geometry_cka")
    args = ap.parse_args()

    device = pick_device("auto"); tok = get_tok()
    base = Path("outputs/_local_probe")
    dm, cfg = load_model(base / f"d160m_dense_{args.load}_s0", args.ckpt, device)
    sm, _ = load_model(base / f"d160m_split_{args.load}_s0", args.ckpt, device)
    recs = records_from_recall_jsonl(args.records_jsonl)[: args.n_fact_entities]

    fact_prompts, fact_answers = [], []
    for r in recs:
        for a in ATTRIBUTES:
            fact_prompts.append(f"{r.name}'s {bios.RELATION_PHRASES[a]} is")
            fact_answers.append(r.attrs[a])
    reason = reasoning_texts(cfg, len(fact_prompts))

    # --- Test B: fact vs reasoning cross-arm last-layer CKA ---
    Af = last_layer(dm, tok, fact_prompts, device); Bf = last_layer(sm, tok, fact_prompts, device)
    Ar = last_layer(dm, tok, reason, device); Br = last_layer(sm, tok, reason, device)
    cka_fact = linear_cka(Af, Bf)
    cka_reason = linear_cka(Ar, Br)

    result = {"load": args.load, "n_fact_prompts": len(fact_prompts),
              "cross_arm_lastlayer_cka_FACT": round(cka_fact, 4),
              "cross_arm_lastlayer_cka_REASONING_baseline": round(cka_reason, 4)}

    # --- Test A: recalled vs failed (only meaningful where dense recalls a lot) ---
    gens = generate_batch(dm, tok, fact_prompts, 40, None, device)  # dense closed-book
    correct = [normalize_answer(ans) in normalize_answer(g) for ans, g in zip(fact_answers, gens)]
    idx_ok = [i for i, c in enumerate(correct) if c]
    idx_no = [i for i, c in enumerate(correct) if not c]
    result["dense_recall_on_this_set"] = round(sum(correct) / len(correct), 4)
    result["n_recalled"], result["n_failed"] = len(idx_ok), len(idx_no)
    if len(idx_ok) >= 25 and len(idx_no) >= 25:
        ok = torch.tensor(idx_ok); no = torch.tensor(idx_no)
        result["cka_FACT_on_dense-RECALLED"] = round(linear_cka(Af[ok], Bf[ok]), 4)
        result["cka_FACT_on_dense-FAILED"] = round(linear_cka(Af[no], Bf[no]), 4)
    else:
        result["testA_note"] = "not enough recalled/failed to split (needs dense to memorize; only n50k qualifies)"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.load}_cka_tests.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
