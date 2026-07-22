#!/usr/bin/env python
"""NR-6 — held-out bits-per-byte on stratified slices for one run.

Slices (all scored as DENSE text on whichever arm, for comparability):
  bio       - fresh entities never seen in training (factual slice)
  reasoning - fresh iGSM + deduction problems (knowledge-free slice)
  bed       - optional held-out natural text (--bed-file)

Prediction (spec §6): split store-OFF bpb rises on the BIO slice specifically
(facts externalized), while bed/reasoning bpb stay comparable to dense.

Usage:
  python scripts/run_ppl_slices.py --run outputs/d160m_split_n200k_s0 \
      [--ckpt snapshots/stepXXXX.pt] [--n-docs 300] [--bed-file held_out.txt]

Needs a checkpoint (cluster / outputs/pulled/). Writes <run>/evals/ppl_slices.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from evals.perplexity import fact_value_nll, slice_bits_per_byte
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _fresh_records(cfg, n_docs):
    n = cfg["n_entities"]
    seed = cfg.get("seed", 0)
    fresh = bios.generate_records(n + cfg.get("n_fresh_entities", 200), seed)[n:]
    return fresh[:n_docs]


def _reasoning_slice(cfg, n_docs):
    seed = cfg.get("seed", 0) * 1000 + 909  # distinct from any training seed
    igsm = importlib.import_module("corpusgen.igsm_lite")
    ded = importlib.import_module("corpusgen.deduction")
    op = tuple(cfg.get("igsm_op", (1, 4)))
    depth = tuple(cfg.get("deduction_depth", (1, 2)))
    half = max(1, n_docs // 2)
    docs = igsm.generate_igsm_docs(half, op[0], op[1], seed)
    docs += ded.generate_deduction_docs(half, depth[0], depth[1], seed + 1)
    return [d.dense_text() for d in docs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-docs", type=int, default=300)
    ap.add_argument("--bed-file", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    arm = cfg.get("arm", "dense")
    device = pick_device("auto")
    tok = get_tok()

    model_cfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    model = GPT(model_cfg)
    state = torch.load(run_dir / (args.ckpt or "ckpt.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    # bpb slices use renderings IDENTICAL across arms (reasoning; optional bed),
    # so they are cleanly comparable. The factual signal uses value-only NLL on
    # fresh entities (L27) rather than full dense bio prose (diluted + a format
    # confound for the split arm).
    slices = {"reasoning": _reasoning_slice(cfg, args.n_docs)}
    if args.bed_file:
        text = Path(args.bed_file).read_text()
        slices["bed"] = [d.strip() for d in text.split("\n\n") if d.strip()][: args.n_docs]

    result = {"run": run_dir.name, "arm": arm, "slices": {}}
    for name, texts in slices.items():
        result["slices"][name] = slice_bits_per_byte(model, tok, texts, device)
        print(f"{name}: bpb={result['slices'][name]['bpb']} "
              f"(n_docs={result['slices'][name]['n_docs']})")

    # factual slice: value-only NLL on fresh entities (clean, identical context
    # both arms). Prediction: split (store OFF) ≫ dense.
    fresh = _fresh_records(cfg, args.n_docs)
    result["fact_value_nll"] = fact_value_nll(model, tok, fresh, device)
    print(f"fact_value_nll: {result['fact_value_nll']['nll_per_token']} "
          f"nats/token (n={result['fact_value_nll']['n']})")

    out = run_dir / "evals"
    out.mkdir(exist_ok=True)
    (out / "ppl_slices.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
