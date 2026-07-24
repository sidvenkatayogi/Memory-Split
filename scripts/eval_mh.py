"""Evaluate the v2 multi-hop run: DENSE vs SPLIT x {closed-book, +context}.

    python scripts/eval_mh.py --corpus data/mh/n20m \
        --dense-run runs/d160m_n20000000_dense \
        --split-run runs/d160m_n20000000_split --judge --device auto

Loads eval/multihop.jsonl (+ eval/factqa.jsonl) from --corpus, generates answers
for each arm under both conditions, grades with the Claude judge (or string-match
fallback), writes <corpus>/eval_mh_results.json, and prints a headline table.
Facts are supplied ONLY via the Context block (the "optimal retriever"
abstraction) — no DB/retriever. The headline is DENSE+ctx vs SPLIT+ctx.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from evals import context_eval
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def load_model(run_dir: str, device: str):
    state = torch.load(Path(run_dir) / "ckpt.pt", map_location=device, weights_only=False)
    mc = state["cfg"]["model"]
    model_cfg = PRESETS[mc] if isinstance(mc, str) else GPTConfig(**mc)
    if "ctx" in state["cfg"]:
        model_cfg.ctx = state["cfg"]["ctx"]
    model = GPT(model_cfg)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(ln) for ln in open(p)] if p.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--dense-run", required=True)
    ap.add_argument("--split-run", required=True)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=192)  # room for the CoT trace
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = get_tok()
    corpus = Path(args.corpus)
    mh = _load_jsonl(corpus / "eval" / "multihop.jsonl")
    fq = _load_jsonl(corpus / "eval" / "factqa.jsonl")
    if args.limit:
        mh, fq = mh[: args.limit], fq[: args.limit]
    items = mh + fq
    if not items:
        raise SystemExit(f"no eval items under {corpus}/eval")

    client = None
    if args.judge:
        if os.environ.get("OPENAI_API_KEY"):
            from evals.gpt_oracle import GatewayClient
            client = GatewayClient(model=os.environ.get("POC_GPT_MODEL"))
        else:
            print("[eval-mh] --judge set but no OPENAI_API_KEY; using string-match")

    dense = load_model(args.dense_run, device)
    split = load_model(args.split_run, device)
    conds = {
        "dense_closed": context_eval.score(dense, tok, items, device, context=False,
                                            client=client, max_new=args.max_new),
        "dense_context": context_eval.score(dense, tok, items, device, context=True,
                                             client=client, max_new=args.max_new),
        "split_closed": context_eval.score(split, tok, items, device, context=False,
                                            client=client, max_new=args.max_new),
        "split_context": context_eval.score(split, tok, items, device, context=True,
                                             client=client, max_new=args.max_new),
    }
    results = {"corpus": str(corpus), "n_multihop": len(mh), "n_factqa": len(fq),
               "judge": client is not None, "conditions": conds}
    out = Path(args.out) if args.out else corpus / "eval_mh_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== multi-hop accuracy (mh_chain + mh_agg), judge={client is not None} ===")
    print(f"{'condition':<20}{'all':>8}{'seen':>9}{'heldout':>9}")
    labels = [("dense_closed", "DENSE closed"), ("dense_context", "DENSE +ctx"),
              ("split_closed", "SPLIT closed"), ("split_context", "SPLIT +ctx")]
    for key, label in labels:
        cells = []
        for sk in ("all", "seen", "heldout"):
            accs = [conds[key][t][sk]["acc"] for t in ("mh_chain", "mh_agg")
                    if conds[key].get(t, {}).get(sk)]
            cells.append(f"{sum(accs) / len(accs) * 100:.0f}%" if accs else "-")
        print(f"{label:<20}{cells[0]:>8}{cells[1]:>9}{cells[2]:>9}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
