#!/usr/bin/env python
"""Alignment-FREE cross-arm neuron comparison (no index assumption).

The mechanism probe's cross-arm number compares neuron (L,j) in dense to the SAME index
(L,j) in split — which assumes index = feature. This drops that assumption: for each of
dense's top-64 fact-neurons we find its BEST-matching split neuron by activation
correlation over the same prompts, and report (a) how good that best match is and (b)
whether the matched split neuron is itself fact-selective. We also report each arm's OWN
top-64 selectivity (each arm ranked on itself — no cross-model index needed).

Reads: dense n50k is the informative load (it actually memorized). Frozen SEEN people.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from corpusgen.records import ATTRIBUTES
from evals.frozen import records_from_recall_jsonl
from evals.mechanism import capture_last_token, cohens_d, top_neurons, _DEFAULT_CONTROL
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def load_model(run_dir, ckpt, device):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    m = GPT(mcfg)
    m.load_state_dict(torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)["model"])
    return m.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--ckpt", default="snapshots/step0006100.pt")
    ap.add_argument("--records-jsonl", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-records", type=int, default=100)
    ap.add_argument("--out", default="outputs/mechinterp/mechanism")
    args = ap.parse_args()

    device = pick_device("auto"); tok = get_tok()
    base = Path("outputs/_local_probe")
    dm = load_model(base / f"d160m_dense_{args.load}_s0", args.ckpt, device)
    sm = load_model(base / f"d160m_split_{args.load}_s0", args.ckpt, device)
    recs = records_from_recall_jsonl(args.records_jsonl)[: args.n_records]

    fact_prompts = [f"{r.name}'s {bios.RELATION_PHRASES[a]} is" for r in recs for a in ATTRIBUTES]
    ctrl_prompts = list(_DEFAULT_CONTROL)

    # selectivity maps per arm
    d_fact = capture_last_token(dm, tok, fact_prompts, device)["mlp"]   # [Nf, L, H]
    d_ctrl = capture_last_token(dm, tok, ctrl_prompts, device)["mlp"]
    s_fact = capture_last_token(sm, tok, fact_prompts, device)["mlp"]
    s_ctrl = capture_last_token(sm, tok, ctrl_prompts, device)["mlp"]
    sel_d = cohens_d(d_fact, d_ctrl)   # [L, H]
    sel_s = cohens_d(s_fact, s_ctrl)   # [L, H]
    H = sel_d.size(-1)

    dense_top = top_neurons(sel_d, args.k)               # (L,j,d) in dense
    split_top = top_neurons(sel_s, args.k)               # (L,j,d) in split (own)
    top_flat = [l * H + j for (l, j, _) in dense_top]

    # --- alignment-free matching: correlate over the fact prompts ---
    A = d_fact.reshape(d_fact.size(0), -1)               # [Nf, L*H] dense
    B = s_fact.reshape(s_fact.size(0), -1)               # [Nf, L*H] split
    Ad = A[:, top_flat]                                  # [Nf, k] dense fact-neurons
    def z(x):
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    Adz, Bz = z(Ad), z(B)
    corr = (Adz.t() @ Bz) / Adz.size(0)                 # [k, L*H]  Pearson corr
    best_abs, best_idx = corr.abs().max(dim=1)          # best split partner per dense neuron
    sel_s_flat = sel_s.reshape(-1)
    matched_sel = sel_s_flat[best_idx].abs()            # partner's fact-selectivity

    def m(t):
        return float(t.float().mean())

    result = {
        "load": args.load, "k": args.k, "n_fact_prompts": len(fact_prompts),
        "dense_own_top_mean_abs_d": m(torch.tensor([abs(d) for *_, d in dense_top])),
        "split_own_top_mean_abs_d": m(torch.tensor([abs(d) for *_, d in split_top])),
        "dense_top_same_index_in_split_mean_abs_d": m(sel_s_flat[torch.tensor(top_flat)].abs()),
        "alignment_free": {
            "best_match_correlation_mean": m(best_abs),
            "best_match_correlation_median": float(best_abs.median()),
            "matched_partner_selectivity_mean_abs_d": m(matched_sel),
            "frac_dense_factneurons_with_selective_partner(|d|>2)": m((matched_sel > 2).float()),
            "frac_best_corr_above_0.5": m((best_abs > 0.5).float()),
        },
    }
    out = Path(args.out) / f"{args.load}_seen"
    out.mkdir(parents=True, exist_ok=True)
    (out / "neuron_matching.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
