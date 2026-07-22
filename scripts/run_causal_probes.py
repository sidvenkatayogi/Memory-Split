#!/usr/bin/env python
"""Runnable-today probes on FROZEN (seen) entities: fact_ledger, double_dissociation
(recall side), attention_heads (fact-use side).

- fact_ledger: reconcile recall-bits (generative recall on trained people) with the
  linear probe-bits (from the mechanism report) into one storage statement.
- double_dissociation (recall side): ablate the dense arm's top fact-neurons and show
  recall collapses, while ablating the same number of RANDOM neurons barely dents it —
  causal evidence those units carry stored facts. (The "reasoning unchanged" half is
  gated until reasoning clears chance.)
- attention_heads (fact-use): ablate each layer's attention heads and measure the rise
  in fact-value NLL — which layers' attention is needed to produce a stored fact.

All on the frozen trained people (`evals/frozen.py`); split/dense as noted per probe.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from evals.frozen import records_from_recall_jsonl
from evals.interp import ablate_mlp_neurons, neuron_ablation_effect
from evals.mechanism import capture_last_token, cohens_d, top_neurons, _DEFAULT_CONTROL
from evals.perplexity import fact_value_nll
from evals.recall import bits_in_weights, recall_accuracy
from probe.attention import head_ablation_effect
from probe.ledger import fact_info_ledger
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device

POOL_SIZES = {"birth_date": 27_759.0, "birth_city": 200.0, "university": 300.0,
              "major": 100.0, "employer": 263.0, "current_city": 200.0}


def load_model(run_dir, ckpt, device):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    m = GPT(mcfg)
    m.load_state_dict(torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)["model"])
    return m.to(device).eval(), cfg, mcfg


def recall_per_attr(model, tok, records, device, n, mode, organizer, seed, max_new=32):
    probes = bios.recall_probes(records, min(n, len(records)), seed)
    return recall_accuracy(model, tok, probes, mode, organizer, device, max_new=max_new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--ckpt", default="snapshots/step0006100.pt")
    ap.add_argument("--records-jsonl", required=True)
    ap.add_argument("--do", default="ledger,dissociation,attention")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n-recall", type=int, default=150)
    ap.add_argument("--out", default="outputs/mechinterp")
    args = ap.parse_args()
    do = set(args.do.split(","))
    L = args.load
    device = pick_device("auto"); tok = get_tok()
    base = Path("outputs/_local_probe")
    dense_dir = base / f"d160m_dense_{L}_s0"
    split_dir = base / f"d160m_split_{L}_s0"
    records = records_from_recall_jsonl(args.records_jsonl)
    out_root = Path(args.out)

    # ---------------- fact_ledger (dense) ----------------
    if "ledger" in do:
        dm, cfg, _ = load_model(dense_dir, args.ckpt, device)
        r = recall_per_attr(dm, tok, records, device, args.n_recall, "closed", None, seed=1)
        recall_bits = bits_in_weights(r["per_attribute"], cfg["n_entities"], POOL_SIZES)
        probe_bits = None
        mj = out_root / "mechanism" / f"{L}_seen" / "mechanism.json"
        if mj.exists():
            probe_bits = json.loads(mj.read_text()).get("probe", {}).get("dense", {}).get("bits", {}).get("total_bits")
        ledger = fact_info_ledger(recall_bits=recall_bits, probe_bits=probe_bits,
                                  mc_recall_acc=None, n_entities=cfg["n_entities"])
        ledger.update({"load": L, "arm": "dense", "closed_recall_overall": r["overall"],
                       "recall_per_attribute": r["per_attribute"],
                       "note": "recall-bits from generative recall on trained people; "
                               "probe-bits from mechanism (underpowered); MC omitted."})
        d = out_root / "fact_ledger"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{L}_ledger.json").write_text(json.dumps(ledger, indent=2))
        print(f"[ledger {L}] recall_overall={r['overall']:.4f} recall_bits={recall_bits:,.0f} "
              f"probe_bits={probe_bits}")
        del dm

    # ---------------- double_dissociation: recall side (dense) ----------------
    if "dissociation" in do:
        dm, cfg, mcfg = load_model(dense_dir, args.ckpt, device)
        # identify dense fact-neurons: cohens_d(fact vs neutral control) on MLP units
        fact_prompts = [f"{r.name}'s {bios.RELATION_PHRASES[a]} is"
                        for r in records[:300] for a in bios.ATTRIBUTES] \
            if hasattr(bios, "ATTRIBUTES") else \
            [f"{r.name}'s {bios.RELATION_PHRASES[a]} is"
             for r in records[:300] for a in bios.RELATION_PHRASES]
        fact_mlp = capture_last_token(dm, tok, fact_prompts, device)["mlp"]
        ctrl_mlp = capture_last_token(dm, tok, list(_DEFAULT_CONTROL), device)["mlp"]
        sel = cohens_d(fact_mlp, ctrl_mlp)                    # [L, H]
        fact_neurons = [(l, n) for (l, n, _d) in top_neurons(sel, args.k)]
        H = sel.size(-1); n_layer = sel.size(0)
        rng = random.Random(0)
        rand_neurons = [(rng.randrange(n_layer), rng.randrange(H)) for _ in range(args.k)]

        probes = bios.recall_probes(records, min(args.n_recall, len(records)), seed=2)

        def scorer(m):
            return recall_accuracy(m, tok, probes, "closed", None, device, max_new=32)["overall"]

        fact_eff = neuron_ablation_effect(dm, fact_neurons, scorer)
        rand_eff = neuron_ablation_effect(dm, rand_neurons, scorer)
        res = {"load": L, "arm": "dense", "k": args.k, "n_recall_probes": len(probes),
               "baseline_recall": fact_eff["baseline"],
               "recall_after_ablating_fact_neurons": fact_eff["ablated"],
               "recall_drop_fact_neurons": fact_eff["delta"],
               "recall_after_ablating_random_neurons": rand_eff["ablated"],
               "recall_drop_random_neurons": rand_eff["delta"],
               "top_fact_neurons": fact_neurons[:15]}
        d = out_root / "double_dissociation"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{L}_dissociation.json").write_text(json.dumps(res, indent=2))
        print(f"[dissociation {L}] baseline={fact_eff['baseline']:.3f} "
              f"fact-ablated={fact_eff['ablated']:.3f} (drop {fact_eff['delta']:.3f}) | "
              f"random-ablated={rand_eff['ablated']:.3f} (drop {rand_eff['delta']:.3f})")
        del dm

    # ---------------- attention_heads: fact-use side (dense) ----------------
    if "attention" in do:
        dm, cfg, mcfg = load_model(dense_dir, args.ckpt, device)
        n_head = mcfg.n_head; n_layer = mcfg.n_layer
        sample = records[:200]

        def nll_scorer(m):
            return -fact_value_nll(m, tok, sample, device)["nll_per_token"]

        base_nll = -nll_scorer(dm)
        per_layer = {}
        for layer in range(n_layer):
            heads = [(layer, h) for h in range(n_head)]
            eff = head_ablation_effect(dm, heads, nll_scorer)
            per_layer[str(layer)] = round(-eff["ablated"] - base_nll, 4)  # ΔNLL when layer's heads removed
        res = {"load": L, "arm": "dense", "baseline_fact_value_nll": round(base_nll, 4),
               "delta_nll_when_layer_heads_ablated": per_layer,
               "note": "positive ΔNLL = that layer's attention heads are needed to produce "
                       "the stored fact value (fact-use). Reasoning-side head tracing is gated."}
        d = out_root / "attention_heads"; d.mkdir(parents=True, exist_ok=True)
        (d / f"{L}_attention.json").write_text(json.dumps(res, indent=2))
        top = sorted(per_layer.items(), key=lambda kv: -kv[1])[:3]
        print(f"[attention {L}] base_nll={base_nll:.3f} top layers by ΔNLL: {top}")
        del dm


if __name__ == "__main__":
    main()
