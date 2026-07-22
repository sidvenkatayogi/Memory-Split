#!/usr/bin/env python
"""Runnable representational/weight probes (probe.*) for one dense+split pair.

Covers the NO-deferred-dep, non-gated probe.* set: geometry_cka, weight_spectral,
superposition, value_injection (split), fact_weight_attribution. Writes one JSON
per probe into outputs/mechinterp/<probe>/<load>.json. Read-only wrt training.

Usage:
  python scripts/run_probe_suite.py --dense outputs/<dense> --split outputs/<split> \
      --load n200k [--ckpt snapshots/step0006100.pt] [--n-records 300] \
      [--out-root outputs/mechinterp]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen import bios
from corpusgen.bios import RELATION_PHRASES
from evals.mechanism import capture_last_token
from probe.geometry import cross_arm_cka, weight_spectral
from probe.weights import fact_superposition, fact_weight_attribution
from probe.splice import value_injection_profile
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _load(run_dir: Path, ckpt_rel: str, device: str):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    mcfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    m = GPT(mcfg)
    state = torch.load(run_dir / ckpt_rel, map_location="cpu", weights_only=False)
    m.load_state_dict(state["model"])
    return m.to(device).eval(), state["model"], cfg


def _write(out_root: Path, probe: str, load: str, obj: dict, tag: str = "") -> None:
    d = out_root / probe
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{load}{tag}.json").write_text(json.dumps(obj, indent=2))
    print(f"  -> {probe}/{load}{tag}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--load", required=True, help="load tag, e.g. n200k")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--n-records", type=int, default=300)
    ap.add_argument("--out-root", default="outputs/mechinterp")
    ap.add_argument("--records-jsonl", default=None,
                    help="load FROZEN trained entities (SEEN) from this recall.jsonl "
                         "instead of regenerating (UNSEEN, drifted generator).")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. '_seen' -> <load>_seen.json")
    args = ap.parse_args()

    device = pick_device("auto")
    tok = get_tok()
    out_root = Path(args.out_root)
    dense_dir, split_dir = Path(args.dense), Path(args.split)

    dm, dsd, dcfg = _load(dense_dir, args.ckpt, device)
    sm, ssd, scfg = _load(split_dir, args.ckpt, device)
    n_ent = dcfg["n_entities"]; seed = dcfg.get("seed", 0)
    if args.records_jsonl:
        from evals.frozen import records_from_recall_jsonl
        recs = records_from_recall_jsonl(args.records_jsonl)[: args.n_records]
        entity_source = "frozen_seen"
    else:
        recs = bios.generate_records(min(n_ent, max(args.n_records, 50)) + 0, seed)[: args.n_records]
        entity_source = "regenerated_unseen"
    print(f"[probe_suite] load={args.load} device={device} records={len(recs)} "
          f"ckpt={args.ckpt} entity_source={entity_source} tag={args.tag!r}")

    # shared prompts for CKA / superposition: "{name}'s {relation} is"
    attrs = tuple(RELATION_PHRASES)
    prompts = [f"{r.name}'s {RELATION_PHRASES[a]} is" for r in recs for a in attrs]

    # --- geometry_cka: per-layer dense<->split representational similarity ---
    try:
        A = capture_last_token(dm, tok, prompts, device)["resid"]
        B = capture_last_token(sm, tok, prompts, device)["resid"]
        cka = cross_arm_cka(A, B)
        _write(out_root, "geometry_cka", args.load,
               {"load": args.load, "ckpt": args.ckpt, "entity_source": entity_source,
                "n_prompts": len(prompts),
                "cka_by_layer": {str(k): v for k, v in cka.items()},
                "min_layer": min(cka, key=cka.get), "min_cka": min(cka.values())},
               tag=args.tag)
        # --- superposition: PR of last-layer fact activations, per arm ---
        _write(out_root, "superposition", args.load,
               {"load": args.load, "ckpt": args.ckpt, "entity_source": entity_source,
                "n_prompts": len(prompts),
                "dense": fact_superposition(A[:, -1, :]),
                "split": fact_superposition(B[:, -1, :])},
               tag=args.tag)
    except Exception as e:  # noqa: BLE001
        print(f"  geometry_cka/superposition FAIL: {e}")

    # --- weight_spectral: MLP eff-rank/norm per arm (from state_dict) ---
    try:
        _write(out_root, "weight_spectral", args.load,
               {"load": args.load, "dense": {str(k): v for k, v in weight_spectral(dsd).items()},
                "split": {str(k): v for k, v in weight_spectral(ssd).items()}},
               tag=args.tag)
    except Exception as e:  # noqa: BLE001
        print(f"  weight_spectral FAIL: {e}")

    # --- value_injection: where the value enters the stream (split arm) ---
    try:
        vi = value_injection_profile(sm, tok, recs[: min(len(recs), 200)], device)
        vi["profile"] = {str(k): v for k, v in vi.get("profile", {}).items()}
        _write(out_root, "value_injection", args.load,
               {"load": args.load, "entity_source": entity_source, **vi}, tag=args.tag)
    except Exception as e:  # noqa: BLE001
        print(f"  value_injection FAIL: {e}")

    # --- fact_weight_attribution: which layers store a fact, dense vs split ---
    try:
        sample = recs[: min(len(recs), 20)]
        def attr_sum(model):
            agg: dict[int, float] = {}
            for r in sample:
                a = fact_weight_attribution(model, tok, f"{r.name}'s major is",
                                            r.attrs["major"], device)
                for l, v in a.items():
                    agg[l] = agg.get(l, 0.0) + v
            return {str(l): v / len(sample) for l, v in sorted(agg.items())}
        _write(out_root, "fact_weight_attribution", args.load,
               {"load": args.load, "ckpt": args.ckpt, "entity_source": entity_source,
                "n_facts": len(sample),
                "dense": attr_sum(dm), "split": attr_sum(sm)}, tag=args.tag)
    except Exception as e:  # noqa: BLE001
        print(f"  fact_weight_attribution FAIL: {e}")

    print("[probe_suite] done")


if __name__ == "__main__":
    main()
