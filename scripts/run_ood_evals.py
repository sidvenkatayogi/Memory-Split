#!/usr/bin/env python
"""E1 — out-of-distribution reasoning generalization (train-easy -> eval-hard).

Both arms train on the SAME in-distribution reasoning band, so OOD accuracy
measures which arm learned the *procedure* (generalizes) vs memorized templates —
H1 measured more rigorously (iGSM op/length extrapolation, Physics 2.1;
RuleTaker/ProofWriter depth). The spec asks for this (iGSM op 9-12, deduction
depth 5-6) but the corpus builder only emits the in-distribution eval.

OOD bands are disjoint from the training band, so their structure hashes cannot
collide with training — no exclusion set is needed. Deterministic: eval items are
regenerated from a fixed seed.

Usage:
  python scripts/run_ood_evals.py --run outputs/d160m_dense_n200k_s0 \
      [--ckpt snapshots/stepNNNN.pt] [--n-items 2000] [--igsm-ood 5 8] \
      [--deduction-ood 5 6] [--batch-size 16] [--continuous]

Writes <run>/evals[/ckpt-<tag>]/ood_summary.json with overall + per-op / per-depth
accuracy (the generalization curve).
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from evals.scorers import score_items
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _by_key(rows, key):
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        k = r["meta"].get(key)
        agg[k][0] += int(bool(r["correct"]))
        agg[k][1] += 1
    return {str(k): {"acc": c / n, "n": n} for k, (c, n) in sorted(agg.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-items", type=int, default=2000)
    ap.add_argument("--igsm-ood", type=int, nargs=2, default=None,
                    help="op_lo op_hi for OOD iGSM (default: train_hi+1 .. train_hi+4)")
    ap.add_argument("--deduction-ood", type=int, nargs=2, default=None,
                    help="depth_lo depth_hi for OOD deduction (default: train_hi+1 .. +2)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--continuous", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    seed = cfg.get("seed", 0)
    device = pick_device("auto")
    tok = get_tok()

    model_cfg = PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    model = GPT(model_cfg)
    state = torch.load(run_dir / (args.ckpt or "ckpt.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    igsm = importlib.import_module("corpusgen.igsm_lite")
    ded = importlib.import_module("corpusgen.deduction")
    train_op = tuple(cfg.get("igsm_op", (1, 4)))
    train_depth = tuple(cfg.get("deduction_depth", (1, 2)))
    igsm_band = tuple(args.igsm_ood) if args.igsm_ood else (train_op[1] + 1, train_op[1] + 4)
    ded_band = tuple(args.deduction_ood) if args.deduction_ood else (train_depth[1] + 1, train_depth[1] + 2)

    # OOD bands are disjoint from training bands => no hash exclusion required.
    igsm_items = igsm.generate_igsm_eval(args.n_items, igsm_band[0], igsm_band[1],
                                         seed * 1000 + 4444, set())
    ded_items = ded.generate_deduction_eval(args.n_items, ded_band[0], ded_band[1],
                                            seed * 1000 + 5555, set())

    summary = {"run": run_dir.name, "arm": cfg.get("arm"),
               "train_op": list(train_op), "igsm_ood_band": list(igsm_band),
               "train_depth": list(train_depth), "deduction_ood_band": list(ded_band)}

    for task, items, key in [("igsm", igsm_items, "op"), ("deduction", ded_items, "depth")]:
        rows, _ = score_items(model, tok, items, None, device, batch_size=args.batch_size)
        acc = sum(r["correct"] for r in rows) / max(1, len(rows))
        summary[f"{task}_ood"] = {
            "overall_acc": acc, "n": len(rows),
            f"by_{key}": _by_key(rows, key),
        }
        print(f"{task} OOD {igsm_band if task=='igsm' else ded_band}: acc={acc:.4f} "
              f"by_{key}=" + json.dumps(summary[f'{task}_ood'][f'by_{key}']))

    from evals.scorers import parse_answer  # noqa: F401 (kept for parity/debug)

    out_root = run_dir / "evals"
    if args.ckpt and Path(args.ckpt).name != "ckpt.pt":
        out_root = out_root / f"ckpt-{Path(args.ckpt).stem}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "ood_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
