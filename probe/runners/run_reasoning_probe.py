#!/usr/bin/env python
"""NR-7 — latent-reasoning probe over one or two arms' checkpoints.

Readout-vs-learning diagnosis: is reasoning at chance because the model never
computes the answer, or computes-but-can't-read-out? Per-layer answer-class probe
+ logit lens at end-of-prompt (copy-free) vs after-gold-CoT (execution control),
with the greedy-accuracy contrast. Pure forward passes; no retraining.

Usage:
  python scripts/run_reasoning_probe.py --run outputs/d160m_dense_n200k_s0 \
      [--split-run outputs/d160m_split_n200k_s0] [--ckpt ckpt.pt] \
      [--tasks igsm deduction] [--limit 500] [--out outputs/reasoning_probe]

Needs checkpoints (cluster / outputs/pulled/). Writes <out>/reasoning_probe.json
+ per-layer figures. See replication/specs/nr7-latent-reasoning-probe.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen.records import QAItem
from evals.reasoning_probe import reasoning_readout_report
from evals.scorers import score_items
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def _load_model(run_dir: Path, ckpt_rel: str, device: str) -> GPT:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model_cfg = (
        PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    )
    model = GPT(model_cfg)
    state = torch.load(run_dir / ckpt_rel, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _load_items(path: Path, limit: int) -> list[QAItem]:
    items = [QAItem(**json.loads(l)) for l in open(path)]
    return items[:limit] if limit else items


def _figure(report: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    plotted = False
    for arm, arm_rep in report.items():
        for site, key in (("end-of-prompt", "probe_end_of_prompt"),
                          ("after-gold-CoT", "probe_after_gold_cot")):
            pr = arm_rep.get(key)
            if not pr:
                continue
            layers = sorted(pr["acc_by_layer"])
            ax.plot(layers, [pr["acc_by_layer"][l] for l in layers],
                    marker="o", label=f"{arm} · {site}")
            plotted = True
            chance = pr["chance"]
    if plotted:
        ax.axhline(chance, ls="--", c="grey", lw=1, label="chance")
        ax.set_xlabel("layer"); ax.set_ylabel("answer-probe accuracy (held-out)")
        ax.set_title("NR-7 latent-reasoning probe: answer decodability by layer")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="an arm's run dir (dense by convention)")
    ap.add_argument("--split-run", default=None, help="optional second arm")
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--tasks", nargs="+", default=["igsm", "deduction"])
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="outputs/reasoning_probe")
    args = ap.parse_args()

    device = pick_device("auto")
    tok = get_tok()
    arms = {"dense": Path(args.run)}
    if args.split_run:
        arms["split"] = Path(args.split_run)

    result: dict = {}
    for arm, run_dir in arms.items():
        cfg = yaml.safe_load(open(run_dir / "config.yaml"))
        data_dir = Path(cfg["data_dir"])
        model = _load_model(run_dir, args.ckpt, device)
        result[arm] = {}
        for task in args.tasks:
            path = data_dir / "eval" / f"{task}.jsonl"
            if not path.exists():
                continue
            items = _load_items(path, args.limit)
            rows, _ = score_items(model, tok, items, None, device, batch_size=args.batch_size)
            greedy = sum(r["correct"] for r in rows) / max(1, len(rows))
            result[arm][task] = reasoning_readout_report(
                model, tok, items, device, greedy_acc=greedy, batch_size=args.batch_size,
            )
            v = result[arm][task]["verdict"]
            print(f"[{arm}/{task}] greedy={greedy:.4f} -> {v}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reasoning_probe.json").write_text(json.dumps(result, indent=2))
    # one figure per task (arms overlaid)
    for task in args.tasks:
        per_task = {arm: result[arm][task] for arm in result if task in result[arm]}
        if per_task:
            _figure(per_task, out / f"probe_{task}.png")
    print(f"reasoning-probe -> {out / 'reasoning_probe.json'}")


if __name__ == "__main__":
    main()
