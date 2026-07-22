#!/usr/bin/env python
"""NR-5 — fact-side mechanism / probing over dense+split checkpoints.

Usage:
  python scripts/run_mechanism.py \
      --dense outputs/d160m_dense_n200k_s0 \
      --split outputs/d160m_split_n200k_s0 \
      [--ckpt ckpt.pt] [--n-entities 2000] [--probe-layer -1] [--k 64] \
      [--out outputs/mechanism]

Loads both arms' weights and the corpus records, runs the fact-side mechanism
report (memorization-neuron localization + linear-probe bits), and writes
<out>/mechanism.json + a small figure. Read-only wrt training. See
replication/specs/nr5-fact-mechanism.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen.bios import generate_records
from evals.mechanism import fact_mechanism_report
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from train.trainer import pick_device


def load_model(run_dir: Path, ckpt_rel: str, device: str) -> GPT:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model_cfg = (
        PRESETS[cfg["model"]] if isinstance(cfg["model"], str) else GPTConfig(**cfg["model"])
    )
    model = GPT(model_cfg)
    state = torch.load(run_dir / ckpt_rel, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _n_entities(run_dir: Path, override: int | None) -> tuple[int, int]:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    total = int(cfg["n_entities"])
    seed = int(cfg.get("data_seed", cfg.get("seed", 0)))
    return (min(total, override) if override else total), seed


def _figure(report: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe = report.get("probe", {})
    arms = [a for a in ("dense", "split") if a in probe]
    if not arms:
        return
    attrs = sorted({a for arm in arms for a in probe[arm]["probe_acc"]})
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    width = 0.38
    for j, arm in enumerate(arms):
        vals = [probe[arm]["probe_acc"].get(a, 0.0) for a in attrs]
        ax.bar([i + j * width for i in range(len(attrs))], vals, width,
               label=f"{arm} (probe acc)")
    ax.set_xticks([i + width / 2 for i in range(len(attrs))])
    ax.set_xticklabels(attrs, rotation=20, ha="right")
    ax.set_ylabel("linear-probe accuracy (held-out)")
    ax.set_title("NR-5 fact-probe recovery: dense vs split")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", default="ckpt.pt")
    ap.add_argument("--n-entities", type=int, default=2000,
                    help="cap entities used for probes (0 = all)")
    ap.add_argument("--probe-layer", type=int, default=-1)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--out", default="outputs/mechanism")
    ap.add_argument("--records-jsonl", default=None,
                    help="load FROZEN trained entities (SEEN) from this recall.jsonl "
                         "instead of regenerating (UNSEEN, drifted generator).")
    args = ap.parse_args()

    device = pick_device("auto")
    tok = get_tok()
    dense_dir, split_dir = Path(args.dense), Path(args.split)

    n_ent, seed = _n_entities(dense_dir, args.n_entities or None)
    if args.records_jsonl:
        from evals.frozen import records_from_recall_jsonl
        records = records_from_recall_jsonl(args.records_jsonl)[:n_ent]
        entity_source = "frozen_seen"
    else:
        records = generate_records(n_ent, seed)
        entity_source = "regenerated_unseen"

    models = {
        "dense": load_model(dense_dir, args.ckpt, device),
        "split": load_model(split_dir, args.ckpt, device),
    }
    report = fact_mechanism_report(
        models, tok, records, device,
        probe_layer=args.probe_layer, k=args.k,
    )
    report["entity_source"] = entity_source
    report["n_entities_probed"] = len(records)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "mechanism.json").write_text(json.dumps(report, indent=2))
    _figure(report, out / "probe_acc.png")
    print(json.dumps(report, indent=2))
    print(f"mechanism -> {out / 'mechanism.json'}, {out / 'probe_acc.png'}")


if __name__ == "__main__":
    main()
