#!/usr/bin/env python
"""End-to-end PoC: does offloading facts to an OPTIMAL (GPT-5.6-sol) retriever
let a split LM beat its dense twin at matched size?

Stages (each independently runnable; `all` runs them in order):
  build       assemble the shared real-fact + reasoning + bed corpus, two arms
  train       train the dense twin, then the split twin (identical init/budget)
  gen-golden  generate golden knowledge for the 50 fact-QA items via GPT-5.6-sol
  eval        SPLIT@GPT-oracle vs DENSE@closed-book on fact-QA (+ reasoning)
  report      print the results table and save a figure

Usage:
  python scripts/poc_run.py --stage all --device auto
  python scripts/poc_run.py --stage build --max-facts 400
  python scripts/poc_run.py --stage gen-golden           # needs TrueFoundry creds
  python scripts/poc_run.py --stage eval --gold-oracle
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpusgen.poc_build import PoCBuildCfg, build_poc_corpus
from corpusgen.records import QAItem
from evals.oracle_scorer import (
    answer_accuracy,
    guard_vocab,
    score_items_closed_book,
    score_items_oracle,
)
from evals.scorers import score_items as generative_score_items
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok
from train.trainer import Trainer, pick_device

ROOT = Path(__file__).resolve().parent.parent
POC_DIR = ROOT / "data" / "poc"
# Checkpoints, the (paid) golden cache, results, and figure persist under
# PERSIST_DIR. In Colab, set POC_PERSIST_DIR to a mounted Google Drive folder
# so a runtime disconnect doesn't lose them. The corpus stays LOCAL (fast
# memmap reads) and is rebuilt byte-identically from the same seeds on
# reconnect, so a checkpoint resumes cleanly against it.
PERSIST_DIR = Path(os.environ.get("POC_PERSIST_DIR", str(POC_DIR)))
CORPUS_DIR = POC_DIR / "corpus"
RUNS_DIR = PERSIST_DIR / "runs"
REALFACTS_PATH = ROOT / "data" / "realfacts" / "popqa_clean.jsonl"
GOLDEN_PATH = PERSIST_DIR / "golden_knowledge.jsonl"
RESULTS_PATH = PERSIST_DIR / "poc_results.json"
FIGURE_PATH = PERSIST_DIR / "poc_figure.png"

# Model presets matched to what the team runs (train/model.py PRESETS /
# scripts/make_relational_manifest.py SCALE_SETTINGS): d160m and d360m are the
# two "protected" science scales; toy is the cheap pilot. micro_batch_size / lr
# mirror the team's per-scale settings; ctx is kept at 512 (realfact traces are
# short, and it fits a single Colab GPU) and the global batch is grad-accumulated
# to TOKENS_PER_STEP. Only the loss mask differs between the two arms.
POC_PRESETS = {
    "toy":   {"dims": {"n_layer": 4,  "n_head": 4,  "d_model": 256},  "micro_bs": 8, "lr": 1.5e-3},
    "d160m": {"dims": {"n_layer": 12, "n_head": 12, "d_model": 768},  "micro_bs": 8, "lr": 1.5e-3},
    "d360m": {"dims": {"n_layer": 20, "n_head": 16, "d_model": 1024}, "micro_bs": 4, "lr": 1.0e-3},
}
CTX = 512
TOKENS_PER_STEP = 16384  # global batch via grad accumulation
WARMUP = 300             # matches the team's warmup_steps
TRAIN_SEED = 7           # SAME for both arms -> identical initialization
EVAL_MAX_NEW = 64        # realfact traces are short
REASON_MAX_NEW = 160
ARMS = ("dense", "split")


def _model_dict(model_name: str) -> dict:
    dims = POC_PRESETS[model_name]["dims"]
    return {**dims, "ctx": CTX, "vocab_size": 50304}


def _run_dir(model_name: str, arm: str) -> Path:
    # Keyed by scale so different --model runs don't collide (and a resume never
    # loads a checkpoint of the wrong shape).
    return RUNS_DIR / model_name / arm


def _trainer_cfg(arm: str, steps: int, device: str, model_name: str,
                 ckpt_minutes: float = 10) -> dict:
    preset = POC_PRESETS[model_name]
    return {
        "run_id": f"poc_{model_name}_{arm}",
        "arm": arm,
        "model": _model_dict(model_name),
        "train_bin": str(CORPUS_DIR / arm / "train.bin"),
        # dense: no mask -> loss everywhere; split: fact values masked.
        "train_mask": str(CORPUS_DIR / arm / "train.mask.bin") if arm == "split" else None,
        "micro_batch_size": preset["micro_bs"],
        "tokens_per_step": TOKENS_PER_STEP,
        "max_steps": steps,
        "lr": preset["lr"],
        "warmup_steps": WARMUP,
        "seed": TRAIN_SEED,
        "device": device,
        "out_dir": str(_run_dir(model_name, arm)),
        "log_every": 25,
        "eval_every": 200,
        "snap_frac": 0.5,
        "ckpt_minutes": ckpt_minutes,
    }


def _load_model(run_dir: Path, device: str) -> GPT:
    import torch
    import yaml

    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = GPT(GPTConfig(**cfg["model"]))
    state = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return guard_vocab(model)


def _load_items(path: Path) -> list[QAItem]:
    with open(path) as f:
        return [QAItem(**json.loads(line)) for line in f if line.strip()]


# ---------------------------------------------------------------- stages


def stage_build(args) -> dict:
    tok = get_tok()
    cfg = PoCBuildCfg(
        realfacts_path=str(REALFACTS_PATH),
        max_facts=args.max_facts,
        n_exposures=args.exposures,
        n_igsm_docs=args.igsm_docs,
        n_deduction_docs=args.deduction_docs,
        n_bed_docs=args.bed_docs,
        n_eval_heldout=args.heldout,
        n_eval_seen=args.seen,
        n_igsm_eval=args.reason_eval,
        n_deduction_eval=args.reason_eval,
    )
    report = build_poc_corpus(cfg, tok, CORPUS_DIR)
    print(f"[build] {report['n_docs']} docs "
          f"(realfact={report['component_docs']['realfact']}, "
          f"igsm={report['component_docs']['igsm']}, "
          f"deduction={report['component_docs']['deduction']}, "
          f"bed={report['component_docs']['bed']}); "
          f"seen={report['n_seen']} heldout={report['n_heldout']}; "
          f"eval factqa={report['eval_counts']['factqa']} "
          f"(heldout {report['eval_counts']['factqa_heldout']} + "
          f"seen {report['eval_counts']['factqa_seen']})")
    for arm in ARMS:
        a = report["arms"][arm]
        print(f"[build] {arm}: {a['n_tokens']} tokens, "
              f"masked {a['masked_token_frac']:.3f}")
    return report


def stage_train(args, arms=ARMS) -> None:
    device = pick_device(args.device)
    print(f"[train] model={args.model} checkpoints -> {RUNS_DIR / args.model} "
          f"({'PERSISTED (Drive)' if os.environ.get('POC_PERSIST_DIR') else 'LOCAL/ephemeral'})")
    for arm in arms:
        cfg = _trainer_cfg(arm, args.steps, device, args.model,
                           ckpt_minutes=args.ckpt_minutes)
        trainer = Trainer(cfg)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        if trainer.ckpt_path.exists() and not args.fresh:
            trainer.load_ckpt()
            print(f"[train] {arm}: resumed at step {trainer.step}")
        print(f"[train] {arm}: {n_params/1e6:.1f}M params, "
              f"micro_bs={cfg['micro_batch_size']} tokens/step={cfg['tokens_per_step']}")
        final = trainer.train_steps()
        print(f"[train] {arm}: done at step {trainer.step} (loss_ema={final:.4f})")


def stage_gen_golden(args) -> dict:
    from evals.gpt_oracle import GatewayClient, generate_golden_knowledge

    items = _load_items(CORPUS_DIR / "eval" / "factqa.jsonl")
    client = GatewayClient(model=args.gpt_model)
    print(f"[gen-golden] model={client.model} for {len(items)} items")
    gs = generate_golden_knowledge(
        items, client, GOLDEN_PATH, ground_truth_fallback=args.ground_truth_fallback
    )
    print(f"[gen-golden] wrote {gs.n} golden values -> {GOLDEN_PATH} "
          f"(GPT fidelity vs PopQA gold = {gs.fidelity:.3f})")
    return {"fidelity": gs.fidelity, "n": gs.n}


def _reasoning_composite(model, tok, device) -> dict:
    out = {}
    for task in ("igsm", "deduction"):
        items = _load_items(CORPUS_DIR / "eval" / f"{task}.jsonl")
        rows, _ = generative_score_items(
            model, tok, items, None, device, max_new=REASON_MAX_NEW, batch_size=16
        )
        acc = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
        out[task] = {"acc": acc, "n": len(rows)}
    out["composite"] = (out["igsm"]["acc"] + out["deduction"]["acc"]) / 2
    return out


def stage_eval(args) -> dict:
    from evals.gpt_oracle import load_golden_knowledge

    device = pick_device(args.device)
    tok = get_tok()
    items = _load_items(CORPUS_DIR / "eval" / "factqa.jsonl")
    if args.limit:
        items = items[: args.limit]

    if not GOLDEN_PATH.exists():
        raise SystemExit(
            f"missing {GOLDEN_PATH}; run `--stage gen-golden` first "
            "(needs TrueFoundry creds)"
        )
    golden = load_golden_knowledge(GOLDEN_PATH)

    dense = _load_model(_run_dir(args.model, "dense"), device)
    split = _load_model(_run_dir(args.model, "split"), device)

    # Headline fact-QA: SPLIT @ GPT-oracle vs DENSE @ closed-book.
    split_oracle = score_items_oracle(split, tok, items, golden, device, max_new=EVAL_MAX_NEW)
    dense_closed = score_items_closed_book(dense, tok, items, device, max_new=EVAL_MAX_NEW)

    results: dict = {
        "n_eval": len(items),
        "factqa": {
            "split_gpt_oracle": _factqa_summary(split_oracle),
            "dense_closed_book": _factqa_summary(dense_closed),
        },
        "reasoning": {
            "dense": _reasoning_composite(dense, tok, device),
            "split": _reasoning_composite(split, tok, device),
        },
    }

    # Upper bound: SPLIT with the exact PopQA gold injected.
    if args.gold_oracle:
        gold_by_qid = {it.qid: it.meta["obj"] for it in items}
        split_gold = score_items_oracle(split, tok, items, gold_by_qid, device, max_new=EVAL_MAX_NEW)
        results["factqa"]["split_gold_oracle"] = _factqa_summary(split_gold)

    if GOLDEN_PATH.exists():
        rows = [json.loads(line) for line in open(GOLDEN_PATH) if line.strip()]
        if rows:
            results["gpt_fidelity"] = sum(r["matched_gold"] for r in rows) / len(rows)

    POC_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {RESULTS_PATH}")
    _print_table(results)
    return results


def _factqa_summary(scored: dict) -> dict:
    agg = scored["aggregates"]
    return {
        split: {
            "answer": answer_accuracy(agg, split),
            "answer_ci": agg.get(split, {}).get("answer_ci", (0.0, 0.0)),
            "no_lookup_rate": agg.get(split, {}).get("no_lookup_rate", 0.0),
            "n": agg.get(split, {}).get("n", 0),
        }
        for split in ("all", "heldout", "seen")
        if split in agg
    }


# ---------------------------------------------------------------- report


def _print_table(results: dict) -> None:
    fq = results["factqa"]
    print("\n=== Fact-QA answer accuracy (optimal retriever) ===")
    print(f"{'condition':<22}{'all':>16}{'held-out':>16}{'seen':>16}")
    labels = {
        "split_gpt_oracle": "SPLIT @ GPT-oracle",
        "dense_closed_book": "DENSE @ closed-book",
        "split_gold_oracle": "SPLIT @ gold (upper)",
    }
    for key, label in labels.items():
        if key not in fq:
            continue
        cells = []
        for split in ("all", "heldout", "seen"):
            v = fq[key].get(split)
            cells.append(f"{v['answer']*100:6.1f}% (n={v['n']})" if v else " - ")
        print(f"{label:<22}" + "".join(f"{c:>16}" for c in cells))
    if "gpt_fidelity" in results:
        print(f"\nGPT-5.6-sol fidelity vs PopQA gold: {results['gpt_fidelity']*100:.1f}%")
    r = results["reasoning"]
    print("\n=== Knowledge-free reasoning composite (supporting) ===")
    print(f"{'arm':<10}{'igsm':>10}{'deduction':>12}{'composite':>12}")
    for arm in ("dense", "split"):
        ra = r[arm]
        print(f"{arm:<10}{ra['igsm']['acc']*100:9.1f}%{ra['deduction']['acc']*100:11.1f}%"
              f"{ra['composite']*100:11.1f}%")


def stage_report(args) -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"missing {RESULTS_PATH}; run `--stage eval` first")
    results = json.load(open(RESULTS_PATH))
    _print_table(results)
    _make_figure(results, FIGURE_PATH)
    print(f"[report] figure -> {FIGURE_PATH}")


def _make_figure(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fq = results["factqa"]
    splits = ["all", "heldout", "seen"]
    # Brand-neutral, colorblind-safe pair (dataviz palette style).
    series = [
        ("DENSE @ closed-book", "dense_closed_book", "#8c8c8c"),
        ("SPLIT @ GPT-oracle", "split_gpt_oracle", "#3b6fb0"),
    ]
    if "split_gold_oracle" in fq:
        series.append(("SPLIT @ gold (upper)", "split_gold_oracle", "#b0d0f0"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = range(len(splits))
    width = 0.8 / len(series)
    for i, (label, key, color) in enumerate(series):
        vals = [fq.get(key, {}).get(s, {}).get("answer", 0.0) * 100 for s in splits]
        ax1.bar([xi + i * width for xi in x], vals, width, label=label, color=color)
    ax1.set_xticks([xi + width * (len(series) - 1) / 2 for xi in x])
    ax1.set_xticklabels(["all", "held-out", "seen"])
    ax1.set_ylabel("answer accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Fact-QA: optimal retriever vs parametric recall")
    ax1.legend(fontsize=8, frameon=False)

    r = results["reasoning"]
    arms = ["dense", "split"]
    comp = [r[a]["composite"] * 100 for a in arms]
    ax2.bar(arms, comp, color=["#8c8c8c", "#3b6fb0"], width=0.6)
    ax2.set_ylabel("composite accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("Knowledge-free reasoning (iGSM + deduction)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "build", "train", "gen-golden", "eval", "report"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--model", default="d160m", choices=list(POC_PRESETS),
                    help="model scale (matches the team: d160m/d360m; toy = pilot). "
                         "d360m needs an A100-class GPU")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--ckpt-minutes", type=float, default=10,
                    help="wall-clock checkpoint cadence (lower = less lost on a "
                         "Colab disconnect); a final ckpt always saves at the end")
    ap.add_argument("--fresh", action="store_true", help="ignore existing ckpt")
    # corpus knobs
    ap.add_argument("--max-facts", type=int, default=None,
                    help="cap total PopQA facts before the split (default: all)")
    ap.add_argument("--exposures", type=int, default=6)
    ap.add_argument("--igsm-docs", type=int, default=1500)
    ap.add_argument("--deduction-docs", type=int, default=1500)
    ap.add_argument("--bed-docs", type=int, default=800)
    ap.add_argument("--heldout", type=int, default=25, help="held-out fact-QA eval items")
    ap.add_argument("--seen", type=int, default=25, help="seen fact-QA eval items")
    ap.add_argument("--reason-eval", type=int, default=50,
                    help="igsm/deduction eval items each")
    # eval knobs
    ap.add_argument("--limit", type=int, default=0, help="cap fact-QA eval items")
    ap.add_argument("--gold-oracle", action="store_true",
                    help="also report the SPLIT upper bound with PopQA gold injected")
    # gpt knobs
    ap.add_argument("--gpt-model", default=None, help="override gateway model path")
    ap.add_argument("--ground-truth-fallback", action="store_true",
                    help="inject PopQA gold when GPT disagrees (exact optimal oracle)")
    args = ap.parse_args()

    print(f"[poc] stage={args.stage} device={pick_device(args.device)}")
    if args.stage == "build":
        stage_build(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "gen-golden":
        stage_gen_golden(args)
    elif args.stage == "eval":
        stage_eval(args)
    elif args.stage == "report":
        stage_report(args)
    else:  # all
        stage_build(args)
        stage_train(args)
        stage_gen_golden(args)
        stage_eval(args)
        stage_report(args)


if __name__ == "__main__":
    main()
