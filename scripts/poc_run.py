#!/usr/bin/env python
"""End-to-end PoC: facts in context (no DB retrieval). Does a model that offloads
facts to context (SPLIT — fact values loss-masked so it never memorizes them)
beat a dense twin that stores them in weights, at matched size?

One corpus, two arms, one toggle: SPLIT masks the loss on fact VALUES in the
context; DENSE gets loss everywhere. Both tasks are open-book (fact-QA + a
reason-over-facts comparison). Evaluated in three fair conditions:
  - DENSE @ closed-book   (no context; recall from weights)
  - DENSE + context       (facts given in context)
  - SPLIT + context       (facts given in context; its trained mode)

Stages: build -> train (both arms) -> eval -> report.
  python scripts/poc_run.py --stage all --model d160m --steps 4000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from corpusgen.poc_build import PoCBuildCfg, build_poc_corpus
from evals import context_eval
from train.model import GPT, GPTConfig
from train.tokenizer import get_tok
from train.trainer import Trainer, pick_device

ROOT = Path(__file__).resolve().parent.parent
POC_DIR = ROOT / "data" / "poc"
PERSIST_DIR = Path(os.environ.get("POC_PERSIST_DIR", str(POC_DIR)))
CORPUS_DIR = POC_DIR / "corpus"            # local (rebuilt deterministically)
RUNS_DIR = PERSIST_DIR / "runs"            # checkpoints (persist to Drive)
REALFACTS_PATH = ROOT / "data" / "realfacts" / "popqa_clean.jsonl"
RESULTS_PATH = PERSIST_DIR / "poc_results.json"
FIGURE_PATH = PERSIST_DIR / "poc_figure.png"

# Model presets matched to the team (train/model.py). Only the loss mask differs
# between arms. ctx 512 fits a single GPU; global batch via grad accumulation.
POC_PRESETS = {
    "toy":   {"dims": {"n_layer": 4,  "n_head": 4,  "d_model": 256},  "micro_bs": 8, "lr": 1.5e-3},
    "d160m": {"dims": {"n_layer": 12, "n_head": 12, "d_model": 768},  "micro_bs": 8, "lr": 1.5e-3},
    "d360m": {"dims": {"n_layer": 20, "n_head": 16, "d_model": 1024}, "micro_bs": 4, "lr": 1.0e-3},
}
CTX = 512
TOKENS_PER_STEP = 16384
WARMUP = 300
TRAIN_SEED = 7          # SAME for both arms -> identical init
MAX_NEW = 48
ARMS = ("dense", "split")


def _model_dict(name: str) -> dict:
    return {**POC_PRESETS[name]["dims"], "ctx": CTX, "vocab_size": 50304}


def _run_dir(model_name: str, arm: str) -> Path:
    # '_incontext' namespace so these never collide with (or wrongly reuse) the
    # old DB-design checkpoints.
    return RUNS_DIR / f"{model_name}_incontext" / arm


def _trainer_cfg(arm, steps, device, model_name, ckpt_minutes=10) -> dict:
    p = POC_PRESETS[model_name]
    return {
        "run_id": f"poc_{model_name}_incontext_{arm}",
        "arm": arm,
        "model": _model_dict(model_name),
        "train_bin": str(CORPUS_DIR / arm / "train.bin"),
        "train_mask": str(CORPUS_DIR / arm / "train.mask.bin") if arm == "split" else None,
        "micro_batch_size": p["micro_bs"],
        "tokens_per_step": TOKENS_PER_STEP,
        "max_steps": steps,
        "lr": p["lr"],
        "warmup_steps": WARMUP,
        "seed": TRAIN_SEED,
        "device": device,
        "out_dir": str(_run_dir(model_name, arm)),
        "log_every": 25, "eval_every": 200, "snap_frac": 0.5,
        "ckpt_minutes": ckpt_minutes,
    }


def _trained_step(run_dir: Path) -> int:
    log = Path(run_dir) / "log.jsonl"
    if not log.exists():
        return -1
    last = ""
    with open(log) as f:
        for line in f:
            if line.strip():
                last = line
    try:
        return int(json.loads(last).get("step", -1)) if last else -1
    except (ValueError, json.JSONDecodeError):
        return -1


def _load_model(run_dir: Path, device: str) -> GPT:
    import torch
    import yaml
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    model = GPT(GPTConfig(**cfg["model"]))
    state = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def _load_items(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- stages


def stage_build(args) -> dict:
    tok = get_tok()
    cfg = PoCBuildCfg(
        realfacts_path=str(REALFACTS_PATH), max_facts=args.max_facts,
        n_exposures=args.exposures, n_reason_train=args.reason_train,
        n_bed_docs=args.bed_docs, n_factqa_heldout=args.heldout,
        n_factqa_seen=args.seen, n_reason_eval=args.reason_eval,
    )
    r = build_poc_corpus(cfg, tok, CORPUS_DIR)
    print(f"[build] docs: factqa={r['component_docs']['factqa']} "
          f"reason={r['component_docs']['reason']} bed={r['component_docs']['bed']}; "
          f"seen={r['n_seen']} heldout={r['n_heldout']}; "
          f"eval factqa={r['eval_counts']['factqa']} reason={r['eval_counts']['reason']}")
    for arm in ARMS:
        a = r["arms"][arm]
        print(f"[build] {arm}: {a['n_tokens']} tokens, masked {a['masked_token_frac']:.3f}")
    return r


def stage_train(args, arms=ARMS) -> None:
    device = pick_device(args.device)
    print(f"[train] model={args.model} checkpoints -> {RUNS_DIR / (args.model + '_incontext')} "
          f"({'PERSISTED (Drive)' if os.environ.get('POC_PERSIST_DIR') else 'LOCAL/ephemeral'})")
    for arm in arms:
        run_dir = _run_dir(args.model, arm)
        done = _trained_step(run_dir)
        if (run_dir / "ckpt.pt").exists() and not args.fresh and done >= args.steps:
            print(f"[train] {arm}: found trained checkpoint at step {done} "
                  f"(>= {args.steps}); reusing it, skipping (--fresh to retrain)")
            continue
        cfg = _trainer_cfg(arm, args.steps, device, args.model, ckpt_minutes=args.ckpt_minutes)
        trainer = Trainer(cfg)
        n_params = sum(p.numel() for p in trainer.model.parameters())
        if trainer.ckpt_path.exists() and not args.fresh:
            trainer.load_ckpt()
            print(f"[train] {arm}: resumed at step {trainer.step}")
        print(f"[train] {arm}: {n_params/1e6:.1f}M params, "
              f"micro_bs={cfg['micro_batch_size']} tokens/step={cfg['tokens_per_step']}")
        final = trainer.train_steps()
        print(f"[train] {arm}: done at step {trainer.step} (loss_ema={final:.4f})")


def _majority_baseline(reason_items) -> float:
    n = len(reason_items)
    if not n:
        return 0.0
    n_yes = sum(1 for it in reason_items if it["answer"] == "yes")
    return max(n_yes, n - n_yes) / n


def stage_eval(args) -> dict:
    device = pick_device(args.device)
    tok = get_tok()
    factqa = _load_items(CORPUS_DIR / "eval" / "factqa.jsonl")
    reason = _load_items(CORPUS_DIR / "eval" / "reason.jsonl")
    if args.limit:
        factqa = factqa[: args.limit]
        reason = reason[: args.limit]
    items = factqa + reason

    client = None
    if args.judge:
        if os.environ.get("OPENAI_API_KEY"):
            from evals.gpt_oracle import GatewayClient
            client = GatewayClient(model=args.gpt_model)
        else:
            print("[eval] --judge set but no OPENAI_API_KEY; using string-match")

    dense = _load_model(_run_dir(args.model, "dense"), device)
    split = _load_model(_run_dir(args.model, "split"), device)

    conditions = {
        "dense_closed": context_eval.score(dense, tok, items, device, context=False,
                                            client=client, max_new=MAX_NEW),
        "dense_context": context_eval.score(dense, tok, items, device, context=True,
                                             client=client, max_new=MAX_NEW),
        "split_context": context_eval.score(split, tok, items, device, context=True,
                                             client=client, max_new=MAX_NEW),
    }
    results = {
        "model": args.model,
        "n_factqa": len(factqa), "n_reason": len(reason),
        "reason_majority_baseline": _majority_baseline(reason),
        "gpt_judge": client is not None,
        "conditions": conditions,
    }
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] wrote {RESULTS_PATH}")
    _print_table(results)
    return results


# ---------------------------------------------------------------- report

_COND_LABELS = {"dense_closed": "DENSE @ closed-book",
                "dense_context": "DENSE + context",
                "split_context": "SPLIT + context"}


def _print_table(results: dict) -> None:
    conds = results["conditions"]
    grade = "judge-graded" if results.get("gpt_judge") else "string-match"
    print(f"\n=== Fact-QA answer accuracy ({grade}) ===")
    print(f"{'condition':<22}{'all':>14}{'held-out':>14}{'seen':>14}")
    for key, label in _COND_LABELS.items():
        fq = conds.get(key, {}).get("factqa", {})
        cells = []
        for s in ("all", "heldout", "seen"):
            v = fq.get(s)
            cells.append(f"{v['acc']*100:6.1f}% (n={v['n']})" if v else " - ")
        print(f"{label:<22}" + "".join(f"{c:>14}" for c in cells))

    print(f"\n=== Reason-over-facts (yes/no; combine 2 facts) ===")
    print(f"(majority-class baseline {results.get('reason_majority_baseline',0)*100:.0f}%)")
    print(f"{'condition':<22}{'accuracy':>12}")
    for key, label in _COND_LABELS.items():
        rv = conds.get(key, {}).get("reason", {}).get("all")
        print(f"{label:<22}{(rv['acc']*100 if rv else 0):11.1f}%")


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

    conds = list(_COND_LABELS)
    colors = {"dense_closed": "#8c8c8c", "dense_context": "#c48a2c",
              "split_context": "#3b6fb0"}
    c = results["conditions"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # fact-QA: all/held-out/seen
    splits = ["all", "heldout", "seen"]
    w = 0.8 / len(conds)
    for i, k in enumerate(conds):
        fq = c.get(k, {}).get("factqa", {})
        vals = [fq.get(s, {}).get("acc", 0.0) * 100 for s in splits]
        ax1.bar([x + i * w for x in range(len(splits))], vals, w,
                label=_COND_LABELS[k], color=colors[k])
    ax1.set_xticks([x + w for x in range(len(splits))])
    ax1.set_xticklabels(["all", "held-out", "seen"])
    ax1.set_ylabel("answer accuracy (%)"); ax1.set_ylim(0, 100)
    ax1.set_title("Fact-QA (in-context)")
    ax1.legend(fontsize=8, frameon=False)
    # reason: one bar per condition + baseline
    rvals = [c.get(k, {}).get("reason", {}).get("all", {}).get("acc", 0.0) * 100 for k in conds]
    ax2.bar(range(len(conds)), rvals, 0.6, color=[colors[k] for k in conds])
    ax2.axhline(results.get("reason_majority_baseline", 0.5) * 100, ls="--", lw=1,
                color="#c0392b", label="majority baseline")
    ax2.set_xticks(range(len(conds)))
    ax2.set_xticklabels([_COND_LABELS[k].replace(" @ ", "\n").replace(" + ", "\n+")
                         for k in conds], fontsize=8)
    ax2.set_ylabel("accuracy (%)"); ax2.set_ylim(0, 100)
    ax2.set_title("Reason-over-facts")
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "build", "train", "eval", "report"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--model", default="d160m", choices=list(POC_PRESETS),
                    help="scale (team: d160m/d360m; toy = pilot). d360m needs A100")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--ckpt-minutes", type=float, default=10)
    ap.add_argument("--fresh", action="store_true", help="ignore existing ckpt")
    # corpus knobs
    ap.add_argument("--max-facts", type=int, default=None)
    ap.add_argument("--exposures", type=int, default=6)
    ap.add_argument("--reason-train", type=int, default=3000)
    ap.add_argument("--bed-docs", type=int, default=800)
    ap.add_argument("--heldout", type=int, default=25)
    ap.add_argument("--seen", type=int, default=25)
    ap.add_argument("--reason-eval", type=int, default=50)
    # eval knobs
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge", action="store_true",
                    help="LLM-judge fact-QA grading (credits paraphrases); needs creds")
    ap.add_argument("--gpt-model", default=None)
    args = ap.parse_args()

    print(f"[poc] stage={args.stage} device={pick_device(args.device)} model={args.model}")
    if args.stage == "build":
        stage_build(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "eval":
        stage_eval(args)
    elif args.stage == "report":
        stage_report(args)
    else:
        stage_build(args)
        stage_train(args)
        stage_eval(args)
        stage_report(args)


if __name__ == "__main__":
    main()
