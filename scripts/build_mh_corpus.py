"""Build a Wikidata5M multi-hop corpus at a DOSE and emit DDP train configs.

    python scripts/build_mh_corpus.py --wikidata-root DIR --dose 20000000 \
        --exposures 30 --model d160m --micro-bs 8 --out data/mh/n20m

Writes  <out>/{dense,split}/train.bin (+ .mask.bin)  and
        configs/mh/<tag>_{dense,split}.yaml  ready for:
        NPROC=8 cluster/torchrun_poc.sh configs/mh/<tag>_split.yaml

`--exposures` sets training passes over the corpus (total_tokens = corpus_tokens
x exposures) -- the fixed-exposure knob for the dose ladder. `--micro-bs` should
be 8 on 40GB A100s, 16/32 on 80GB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from corpusgen.mh_build import MHBuildCfg, build_mh_corpus
from train.tokenizer import get_tok


def _config(model, condition, ctx, micro_bs, tokens_per_step, total_tokens,
            lr, seed, train_bin, train_mask, out_dir, tag, s3_ckpt=None) -> dict:
    conf = {
        "schema_version": 1,
        "run_id": f"{tag}_{condition}_s{seed}",
        "model": model,
        "condition": condition,
        "seed": seed,
        "ctx": ctx,
        "micro_batch_size": micro_bs,
        "tokens_per_step": tokens_per_step,
        "total_tokens": int(total_tokens),
        "lr": lr,
        "warmup_steps": 300,
        "weight_decay": 0.1,
        "compile": True,
        "device": "cuda",
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": 0.1,
        "ckpt_minutes": 15,
        "train_bin": str(train_bin),
        "train_mask": str(train_mask),
        "out_dir": str(out_dir),
    }
    if s3_ckpt:
        conf["s3_ckpt"] = s3_ckpt          # ckpt/snapshots pushed here for durability
        conf["s3_region"] = "us-east-1"
    return conf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikidata-root", required=True)
    ap.add_argument("--dose", type=int, default=20_000_000, help="n_entities")
    ap.add_argument("--max-triples", type=int, default=None)
    ap.add_argument("--exposures", type=int, default=30, help="training passes over corpus")
    ap.add_argument("--n-mh-train", type=int, default=1_000_000)
    ap.add_argument("--n-mh-eval", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--atomic-exposures", type=int, default=1)
    ap.add_argument("--n-bed-docs", type=int, default=200_000)
    ap.add_argument("--model", default="d160m")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--micro-bs", type=int, default=8)
    ap.add_argument("--tokens-per-step", type=int, default=524288)
    ap.add_argument("--lr", type=float, default=0.0015)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-dir", default="configs/mh")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--s3-ckpt-prefix", default=None,
                    help="e.g. s3://memorysplit-sid-056956104102/runs ; per-arm "
                         "ckpt+snapshots+log sync here so the run survives interruption")
    args = ap.parse_args()

    tag = args.tag or f"{args.model}_n{args.dose}"
    out = Path(args.out)
    cfg = MHBuildCfg(
        wikidata_root=args.wikidata_root, n_entities=args.dose,
        max_triples=args.max_triples, n_mh_train=args.n_mh_train,
        n_mh_eval=args.n_mh_eval, max_depth=args.max_depth,
        atomic_exposures=args.atomic_exposures, n_bed_docs=args.n_bed_docs,
        split_seed=args.seed, render_seed=args.seed,
    )
    print(f"[build_mh] dose={args.dose} model={args.model} -> {out}", flush=True)
    rep = build_mh_corpus(cfg, get_tok(), out)
    print(json.dumps(rep, indent=2), flush=True)

    cfg_dir = Path(args.config_dir)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for arm in ("dense", "split"):
        n_tokens = rep["arms"][arm]["n_tokens"]
        total_tokens = n_tokens * args.exposures
        s3ck = (args.s3_ckpt_prefix.rstrip("/") + f"/{tag}_{arm}"
                if args.s3_ckpt_prefix else None)
        conf = _config(args.model, arm, args.ctx, args.micro_bs,
                       args.tokens_per_step, total_tokens, args.lr, args.seed,
                       out / arm / "train.bin", out / arm / "train.mask.bin",
                       Path("runs") / f"{tag}_{arm}", tag, s3ck)
        path = cfg_dir / f"{tag}_{arm}.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(conf, f, sort_keys=False)
        steps = total_tokens // args.tokens_per_step
        print(f"[build_mh] {arm}: {n_tokens:,} tok x{args.exposures} = "
              f"{total_tokens:,} tok ({steps:,} steps) -> {path}", flush=True)


if __name__ == "__main__":
    main()
