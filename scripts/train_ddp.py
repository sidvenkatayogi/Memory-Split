"""torchrun entrypoint: DDP-train one arm from a config yaml.

    NPROC=8 cluster/torchrun_poc.sh configs/mh/d160m_split_n20m_s0.yaml [--max-steps N]

The config carries the Trainer schema (model, lr, tokens_per_step, micro_batch_size,
ctx, ...) plus data paths (train_bin, train_mask, out_dir). Run WITHOUT torchrun
and it trains single-process (WORLD_SIZE defaults to 1) — same code path as
scripts/poc_run.py. The global batch (tokens_per_step) must divide exactly across
world_size x micro_batch_size x ctx; we assert that here (the "EXACT" gate) so a
misconfigured run fails immediately instead of silently changing the batch.
"""

from __future__ import annotations

import argparse
import os

import yaml

from train.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", default="auto", choices=["auto", "none"])
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    ctx = cfg.get("ctx", 1024)
    denom = cfg["micro_batch_size"] * ctx * ws
    if cfg["tokens_per_step"] % denom != 0:
        raise SystemExit(
            f"NOT-EXACT: tokens_per_step={cfg['tokens_per_step']} not divisible by "
            f"micro_bs({cfg['micro_batch_size']}) x ctx({ctx}) x world_size({ws})={denom}"
        )
    accum = cfg["tokens_per_step"] // denom
    if os.environ.get("RANK", "0") == "0":
        print(f"EXACT global batch: {cfg['tokens_per_step']} tok/step "
              f"= mbs {cfg['micro_batch_size']} x ctx {ctx} x ws {ws} x accum {accum}")

    trainer = Trainer(cfg)
    # Resume: rank-0 pulls the last ckpt from S3 (if configured + none local),
    # then a barrier so every rank loads the same shared file → identical weights.
    if trainer.is_main:
        trainer.maybe_pull_ckpt_from_s3()
    if trainer.ddp:
        import torch.distributed as dist

        dist.barrier()
    if args.resume == "auto" and trainer.ckpt_path.exists():
        trainer.load_ckpt()
        if trainer.is_main:
            print(f"resumed from step {trainer.step}")
    try:
        trainer.train_steps()
    finally:
        if trainer.ddp:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
