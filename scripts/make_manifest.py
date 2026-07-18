#!/usr/bin/env python
"""Generate run configs + a manifest TSV for a battery stage.

Usage:
  python scripts/make_manifest.py --stage gates|sweep|confirm|stretch \
      --data-root /scratch/users/syz/memorysplit_data [--top-load n800k]

Writes configs/gen/{stage}_{arm}_{load}_s{seed}.yaml and
outputs/manifests/{stage}.tsv (one config path per line).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

LOADS = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}

SCALE = {
    # preset: (total_tokens, tokens/step, micro_bs, lr, warmup)
    "d160m": (3_200_000_000, 524_288, 16, 1.5e-3, 300),
    "d410m": (8_000_000_000, 524_288, 8, 1.0e-3, 300),
    "d1b": (10_000_000_000, 524_288, 4, 6.0e-4, 300),
}

GATE_TOKENS = 800_000_000  # short-budget pilots for gates A-C


def make_cfg(preset, arm, load, seed, data_root, out_root, total_tokens=None):
    tokens, tps, mbs, lr, warmup = SCALE[preset]
    if total_tokens is not None:
        tokens = total_tokens
    run_id = f"{preset}_{arm}_{load}_s{seed}" + ("" if total_tokens is None else "_gate")
    data_dir = Path(data_root) / load
    return run_id, {
        "run_id": run_id,
        "model": preset,
        "arm": arm,
        "load": load,
        "n_entities": LOADS[load],
        "train_bin": str(data_dir / arm / "train.bin"),
        "train_mask": str(data_dir / arm / "train.mask.bin") if arm == "split" else None,
        "data_dir": str(data_dir),
        "total_tokens": tokens,
        "tokens_per_step": tps,
        "micro_batch_size": mbs,
        "lr": lr,
        "warmup_steps": warmup,
        "weight_decay": 0.1,
        "seed": seed,
        "compile": True,
        "device": "auto",
        "out_dir": str(Path(out_root) / run_id),
        "log_every": 20,
        "eval_every": 250,
        "snap_frac": 0.10,
        "ckpt_minutes": 30,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gates", "sweep", "confirm", "stretch"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--top-load", default="n800k", choices=list(LOADS))
    args = ap.parse_args()

    jobs: list[tuple[str, dict]] = []
    if args.stage == "gates":
        for load in LOADS:  # gate A+B: dense across loads
            jobs.append(make_cfg("d160m", "dense", load, 0, args.data_root, args.out_root, GATE_TOKENS))
        jobs.append(make_cfg("d160m", "split", "n200k", 0, args.data_root, args.out_root, GATE_TOKENS))
    elif args.stage == "sweep":
        for load in LOADS:
            for arm in ("dense", "split"):
                for seed in (0, 1):
                    jobs.append(make_cfg("d160m", arm, load, seed, args.data_root, args.out_root))
    elif args.stage == "confirm":
        for arm in ("dense", "split"):
            for seed in (0, 1, 2):
                jobs.append(make_cfg("d410m", arm, args.top_load, seed, args.data_root, args.out_root))
    elif args.stage == "stretch":
        for arm in ("dense", "split"):
            jobs.append(make_cfg("d1b", arm, args.top_load, 0, args.data_root, args.out_root))

    gen_dir = Path("configs/gen")
    gen_dir.mkdir(parents=True, exist_ok=True)
    man_dir = Path("outputs/manifests")
    man_dir.mkdir(parents=True, exist_ok=True)
    manifest = man_dir / f"{args.stage}.tsv"
    with open(manifest, "w") as mf:
        for run_id, cfg in jobs:
            cfg_path = gen_dir / f"{run_id}.yaml"
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            mf.write(str(cfg_path) + "\n")
    print(f"{len(jobs)} configs -> {manifest}")


if __name__ == "__main__":
    main()
