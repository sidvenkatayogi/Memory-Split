#!/usr/bin/env python
"""Evaluate EVERY snapshot of a run (+ the final ckpt) to populate H4 curves.

Scales the NR-2 flow to any training run: `run_evals.py --ckpt` already routes
each snapshot's eval to `<run>/evals/ckpt-<tag>/summary.json` (with step/tokens),
which `evals.curves.collect_metric_curve` then reads. This driver just sweeps all
snapshots so the full per-snapshot curve (accuracy / recall / continuous), not
only the log-derived loss/masked-CE curve, is available for any run.

Usage:
  python scripts/eval_all_snapshots.py --run outputs/d160m_split_n200k_s0 \
      [--continuous] [--natural] [--limit 500] [--batch-size 16] [--skip-existing]

Each snapshot is evaluated in a fresh subprocess (isolation + clean GPU memory).
Composes with scripts/plot_run_curves.py (single run) or scripts/analyze.py
(multi-run battery) to produce the figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from evals.curves import snapshot_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--continuous", action="store_true")
    ap.add_argument("--natural", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip snapshots whose evals/ckpt-<tag>/summary.json exists")
    ap.add_argument("--include-final", action="store_true",
                    help="also (re)evaluate the final ckpt.pt into evals/")
    args = ap.parse_args()

    run_dir = Path(args.run)
    evals_root = run_dir / "evals"
    snaps = sorted((run_dir / "snapshots").glob("step*.pt"), key=lambda p: snapshot_step(p.name) or 0)
    if not snaps and not args.include_final:
        print(f"no snapshots in {run_dir/'snapshots'}")
        return

    # (ckpt_rel, output_marker) work items; None ckpt = final into evals/
    items: list[str | None] = [f"snapshots/{p.name}" for p in snaps]
    if args.include_final:
        items.append(None)

    passthrough = []
    if args.continuous:
        passthrough.append("--continuous")
    if args.natural:
        passthrough.append("--natural")
    if args.limit:
        passthrough += ["--limit", str(args.limit)]
    passthrough += ["--batch-size", str(args.batch_size)]

    n_done = 0
    for ckpt_rel in items:
        tag = "final" if ckpt_rel is None else snapshot_step(ckpt_rel)
        marker = (evals_root / "summary.json") if ckpt_rel is None \
            else (evals_root / f"ckpt-step{int(tag):07d}" / "summary.json") if isinstance(tag, int) \
            else None
        if args.skip_existing and marker is not None and marker.exists():
            print(f"[skip] {ckpt_rel or 'final'} (summary exists)")
            continue
        cmd = [sys.executable, "scripts/run_evals.py", "--run", str(run_dir)]
        if ckpt_rel is not None:
            cmd += ["--ckpt", ckpt_rel]
        cmd += passthrough
        print(f"[eval] {ckpt_rel or 'final'} -> {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        n_done += 1
    print(f"done: evaluated {n_done} checkpoint(s) for {run_dir.name}")


if __name__ == "__main__":
    main()
