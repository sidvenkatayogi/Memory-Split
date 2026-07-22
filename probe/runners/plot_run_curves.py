#!/usr/bin/env python
"""Plot H4 learning curves for ONE experiment (1-2 arms), from any run(s).

Two curve sources, both handled:
  * snapshot metrics (accuracy / composite / recall / continuous) — read from
    each run's `evals/ckpt-*/summary.json` + final `evals/summary.json` via
    `evals.curves.collect_metric_curve` (populate these with
    `scripts/eval_all_snapshots.py`).
  * training-log metrics (loss, masked-value CE) — read from `log.jsonl` via
    `evals.curves.log_series` (available for ANY run with no extra eval passes).

Produces one figure per metric (arms overlaid) + a tokens-to-milestone table.
Generalizes the one-off NR-2 log-curve script to arbitrary runs/metrics.

Usage:
  python scripts/plot_run_curves.py --runs outputs/d160m_dense_n200k_s0 \
      outputs/d160m_split_n200k_s0 --out outputs/analysis/curves \
      [--metrics composite_knowledge_free igsm.acc deduction.acc recall.on] \
      [--xkey tokens]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from evals.curves import (
    METRIC_DIRECTION,
    PRIMARY_MILESTONE,
    collect_metric_curve,
    log_series,
    tokens_to_threshold,
)
from evals.figures import learning_curve_figure

# milestone thresholds + directions live in evals.curves (single source of truth
# shared with scripts/analyze.py; NR-2 reconciliation). Defaults below preserve
# the prior behavior for any metric not registered there.
def _milestone(metric: str) -> tuple[bool, float]:
    return METRIC_DIRECTION.get(metric, True), PRIMARY_MILESTONE.get(metric, 0.5)


def _arm_of(run_dir: Path) -> str:
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    return cfg.get("arm", run_dir.name)


def _tps(run_dir: Path):
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    return cfg.get("tokens_per_step")


def _snapshot_series(run_dir: Path, metric: str, xkey: str):
    pts = collect_metric_curve(run_dir, metric)
    out = []
    for p in pts:
        x = p.get(xkey) if p.get(xkey) is not None else p.get("step")
        if x is not None:
            out.append((float(x), float(p["value"])))
    return out


def _log_series_x(run_dir: Path, ykey: str, xkey: str):
    tps = _tps(run_dir)
    pairs = log_series(run_dir / "log.jsonl", ykey)
    if xkey == "tokens" and tps:
        return [(s * tps, v) for s, v in pairs]
    return [(float(s), float(v)) for s, v in pairs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/analysis/curves")
    ap.add_argument("--metrics", nargs="+",
                    default=["composite_knowledge_free", "igsm.acc", "deduction.acc",
                             "recall.on", "recall.closed", "loss", "loss_masked_values"])
    ap.add_argument("--xkey", default="tokens", choices=["tokens", "step"])
    args = ap.parse_args()

    runs = [Path(r) for r in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xlabel = "training tokens" if args.xkey == "tokens" else "training step"
    log_metrics = {"loss", "loss_masked_values"}

    milestones: dict = {}
    for metric in args.metrics:
        by_arm: dict[str, list] = {}
        for run in runs:
            arm = _arm_of(run)
            series = (_log_series_x(run, metric, args.xkey) if metric in log_metrics
                      else _snapshot_series(run, metric, args.xkey))
            if series:
                by_arm[arm] = series
        if not by_arm:
            continue
        safe = metric.replace(".", "_")
        learning_curve_figure(by_arm, out / f"curve_{safe}.png",
                              ylabel=metric, xlabel=xlabel)
        higher, thr = _milestone(metric)
        milestones[metric] = {
            arm: tokens_to_threshold(
                [{"step": None, args.xkey: x, "value": v} for x, v in series],
                thr, xkey=args.xkey, higher_is_better=higher)
            for arm, series in by_arm.items()
        }
        print(f"figure -> {out / f'curve_{safe}.png'}  (arms: {list(by_arm)})")

    (out / "milestones.json").write_text(json.dumps(milestones, indent=2))
    print(f"milestones -> {out / 'milestones.json'}")


if __name__ == "__main__":
    main()
