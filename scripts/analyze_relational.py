#!/usr/bin/env python
"""Aggregate the exact 21-run relational matrix and apply the frozen verdict."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from evals.relational_metrics import (
    EXPECTED_TASKS,
    factual_job_guardrail,
    internal_knowledge_guardrail,
    language_bpb_guardrail,
    recognition_guardrails,
)
from evals.relational_stats import (
    SEEDS,
    VerdictInputs,
    decide_verdict,
    difference_in_differences,
    paired_t_interval,
    pooled_seed_sigma,
    stratum_mean_deltas,
)

LOW_LOAD = "n50k"
HIGH_LOAD = "n800k"
CONFIRM_LOAD = "n1p8m"


def expected_run_keys() -> set[tuple[str, str, str, int]]:
    expected = {
        ("d160m", condition, load, seed)
        for condition in ("dense", "split")
        for load in (LOW_LOAD, HIGH_LOAD)
        for seed in SEEDS
    }
    expected.update(
        ("d160m", "random", HIGH_LOAD, seed) for seed in SEEDS
    )
    expected.update(
        ("d360m", condition, CONFIRM_LOAD, seed)
        for condition in ("dense", "split")
        for seed in SEEDS
    )
    return expected


def _require_expected_run_matrix(runs) -> None:
    actual = set(runs)
    expected = expected_run_keys()
    if actual != expected:
        raise ValueError(
            "run matrix mismatch; "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _load_complete_runs(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(root)
    runs = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        config_path = directory / "config.yaml"
        if not config_path.exists():
            continue
        mode_paths = {
            mode: directory
            / "evals"
            / f"memory_{mode}"
            / "summary.json"
            for mode in ("on", "off")
        }
        guardrail_path = directory / "evals" / "guardrails.json"
        missing = [
            str(path)
            for path in (*mode_paths.values(), guardrail_path)
            if not path.is_file()
        ]
        if missing:
            raise ValueError(
                f"incomplete relational evaluation for {directory.name}: "
                f"{missing}"
            )
        cfg = yaml.safe_load(config_path.read_text())
        key = (
            str(cfg["model"]),
            str(cfg["condition"]),
            str(cfg["load"]),
            int(cfg["seed"]),
        )
        if key in runs:
            raise ValueError(f"duplicate run key: {key}")
        runs[key] = {
            "cfg": cfg,
            "on": json.loads(mode_paths["on"].read_text()),
            "off": json.loads(mode_paths["off"].read_text()),
            "guardrails": json.loads(guardrail_path.read_text()),
            "directory": str(directory),
        }
    _require_expected_run_matrix(runs)
    return runs


def _task_score(run: dict, mode: str, task: str) -> float:
    summary = run[mode]
    if summary["memory"] != mode:
        raise ValueError("memory summary label does not match its directory")
    tasks = summary["tasks"]
    if set(tasks) != set(EXPECTED_TASKS):
        raise ValueError("summary task set does not match frozen strata")
    score = float(tasks[task]["counterfactual_pair_accuracy"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("counterfactual pair accuracy must be in [0, 1]")
    if tasks[task]["n_pairs"] != 10_000:
        raise ValueError("protected summaries require 10,000 pairs per task")
    return score


def _composite(run: dict) -> float:
    computed = sum(
        _task_score(run, "on", task) for task in EXPECTED_TASKS
    ) / len(EXPECTED_TASKS)
    declared = float(run["on"]["primary_composite"])
    if not math.isclose(computed, declared, abs_tol=1e-12):
        raise ValueError("declared primary composite is inconsistent")
    return computed


def _condition_values(
    runs: dict,
    model: str,
    condition: str,
    load: str,
) -> dict[int, float]:
    return {
        seed: _composite(runs[(model, condition, load, seed)])
        for seed in SEEDS
    }


def _flatten_static_guardrails(key, run, combined) -> None:
    within = run["guardrails"]["within_run_guardrails"]
    if set(within) != {"route", "mask"}:
        raise ValueError("within-run guardrails require route and mask")
    route = within["route"]
    if not route:
        raise ValueError("route guardrails must not be empty")
    for name, measurement in route.items():
        combined[f"{key}:route:{name}"] = measurement
    mask = within["mask"]
    if mask["condition"] != key[1]:
        raise ValueError("mask guardrail condition does not match run key")
    combined[f"{key}:mask"] = mask


def _collect_guardrails(runs: dict) -> dict[str, dict]:
    combined: dict[str, dict] = {}
    for key, run in runs.items():
        _flatten_static_guardrails(key, run, combined)

    paired_cells = [
        ("d160m", LOW_LOAD, seed) for seed in SEEDS
    ] + [
        ("d160m", HIGH_LOAD, seed) for seed in SEEDS
    ] + [
        ("d360m", CONFIRM_LOAD, seed) for seed in SEEDS
    ]
    for model, load, seed in paired_cells:
        dense = runs[(model, "dense", load, seed)]["guardrails"]
        split = runs[(model, "split", load, seed)]["guardrails"]
        prefix = f"{model}:{load}:seed{seed}"
        recognition = recognition_guardrails(
            dense["recognition_store_off"],
            split["recognition_store_off"],
        )
        combined[f"{prefix}:burden"] = recognition["burden"]
        combined[f"{prefix}:leakage"] = recognition["leakage"]
        combined[f"{prefix}:factual"] = factual_job_guardrail(
            split["factual_recall"]["on"],
            dense["factual_recall"]["off"],
        )
        combined[f"{prefix}:internal"] = internal_knowledge_guardrail(
            split["internal_accuracy"],
            dense["internal_accuracy"],
        )
        combined[f"{prefix}:language"] = language_bpb_guardrail(
            split["language"],
            dense["language"],
        )
    if not combined:
        raise ValueError("no guardrail measurements were collected")
    return combined


def analyze_runs(runs: dict) -> dict:
    _require_expected_run_matrix(runs)
    dense_low = _condition_values(
        runs, "d160m", "dense", LOW_LOAD
    )
    split_low = _condition_values(
        runs, "d160m", "split", LOW_LOAD
    )
    dense_high = _condition_values(
        runs, "d160m", "dense", HIGH_LOAD
    )
    split_high = _condition_values(
        runs, "d160m", "split", HIGH_LOAD
    )
    interaction = difference_in_differences(
        dense_low=dense_low,
        split_low=split_low,
        dense_high=dense_high,
        split_high=split_high,
    )
    by_load = {
        "low": interaction["split_minus_dense_low"],
        "high": interaction["split_minus_dense_high"],
    }
    sigma = pooled_seed_sigma(by_load)

    dense_360 = _condition_values(
        runs, "d360m", "dense", CONFIRM_LOAD
    )
    split_360 = _condition_values(
        runs, "d360m", "split", CONFIRM_LOAD
    )
    delta_360 = tuple(
        split_360[seed] - dense_360[seed] for seed in SEEDS
    )
    split_by_task = {
        task: {
            seed: _task_score(
                runs[("d360m", "split", CONFIRM_LOAD, seed)],
                "on",
                task,
            )
            for seed in SEEDS
        }
        for task in EXPECTED_TASKS
    }
    dense_by_task = {
        task: {
            seed: _task_score(
                runs[("d360m", "dense", CONFIRM_LOAD, seed)],
                "on",
                task,
            )
            for seed in SEEDS
        }
        for task in EXPECTED_TASKS
    }
    stratum_means = stratum_mean_deltas(
        split_by_task, dense_by_task
    )
    random_high = _condition_values(
        runs, "d160m", "random", HIGH_LOAD
    )
    split_minus_random = tuple(
        split_high[seed] - random_high[seed] for seed in SEEDS
    )
    guardrails = _collect_guardrails(runs)
    inputs = VerdictInputs(
        delta_360=delta_360,
        pooled_sigma=sigma,
        interaction_ci=(
            interaction["ci_lo"],
            interaction["ci_hi"],
        ),
        stratum_means=stratum_means,
        split_minus_random=split_minus_random,
        guardrails=guardrails,
    )
    return {
        "verdict": decide_verdict(inputs),
        "paired_360_interval": {
            key: value
            for key, value in zip(
                ("mean", "ci_lo", "ci_hi"),
                paired_t_interval(delta_360),
            )
        },
        "pooled_seed_sigma": sigma,
        "difference_in_differences": interaction,
        "inputs": asdict(inputs),
        "run_count": len(runs),
    }


def _write_summary(result: dict, path: Path) -> None:
    interaction = result["difference_in_differences"]
    interval = result["paired_360_interval"]
    lines = [
        "# Relational MemorySplit protected analysis",
        "",
        f"- Verdict: **{result['verdict']}**",
        f"- Runs: {result['run_count']}",
        (
            "- Pooled 160M within-load seed sigma "
            "(used for the 360M margin): "
            f"{result['pooled_seed_sigma']:.6f}"
        ),
        (
            "- Two-load difference-in-differences: "
            f"{interaction['mean']:+.6f} "
            f"[{interaction['ci_lo']:+.6f}, "
            f"{interaction['ci_hi']:+.6f}]"
        ),
        (
            "- 360M Split-Dense: "
            f"{interval['mean']:+.6f} "
            f"[{interval['ci_lo']:+.6f}, "
            f"{interval['ci_hi']:+.6f}]"
        ),
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    runs = _load_complete_runs(Path(args.runs_root))
    result = analyze_runs(runs)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    _write_summary(result, output / "summary.md")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
