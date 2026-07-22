from pathlib import Path
import subprocess
import sys

import pytest

from evals.relational_stats import (
    VerdictInputs,
    decide_verdict,
    difference_in_differences,
    paired_t_interval,
    pooled_seed_sigma,
    stratum_mean_deltas,
)
from scripts.analyze_relational import (
    _require_expected_run_matrix,
    analyze_runs,
    expected_run_keys,
)


def test_pooled_sigma_uses_two_loads_and_four_degrees_of_freedom():
    values = {
        "low": [-0.01, 0.00, 0.01],
        "high": [0.04, 0.05, 0.06],
    }
    assert pooled_seed_sigma(values) == pytest.approx(0.01)

    with pytest.raises(ValueError, match="exactly two loads"):
        pooled_seed_sigma({**values, "middle": [0.01, 0.02, 0.03]})


def test_paired_t_interval_uses_three_paired_seeds():
    mean, low, high = paired_t_interval([0.02, 0.03, 0.04])
    assert mean == pytest.approx(0.03)
    assert low < mean < high

    with pytest.raises(ValueError, match="three paired seeds"):
        paired_t_interval([0.02, 0.03])


def test_two_load_difference_in_differences_is_seed_paired():
    result = difference_in_differences(
        dense_low={0: 0.50, 1: 0.50, 2: 0.50},
        split_low={0: 0.51, 1: 0.51, 2: 0.51},
        dense_high={0: 0.50, 1: 0.50, 2: 0.50},
        split_high={0: 0.54, 1: 0.55, 2: 0.56},
    )

    assert result["dose_effect"] == pytest.approx([0.03, 0.04, 0.05])
    assert result["mean"] == pytest.approx(0.04)
    assert result["ci_lo"] < result["mean"] < result["ci_hi"]
    assert result["seeds"] == [0, 1, 2]


def test_stratum_means_require_all_three_tasks_and_seeds():
    split = {
        task: {seed: 0.60 + 0.01 * seed for seed in range(3)}
        for task in (
            "path_composition",
            "date_ordering",
            "balanced_equality",
        )
    }
    dense = {
        task: {seed: 0.55 + 0.01 * seed for seed in range(3)}
        for task in split
    }
    means = stratum_mean_deltas(split, dense)
    assert means == {
        "path_composition": pytest.approx(0.05),
        "date_ordering": pytest.approx(0.05),
        "balanced_equality": pytest.approx(0.05),
    }

    del split["balanced_equality"]
    with pytest.raises(ValueError, match="strata"):
        stratum_mean_deltas(split, dense)


def _validated_inputs(**changes):
    values = {
        "delta_360": (0.08, 0.09, 0.07),
        "pooled_sigma": 0.01,
        "interaction_ci": (0.01, 0.04),
        "stratum_means": {
            "path_composition": 0.07,
            "date_ordering": 0.08,
            "balanced_equality": 0.09,
        },
        "split_minus_random": (0.03, 0.02, 0.04),
        "guardrails": {
            "burden": {"passed": True},
            "leakage": {"passed": True},
            "factual": {"passed": True},
            "language": {"passed": True},
        },
    }
    values.update(changes)
    return VerdictInputs(**values)


def test_validate_requires_every_frozen_rule():
    assert decide_verdict(_validated_inputs()) == "validated"
    assert (
        decide_verdict(
            _validated_inputs(split_minus_random=(0.03, 0.01, 0.04))
        )
        == "inconclusive"
    )
    assert (
        decide_verdict(_validated_inputs(delta_360=(0.08, 0.0, 0.07)))
        == "inconclusive"
    )


def test_failed_instrument_guardrail_is_invalid():
    assert (
        decide_verdict(
            _validated_inputs(
                guardrails={
                    "burden": {"passed": True},
                    "leakage": {"passed": False},
                }
            )
        )
        == "invalid"
    )
    with pytest.raises(ValueError, match="guardrails"):
        decide_verdict(_validated_inputs(guardrails={}))


def test_reject_requires_all_three_futility_boundaries():
    rejected = _validated_inputs(
        delta_360=(0.0, 0.0, 0.0),
        interaction_ci=(-0.02, 0.0),
        split_minus_random=(0.01, 0.005, -0.01),
    )
    assert decide_verdict(rejected) == "rejected"

    assert (
        decide_verdict(
            _validated_inputs(
                delta_360=(0.0, 0.0, 0.0),
                interaction_ci=(-0.02, 0.001),
                split_minus_random=(0.01, 0.005, -0.01),
            )
        )
        == "inconclusive"
    )


def test_missing_or_nonfinite_verdict_data_raises():
    with pytest.raises(ValueError, match="delta_360"):
        decide_verdict(_validated_inputs(delta_360=(0.1, 0.2)))
    with pytest.raises(ValueError, match="finite"):
        decide_verdict(
            _validated_inputs(split_minus_random=(0.1, float("nan"), 0.2))
        )


def test_expected_protected_matrix_is_exactly_twenty_one_runs():
    expected = expected_run_keys()
    assert len(expected) == 21
    assert {
        key[2]
        for key in expected
        if key[0] == "d160m" and key[1] in {"dense", "split"}
    } == {"n50k", "n800k"}
    _require_expected_run_matrix({key: {} for key in expected})

    missing = {key: {} for key in expected}
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="run matrix mismatch"):
        _require_expected_run_matrix(missing)


def test_relational_analysis_command_is_repo_relative():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/analyze_relational.py", "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--runs-root" in completed.stdout


def test_complete_matrix_analysis_applies_two_load_and_360m_rules():
    def mode_summary(mode, score):
        return {
            "memory": mode,
            "tasks": {
                task: {
                    "counterfactual_pair_accuracy": score,
                    "n_pairs": 10_000,
                }
                for task in (
                    "path_composition",
                    "date_ordering",
                    "balanced_equality",
                )
            },
            "primary_composite": score,
        }

    def guardrails(condition):
        recognition = (
            {
                "accuracy": 0.4,
                "ci_lo": 0.31,
                "ci_hi": 0.5,
                "n": 100,
            }
            if condition == "dense"
            else {
                "accuracy": 0.2,
                "ci_lo": 0.1,
                "ci_hi": 0.29,
                "n": 100,
            }
        )
        return {
            "within_run_guardrails": {
                "route": {
                    "route_rate": {"passed": True},
                    "tail_external": {"passed": True},
                    "structure_internal": {"passed": True},
                },
                "mask": {"passed": True},
            },
            "recognition_store_off": recognition,
            "factual_recall": {
                "on": {
                    "accuracy": 0.79 if condition == "split" else 0.8,
                    "n": 100,
                },
                "off": {"accuracy": 0.8, "n": 100},
            },
            "internal_accuracy": {
                "accuracy": 0.79 if condition == "split" else 0.8,
                "n": 100,
            },
            "language": {"bpb": 1.0, "total_utf8_bytes": 1_000},
        }

    runs = {}
    for key in expected_run_keys():
        model, condition, load, _ = key
        if model == "d360m":
            score = 0.58 if condition == "split" else 0.50
        elif condition == "dense":
            score = 0.50
        elif condition == "random":
            score = 0.52
        elif load == "n50k":
            score = 0.51
        else:
            score = 0.54
        runs[key] = {
            "on": mode_summary("on", score),
            "off": mode_summary("off", score),
            "guardrails": guardrails(condition),
        }

    result = analyze_runs(runs)

    assert result["verdict"] == "validated"
    assert result["difference_in_differences"]["mean"] == pytest.approx(0.03)
    assert result["run_count"] == 21
