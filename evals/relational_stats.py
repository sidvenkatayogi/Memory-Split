"""Frozen seed-level statistics and preregistered verdict rules."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evals.relational_metrics import EXPECTED_TASKS

SEEDS = (0, 1, 2)
T_975_DF2 = 4.302652729911275


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} values must be finite")
    return result


def _three_values(values, name: str) -> tuple[float, float, float]:
    if isinstance(values, Mapping):
        if set(values) != set(SEEDS):
            raise ValueError(f"{name} requires seeds 0, 1, and 2")
        raw = [values[seed] for seed in SEEDS]
    else:
        raw = list(values)
        if len(raw) != 3:
            raise ValueError(f"{name} requires three paired seeds")
    return tuple(_finite(value, name) for value in raw)


def paired_t_interval(values) -> tuple[float, float, float]:
    samples = _three_values(values, "paired t interval")
    mean = sum(samples) / 3
    variance = sum((value - mean) ** 2 for value in samples) / 2
    radius = T_975_DF2 * math.sqrt(variance) / math.sqrt(3)
    return mean, mean - radius, mean + radius


def pooled_seed_sigma(by_load: Mapping) -> float:
    """Pool within-load seed variance over exactly two loads (four df)."""

    if not isinstance(by_load, Mapping) or len(by_load) != 2:
        raise ValueError("pooled seed sigma requires exactly two loads")
    numerator = 0.0
    degrees = 0
    for load, raw_values in by_load.items():
        values = _three_values(raw_values, f"load {load!r}")
        mean = sum(values) / len(values)
        numerator += sum((value - mean) ** 2 for value in values)
        degrees += len(values) - 1
    if degrees != 4:
        raise AssertionError("two three-seed loads must provide four degrees")
    return math.sqrt(numerator / degrees)


pooled_sigma = pooled_seed_sigma


def _difference_in_differences_from_deltas(
    low,
    high,
) -> dict:
    low_values = _three_values(low, "low-load paired deltas")
    high_values = _three_values(high, "high-load paired deltas")
    effects = [
        high_values[index] - low_values[index] for index in range(3)
    ]
    mean, low_bound, high_bound = paired_t_interval(effects)
    return {
        "dose_effect": effects,
        "mean": mean,
        "ci_lo": low_bound,
        "ci_hi": high_bound,
        "seeds": list(SEEDS),
        "n_seeds": 3,
    }


def difference_in_differences(
    *,
    dense_low,
    split_low,
    dense_high,
    split_high,
) -> dict:
    dense_low_values = _three_values(dense_low, "dense low")
    split_low_values = _three_values(split_low, "split low")
    dense_high_values = _three_values(dense_high, "dense high")
    split_high_values = _three_values(split_high, "split high")
    low_delta = [
        split_low_values[index] - dense_low_values[index]
        for index in range(3)
    ]
    high_delta = [
        split_high_values[index] - dense_high_values[index]
        for index in range(3)
    ]
    result = _difference_in_differences_from_deltas(low_delta, high_delta)
    result["split_minus_dense_low"] = low_delta
    result["split_minus_dense_high"] = high_delta
    return result


def load_interaction(by_load: Mapping) -> dict:
    """Compatibility wrapper for already paired low/high arm deltas."""

    if not isinstance(by_load, Mapping) or set(by_load) != {"low", "high"}:
        raise ValueError("load interaction requires low and high paired deltas")
    return _difference_in_differences_from_deltas(
        by_load["low"], by_load["high"]
    )


def stratum_mean_deltas(
    split_by_task: Mapping[str, Mapping[int, float]],
    dense_by_task: Mapping[str, Mapping[int, float]],
) -> dict[str, float]:
    expected = set(EXPECTED_TASKS)
    if set(split_by_task) != expected or set(dense_by_task) != expected:
        raise ValueError("strata must exactly match the three frozen tasks")
    result = {}
    for task in EXPECTED_TASKS:
        split = _three_values(split_by_task[task], f"split {task}")
        dense = _three_values(dense_by_task[task], f"dense {task}")
        result[task] = sum(
            split[index] - dense[index] for index in range(3)
        ) / 3
    return result


@dataclass(frozen=True)
class VerdictInputs:
    delta_360: Sequence[float]
    pooled_sigma: float
    interaction_ci: Sequence[float]
    stratum_means: Mapping[str, float] | Sequence[float]
    split_minus_random: Sequence[float]
    guardrails: Mapping[str, bool | Mapping]


def _validate_interval(values, name: str) -> tuple[float, float]:
    raw = list(values)
    if len(raw) != 2:
        raise ValueError(f"{name} requires lower and upper bounds")
    low, high = (_finite(value, name) for value in raw)
    if low > high:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return low, high


def _stratum_values(values) -> tuple[float, float, float]:
    if isinstance(values, Mapping):
        if set(values) != set(EXPECTED_TASKS):
            raise ValueError(
                "stratum_means must exactly match the three frozen tasks"
            )
        raw = [values[task] for task in EXPECTED_TASKS]
    else:
        raw = list(values)
        if len(raw) != 3:
            raise ValueError("stratum_means requires three frozen strata")
    return tuple(_finite(value, "stratum_means") for value in raw)


def _guardrails_pass(guardrails: Mapping) -> bool:
    if not isinstance(guardrails, Mapping) or not guardrails:
        raise ValueError("guardrails must contain measured instrument gates")
    passed = []
    for name, measurement in guardrails.items():
        if isinstance(measurement, bool):
            value = measurement
        elif isinstance(measurement, Mapping):
            if "passed" not in measurement:
                raise ValueError(
                    f"guardrail {name!r} is missing its passed measurement"
                )
            value = measurement["passed"]
            if not isinstance(value, bool):
                raise ValueError(
                    f"guardrail {name!r} passed value must be Boolean"
                )
        else:
            raise ValueError(
                f"guardrail {name!r} must be Boolean or a measurement"
            )
        passed.append(value)
    return all(passed)


def decide_verdict(value: VerdictInputs) -> str:
    delta_360 = _three_values(value.delta_360, "delta_360")
    sigma = _finite(value.pooled_sigma, "pooled_sigma")
    if sigma < 0:
        raise ValueError("pooled_sigma must be non-negative")
    interaction_low, interaction_high = _validate_interval(
        value.interaction_ci, "interaction_ci"
    )
    strata = _stratum_values(value.stratum_means)
    random_contrast = _three_values(
        value.split_minus_random, "split_minus_random"
    )

    if not _guardrails_pass(value.guardrails):
        return "invalid"

    mean_360, _, upper_360 = paired_t_interval(delta_360)
    margin = max(0.02, 2 * sigma)
    validates = (
        all(delta > 0 for delta in delta_360)
        and mean_360 > margin
        and all(delta > 0 for delta in strata)
        and interaction_low > 0
        and all(delta > 0.01 for delta in random_contrast)
    )
    if validates:
        return "validated"

    rejects = (
        upper_360 < 0.02
        and interaction_high <= 0
        and max(random_contrast) <= 0.01
    )
    return "rejected" if rejects else "inconclusive"
