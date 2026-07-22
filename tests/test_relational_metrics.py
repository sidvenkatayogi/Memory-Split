import math

import pytest

from evals.relational_metrics import (
    assert_expected_counts,
    counterfactual_pair_accuracy,
    exact_accuracy,
    factual_job_guardrail,
    internal_knowledge_guardrail,
    language_bpb_guardrail,
    mask_ledger_guardrail,
    path_diagnostics,
    path_metrics,
    recognition_accuracy,
    recognition_guardrails,
    route_guardrails,
    shared_text_bpb,
    validate_eval_structure,
    wilson_interval,
)


def _result(pair_id, variant, correct, task="path_composition"):
    suffix = "o" if variant == "original" else "c"
    return {
        "qid": f"{pair_id}-{suffix}",
        "pair_id": pair_id,
        "variant": variant,
        "task": task,
        "correct": correct,
    }


def test_structure_validator_accepts_unscored_rows_and_rejects_duplicate_qids():
    rows_by_task = {}
    for task in (
        "path_composition",
        "date_ordering",
        "balanced_equality",
    ):
        rows_by_task[task] = [
            {
                key: value
                for key, value in _result(
                    f"{task}-{pair}",
                    variant,
                    True,
                    task,
                ).items()
                if key != "correct"
            }
            for pair in range(2)
            for variant in ("original", "counterfactual")
        ]

    validate_eval_structure(rows_by_task, n_pairs=2)
    assert_expected_counts(rows_by_task, n_pairs=2)

    rows_by_task["date_ordering"][0]["qid"] = rows_by_task[
        "path_composition"
    ][0]["qid"]
    with pytest.raises(ValueError, match="duplicate eval qid"):
        validate_eval_structure(rows_by_task, n_pairs=2)


def test_counterfactual_pair_accuracy_requires_both_distinct_variants():
    rows = [
        _result("p0", "original", True),
        _result("p0", "counterfactual", True),
        _result("p1", "original", True),
        _result("p1", "counterfactual", False),
    ]
    assert counterfactual_pair_accuracy(rows, expected_pairs=2) == 0.5

    duplicate = [
        _result("p0", "original", True),
        _result("p0", "original", True),
    ]
    with pytest.raises(ValueError, match="original and counterfactual"):
        counterfactual_pair_accuracy(duplicate)


def test_counterfactual_pair_accuracy_rejects_missing_or_unexpected_counts():
    with pytest.raises(ValueError, match="requires original and counterfactual"):
        counterfactual_pair_accuracy([_result("p0", "original", True)])

    complete = [
        _result("p0", "original", True),
        _result("p0", "counterfactual", True),
    ]
    with pytest.raises(ValueError, match="expected 2 pairs"):
        counterfactual_pair_accuracy(complete, expected_pairs=2)


def test_counterfactual_pair_accuracy_requires_scores_for_both_variants():
    rows = [
        _result("p0", "original", False),
        {
            key: value
            for key, value in _result(
                "p0",
                "counterfactual",
                True,
            ).items()
            if key != "correct"
        },
    ]

    with pytest.raises(ValueError, match="requires scored rows"):
        counterfactual_pair_accuracy(rows, expected_pairs=1)


def test_expected_counts_are_exact_per_frozen_stratum():
    rows_by_task = {}
    for task in (
        "path_composition",
        "date_ordering",
        "balanced_equality",
    ):
        rows_by_task[task] = [
            _result(f"{task}-{pair}", variant, True, task)
            for pair in range(2)
            for variant in ("original", "counterfactual")
        ]
    assert_expected_counts(rows_by_task, n_pairs=2)

    rows_by_task["date_ordering"].pop()
    with pytest.raises(ValueError, match="date_ordering: expected 4 rows"):
        assert_expected_counts(rows_by_task, n_pairs=2)


def test_path_metrics_report_exact_hop_referent_and_failure_rates():
    rows = [
        {
            "actions": ["a", "b", "c"],
            "gold_actions": ["a", "b", "x"],
            "correct_referents": [True, True, False],
            "misses": 1,
            "malformed": 1,
            "excess_reads": 2,
            "halt_step": 4,
            "n_steps": 6,
        }
    ]

    out = path_diagnostics(rows)

    assert out["full_path_exact"] == 0.0
    assert out["per_hop_accuracy"] == pytest.approx(2 / 3)
    assert out["correct_referent_rate"] == pytest.approx(2 / 3)
    assert out["miss_rate"] == pytest.approx(1 / 3)
    assert out["excess_read_rate"] == pytest.approx(2 / 3)
    assert out["malformed_rate"] == pytest.approx(1 / 6)
    assert out["mean_halt_step"] == 4.0


def test_path_metrics_reject_missing_hop_evidence():
    with pytest.raises(ValueError, match="correct_referents"):
        path_diagnostics(
            [
                {
                    "actions": ["a"],
                    "gold_actions": ["a", "b"],
                    "correct_referents": [True],
                    "misses": 0,
                    "malformed": 0,
                    "excess_reads": 0,
                    "halt_step": None,
                    "n_steps": 6,
                }
            ]
        )


def test_core_path_metrics_require_only_the_frozen_action_and_miss_fields():
    out = path_metrics(
        [
            {
                "actions": ["a", "b", "c"],
                "gold_actions": ["a", "b", "x"],
                "misses": 1,
            }
        ]
    )
    assert out == {
        "full_path_exact": 0.0,
        "per_hop_accuracy": pytest.approx(2 / 3),
        "miss_rate": pytest.approx(1 / 3),
    }


def test_wilson_and_four_way_recognition_are_exact():
    items = [
        {
            "prompt": "first",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 2,
        },
        {
            "prompt": "second",
            "choices": ["a", "b", "c", "d"],
            "answer_index": 0,
        },
    ]

    def scores(prompt, choices):
        del choices
        return [0.0, 0.1, 0.9, 0.2] if prompt == "first" else [0.0, 1.0, 0.5, 0.2]

    result = recognition_accuracy(scores, items, expected_count=2)
    expected_lo, expected_hi = wilson_interval(1, 2)

    assert result == {
        "accuracy": 0.5,
        "ci_lo": expected_lo,
        "ci_hi": expected_hi,
        "n": 2,
        "correct": 1,
    }


def test_recognition_rejects_non_four_way_or_missing_items():
    with pytest.raises(ValueError, match="four choices"):
        recognition_accuracy(
            lambda prompt, choices: [1.0, 0.0],
            [{"prompt": "x", "choices": ["a", "b"], "answer_index": 0}],
        )
    with pytest.raises(ValueError, match="requires at least one item"):
        recognition_accuracy(lambda prompt, choices: [], [])


def test_shared_text_bpb_uses_utf8_bytes_and_natural_log_units():
    assert shared_text_bpb(8 * math.log(2), 4) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="must contain bytes"):
        shared_text_bpb(0.0, 0)


def test_burden_and_leakage_use_wilson_bounds_not_point_estimates():
    dense = exact_accuracy(
        [{"correct": index < 40} for index in range(100)],
        expected_count=100,
    )
    split = exact_accuracy(
        [{"correct": index < 20} for index in range(100)],
        expected_count=100,
    )

    result = recognition_guardrails(dense, split)

    assert result["burden"]["passed"]
    assert result["burden"]["value"] == dense["ci_lo"]
    assert result["leakage"]["passed"]
    assert result["leakage"]["value"] == split["ci_hi"]


def test_factual_internal_and_language_guardrails_measure_paired_deltas():
    split = {"accuracy": 0.78, "n": 100}
    dense = {"accuracy": 0.80, "n": 100}

    factual = factual_job_guardrail(split, dense)
    internal = internal_knowledge_guardrail(split, dense)
    language = language_bpb_guardrail(
        {"bpb": 1.01, "total_utf8_bytes": 10_000},
        {"bpb": 1.00, "total_utf8_bytes": 10_000},
    )

    assert factual["passed"] and factual["value"] == pytest.approx(-0.02)
    assert internal["passed"] and internal["value"] == pytest.approx(-0.02)
    assert factual["rule"] == "split >= dense - 0.02"
    assert internal["rule"] == "split >= dense - 0.02"
    assert factual["test"] == "one-sided noninferiority"
    assert language["passed"] and language["value"] == pytest.approx(1.01)


def test_paired_guardrails_reject_unshared_evaluation_denominators():
    with pytest.raises(ValueError, match="same item count"):
        factual_job_guardrail(
            {"accuracy": 0.8, "n": 99},
            {"accuracy": 0.8, "n": 100},
        )
    with pytest.raises(ValueError, match="same shared text"):
        language_bpb_guardrail(
            {"bpb": 1.0, "total_utf8_bytes": 99},
            {"bpb": 1.0, "total_utf8_bytes": 100},
        )


def test_route_and_mask_guardrails_require_measured_counts():
    route = route_guardrails(
        {
            "route_rate": 0.5,
            "route_total": 100,
            "low_use_high_entropy_external_rate": 0.8,
            "low_use_high_entropy_total": 25,
            "rules_top_centrality_internal_rate": 0.9,
            "rules_top_centrality_total": 20,
        }
    )
    mask = mask_ledger_guardrail(
        {
            "unmasked_external_payloads": 0,
            "external_payload_occurrences": 50,
            "masked_rule_action_answer_targets": 0,
            "rule_action_answer_targets": 200,
        },
        condition="split",
    )

    assert all(value["passed"] for value in route.values())
    assert mask["passed"]

    with pytest.raises(KeyError):
        route_guardrails({"route_rate": 0.5})


@pytest.mark.parametrize("condition", ["dense", "random"])
def test_non_split_ledgers_allow_factual_targets_to_remain_unmasked(condition):
    faithful = mask_ledger_guardrail(
        {
            "unmasked_external_payloads": 50,
            "external_payload_occurrences": 50,
            "masked_rule_action_answer_targets": 0,
            "rule_action_answer_targets": 200,
        },
        condition=condition,
    )
    split = mask_ledger_guardrail(
        {
            "unmasked_external_payloads": 50,
            "external_payload_occurrences": 50,
            "masked_rule_action_answer_targets": 0,
            "rule_action_answer_targets": 200,
        },
        condition="split",
    )

    assert faithful["passed"] and not faithful["external_mask_applicable"]
    assert not split["passed"] and split["external_mask_applicable"]
