"""Exact relational endpoints and frozen guardrail measurements."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch

EXPECTED_TASKS = (
    "path_composition",
    "date_ordering",
    "balanced_equality",
)
EXPECTED_PAIRS_PER_TASK = 10_000
_Z_975 = 1.959963984540054
_TOLERANCE = 1e-12


def _field(value, name: str):
    if isinstance(value, Mapping):
        return value[name]
    return getattr(value, name)


def _row_pair_id(row) -> str:
    if isinstance(row, Mapping) and "pair_id" in row:
        return str(row["pair_id"])
    meta = _field(row, "meta")
    return str(meta["pair_id"])


def _row_variant(row) -> str:
    if isinstance(row, Mapping) and "variant" in row:
        variant = str(row["variant"])
    else:
        meta = _field(row, "meta")
        if "variant" in meta:
            variant = str(meta["variant"])
        else:
            qid = str(_field(row, "qid"))
            if qid.endswith(("-o", "_o", "-original")):
                variant = "original"
            elif qid.endswith(("-c", "_c", "-counterfactual")):
                variant = "counterfactual"
            else:
                raise ValueError(
                    "every pair requires original and counterfactual variants"
                )
    if variant not in ("original", "counterfactual"):
        raise ValueError(f"unexpected pair variant: {variant}")
    return variant


def counterfactual_pair_accuracy(
    rows,
    *,
    expected_pairs: int | None = None,
) -> float:
    materialized = list(rows)
    if not materialized:
        raise ValueError("counterfactual accuracy requires at least one pair")
    if expected_pairs is not None and expected_pairs <= 0:
        raise ValueError("expected_pairs must be positive")

    grouped: dict[str, dict[str, Mapping]] = defaultdict(dict)
    seen_qids = set()
    for row in materialized:
        pair_id = _row_pair_id(row)
        variant = _row_variant(row)
        qid = str(_field(row, "qid"))
        if variant in grouped[pair_id]:
            raise ValueError(
                "every pair requires distinct original and counterfactual variants"
            )
        if qid in seen_qids:
            raise ValueError(f"duplicate eval qid: {qid}")
        seen_qids.add(qid)
        grouped[pair_id][variant] = row

    if expected_pairs is not None and len(grouped) != expected_pairs:
        raise ValueError(
            f"expected {expected_pairs} pairs, got {len(grouped)}"
        )
    required = {"original", "counterfactual"}
    if any(set(pair) != required for pair in grouped.values()):
        raise ValueError(
            "every pair requires original and counterfactual variants"
        )
    successes = sum(
        bool(_field(pair["original"], "correct"))
        and bool(_field(pair["counterfactual"], "correct"))
        for pair in grouped.values()
    )
    return successes / len(grouped)


def assert_expected_counts(
    rows_by_task: Mapping[str, Sequence],
    n_pairs: int = EXPECTED_PAIRS_PER_TASK,
) -> None:
    if n_pairs <= 0:
        raise ValueError("n_pairs must be positive")
    actual_tasks = set(rows_by_task)
    expected_tasks = set(EXPECTED_TASKS)
    if actual_tasks != expected_tasks:
        raise ValueError(
            "task set mismatch; "
            f"missing={sorted(expected_tasks - actual_tasks)}, "
            f"extra={sorted(actual_tasks - expected_tasks)}"
        )
    expected_rows = 2 * n_pairs
    for task in EXPECTED_TASKS:
        rows = list(rows_by_task[task])
        if len(rows) != expected_rows:
            raise ValueError(
                f"{task}: expected {expected_rows} rows, got {len(rows)}"
            )
        if any(str(_field(row, "task")) != task for row in rows):
            raise ValueError(f"{task}: row task labels do not match stratum")
        counterfactual_pair_accuracy(rows, expected_pairs=n_pairs)


def _nonnegative_int(row, name: str) -> int:
    value = _field(row, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def path_metrics(rows) -> dict:
    materialized = list(rows)
    if not materialized:
        raise ValueError("path metrics require at least one row")

    hop_correct = 0
    hop_total = 0
    exact = 0
    misses = 0
    for row in materialized:
        actions = list(_field(row, "actions"))
        gold = list(_field(row, "gold_actions"))
        if not gold:
            raise ValueError("every path row requires at least one gold action")
        exact += actions == gold
        hop_total += len(gold)
        hop_correct += sum(
            predicted == expected
            for predicted, expected in zip(actions, gold)
        )
        misses += _nonnegative_int(row, "misses")

    n_rows = len(materialized)
    return {
        "full_path_exact": exact / n_rows,
        "per_hop_accuracy": hop_correct / hop_total,
        "miss_rate": misses / hop_total,
    }


def path_diagnostics(rows) -> dict:
    materialized = list(rows)
    core = path_metrics(materialized)
    referent_correct = 0
    hop_total = 0
    excess_reads = 0
    malformed = 0
    action_steps = 0
    halt_steps = []
    halted = 0
    for row in materialized:
        gold = list(_field(row, "gold_actions"))
        referents = list(_field(row, "correct_referents"))
        if len(referents) != len(gold):
            raise ValueError(
                "correct_referents must contain one value per gold action"
            )
        n_steps = _nonnegative_int(row, "n_steps")
        if n_steps != 6:
            raise ValueError("every relational decode must contain six steps")
        halt_step = _field(row, "halt_step")
        if halt_step is not None:
            if (
                isinstance(halt_step, bool)
                or not isinstance(halt_step, int)
                or not 1 <= halt_step <= 6
            ):
                raise ValueError("halt_step must be in [1, 6] or None")
            halted += 1
            halt_steps.append(halt_step)
        else:
            halt_steps.append(7)
        hop_total += len(gold)
        referent_correct += sum(bool(value) for value in referents)
        excess_reads += _nonnegative_int(row, "excess_reads")
        malformed += _nonnegative_int(row, "malformed")
        action_steps += n_steps

    n_rows = len(materialized)
    return {
        **core,
        "correct_referent_rate": referent_correct / hop_total,
        "excess_read_rate": excess_reads / hop_total,
        "malformed_rate": malformed / action_steps,
        "mean_excess_reads": excess_reads / n_rows,
        "mean_halt_step": sum(halt_steps) / n_rows,
        "halt_rate": halted / n_rows,
        "n_rows": n_rows,
        "n_gold_hops": hop_total,
    }


def wilson_interval(
    successes: int,
    total: int,
    z: float = _Z_975,
) -> tuple[float, float]:
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
    ):
        raise ValueError("total must be positive")
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or not 0 <= successes <= total
    ):
        raise ValueError("successes must be between zero and total")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * (
            p * (1 - p) / total
            + z * z / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return center - radius, center + radius


def exact_accuracy(rows, *, expected_count: int | None = None) -> dict:
    materialized = list(rows)
    if not materialized:
        raise ValueError("accuracy requires at least one row")
    if expected_count is not None and len(materialized) != expected_count:
        raise ValueError(
            f"expected {expected_count} rows, got {len(materialized)}"
        )
    correct = sum(bool(_field(row, "correct")) for row in materialized)
    low, high = wilson_interval(correct, len(materialized))
    return {
        "accuracy": correct / len(materialized),
        "ci_lo": low,
        "ci_hi": high,
        "n": len(materialized),
        "correct": correct,
    }


def _score_value(value) -> float:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError("choice score tuples must not be empty")
        value = value[0]
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("choice scores must be finite")
    return score


def recognition_accuracy(
    score_choices,
    items,
    *,
    expected_count: int | None = None,
) -> dict:
    materialized = list(items)
    if not materialized:
        raise ValueError("recognition requires at least one item")
    if expected_count is not None and len(materialized) != expected_count:
        raise ValueError(
            f"expected {expected_count} recognition items, "
            f"got {len(materialized)}"
        )
    correct = 0
    for item in materialized:
        prompt = str(_field(item, "prompt"))
        choices = list(_field(item, "choices"))
        if len(choices) != 4:
            raise ValueError("recognition items require exactly four choices")
        answer_index = _field(item, "answer_index")
        if (
            isinstance(answer_index, bool)
            or not isinstance(answer_index, int)
            or not 0 <= answer_index < 4
        ):
            raise ValueError("answer_index must select one of four choices")
        raw_scores = list(score_choices(prompt, choices))
        if len(raw_scores) != 4:
            raise ValueError("score_choices must return one score per choice")
        scores = [_score_value(value) for value in raw_scores]
        correct += (
            max(range(len(scores)), key=scores.__getitem__) == answer_index
        )
    low, high = wilson_interval(correct, len(materialized))
    return {
        "accuracy": correct / len(materialized),
        "ci_lo": low,
        "ci_hi": high,
        "n": len(materialized),
        "correct": correct,
    }


def score_choice_loglikelihoods(
    model,
    tok,
    prompt: str,
    choices,
    device=None,
) -> list[float]:
    """Score choices with ``GPT.forward_step`` and no padded batches."""

    resolved_device = torch.device(
        device if device is not None else getattr(model, "device", "cpu")
    )
    context_ids = tok.encode(prompt) or [tok.EOT]
    scores = []
    for choice in choices:
        choice_ids = tok.encode(str(choice))
        if not choice_ids:
            raise ValueError("every choice must encode to at least one token")
        input_ids = context_ids + choice_ids[:-1]
        value = torch.tensor(
            [input_ids], dtype=torch.long, device=resolved_device
        )
        with torch.no_grad():
            logits, _ = model.forward_step(value, None)
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        positions = torch.arange(
            len(context_ids) - 1,
            len(context_ids) - 1 + len(choice_ids),
            device=resolved_device,
        )
        targets = torch.tensor(
            choice_ids, dtype=torch.long, device=resolved_device
        )
        scores.append(float(log_probs[positions, targets].sum().item()))
    return scores


def shared_text_bpb(
    total_nll_nats: float,
    total_utf8_bytes: int,
) -> float:
    if (
        isinstance(total_utf8_bytes, bool)
        or not isinstance(total_utf8_bytes, int)
        or total_utf8_bytes <= 0
    ):
        raise ValueError("held-out text must contain bytes")
    if not math.isfinite(total_nll_nats) or total_nll_nats < 0:
        raise ValueError("total_nll_nats must be finite and non-negative")
    return total_nll_nats / (total_utf8_bytes * math.log(2))


def measure_shared_text_bpb(model, tok, texts, device=None) -> dict:
    """Measure UTF-8 BPB using only unpadded ``forward_step`` calls."""

    materialized = [str(text) for text in texts]
    if not materialized:
        raise ValueError("shared-text measurement requires at least one text")
    resolved_device = torch.device(
        device if device is not None else getattr(model, "device", "cpu")
    )
    total_nll = 0.0
    total_bytes = 0
    for text in materialized:
        token_ids = tok.encode(text)
        if not token_ids:
            if text.encode("utf-8"):
                raise ValueError("non-empty text encoded to no tokens")
            continue
        inputs = [tok.EOT] + token_ids[:-1]
        value = torch.tensor(
            [inputs], dtype=torch.long, device=resolved_device
        )
        targets = torch.tensor(
            token_ids, dtype=torch.long, device=resolved_device
        )
        with torch.no_grad():
            logits, _ = model.forward_step(value, None)
            log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        positions = torch.arange(len(token_ids), device=resolved_device)
        total_nll -= float(log_probs[positions, targets].sum().item())
        total_bytes += len(text.encode("utf-8"))
    return {
        "bpb": shared_text_bpb(total_nll, total_bytes),
        "total_nll_nats": total_nll,
        "total_utf8_bytes": total_bytes,
        "n": len(materialized),
    }


def _validated_accuracy(measurement: Mapping, name: str) -> tuple[float, int]:
    accuracy = float(measurement["accuracy"])
    n = measurement["n"]
    if not 0 <= accuracy <= 1 or not math.isfinite(accuracy):
        raise ValueError(f"{name} accuracy must be in [0, 1]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"{name} n must be positive")
    return accuracy, n


def recognition_guardrails(
    dense_recognition: Mapping,
    split_store_off_recognition: Mapping,
) -> dict[str, dict]:
    dense_accuracy, dense_n = _validated_accuracy(
        dense_recognition, "dense recognition"
    )
    split_accuracy, split_n = _validated_accuracy(
        split_store_off_recognition, "split recognition"
    )
    dense_low = float(dense_recognition["ci_lo"])
    split_high = float(split_store_off_recognition["ci_hi"])
    if dense_n != split_n:
        raise ValueError("recognition arms must use the same item count")
    if (
        not math.isfinite(dense_low)
        or not math.isfinite(split_high)
        or not 0 <= dense_low <= 1
        or not 0 <= split_high <= 1
    ):
        raise ValueError("recognition confidence bounds must be in [0, 1]")
    return {
        "burden": {
            "value": dense_low,
            "estimate": dense_accuracy,
            "threshold": 0.30,
            "comparison": ">",
            "passed": dense_low > 0.30,
            "n": dense_n,
        },
        "leakage": {
            "value": split_high,
            "estimate": split_accuracy,
            "threshold": 0.30,
            "comparison": "<",
            "passed": split_high < 0.30,
            "n": split_n,
        },
    }


def _paired_accuracy_guardrail(
    split: Mapping,
    dense: Mapping,
    name: str,
) -> dict:
    split_accuracy, split_n = _validated_accuracy(split, f"split {name}")
    dense_accuracy, dense_n = _validated_accuracy(dense, f"dense {name}")
    if split_n != dense_n:
        raise ValueError(f"{name} arms must use the same item count")
    delta = split_accuracy - dense_accuracy
    return {
        "value": delta,
        "split": split_accuracy,
        "dense": dense_accuracy,
        "threshold": -0.02,
        "comparison": ">=",
        "test": "one-sided noninferiority",
        "rule": "split >= dense - 0.02",
        "passed": delta >= -0.02 - _TOLERANCE,
        "n": min(split_n, dense_n),
        "n_split": split_n,
        "n_dense": dense_n,
    }


def factual_job_guardrail(
    split_memory_on: Mapping,
    dense_memory_off: Mapping,
) -> dict:
    """One-sided noninferiority: Split ON may trail Dense OFF by at most .02."""

    return _paired_accuracy_guardrail(
        split_memory_on, dense_memory_off, "factual recall"
    )


def internal_knowledge_guardrail(
    split_internal: Mapping,
    dense_internal: Mapping,
) -> dict:
    """One-sided noninferiority: Split may trail Dense by at most .02."""

    return _paired_accuracy_guardrail(
        split_internal, dense_internal, "internal knowledge"
    )


def language_bpb_guardrail(
    split_language: Mapping,
    dense_language: Mapping,
) -> dict:
    split_bpb = float(split_language["bpb"])
    dense_bpb = float(dense_language["bpb"])
    split_bytes = split_language["total_utf8_bytes"]
    dense_bytes = dense_language["total_utf8_bytes"]
    if (
        not math.isfinite(split_bpb)
        or not math.isfinite(dense_bpb)
        or split_bpb < 0
        or dense_bpb <= 0
    ):
        raise ValueError("language BPB values must be finite and positive")
    if (
        isinstance(split_bytes, bool)
        or not isinstance(split_bytes, int)
        or split_bytes <= 0
        or isinstance(dense_bytes, bool)
        or not isinstance(dense_bytes, int)
        or dense_bytes <= 0
    ):
        raise ValueError("language byte counts must be positive")
    if split_bytes != dense_bytes:
        raise ValueError("language arms must score the same shared text")
    ratio = split_bpb / dense_bpb
    return {
        "value": ratio,
        "split_bpb": split_bpb,
        "dense_bpb": dense_bpb,
        "threshold": 1.01,
        "comparison": "<=",
        "passed": ratio <= 1.01 + _TOLERANCE,
        "n": min(split_bytes, dense_bytes),
        "split_utf8_bytes": split_bytes,
        "dense_utf8_bytes": dense_bytes,
    }


def _rate_measurement(
    value: float,
    *,
    threshold,
    comparison: str,
    passed: bool,
    n: int,
) -> dict:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("guardrail rates must be in [0, 1]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("guardrail counts must be positive")
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
        "n": n,
    }


def route_guardrails(audit: Mapping) -> dict[str, dict]:
    route = float(audit["route_rate"])
    tail = float(audit["low_use_high_entropy_external_rate"])
    structure = float(audit["rules_top_centrality_internal_rate"])
    return {
        "route_rate": _rate_measurement(
            route,
            threshold=[0.40, 0.60],
            comparison="inside",
            passed=0.40 <= route <= 0.60,
            n=audit["route_total"],
        ),
        "tail_external": _rate_measurement(
            tail,
            threshold=0.80,
            comparison=">=",
            passed=tail >= 0.80,
            n=audit["low_use_high_entropy_total"],
        ),
        "structure_internal": _rate_measurement(
            structure,
            threshold=0.80,
            comparison=">=",
            passed=structure >= 0.80,
            n=audit["rules_top_centrality_total"],
        ),
    }


def mask_ledger_guardrail(audit: Mapping, *, condition: str) -> dict:
    if condition not in ("dense", "split", "random"):
        raise ValueError(f"unexpected training condition: {condition}")
    unmasked = audit["unmasked_external_payloads"]
    external_total = audit["external_payload_occurrences"]
    masked_protected = audit["masked_rule_action_answer_targets"]
    protected_total = audit["rule_action_answer_targets"]
    for name, value in (
        ("unmasked_external_payloads", unmasked),
        ("external_payload_occurrences", external_total),
        ("masked_rule_action_answer_targets", masked_protected),
        ("rule_action_answer_targets", protected_total),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if external_total <= 0 or protected_total <= 0:
        raise ValueError("mask ledger denominators must be positive")
    if unmasked > external_total or masked_protected > protected_total:
        raise ValueError("mask ledger violations exceed audited occurrences")
    external_mask_applicable = condition == "split"
    violations = masked_protected + (
        unmasked if external_mask_applicable else 0
    )
    return {
        "value": violations,
        "condition": condition,
        "external_mask_applicable": external_mask_applicable,
        "unmasked_external_payloads": unmasked,
        "masked_rule_action_answer_targets": masked_protected,
        "threshold": 0,
        "comparison": "==",
        "passed": violations == 0,
        "n": external_total + protected_total,
    }
