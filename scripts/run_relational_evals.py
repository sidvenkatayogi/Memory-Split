#!/usr/bin/env python
"""Evaluate one standard-GPT run in both atomic-memory modes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.records import QAItem
from evals.relational_generate import (
    GraphDecodeState,
    OverlayStore,
    decode_items,
)
from evals.relational_metrics import (
    EXPECTED_TASKS,
    assert_expected_counts,
    counterfactual_pair_accuracy,
    mask_ledger_guardrail,
    path_diagnostics,
    path_metrics,
    route_guardrails,
)
from evals.scorers import normalize_answer
from organizer.graph_store import AtomicGraphStore
from train.model import GPT, GPTConfig, PRESETS
from train.trainer import pick_device


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL input is empty: {path}")
    return rows


def _load_eval_items(data_dir: Path, expected_pairs: int) -> list[QAItem]:
    originals = [
        QAItem(**row)
        for row in _read_jsonl(data_dir / "eval" / "original.jsonl")
    ]
    counterfactuals = [
        QAItem(**row)
        for row in _read_jsonl(
            data_dir / "eval" / "counterfactual.jsonl"
        )
    ]
    items = originals + counterfactuals
    rows_by_task = {
        task: [item.__dict__ for item in items if item.task == task]
        for task in EXPECTED_TASKS
    }
    assert_expected_counts(rows_by_task, expected_pairs)
    return items


def _item_meta(item) -> dict:
    meta = item["meta"] if isinstance(item, dict) else item.meta
    if not isinstance(meta, dict):
        raise ValueError("eval item meta must be a mapping")
    return meta


def store_for_item(
    base: AtomicGraphStore,
    item,
    *,
    memory_on: bool,
) -> AtomicGraphStore | OverlayStore | None:
    """Return the only evaluator toggle: base/overlay store, or ``None``."""

    if not isinstance(memory_on, bool):
        raise TypeError("memory_on must be Boolean")
    meta = _item_meta(item)
    variant = meta["variant"]
    changed = meta["changed_row"]
    if variant == "original":
        if changed is not None:
            raise ValueError("original eval items cannot contain a changed row")
        selected = base
    elif variant == "counterfactual":
        if not isinstance(changed, dict):
            raise ValueError(
                "counterfactual eval items require one changed row"
            )
        selected = OverlayStore(base, GraphRow.from_json(changed))
    else:
        raise ValueError(f"unexpected eval variant: {variant}")
    return selected if memory_on else None


def _action_json(action: GraphAction) -> list:
    return [
        action.source_slot,
        action.relation_id,
        action.direction,
        action.read,
        action.halt,
    ]


def _gold_actions(item) -> list[GraphAction]:
    meta = _item_meta(item)
    task = item["task"] if isinstance(item, dict) else item.task
    addresses = meta["gold_addresses"]
    if not isinstance(addresses, list) or not addresses:
        raise ValueError("gold_addresses must contain at least one hop")
    if task not in EXPECTED_TASKS:
        raise ValueError(f"unexpected relational task: {task}")
    actions = []
    for index, raw in enumerate(addresses):
        if not isinstance(raw, list) or len(raw) != 3:
            raise ValueError("gold graph addresses require three fields")
        _, relation, direction = raw
        source_slot = 0 if task == "path_composition" else index
        actions.append(
            GraphAction(
                source_slot=source_slot,
                relation_id=str(relation),
                direction=direction,
                read=True,
                halt=False,
            )
        )
    return actions


def _states_to_rows(items, states: list[GraphDecodeState]) -> list[dict]:
    materialized = list(items)
    if len(materialized) != len(states):
        raise ValueError("every eval item requires exactly one decoded state")
    rows = []
    for item, state in zip(materialized, states):
        if (
            len(state.actions) != 6
            or len(state.rows) != 6
            or len(state.provisional_answers) != 6
        ):
            raise ValueError("every decoded state must contain six steps")
        meta = _item_meta(item)
        gold_actions = _gold_actions(item)
        gold_addresses = [
            GraphAddress(int(source), str(relation), direction)
            for source, relation, direction in meta["gold_addresses"]
        ]
        read_pairs = [
            (action, row)
            for action, row in zip(state.actions, state.rows)
            if action.read
        ]
        correct_referents = []
        for index, address in enumerate(gold_addresses):
            returned = (
                read_pairs[index][1] if index < len(read_pairs) else None
            )
            correct_referents.append(
                returned is not None and returned.address == address
            )
        prediction = state.provisional_answers[-1]
        answer = item["answer"] if isinstance(item, dict) else item.answer
        qid = item["qid"] if isinstance(item, dict) else item.qid
        task = item["task"] if isinstance(item, dict) else item.task
        predicted_reads = [action for action, _ in read_pairs]
        rows.append(
            {
                "qid": qid,
                "task": task,
                "pair_id": str(meta["pair_id"]),
                "variant": str(meta["variant"]),
                "correct": (
                    normalize_answer(prediction)
                    == normalize_answer(str(answer))
                ),
                "pred": prediction,
                "answer": answer,
                "actions": [
                    _action_json(action) for action in predicted_reads
                ],
                "all_actions": [
                    _action_json(action) for action in state.actions
                ],
                "gold_actions": [
                    _action_json(action) for action in gold_actions
                ],
                "correct_referents": correct_referents,
                "misses": state.misses,
                "malformed": state.malformed,
                "excess_reads": max(
                    0, len(predicted_reads) - len(gold_actions)
                ),
                "halt_step": state.halt_step,
                "n_steps": len(state.actions),
                "meta": meta,
            }
        )
    return rows


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def _summary(rows: list[dict], expected_pairs: int, memory: str) -> dict:
    rows_by_task = {
        task: [row for row in rows if row["task"] == task]
        for task in EXPECTED_TASKS
    }
    assert_expected_counts(rows_by_task, expected_pairs)
    task_summary = {
        task: {
            "counterfactual_pair_accuracy": counterfactual_pair_accuracy(
                task_rows, expected_pairs=expected_pairs
            ),
            "path": path_metrics(task_rows),
            "path_diagnostics": path_diagnostics(task_rows),
            "n_rows": len(task_rows),
            "n_pairs": expected_pairs,
        }
        for task, task_rows in rows_by_task.items()
    }
    composite = sum(
        task_summary[task]["counterfactual_pair_accuracy"]
        for task in EXPECTED_TASKS
    ) / len(EXPECTED_TASKS)
    return {
        "memory": memory,
        "tasks": task_summary,
        "primary_composite": composite,
        "n_rows": len(rows),
        "n_pairs_per_task": expected_pairs,
    }


def _validate_accuracy(value: dict, name: str) -> None:
    accuracy = float(value["accuracy"])
    n = value["n"]
    if not 0 <= accuracy <= 1:
        raise ValueError(f"{name} accuracy must be in [0, 1]")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"{name} n must be positive")


def _load_guardrail_measurements(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    required = {
        "route",
        "mask",
        "recognition_store_off",
        "factual_recall",
        "internal_accuracy",
        "language",
    }
    if set(value) != required:
        raise ValueError(
            "guardrail measurement keys mismatch; "
            f"missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    route = route_guardrails(value["route"])
    mask = mask_ledger_guardrail(value["mask"])
    recognition = value["recognition_store_off"]
    _validate_accuracy(recognition, "recognition_store_off")
    for mode in ("on", "off"):
        _validate_accuracy(
            value["factual_recall"][mode],
            f"factual_recall.{mode}",
        )
    _validate_accuracy(value["internal_accuracy"], "internal_accuracy")
    language = value["language"]
    if float(language["bpb"]) < 0:
        raise ValueError("language BPB must be non-negative")
    if (
        isinstance(language["total_utf8_bytes"], bool)
        or not isinstance(language["total_utf8_bytes"], int)
        or language["total_utf8_bytes"] <= 0
    ):
        raise ValueError("language byte count must be positive")
    return {
        **value,
        "within_run_guardrails": {
            "route": route,
            "mask": mask,
        },
    }


def _resolve_data_dir(cfg: dict, override: str | None) -> Path:
    if override is not None:
        return Path(override)
    if "data_dir" in cfg:
        return Path(cfg["data_dir"])
    if "data_rel" in cfg:
        root = os.environ.get("DATA_ROOT")
        if root is None:
            raise ValueError("DATA_ROOT is required when config uses data_rel")
        return Path(root) / cfg["data_rel"]
    raise KeyError("run config requires data_dir or data_rel")


def _load_model(
    run: Path,
    checkpoint: str,
    device: str,
) -> tuple[GPT, dict]:
    config_path = run / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    model_value = cfg["model"]
    if isinstance(model_value, str):
        if model_value not in PRESETS:
            raise ValueError(f"unknown model preset: {model_value}")
        model_cfg = replace(PRESETS[model_value])
    elif isinstance(model_value, dict):
        model_cfg = GPTConfig(**model_value)
    else:
        raise ValueError("model config must be a preset name or mapping")
    if "ctx" in cfg:
        model_cfg.ctx = int(cfg["ctx"])
    model = GPT(model_cfg)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run / checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--data-dir")
    parser.add_argument("--guardrails-json", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--expected-pairs",
        type=int,
        default=10_000,
        help="frozen value is 10000; smaller values are for smoke tests",
    )
    args = parser.parse_args()

    if args.expected_pairs <= 0:
        raise ValueError("expected-pairs must be positive")
    run = Path(args.run)
    device = pick_device(args.device)
    model, cfg = _load_model(run, args.checkpoint, device)
    data_dir = _resolve_data_dir(cfg, args.data_dir)
    items = _load_eval_items(data_dir, args.expected_pairs)
    base_store = AtomicGraphStore.load(data_dir / "eval" / "graph.jsonl")
    measurements = _load_guardrail_measurements(
        Path(args.guardrails_json)
    )

    from train.tokenizer import get_tok

    tok = get_tok()
    output = run / "evals"
    output.mkdir(parents=True, exist_ok=True)
    mode_summaries = {}
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            model,
            tok,
            items,
            lambda item, enabled=memory_on: store_for_item(
                base_store, item, memory_on=enabled
            ),
            device=device,
            batch_size=args.batch_size,
        )
        rows = _states_to_rows(items, states)
        mode_dir = output / f"memory_{memory}"
        _write_jsonl(mode_dir / "rows.jsonl", rows)
        summary = _summary(rows, args.expected_pairs, memory)
        (mode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        mode_summaries[memory] = summary

    (output / "guardrails.json").write_text(
        json.dumps(measurements, indent=2, sort_keys=True) + "\n"
    )
    combined = {
        "condition": cfg["condition"],
        "modes": mode_summaries,
        "guardrails": measurements,
    }
    (output / "relational_summary.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(combined, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
