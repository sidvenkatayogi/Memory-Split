#!/usr/bin/env python
"""Run the deterministic two-step relational pipeline on CPU."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from corpusgen.relational_build import (
    RelationalBuildConfig,
    build_relational_corpus,
)
from corpusgen.records import QAItem
from evals.relational_generate import decode_items
from evals.relational_metrics import EXPECTED_TASKS
from organizer.graph_store import AtomicGraphStore
from scripts.run_relational_evals import (
    _states_to_rows,
    _summary,
    _write_jsonl,
    store_for_item,
)
from train.tokenizer import get_tok
from train.trainer import Trainer


SMOKE_FIXTURE = {
    "n_entities": 32,
    "total_tokens": 40_000,
    "data_seed": 1,
    "world_size": 32,
    "eval_pairs_per_task": 4,
    "eval_pairs_per_world": 4,
    "route_stats_pairs_per_task": 64,
    "guardrail_items": 4,
    "shared_text_eval_count": 4,
}
SMOKE_STEPS = 2
_MODEL = {
    "n_layer": 1,
    "n_head": 1,
    "d_model": 8,
    "ctx": 320,
    "vocab_size": 50_304,
}
_BED = (
    "Glaciers carved the valley and left long ridges of gravel behind.",
    "Wind turbines convert moving air into electricity for the local grid.",
    "The old observatory records each comet crossing the night sky.",
    "Bees communicate the location of food through patterned movements.",
)


def _bed_stream():
    for index in itertools.count():
        yield f"{_BED[index % len(_BED)]} Deterministic passage {index}."


def _trainer_config(
    root: Path,
    corpus: Path,
    arm: str,
    *,
    steps: int,
    device: str,
) -> dict:
    return {
        "condition": arm,
        "model": dict(_MODEL),
        "train_bin": str(corpus / "train.bin"),
        "train_weights": str(corpus / f"{arm}.weights.bin"),
        "micro_batch_size": 1,
        "tokens_per_step": _MODEL["ctx"],
        "max_steps": steps,
        "lr": 1e-3,
        "warmup_steps": 1,
        "seed": 19,
        "device": device,
        "out_dir": str(root / "runs" / arm),
        "log_every": steps,
        "eval_every": steps,
        "snap_frac": 1.0,
        "ckpt_minutes": 999,
    }


def _same_state(left, right) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name]) for name in left_state
    )


def _resume_is_exact(trainer: Trainer, root: Path) -> bool:
    resumed_cfg = dict(
        trainer.cfg,
        out_dir=str(root / "runs" / "resume-check"),
    )
    resumed = Trainer(resumed_cfg)
    resumed.load_ckpt(trainer.ckpt_path)
    if (
        resumed.step != trainer.step
        or resumed.data.state_dict() != trainer.data.state_dict()
        or not _same_state(trainer.model, resumed.model)
    ):
        return False

    expected_batch = trainer.data.next_weighted_batch()
    resumed_batch = resumed.data.next_weighted_batch()
    if not all(
        torch.equal(expected, actual)
        for expected, actual in zip(expected_batch, resumed_batch)
    ):
        return False

    with torch.no_grad():
        _, expected_loss = trainer.model(
            expected_batch[0],
            expected_batch[1],
            target_weights=expected_batch[2],
        )
        _, resumed_loss = resumed.model(
            resumed_batch[0],
            resumed_batch[1],
            target_weights=resumed_batch[2],
        )
    return torch.equal(expected_loss, resumed_loss)


def _load_smoke_items(corpus: Path) -> list[QAItem]:
    items = []
    for name in ("original.jsonl", "counterfactual.jsonl"):
        items.extend(
            QAItem(**json.loads(line))
            for line in (corpus / "eval" / name).read_text().splitlines()
            if line.strip()
        )
    expected_pairs = SMOKE_FIXTURE["eval_pairs_per_task"]
    for task in EXPECTED_TASKS:
        task_items = [item for item in items if item.task == task]
        pairs: dict[str, set[str]] = {}
        for item in task_items:
            pair_id = str(item.meta["pair_id"])
            pairs.setdefault(pair_id, set()).add(str(item.meta["variant"]))
        if (
            len(task_items) != 2 * expected_pairs
            or len(pairs) != expected_pairs
            or any(
                variants != {"original", "counterfactual"}
                for variants in pairs.values()
            )
        ):
            raise ValueError(f"incomplete smoke eval pairs for {task}")
    return items


def _evaluate_modes(
    root: Path,
    corpus: Path,
    trainer: Trainer,
    tok,
) -> list[str]:
    items = _load_smoke_items(corpus)
    base_store = AtomicGraphStore.load(corpus / "eval" / "graph.jsonl")
    modes = []
    trainer.model.eval()
    for memory in ("off", "on"):
        memory_on = memory == "on"
        states = decode_items(
            trainer.model,
            tok,
            items,
            lambda item, enabled=memory_on: store_for_item(
                base_store,
                item,
                memory_on=enabled,
            ),
            device="cpu",
            batch_size=8,
        )
        rows = _states_to_rows(items, states)
        mode_dir = root / "evals" / f"memory_{memory}"
        _write_jsonl(mode_dir / "rows.jsonl", rows)
        summary = _summary(
            rows,
            SMOKE_FIXTURE["eval_pairs_per_task"],
            memory,
        )
        (mode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        modes.append(memory)
    return modes


def run_smoke(
    out_dir: Path | str,
    *,
    steps: int = SMOKE_STEPS,
    device: str = "cpu",
) -> dict:
    """Build, train, resume, and evaluate the real tiny relational pipeline."""

    if steps != SMOKE_STEPS:
        raise ValueError("the local smoke contract requires exactly two steps")
    if device != "cpu":
        raise ValueError("the local smoke contract requires device='cpu'")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"smoke output directory must be empty: {root}")

    tok = get_tok()
    corpus = root / "corpus"
    build_report = build_relational_corpus(
        RelationalBuildConfig(**SMOKE_FIXTURE),
        tok,
        _bed_stream(),
        corpus,
    )
    if not all(build_report["checks"].values()):
        raise AssertionError("smoke corpus failed relational build checks")

    dense = Trainer(
        _trainer_config(root, corpus, "dense", steps=steps, device=device)
    )
    split = Trainer(
        _trainer_config(root, corpus, "split", steps=steps, device=device)
    )
    initial_state = {
        name: value.detach().clone()
        for name, value in dense.model.state_dict().items()
    }
    dense.model.load_state_dict(initial_state)
    split.model.load_state_dict(initial_state)
    if not _same_state(dense.model, split.model):
        raise AssertionError("Dense and Split initial states differ")

    dense.train_steps(steps)
    split.train_steps(steps)
    resume_exact = _resume_is_exact(dense, root)
    modes = _evaluate_modes(root, corpus, split, tok)
    pair_count = SMOKE_FIXTURE["eval_pairs_per_task"]
    pairs_complete = all(
        json.loads(
            (root / "evals" / f"memory_{mode}" / "summary.json").read_text()
        )["n_pairs_per_task"]
        == pair_count
        for mode in modes
    )

    report = {
        "shared_stream": (
            dense.cfg["train_bin"] == split.cfg["train_bin"]
            and dense.data.n_tokens == split.data.n_tokens
        ),
        "dense_steps": dense.step,
        "split_steps": split.step,
        "resume_exact": resume_exact,
        "memory_modes": modes,
        "pairs_complete": pairs_complete,
    }
    if not (
        report["shared_stream"]
        and report["dense_steps"] == steps
        and report["split_steps"] == steps
        and report["resume_exact"]
        and report["memory_modes"] == ["off", "on"]
        and report["pairs_complete"]
    ):
        raise AssertionError(f"local relational smoke failed: {report}")
    (root / "smoke-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic two-step relational CPU smoke."
    )
    parser.add_argument("--out", default="outputs/relational-smoke")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)
    report = run_smoke(args.out, device=args.device)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
