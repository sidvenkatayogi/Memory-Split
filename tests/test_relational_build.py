from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from corpusgen import relational_build as relational
from corpusgen.relational_build import (
    EncodedSpan,
    FactCost,
    RelationalBuildConfig,
    RoutePolicy,
    _encode_record,
    build_relational_corpus,
    calibrate_write_cost,
    derive_weights,
    load_route_policy,
)
from corpusgen.graph_records import RenderedRecord, ScheduleEntry, TaggedSegment
from train.tokenizer import get_tok


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_SHA256 = (
    "0214cd5dd63e7534dc786569f8b789b6c614ffbe219c84887bd3a71b57bcf058"
)


@pytest.fixture(scope="session")
def route_policy_fixture(tmp_path_factory):
    path = tmp_path_factory.mktemp("route-policy") / "fixture-policy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": {
                    "hop_cost": 0.25,
                    "read_cost": 0.25,
                    "write_cost": 1.0,
                },
                "policy_sha256": POLICY_SHA256,
                "calibration": {"source": "explicit test fixture"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def test_write_cost_is_selected_without_semantic_labels():
    facts = [
        FactCost("a", entropy=12, exposures=1, expected_reads=1, expected_hops=1),
        FactCost("b", entropy=8, exposures=1, expected_reads=1, expected_hops=1),
        FactCost(
            "c",
            entropy=1,
            exposures=16,
            expected_reads=20,
            expected_hops=2,
        ),
        FactCost(
            "d",
            entropy=1,
            exposures=16,
            expected_reads=20,
            expected_hops=2,
        ),
    ]

    policy = calibrate_write_cost(facts)

    assert 0.40 <= policy.route_rate(facts) <= 0.60
    assert policy.is_external(facts[0])
    assert not policy.is_external(facts[-1])


def test_split_and_random_weights_mask_only_their_allowed_spans():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 10, "action"),
        EncodedSpan(10, 13, "payload", "external", external),
        EncodedSpan(13, 15, "action"),
        EncodedSpan(15, 18, "random_control"),
        EncodedSpan(18, 25, "rule"),
        EncodedSpan(25, 35, "provisional_answer"),
        EncodedSpan(35, 50, "action"),
        EncodedSpan(50, 53, "plain"),
        EncodedSpan(53, 100, "final_answer"),
    ]
    policy = RoutePolicy(write_cost=1)

    split = derive_weights("split", spans, policy, random.Random(7))
    random_control = derive_weights("random", spans, policy, random.Random(7))

    assert np.flatnonzero(split == 0).tolist() == [10, 11, 12]
    assert np.flatnonzero(random_control == 0).tolist() == [15, 16, 17]
    assert random_control[50:53].tolist() == [1, 1, 1]


def test_random_matching_samples_deterministically_within_exact_key_pool():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 10, "action"),
        EncodedSpan(10, 13, "payload", "external", external),
        EncodedSpan(13, 15, "action"),
        EncodedSpan(15, 18, "random_control"),
        EncodedSpan(18, 21, "random_control"),
        EncodedSpan(21, 100, "action"),
    ]
    policy = RoutePolicy(write_cost=1)

    first = derive_weights("random", spans, policy, random.Random(23))
    second = derive_weights("random", spans, policy, random.Random(23))

    assert np.array_equal(first, second)
    assert np.flatnonzero(first == 0).tolist() in (
        [15, 16, 17],
        [18, 19, 20],
    )


def test_expected_external_ranges_are_collected_before_weight_derivation():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 5, "action"),
        EncodedSpan(5, 8, "payload", "external", external),
        EncodedSpan(8, 12, "plain"),
    ]

    expected = relational.collect_expected_external_ranges(
        spans,
        RoutePolicy(write_cost=1),
    )

    assert [(item.start, item.end, item.fact_id) for item in expected] == [
        (5, 8, "external")
    ]


def test_split_coverage_validation_rejects_missing_or_extra_zeros():
    external = FactCost(
        "external",
        entropy=12,
        exposures=1,
        expected_reads=0,
        expected_hops=0,
    )
    spans = [
        EncodedSpan(0, 5, "action"),
        EncodedSpan(5, 8, "payload", "external", external),
        EncodedSpan(8, 12, "plain"),
    ]
    expected = relational.collect_expected_external_ranges(
        spans,
        RoutePolicy(write_cost=1),
    )

    with pytest.raises(ValueError, match="expected external payload"):
        relational.validate_split_coverage(
            expected,
            spans,
            np.ones(12, dtype=np.uint8),
            [],
        )

    weights = np.ones(12, dtype=np.uint8)
    weights[5:8] = 0
    weights[9] = 0
    actual = [(5, 8, spans[1])]
    with pytest.raises(ValueError, match="protected nonpayload"):
        relational.validate_split_coverage(expected, spans, weights, actual)


def test_record_encoding_rejects_token_ids_outside_uint16():
    class OversizedTokenizer:
        EOT = 1

        @staticmethod
        def encode_tagged_segments(segments):
            del segments
            return [70_000], ["plain"], [None]

    record = RenderedRecord(
        segments=(TaggedSegment("ignored", "plain"),),
        schedule=ScheduleEntry("bed", "bed-0", 0, 0),
    )

    with pytest.raises(ValueError, match="token id does not fit uint16"):
        _encode_record(OversizedTokenizer(), record, {})


def test_eval_pair_count_must_be_positive():
    with pytest.raises(
        ValueError,
        match="eval_pairs_per_task must be positive",
    ):
        RelationalBuildConfig(
            n_entities=64,
            total_tokens=60_000,
            data_seed=17,
            eval_pairs_per_task=0,
        )


def _bed_stream():
    passages = (
        "Glaciers carved the valley and left long ridges of gravel behind.",
        "Wind turbines convert moving air into electricity for the local grid.",
        "The old observatory records the path of each comet across the sky.",
        "Bees communicate the location of food through a sequence of movements.",
    )
    index = 0
    while True:
        yield f"{passages[index % len(passages)]} Passage {index}."
        index += 1


@pytest.fixture(scope="module")
def built(tmp_path_factory, route_policy_fixture):
    out = tmp_path_factory.mktemp("relational-corpus")
    cfg = RelationalBuildConfig(
        n_entities=64,
        total_tokens=60_000,
        data_seed=17,
        world_size=64,
        eval_pairs_per_task=4,
        guardrail_items=8,
        shared_text_eval_count=4,
    )
    report = build_relational_corpus(
        cfg,
        get_tok(),
        _bed_stream(),
        out,
        route_policy_path=route_policy_fixture,
        expected_policy_sha256=POLICY_SHA256,
    )
    return out, cfg, report


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_shared_stream_sidecars_are_aligned_and_checks_pass(built):
    out, _, report = built
    tokens = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    dense = np.memmap(out / "dense.weights.bin", dtype=np.uint8, mode="r")
    split = np.memmap(out / "split.weights.bin", dtype=np.uint8, mode="r")
    random_control = np.memmap(
        out / "random.weights.bin",
        dtype=np.uint8,
        mode="r",
    )

    assert len(tokens) == len(dense) == len(split) == len(random_control)
    assert len(tokens) >= 60_000
    assert (dense == 1).all()
    assert report["checks"] == {
        "external_payload_coverage": True,
        "external_payload_ranges_exact": True,
        "random_mass_within_1pct": True,
        "random_span_histogram_within_1pct": True,
        "random_position_histogram_within_1pct": True,
        "mixture_within_1pct": True,
        "graph_mixture_exact": True,
        "protected_roles_unmasked": True,
        "eval_validity": True,
        "schedule_hash_stable": True,
        "sidecars_aligned": True,
        "manifests_relative": True,
    }


def test_schedule_preserves_mixtures_curriculum_and_six_step_traces(built):
    out, cfg, _ = built
    tok = get_tok()
    tokens = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    schedule = _read_jsonl(out / "schedule.jsonl")
    components = Counter(row["component"] for row in schedule)
    graph_kinds = Counter(
        row["graph_subcomponent"]
        for row in schedule
        if row["component"] == "graph"
    )

    assert components.keys() == {"bed", "graph", "reasoning"}
    assert graph_kinds == {
        "peripheral": graph_kinds.total() * 7 // 10,
        "central": graph_kinds.total() * 2 // 10,
        "rule": graph_kinds.total() // 10,
    }

    for row in schedule:
        if row["component"] != "reasoning":
            continue
        record_tokens = tokens[row["token_start"] : row["token_end"]]
        assert int((record_tokens == tok.GRAPH_START).sum()) == 6
        assert int((record_tokens == tok.ANSWER_STATE).sum()) == 6
        position = row["token_start"] / cfg.total_tokens
        if position < 0.20:
            assert row["curriculum_band"] == 1
        elif position < 0.50:
            assert row["curriculum_band"] in {1, 2}
        else:
            assert row["curriculum_band"] in {1, 2, 4}


def test_mask_ledger_proves_coverage_and_matched_random_histogram(built):
    out, _, _ = built
    split = np.memmap(out / "split.weights.bin", dtype=np.uint8, mode="r")
    random_control = np.memmap(
        out / "random.weights.bin",
        dtype=np.uint8,
        mode="r",
    )
    ledger = _read_jsonl(out / "mask-ledger.jsonl")
    expected_rows = [
        row for row in ledger if row["condition"] == "expected_split"
    ]
    split_rows = [row for row in ledger if row["condition"] == "split"]
    random_rows = [row for row in ledger if row["condition"] == "random"]

    assert split_rows
    assert [
        (row["start"], row["end"], row["fact_id"])
        for row in expected_rows
    ] == [
        (row["start"], row["end"], row["fact_id"])
        for row in split_rows
    ]
    assert all("position_bin" in row for row in split_rows + random_rows)
    split_histogram = Counter(
        (row["length"], row["position_bin"]) for row in split_rows
    )
    random_histogram = Counter(
        (row["length"], row["position_bin"]) for row in random_rows
    )
    mismatch = sum(
        abs(split_histogram[key] - random_histogram[key])
        for key in set(split_histogram) | set(random_histogram)
    )
    assert mismatch / len(split_rows) <= 0.01
    assert (
        abs(int((split == 0).sum()) - int((random_control == 0).sum()))
        / int((split == 0).sum())
        <= 0.01
    )
    for row in split_rows:
        assert row["role"] == "payload"
        assert not split[row["start"] : row["end"]].any()
    for row in random_rows:
        assert row["role"] == "random_control"
        assert not random_control[row["start"] : row["end"]].any()
    selected_random_positions = np.zeros(len(random_control), dtype=bool)
    for row in random_rows:
        selected_random_positions[row["start"] : row["end"]] = True
    assert np.array_equal(random_control == 0, selected_random_positions)


def test_graph_policy_eval_and_manifests_are_portable(
    built,
    route_policy_fixture,
):
    out, cfg, _ = built
    graph = _read_jsonl(out / "graph.jsonl")
    policy = json.loads((out / "route-policy.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    eval_manifest = json.loads((out / "eval-manifest.json").read_text())
    originals = _read_jsonl(out / "eval" / "original.jsonl")
    counterfactuals = _read_jsonl(out / "eval" / "counterfactual.jsonl")
    eval_graph = _read_jsonl(out / "eval" / "graph.jsonl")

    assert len(graph) == cfg.n_entities * 6
    assert (out / "route-policy.json").read_bytes() == (
        route_policy_fixture.read_bytes()
    )
    assert policy["policy_sha256"] == POLICY_SHA256
    policy_text = json.dumps(policy, sort_keys=True)
    for forbidden in ("audit_class", "answer", "target", "task", "outcome"):
        assert forbidden not in policy_text

    assert len(originals) == len(counterfactuals) == 3 * cfg.eval_pairs_per_task
    assert {row["meta"]["pair_id"] for row in originals} == {
        row["meta"]["pair_id"] for row in counterfactuals
    }
    assert all(
        original["answer"] != counterfactual["answer"]
        for original, counterfactual in zip(originals, counterfactuals)
    )
    assert {row["source_id"] for row in graph}.isdisjoint(
        {row["source_id"] for row in eval_graph}
    )
    assert eval_manifest["checks"] == {
        "exact_task_counts": True,
        "two_variants_per_pair": True,
        "answer_flips": True,
        "changed_supporting_row": True,
        "explicit_gold_actions": True,
        "fresh_sources_disjoint": True,
    }

    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert _sha256(out / relative) == artifact["sha256"]


def test_standalone_evaluator_loads_builder_produced_raw_eval_items(built):
    from scripts.run_relational_evals import _load_eval_items

    out, cfg, _ = built
    items = _load_eval_items(out, cfg.eval_pairs_per_task)

    assert len(items) == 6 * cfg.eval_pairs_per_task
    assert all(not hasattr(item, "correct") for item in items)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unexpected", "task set mismatch"),
        ("missing", "task set mismatch"),
        ("misstratified", "expected 8 rows"),
    ],
)
def test_standalone_evaluator_rejects_observed_task_stratum_violations(
    built,
    tmp_path,
    mutation,
    message,
):
    from scripts.run_relational_evals import _load_eval_items

    out, cfg, _ = built
    copied = tmp_path / mutation
    eval_dir = copied / "eval"
    eval_dir.mkdir(parents=True)
    originals = _read_jsonl(out / "eval" / "original.jsonl")
    counterfactuals = _read_jsonl(
        out / "eval" / "counterfactual.jsonl"
    )

    if mutation == "unexpected":
        for rows, variant in (
            (originals, "original"),
            (counterfactuals, "counterfactual"),
        ):
            extra = json.loads(json.dumps(rows[0]))
            extra["qid"] = f"unexpected-0-{variant}"
            extra["task"] = "unexpected_task"
            extra["meta"]["pair_id"] = "unexpected-0"
            extra["meta"]["variant"] = variant
            rows.append(extra)
    elif mutation == "missing":
        originals = [
            row for row in originals if row["task"] != "date_ordering"
        ]
        counterfactuals = [
            row
            for row in counterfactuals
            if row["task"] != "date_ordering"
        ]
    else:
        source_task = counterfactuals[0]["task"]
        counterfactuals[0]["task"] = next(
            task
            for task in (
                "path_composition",
                "date_ordering",
                "balanced_equality",
            )
            if task != source_task
        )

    for name, rows in (
        ("original.jsonl", originals),
        ("counterfactual.jsonl", counterfactuals),
    ):
        (eval_dir / name).write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    with pytest.raises(ValueError, match=message):
        _load_eval_items(copied, cfg.eval_pairs_per_task)


def test_builder_commits_all_guardrail_eval_inputs(built):
    out, cfg, report = built
    recognition = _read_jsonl(out / "eval" / "recognition.jsonl")
    factual = _read_jsonl(out / "eval" / "factual.jsonl")
    factual_graph = _read_jsonl(out / "eval" / "factual-graph.jsonl")
    internal = _read_jsonl(out / "eval" / "internal.jsonl")
    shared_text = _read_jsonl(out / "eval" / "shared_text.jsonl")
    route = json.loads((out / "eval" / "route-audit.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    artifact_names = {item["path"] for item in manifest["artifacts"]}

    assert len(recognition) == cfg.guardrail_items
    assert len(factual) == cfg.guardrail_items
    assert len(internal) == cfg.guardrail_items
    assert len(shared_text) == cfg.shared_text_eval_count
    assert all(
        len(item["choices"]) == 4
        and 0 <= item["answer_index"] < 4
        for item in recognition + internal
    )
    assert all(
        item["task"] == "factual_recall"
        and len(item["meta"]["gold_actions"]) == 6
        and item["meta"]["route"] == "external"
        for item in factual
    )
    factual_addresses = {
        (
            row["source_id"],
            row["relation_id"],
            row["direction"],
        )
        for row in factual_graph
    }
    assert all(
        tuple(item["meta"]["gold_addresses"][0]) in factual_addresses
        for item in factual
    )
    assert {item["kind"] for item in internal} == {
        "rule",
        "central_fact",
    }
    assert set(route) == {
        "route_rate",
        "route_total",
        "low_use_high_entropy_external_rate",
        "low_use_high_entropy_total",
        "rules_top_centrality_internal_rate",
        "rules_top_centrality_total",
    }
    assert 0.40 <= route["route_rate"] <= 0.60
    assert route["low_use_high_entropy_external_rate"] >= 0.80
    assert route["rules_top_centrality_internal_rate"] >= 0.80
    assert report["eval"]["guardrail_items"] == cfg.guardrail_items
    assert {
        "eval/recognition.jsonl",
        "eval/factual.jsonl",
        "eval/factual-graph.jsonl",
        "eval/internal.jsonl",
        "eval/shared_text.jsonl",
        "eval/route-audit.json",
    } <= artifact_names


def test_real_eval_answer_choices_are_token_prefix_free(built):
    out, _, _ = built
    tok = get_tok()
    choice_sets = [
        item["choices"]
        for name in ("recognition.jsonl", "internal.jsonl")
        for item in _read_jsonl(out / "eval" / name)
    ]
    choice_sets.extend(
        item["meta"]["answer_choices"]
        for name in ("original.jsonl", "counterfactual.jsonl", "factual.jsonl")
        for item in _read_jsonl(out / "eval" / name)
    )

    for choices in choice_sets:
        encoded = [tuple(tok.encode(choice)) for choice in choices]
        assert all(encoded)
        assert len(set(encoded)) == len(encoded)
        assert all(
            not (
                len(left) < len(right)
                and right[: len(left)] == left
            )
            for left in encoded
            for right in encoded
            if left != right
        )


def test_evaluator_produces_guardrails_consumed_by_analysis(built):
    from scripts.analyze_relational import analyze_runs, expected_run_keys
    from scripts.run_relational_evals import produce_guardrail_measurements
    from train.model import GPT, GPTConfig

    out, _, _ = built
    torch.manual_seed(5)
    model = GPT(
        GPTConfig(n_layer=1, n_head=1, d_model=32, ctx=1024)
    ).eval()
    tok = get_tok()
    produced = {
        condition: produce_guardrail_measurements(
            model,
            tok,
            out,
            condition=condition,
            device="cpu",
            batch_size=4,
        )
        for condition in ("dense", "split", "random")
    }

    assert all(
        set(value)
        == {
            "within_run_guardrails",
            "recognition_store_off",
            "factual_recall",
            "internal_accuracy",
            "language",
        }
        for value in produced.values()
    )
    assert produced["split"]["within_run_guardrails"]["mask"]["passed"]
    assert produced["dense"]["within_run_guardrails"]["mask"]["passed"]
    assert not produced["dense"]["within_run_guardrails"]["mask"][
        "external_mask_applicable"
    ]
    assert produced["random"]["within_run_guardrails"]["mask"]["passed"]
    assert set(produced["split"]["internal_accuracy"]["per_kind"]) == {
        "rule",
        "central_fact",
    }

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

    runs = {}
    for key in expected_run_keys():
        _, condition, _, _ = key
        runs[key] = {
            "on": mode_summary("on", 0.5),
            "off": mode_summary("off", 0.5),
            "guardrails": produced[condition],
        }

    result = analyze_runs(runs)
    assert result["run_count"] == 21
    assert result["verdict"] in {
        "invalid",
        "validated",
        "rejected",
        "inconclusive",
    }


def test_guardrail_producer_raises_on_missing_committed_artifact(
    built,
    tmp_path,
):
    from scripts.run_relational_evals import produce_guardrail_measurements
    from train.model import GPT, GPTConfig

    out, _, _ = built
    copied = tmp_path / "incomplete"
    shutil.copytree(out, copied)
    (copied / "eval" / "recognition.jsonl").unlink()
    model = GPT(
        GPTConfig(n_layer=1, n_head=1, d_model=16, ctx=1024)
    ).eval()

    with pytest.raises(FileNotFoundError, match="recognition.jsonl"):
        produce_guardrail_measurements(
            model,
            get_tok(),
            copied,
            condition="dense",
            device="cpu",
            batch_size=4,
        )


def test_eval_validator_rejects_a_nonflipping_twin(built, tmp_path):
    out, cfg, _ = built
    copied = tmp_path / "eval-copy"
    copied.mkdir()
    for name in ("graph.jsonl", "original.jsonl", "counterfactual.jsonl"):
        shutil.copy(out / "eval" / name, copied / name)
    training_graph = tmp_path / "graph.jsonl"
    shutil.copy(out / "graph.jsonl", training_graph)

    counterfactuals = _read_jsonl(copied / "counterfactual.jsonl")
    originals = _read_jsonl(copied / "original.jsonl")
    counterfactuals[0]["answer"] = originals[0]["answer"]
    (copied / "counterfactual.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in counterfactuals)
    )

    with pytest.raises(ValueError, match="answers must flip"):
        relational.validate_eval_sets(
            cfg,
            training_graph,
            copied / "graph.jsonl",
            copied / "original.jsonl",
            copied / "counterfactual.jsonl",
        )


def test_eval_validator_rejects_missing_explicit_gold_actions(
    built,
    tmp_path,
):
    out, cfg, _ = built
    copied = tmp_path / "eval-gold-copy"
    copied.mkdir()
    for name in ("graph.jsonl", "original.jsonl", "counterfactual.jsonl"):
        shutil.copy(out / "eval" / name, copied / name)
    training_graph = tmp_path / "graph.jsonl"
    shutil.copy(out / "graph.jsonl", training_graph)
    originals = _read_jsonl(copied / "original.jsonl")
    del originals[0]["meta"]["gold_actions"]
    (copied / "original.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in originals)
    )

    with pytest.raises(ValueError, match="gold actions"):
        relational.validate_eval_sets(
            cfg,
            training_graph,
            copied / "graph.jsonl",
            copied / "original.jsonl",
            copied / "counterfactual.jsonl",
        )


def test_same_seed_rebuild_has_identical_artifact_hashes(
    built,
    tmp_path,
    route_policy_fixture,
):
    first_out, cfg, _ = built
    second_out = tmp_path / "rerun"
    build_relational_corpus(
        cfg,
        get_tok(),
        _bed_stream(),
        second_out,
        route_policy_path=route_policy_fixture,
        expected_policy_sha256=POLICY_SHA256,
    )

    first = json.loads((first_out / "manifest.json").read_text())
    second = json.loads((second_out / "manifest.json").read_text())
    first_hashes = {
        artifact["path"]: artifact["sha256"]
        for artifact in first["artifacts"]
    }
    second_hashes = {
        artifact["path"]: artifact["sha256"]
        for artifact in second["artifacts"]
    }
    assert first_hashes == second_hashes


def test_all_frozen_seeds_and_loads_use_the_committed_policy_hash():
    from scripts.make_relational_manifest import make_jobs

    jobs = [
        job
        for scale in ("160m", "360m")
        for job in make_jobs(scale)
    ]
    expected_hashes = {job["route_policy_sha256"] for job in jobs}
    assert expected_hashes == {POLICY_SHA256}

    policy_path = REPO_ROOT / "configs" / "route-policy.json"
    loaded_hashes = {
        load_route_policy(
            policy_path,
            expected_policy_sha256=job["route_policy_sha256"],
        )[0].sha256()
        for job in jobs
    }
    assert loaded_hashes == {POLICY_SHA256}


def test_policy_hash_mismatch_fails_before_any_corpus_work(
    tmp_path,
    route_policy_fixture,
):
    cfg = RelationalBuildConfig(
        n_entities=16,
        total_tokens=1,
        data_seed=0,
        world_size=16,
        eval_pairs_per_task=1,
        guardrail_items=1,
        shared_text_eval_count=1,
    )
    out = tmp_path / "must-not-exist"

    def forbidden_bed_stream():
        raise AssertionError("corpus work started before policy validation")
        yield

    with pytest.raises(ValueError, match="expected route policy SHA-256"):
        build_relational_corpus(
            cfg,
            get_tok(),
            forbidden_bed_stream(),
            out,
            route_policy_path=route_policy_fixture,
            expected_policy_sha256="0" * 64,
        )

    assert not out.exists()


def test_bed_jsonl_stream_rewinds_deterministically(tmp_path):
    from scripts.build_relational_corpus import iter_bed_jsonl

    source = tmp_path / "bed.jsonl"
    source.write_text(
        '{"text":"first pinned passage"}\n'
        '{"text":"second pinned passage"}\n'
    )

    assert list(islice(iter_bed_jsonl(source), 5)) == [
        "first pinned passage",
        "second pinned passage",
        "first pinned passage",
        "second pinned passage",
        "first pinned passage",
    ]


def test_corpus_command_runs_as_a_repo_relative_script():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_relational_corpus.py",
            "--help",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--bed-jsonl" in completed.stdout
    assert "--route-policy" in completed.stdout
    assert "--route-policy-sha256" in completed.stdout
    assert "--guardrail-items" in completed.stdout
    assert "--shared-text-eval-count" in completed.stdout


def test_corpus_command_requires_policy_path_and_expected_hash():
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_relational_corpus.py",
            "--out",
            "unused",
            "--entities",
            "16",
            "--tokens",
            "1",
            "--data-seed",
            "0",
            "--bed-jsonl",
            "unused.jsonl",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--route-policy" in completed.stderr
    assert "--route-policy-sha256" in completed.stderr
