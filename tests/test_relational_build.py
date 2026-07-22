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
)
from corpusgen.graph_records import RenderedRecord, ScheduleEntry, TaggedSegment
from train.tokenizer import get_tok


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
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("relational-corpus")
    cfg = RelationalBuildConfig(
        n_entities=64,
        total_tokens=60_000,
        data_seed=17,
        world_size=64,
        eval_pairs_per_task=4,
    )
    report = build_relational_corpus(cfg, get_tok(), _bed_stream(), out)
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


def test_graph_policy_eval_and_manifests_are_portable(built):
    out, cfg, _ = built
    graph = _read_jsonl(out / "graph.jsonl")
    policy = json.loads((out / "route-policy.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    eval_manifest = json.loads((out / "eval-manifest.json").read_text())
    originals = _read_jsonl(out / "eval" / "original.jsonl")
    counterfactuals = _read_jsonl(out / "eval" / "counterfactual.jsonl")
    eval_graph = _read_jsonl(out / "eval" / "graph.jsonl")

    assert len(graph) == cfg.n_entities * 6
    assert 0.40 <= policy["calibration"]["route_rate"] <= 0.60
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
        "fresh_sources_disjoint": True,
    }

    for artifact in manifest["artifacts"]:
        relative = Path(artifact["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert _sha256(out / relative) == artifact["sha256"]


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


def test_same_seed_rebuild_has_identical_artifact_hashes(built, tmp_path):
    first_out, cfg, _ = built
    second_out = tmp_path / "rerun"
    build_relational_corpus(cfg, get_tok(), _bed_stream(), second_out)

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
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_relational_corpus.py",
            "--help",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--bed-jsonl" in completed.stdout
