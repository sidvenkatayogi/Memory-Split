from __future__ import annotations

from collections import Counter, defaultdict
from itertools import islice

import pytest

from corpusgen.graph_records import GraphAddress
from corpusgen.graph_trace import serialize_return
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
    iter_bed_records,
    iter_graph_records,
    iter_reasoning_records,
    iter_worlds,
)
from organizer.graph_store import AtomicGraphStore
from train.tokenizer import get_tok


RELATIONS = {f"r{index}" for index in range(6)}


def _rows_by_address(world):
    return {fact.row.address: fact.row for fact in world.facts}


def _answer_from_evidence(world, pair, item):
    rows = _rows_by_address(world)
    if item.meta["variant"] == "counterfactual":
        rows[pair.changed_row.address] = pair.changed_row
    used = [
        rows[GraphAddress(int(source), relation, direction)]
        for source, relation, direction in item.meta["gold_addresses"]
    ]
    if item.task == "path_composition":
        compose = sum(int(dict(row.qualifiers)["compose"]) for row in used) % 4
        return f"r{compose}"
    if item.task == "date_ordering":
        return "<|slot_0|>" if used[0].target < used[1].target else "<|slot_1|>"
    if item.task == "balanced_equality":
        return "yes" if used[0].target == used[1].target else "no"
    raise AssertionError(f"unexpected task: {item.task}")


def _path_cursor_oracle(world, pair, item):
    """Traverse from slot 0 using relations, never stored gold addresses."""

    rows = _rows_by_address(world)
    if item.meta["variant"] == "counterfactual":
        rows[pair.changed_row.address] = pair.changed_row
    cursor = int(item.meta["entity_slots"][0])
    addresses = []
    compose = 0
    for relation in item.meta["relations"]:
        address = GraphAddress(cursor, relation, "out")
        row = rows[address]
        addresses.append(address)
        compose = (compose + int(dict(row.qualifiers)["compose"])) % 4
        cursor = int(row.target)
    return addresses, f"r{compose}"


def test_world_has_six_unique_functional_facts_per_entity():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    by_source = defaultdict(list)
    for fact in world.facts:
        by_source[fact.row.source_id].append(fact)

    assert len(world.facts) == 64 * 6
    assert set(by_source) == set(range(64))
    assert all(
        {fact.row.relation_id for fact in facts} == RELATIONS
        for facts in by_source.values()
    )
    assert len({fact.row.address for fact in world.facts}) == len(world.facts)
    assert len({fact.fact_id for fact in world.facts}) == len(world.facts)

    store = AtomicGraphStore(fact.row for fact in world.facts)
    assert all(store.lookup(fact.row.address) == fact.row for fact in world.facts)


def test_world_generation_is_seed_deterministic():
    cfg = WorldConfig(n_entities=64, seed=11, entity_id_offset=512)
    assert generate_world(3, cfg) == generate_world(3, cfg)
    assert generate_world(3, cfg) != generate_world(
        3, WorldConfig(n_entities=64, seed=12, entity_id_offset=512)
    )


def test_public_world_ids_have_disjoint_addresses():
    first = generate_world(3, WorldConfig(n_entities=32, seed=7))
    second = generate_world(4, WorldConfig(n_entities=32, seed=7))
    store = AtomicGraphStore(
        [fact.row for fact in first.facts]
        + [fact.row for fact in second.facts]
    )
    assert len(store) == 2 * 32 * 6


@pytest.mark.parametrize(
    ("n_entities", "expected"),
    [
        (16, [16]),
        (32, [32]),
        (63, [63]),
        (65, [49, 16]),
        (70, [54, 16]),
        (127, [63, 64]),
        (130, [64, 50, 16]),
    ],
)
def test_world_stream_preserves_population_without_undersized_tail(
    n_entities,
    expected,
):
    worlds = list(iter_worlds(n_entities, world_size=64, seed=11))
    source_ids_by_world = [
        {fact.row.source_id for fact in world.facts}
        for world in worlds
    ]

    assert [len(world.entity_names) for world in worlds] == expected
    assert sum(len(world.entity_names) for world in worlds) == n_entities
    assert all(len(world.entity_names) >= 16 for world in worlds)
    assert set().union(*source_ids_by_world) == set(range(n_entities))
    assert all(
        source_ids.isdisjoint(other_ids)
        for index, source_ids in enumerate(source_ids_by_world)
        for other_ids in source_ids_by_world[index + 1 :]
    )


def test_world_stream_rejects_totals_below_reasoning_minimum_immediately():
    with pytest.raises(ValueError, match="n_entities must be at least 16"):
        iter_worlds(
            n_entities=15,
            world_size=64,
            seed=23,
        )


def test_world_iterator_does_not_materialize_the_requested_population():
    worlds = iter_worlds(
        n_entities=10**12,
        world_size=16,
        seed=5,
        world_id_offset=7,
    )
    first = next(worlds)
    assert first.world_id == 7
    assert {fact.row.source_id for fact in first.facts} == set(range(112, 128))


def test_world_configuration_keeps_the_fixed_six_relation_schema():
    with pytest.raises(ValueError, match="relation_count must be 4"):
        generate_world(0, WorldConfig(n_entities=16, relation_count=3))


def test_counterfactual_pairs_change_supporting_evidence_and_replay_to_flip():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    original_rows = _rows_by_address(world)
    pairs = generate_eval_pairs(world, n_pairs_per_task=20, seed=17)

    assert len(pairs) == 60
    assert {pair.task for pair in pairs} == {
        "path_composition",
        "date_ordering",
        "balanced_equality",
    }
    for pair in pairs:
        original = pair.original
        counterfactual = pair.counterfactual
        changed_original = original_rows[pair.changed_row.address]
        gold_addresses = {
            GraphAddress(int(source), relation, direction)
            for source, relation, direction in original.meta["gold_addresses"]
        }

        assert original.answer != counterfactual.answer
        assert _answer_from_evidence(world, pair, original) == original.answer
        assert (
            _answer_from_evidence(world, pair, counterfactual)
            == counterfactual.answer
        )
        assert original.meta["pair_id"] == counterfactual.meta["pair_id"]
        assert original.meta["graph_rows"] == counterfactual.meta["graph_rows"]
        assert original.meta["graph_rows"] == len(world.facts)
        assert original.meta["changed_row"] is None
        assert counterfactual.meta["changed_row"] == pair.changed_row.as_json()
        assert pair.changed_row.address in gold_addresses
        assert pair.changed_row.address == changed_original.address
        assert pair.changed_row != changed_original
        assert pair.changed_row.provenance_id == changed_original.provenance_id


def test_independent_cursor_oracle_reconstructs_every_path_without_gold_addresses():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    pairs = [
        pair
        for pair in generate_eval_pairs(world, n_pairs_per_task=20, seed=17)
        if pair.task == "path_composition"
    ]

    for pair in pairs:
        for item in (pair.original, pair.counterfactual):
            derived_addresses, derived_answer = _path_cursor_oracle(
                world,
                pair,
                item,
            )
            stored_addresses = [
                GraphAddress(int(source), relation, direction)
                for source, relation, direction in item.meta["gold_addresses"]
            ]
            assert derived_addresses == stored_addresses
            assert derived_answer == item.answer

            item.meta["gold_addresses"] = [[2**63, "r15", "in"]]
            assert _path_cursor_oracle(world, pair, item) == (
                derived_addresses,
                derived_answer,
            )


def test_eval_items_persist_exact_six_step_gold_actions():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    pairs = generate_eval_pairs(world, n_pairs_per_task=20, seed=17)

    for pair in pairs:
        for item in (pair.original, pair.counterfactual):
            actions = item.meta["gold_actions"]
            reads = [action for action in actions if action["read"]]

            assert len(actions) == 6
            assert len(reads) == len(item.meta["gold_addresses"])
            assert [
                [action["relation_id"], action["direction"]]
                for action in reads
            ] == [
                [relation, direction]
                for _, relation, direction in item.meta["gold_addresses"]
            ]
            halt = actions[len(reads)]
            assert halt["halt"] and not halt["read"]
            assert all(
                not action["read"] and not action["halt"]
                for action in actions[len(reads) + 1 :]
            )


def test_gold_action_slots_cover_multihop_paths_and_two_branch_reads():
    world = generate_world(0, WorldConfig(n_entities=64, seed=13))
    pairs = generate_eval_pairs(world, n_pairs_per_task=30, seed=19)
    path = next(
        pair.original
        for pair in pairs
        if pair.task == "path_composition"
        and len(pair.original.meta["gold_addresses"]) >= 3
    )
    branch = next(
        pair.original for pair in pairs if pair.task == "date_ordering"
    )

    assert {
        action["source_slot"]
        for action in path.meta["gold_actions"]
        if action["read"]
    } == {0}
    assert [
        action["source_slot"]
        for action in branch.meta["gold_actions"]
        if action["read"]
    ] == [0, 1]


def test_balanced_equality_twins_are_counterbalanced_in_both_orientations():
    world = generate_world(0, WorldConfig(n_entities=64, seed=23))
    equality_pairs = [
        pair
        for pair in generate_eval_pairs(world, n_pairs_per_task=20, seed=29)
        if pair.task == "balanced_equality"
    ]
    assert Counter(pair.original.answer for pair in equality_pairs) == {
        "yes": 10,
        "no": 10,
    }
    assert Counter(pair.counterfactual.answer for pair in equality_pairs) == {
        "yes": 10,
        "no": 10,
    }
    assert all(
        {pair.original.answer, pair.counterfactual.answer} == {"yes", "no"}
        for pair in equality_pairs
    )


def test_protected_answers_are_derived_not_payload_copies():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    payloads = {fact.row.target for fact in world.facts}
    pairs = generate_eval_pairs(world, n_pairs_per_task=10, seed=19)
    for pair in pairs:
        assert pair.original.answer not in payloads
        assert pair.counterfactual.answer not in payloads


def test_bed_and_graph_renderers_are_lazy_over_their_inputs():
    def bed():
        yield "first"
        raise AssertionError("bed input was materialized")

    bed_record = next(iter_bed_records(bed()))
    assert bed_record.segments[0].text == "first"
    assert bed_record.schedule.component == "bed"

    world = generate_world(0, WorldConfig(n_entities=16, seed=31))

    def worlds():
        yield world
        raise AssertionError("world input was materialized")

    tok = get_tok()
    graph_record = next(iter_graph_records(tok, worlds))
    payload = next(
        segment for segment in graph_record.segments if segment.role == "payload"
    )
    neutral = graph_record.segments[-1]
    assert graph_record.schedule.component == "graph"
    assert payload.fact_id is not None
    assert neutral.role == "random_control"
    assert len(tok.encode(neutral.text)) == len(tok.encode(payload.text))


def test_graph_controls_counterbalance_length_and_position_bins():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=16, seed=31))
    records = list(
        islice(
            iter_graph_records(tok, lambda: iter((world,))),
            2,
        )
    )
    payload_keys = Counter()
    control_keys = Counter()

    for record in records:
        encoded = [
            (segment, tok.encode(segment.text))
            for segment in record.segments
        ]
        document_length = sum(len(ids) for _, ids in encoded) + 1
        start = 0
        for segment, ids in encoded:
            end = start + len(ids)
            key = (
                len(ids),
                min(9, ((start + end) * 10) // (2 * document_length)),
            )
            if segment.role == "payload":
                payload_keys[key] += 1
            elif segment.role == "random_control":
                control_keys[key] += 1
            start = end

    assert payload_keys == control_keys


@pytest.mark.parametrize("hop_band", [1, 2, 4])
def test_reasoning_records_have_fixed_hops_and_six_supervised_steps(hop_band):
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=37))
    facts_by_id = {fact.fact_id: fact for fact in world.facts}

    def worlds():
        yield world
        raise AssertionError("world input was materialized")

    record = next(iter_reasoning_records(tok, worlds, seed=41, max_hops=hop_band))
    ids, _, _ = tok.encode_tagged_segments(record.segments)
    payloads = [
        segment for segment in record.segments if segment.role == "payload"
    ]

    assert record.schedule.component == "reasoning"
    assert record.schedule.curriculum_band == hop_band
    assert ids.count(tok.GRAPH_START) == 6
    assert ids.count(tok.ANSWER_STATE) == 6
    assert ids.count(tok.GRAPH_READ) == hop_band
    assert ids.count(tok.GRAPH_HALT) == 1
    assert ids.count(tok.GRAPH_NOOP) == 5 - hop_band
    assert ids.count(tok.GRAPH_MISS) == 6 - hop_band
    assert len(payloads) == hop_band
    assert (
        sum(
            segment.role == "provisional_answer"
            for segment in record.segments
        )
        == 6
    )
    assert sum(segment.role == "final_answer" for segment in record.segments) == 1

    for payload in payloads:
        fact = facts_by_id[payload.fact_id]
        expected_payload = next(
            segment
            for segment in serialize_return(fact.row, fact.fact_id)
            if segment.role == "payload"
        )
        assert payload.text == expected_payload.text
    final_answer = next(
        segment.text for segment in record.segments if segment.role == "final_answer"
    )
    assert final_answer not in {fact.row.target for fact in world.facts}


def test_reasoning_records_cover_every_task_with_post_halt_noops_and_controls():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=43))
    facts_by_id = {fact.fact_id: fact for fact in world.facts}
    records = list(
        islice(
            iter_reasoning_records(
                tok,
                lambda: iter((world,)),
                seed=47,
                max_hops=2,
            ),
            24,
        )
    )
    records_by_task = {
        task: next(
            record
            for record in records
            if f"-{task}-" in record.schedule.record_id
        )
        for task in (
            "path_composition",
            "date_ordering",
            "balanced_equality",
        )
    }

    for record in records_by_task.values():
        ids, _, _ = tok.encode_tagged_segments(record.segments)
        action_frames = []
        for segment in record.segments:
            if segment.role != "action":
                continue
            segment_ids = tok.encode(segment.text)
            if segment_ids and segment_ids[0] == tok.GRAPH_START:
                action_frames.append(segment_ids)

        assert len(action_frames) == 6
        assert [frame[4] for frame in action_frames] == [
            tok.GRAPH_READ,
            tok.GRAPH_READ,
            tok.GRAPH_HALT,
            tok.GRAPH_NOOP,
            tok.GRAPH_NOOP,
            tok.GRAPH_NOOP,
        ]
        assert ids.count(tok.ANSWER_STATE) == 6
        assert ids.count(tok.GRAPH_MISS) == 4
        assert (
            sum(
                segment.role == "provisional_answer"
                for segment in record.segments
            )
            == 6
        )
        assert (
            sum(
                segment.role == "final_answer"
                for segment in record.segments
            )
            == 1
        )

        payload_indexes = [
            index
            for index, segment in enumerate(record.segments)
            if segment.role == "payload"
        ]
        assert len(payload_indexes) == 2
        for index in payload_indexes:
            payload = record.segments[index]
            fact = facts_by_id[payload.fact_id]
            expected_payload = next(
                segment
                for segment in serialize_return(fact.row, fact.fact_id)
                if segment.role == "payload"
            )
            assert payload.text == expected_payload.text
            matching_controls = [
                segment
                for segment in record.segments
                if segment.role == "random_control"
                and len(tok.encode(segment.text))
                == len(tok.encode(payload.text))
            ]
            assert matching_controls


def test_reasoning_controls_use_both_counterbalanced_orientations():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=64, seed=43))
    records = list(
        islice(
            iter_reasoning_records(
                tok,
                lambda: iter((world,)),
                seed=47,
                max_hops=1,
            ),
            2,
        )
    )
    orientations = set()

    for record in records:
        payload_index = next(
            index
            for index, segment in enumerate(record.segments)
            if segment.role == "payload"
        )
        control_index = next(
            index
            for index, segment in enumerate(record.segments)
            if segment.role == "random_control"
        )
        orientations.add("control_first" if control_index < payload_index else "payload_first")

    assert orientations == {"control_first", "payload_first"}


def test_reasoning_renderer_rejects_non_curriculum_hop_bands():
    tok = get_tok()
    world = generate_world(0, WorldConfig(n_entities=16, seed=43))
    records = iter_reasoning_records(
        tok,
        lambda: iter((world,)),
        seed=47,
        max_hops=3,
    )
    with pytest.raises(ValueError, match="max_hops must be one of"):
        next(records)
