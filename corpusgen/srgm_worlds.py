from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from corpusgen.graph_records import (
    GraphAction,
    GraphAddress,
    GraphRow,
    RenderedRecord,
    ScheduleEntry,
    SelectorFeatures,
    TaggedSegment,
    relative_position_bin,
)
from corpusgen.graph_trace import serialize_action, serialize_return
from corpusgen.records import QAItem


ENTITY_RELATIONS = tuple(f"r{index}" for index in range(4))
DATE_RELATION = "r4"
CATEGORY_RELATION = "r5"
CURRICULUM_HOPS = (1, 2, 4)
MIN_REASONING_ENTITIES = 16
_DEFAULT_WORLD_ENTITY_STRIDE = 1 << 64


@dataclass(frozen=True)
class WorldConfig:
    n_entities: int = 64
    seed: int = 0
    relation_count: int = 4
    entity_id_offset: int | None = None

    def __post_init__(self) -> None:
        if self.n_entities <= 0:
            raise ValueError("n_entities must be positive")
        if self.relation_count != len(ENTITY_RELATIONS):
            raise ValueError("relation_count must be 4")
        if self.entity_id_offset is not None and self.entity_id_offset < 0:
            raise ValueError("entity_id_offset must be non-negative")
        if (
            self.entity_id_offset is None
            and self.n_entities > _DEFAULT_WORLD_ENTITY_STRIDE
        ):
            raise ValueError("default-offset worlds exceed the entity stride")


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    row: GraphRow
    features: SelectorFeatures
    audit_class: str


@dataclass(frozen=True)
class GraphWorld:
    world_id: int
    entity_names: tuple[str, ...]
    facts: tuple[GraphFact, ...]


@dataclass(frozen=True)
class CounterfactualPair:
    task: str
    original: QAItem
    counterfactual: QAItem
    changed_row: GraphRow


def _features(
    exposure: int,
    entropy: float,
    queries: float,
    centrality: float,
) -> SelectorFeatures:
    return SelectorFeatures(
        log_exposure=math.log1p(exposure),
        payload_entropy=entropy,
        payload_tokens=1.0,
        expected_queries=queries,
        path_centrality=centrality,
    )


def _random_date(rng: random.Random) -> str:
    return (
        f"{1930 + rng.randrange(76):04d}-"
        f"{1 + rng.randrange(12):02d}-"
        f"{1 + rng.randrange(28):02d}"
    )


def _category_values(rng: random.Random, n_entities: int) -> list[str]:
    category_count = min(64, max(2, n_entities // 2))
    labels = rng.sample(range(64), category_count)
    values = [
        f"category-{labels[index % category_count]}"
        for index in range(n_entities)
    ]
    rng.shuffle(values)
    return values


def generate_world(world_id: int, cfg: WorldConfig) -> GraphWorld:
    if world_id < 0:
        raise ValueError("world_id must be non-negative")

    entity_id_offset = (
        world_id * _DEFAULT_WORLD_ENTITY_STRIDE
        if cfg.entity_id_offset is None
        else cfg.entity_id_offset
    )
    rng = random.Random((cfg.seed << 32) ^ world_id)
    names = tuple(
        f"entity-{entity_id_offset + local_id}-{rng.getrandbits(32):08x}"
        for local_id in range(cfg.n_entities)
    )
    facts: list[GraphFact] = []

    for relation_index, relation_id in enumerate(ENTITY_RELATIONS):
        targets = list(range(cfg.n_entities))
        rng.shuffle(targets)
        for source_id, target_id in enumerate(targets):
            global_source = entity_id_offset + source_id
            global_target = entity_id_offset + target_id
            compose = rng.randrange(4)
            row = GraphRow(
                source_id=global_source,
                relation_id=relation_id,
                direction="out",
                target_kind="entity",
                target=str(global_target),
                qualifiers=(("compose", str(compose)),),
                provenance_id=f"world-{world_id}",
            )
            centrality = (
                1.0
                if source_id < max(1, cfg.n_entities // 4)
                else 0.1
            )
            facts.append(
                GraphFact(
                    fact_id=str(global_source * 6 + relation_index),
                    row=row,
                    features=_features(
                        exposure=4,
                        entropy=math.log2(cfg.n_entities),
                        queries=1.0,
                        centrality=centrality,
                    ),
                    audit_class=(
                        "central" if centrality == 1.0 else "peripheral"
                    ),
                )
            )

    dates = [_random_date(rng) for _ in range(cfg.n_entities)]
    if cfg.n_entities > 1 and len(set(dates)) == 1:
        dates[-1] = "2099-12-31"
    categories = _category_values(rng, cfg.n_entities)
    for source_id, (date, category) in enumerate(zip(dates, categories)):
        global_source = entity_id_offset + source_id
        for relation_index, relation_id, value, entropy in (
            (4, DATE_RELATION, date, 14.7),
            (5, CATEGORY_RELATION, category, 6.0),
        ):
            row = GraphRow(
                source_id=global_source,
                relation_id=relation_id,
                direction="out",
                target_kind="literal",
                target=value,
                provenance_id=f"world-{world_id}",
            )
            facts.append(
                GraphFact(
                    fact_id=str(global_source * 6 + relation_index),
                    row=row,
                    features=_features(
                        exposure=2,
                        entropy=entropy,
                        queries=0.25,
                        centrality=0.05,
                    ),
                    audit_class="peripheral",
                )
            )

    return GraphWorld(
        world_id=world_id,
        entity_names=names,
        facts=tuple(facts),
    )


def _iter_world_sizes(
    n_entities: int,
    world_size: int,
) -> Iterator[int]:
    full_worlds, remainder = divmod(n_entities, world_size)
    if remainder == 0:
        yield from (world_size for _ in range(full_worlds))
        return
    if full_worlds == 0:
        yield remainder
        return
    if remainder >= MIN_REASONING_ENTITIES:
        yield from (world_size for _ in range(full_worlds - 1))
        combined = world_size + remainder
        first = combined // 2
        yield first
        yield combined - first
        return

    deficit = MIN_REASONING_ENTITIES - remainder
    donor_size = world_size - deficit
    yield from (world_size for _ in range(max(0, full_worlds - 1)))
    if donor_size >= MIN_REASONING_ENTITIES:
        yield donor_size
        yield MIN_REASONING_ENTITIES
    else:
        yield world_size + remainder


def iter_worlds(
    n_entities: int,
    world_size: int,
    seed: int,
    world_id_offset: int = 0,
) -> Iterator[GraphWorld]:
    if n_entities < MIN_REASONING_ENTITIES:
        raise ValueError(
            f"n_entities must be at least {MIN_REASONING_ENTITIES}"
        )
    if world_size < MIN_REASONING_ENTITIES:
        raise ValueError(
            f"world_size must be at least {MIN_REASONING_ENTITIES}"
        )
    if world_id_offset < 0:
        raise ValueError("world_id_offset must be non-negative")

    def worlds() -> Iterator[GraphWorld]:
        entity_id_offset = world_id_offset * world_size
        for ordinal, size in enumerate(
            _iter_world_sizes(n_entities, world_size)
        ):
            world_id = world_id_offset + ordinal
            yield generate_world(
                world_id,
                WorldConfig(
                    n_entities=size,
                    seed=seed,
                    entity_id_offset=entity_id_offset,
                ),
            )
            entity_id_offset += size

    return worlds()


def _row_map(world: GraphWorld) -> dict[GraphAddress, GraphFact]:
    rows = {fact.row.address: fact for fact in world.facts}
    if len(rows) != len(world.facts):
        raise ValueError("world contains duplicate functional addresses")
    return rows


def _entity_ids(world: GraphWorld) -> tuple[int, ...]:
    ids = tuple(sorted({fact.row.source_id for fact in world.facts}))
    if len(ids) != len(world.entity_names):
        raise ValueError("entity_names must align one-to-one with source ids")
    return ids


def _follow(
    rows: dict[GraphAddress, GraphFact],
    start: int,
    relations: tuple[str, ...],
) -> tuple[int, int, tuple[GraphFact, ...]]:
    current = start
    used: list[GraphFact] = []
    compose = 0
    for relation in relations:
        fact = rows[GraphAddress(current, relation, "out")]
        if fact.row.target_kind != "entity":
            raise ValueError("path relations must target entities")
        used.append(fact)
        compose = (
            compose + int(dict(fact.row.qualifiers)["compose"])
        ) % 4
        current = int(fact.row.target)
    return current, compose, tuple(used)


def _item(
    *,
    qid: str,
    task: str,
    prompt: str,
    answer: str,
    pair_id: str,
    graph_rows: int,
    variant: str,
    entity_slots: tuple[int | None, ...],
    gold_facts: tuple[GraphFact, ...],
    answer_choices: tuple[str, ...],
    changed_row: GraphRow | None = None,
    relations: tuple[str, ...] = (),
) -> QAItem:
    return QAItem(
        qid=qid,
        task=task,
        prompt=prompt,
        answer=answer,
        meta={
            "pair_id": pair_id,
            "template": task,
            "graph_rows": graph_rows,
            "variant": variant,
            "changed_row": (
                None if changed_row is None else changed_row.as_json()
            ),
            "entity_slots": list(entity_slots),
            "gold_addresses": [
                [
                    fact.row.address.source_id,
                    fact.row.address.relation_id,
                    fact.row.address.direction,
                ]
                for fact in gold_facts
            ],
            "gold_fact_ids": [fact.fact_id for fact in gold_facts],
            "answer_choices": list(answer_choices),
            "relations": list(relations),
        },
    )


def _replace(
    row: GraphRow,
    *,
    target: str | None = None,
    compose: int | None = None,
) -> GraphRow:
    qualifiers = row.qualifiers
    if compose is not None:
        if "compose" not in dict(qualifiers):
            raise ValueError("compose replacement requires a compose qualifier")
        qualifiers = tuple(
            (key, str(compose) if key == "compose" else value)
            for key, value in qualifiers
        )
    return GraphRow(
        source_id=row.source_id,
        relation_id=row.relation_id,
        direction=row.direction,
        target_kind=row.target_kind,
        target=row.target if target is None else target,
        qualifiers=qualifiers,
        provenance_id=row.provenance_id,
    )


def _path_relations(
    rng: random.Random,
    hops: int,
) -> tuple[str, ...]:
    while True:
        relations = tuple(rng.choice(ENTITY_RELATIONS) for _ in range(hops))
        counts = Counter(relations)
        if any(count == 1 for count in counts.values()):
            return relations


def _date_pair(
    rng: random.Random,
    entity_ids: tuple[int, ...],
    rows: dict[GraphAddress, GraphFact],
    earlier_in_slot_zero: bool,
) -> tuple[int, int, GraphFact, GraphFact]:
    for _ in range(1000):
        a, b = rng.sample(entity_ids, 2)
        fact_a = rows[GraphAddress(a, DATE_RELATION, "out")]
        fact_b = rows[GraphAddress(b, DATE_RELATION, "out")]
        if fact_a.row.target == fact_b.row.target:
            continue
        if (fact_a.row.target < fact_b.row.target) != earlier_in_slot_zero:
            a, b = b, a
            fact_a, fact_b = fact_b, fact_a
        return a, b, fact_a, fact_b
    raise ValueError("world must contain at least two distinct dates")


def _category_groups(
    entity_ids: tuple[int, ...],
    rows: dict[GraphAddress, GraphFact],
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for entity_id in entity_ids:
        fact = rows[GraphAddress(entity_id, CATEGORY_RELATION, "out")]
        groups[fact.row.target].append(entity_id)
    return dict(groups)


def _category_pair(
    rng: random.Random,
    groups: dict[str, list[int]],
    rows: dict[GraphAddress, GraphFact],
    equal: bool,
) -> tuple[int, int, GraphFact, GraphFact]:
    if equal:
        choices = [value for value, ids in groups.items() if len(ids) >= 2]
        if not choices:
            raise ValueError("world must contain a repeated category")
        category = rng.choice(choices)
        a, b = rng.sample(groups[category], 2)
    else:
        categories = tuple(groups)
        if len(categories) < 2:
            raise ValueError("world must contain at least two categories")
        category_a, category_b = rng.sample(categories, 2)
        a = rng.choice(groups[category_a])
        b = rng.choice(groups[category_b])
    return (
        a,
        b,
        rows[GraphAddress(a, CATEGORY_RELATION, "out")],
        rows[GraphAddress(b, CATEGORY_RELATION, "out")],
    )


def _different_category(value: str) -> str:
    prefix = "category-"
    if value.startswith(prefix):
        try:
            return f"{prefix}{(int(value[len(prefix):]) + 1) % 64}"
        except ValueError:
            pass
    return f"{value}-counterfactual"


def _composition_answer(rows: Iterable[GraphRow]) -> str:
    compose = sum(
        int(dict(row.qualifiers)["compose"]) for row in rows
    ) % 4
    return f"r{compose}"


def _generate_eval_pairs(
    world: GraphWorld,
    n_pairs_per_task: int,
    seed: int,
    *,
    path_hops: int | None,
) -> list[CounterfactualPair]:
    if n_pairs_per_task < 0:
        raise ValueError("n_pairs_per_task must be non-negative")
    if n_pairs_per_task == 0:
        return []
    if path_hops is not None and path_hops not in CURRICULUM_HOPS:
        raise ValueError(f"path_hops must be one of {CURRICULUM_HOPS}")

    entity_ids = _entity_ids(world)
    if len(entity_ids) < MIN_REASONING_ENTITIES:
        raise ValueError(
            "reasoning worlds must contain at least sixteen entities"
        )
    names = dict(zip(entity_ids, world.entity_names))
    rows = _row_map(world)
    category_groups = _category_groups(entity_ids, rows)
    rng = random.Random((seed << 32) ^ world.world_id)
    date_orientation = rng.randrange(2)
    equality_orientation = rng.randrange(2)
    pairs: list[CounterfactualPair] = []

    for task in (
        "path_composition",
        "date_ordering",
        "balanced_equality",
    ):
        for index in range(n_pairs_per_task):
            pair_id = f"{world.world_id}-{seed}-{task}-{index}"
            relations: tuple[str, ...] = ()

            if task == "path_composition":
                a = rng.choice(entity_ids)
                hops = path_hops if path_hops is not None else rng.randint(2, 4)
                relations = _path_relations(rng, hops)
                _, _, used = _follow(rows, a, relations)
                answer = _composition_answer(fact.row for fact in used)

                relation_counts = Counter(relations)
                change_index = next(
                    position
                    for position, relation in enumerate(relations)
                    if relation_counts[relation] == 1
                )
                changed_fact = used[change_index]
                old_code = int(
                    dict(changed_fact.row.qualifiers)["compose"]
                )
                changed = _replace(
                    changed_fact.row,
                    compose=(old_code + 1) % 4,
                )
                changed_rows = [
                    changed
                    if fact.row.address == changed.address
                    else fact.row
                    for fact in used
                ]
                flipped = _composition_answer(changed_rows)
                if flipped == answer:
                    raise AssertionError(
                        "path counterfactual failed to change composition"
                    )

                gold_facts = used
                entity_slots = (a, None, None, None)
                answer_choices = tuple(f"r{i}" for i in range(4))
                prompt = (
                    f"Slot 0 refers to {names[a]}. Start at slot 0 and follow "
                    f"{' '.join(relations)}. Return the composed relation."
                )
            elif task == "date_ordering":
                a, b, fact_a, fact_b = _date_pair(
                    rng,
                    entity_ids,
                    rows,
                    earlier_in_slot_zero=(
                        (index + date_orientation) % 2 == 0
                    ),
                )
                answer = (
                    "<|slot_0|>"
                    if fact_a.row.target < fact_b.row.target
                    else "<|slot_1|>"
                )
                changed = _replace(
                    fact_a.row,
                    target=(
                        "9999-12-31"
                        if answer == "<|slot_0|>"
                        else "0001-01-01"
                    ),
                )
                flipped = (
                    "<|slot_0|>"
                    if changed.target < fact_b.row.target
                    else "<|slot_1|>"
                )
                gold_facts = (fact_a, fact_b)
                entity_slots = (a, b, None, None)
                answer_choices = ("<|slot_0|>", "<|slot_1|>")
                prompt = (
                    f"Slot 0 refers to {names[a]}. "
                    f"Slot 1 refers to {names[b]}. "
                    "Read the dates reached from slots 0 and 1. "
                    "Return the earlier slot."
                )
            else:
                want_equal = (index + equality_orientation) % 2 == 0
                a, b, fact_a, fact_b = _category_pair(
                    rng,
                    category_groups,
                    rows,
                    equal=want_equal,
                )
                answer = "yes" if want_equal else "no"
                changed = _replace(
                    fact_a.row,
                    target=(
                        _different_category(fact_b.row.target)
                        if want_equal
                        else fact_b.row.target
                    ),
                )
                flipped = (
                    "yes"
                    if changed.target == fact_b.row.target
                    else "no"
                )
                gold_facts = (fact_a, fact_b)
                entity_slots = (a, b, None, None)
                answer_choices = ("yes", "no")
                prompt = (
                    f"Slot 0 refers to {names[a]}. "
                    f"Slot 1 refers to {names[b]}. "
                    "Read the categories reached from slots 0 and 1. "
                    "Are they equal?"
                )

            if flipped == answer:
                raise AssertionError(
                    f"{task} counterfactual failed to flip the answer"
                )
            original = _item(
                qid=f"{pair_id}-o",
                task=task,
                prompt=prompt,
                answer=answer,
                pair_id=pair_id,
                graph_rows=len(world.facts),
                variant="original",
                entity_slots=entity_slots,
                gold_facts=gold_facts,
                answer_choices=answer_choices,
                relations=relations,
            )
            counterfactual = _item(
                qid=f"{pair_id}-c",
                task=task,
                prompt=prompt,
                answer=flipped,
                pair_id=pair_id,
                graph_rows=len(world.facts),
                variant="counterfactual",
                entity_slots=entity_slots,
                gold_facts=gold_facts,
                answer_choices=answer_choices,
                changed_row=changed,
                relations=relations,
            )
            pairs.append(
                CounterfactualPair(
                    task=task,
                    original=original,
                    counterfactual=counterfactual,
                    changed_row=changed,
                )
            )
    return pairs


def generate_eval_pairs(
    world: GraphWorld,
    n_pairs_per_task: int,
    seed: int,
) -> list[CounterfactualPair]:
    return _generate_eval_pairs(
        world,
        n_pairs_per_task,
        seed,
        path_hops=None,
    )


def iter_bed_records(bed_iter: Iterable[str]) -> Iterator[RenderedRecord]:
    for index, text in enumerate(bed_iter):
        yield RenderedRecord(
            segments=(TaggedSegment(text, "plain"),),
            schedule=ScheduleEntry(
                component="bed",
                record_id=f"bed-{index}",
                exposure=index,
                curriculum_band=0,
            ),
        )


def _payload_text(row: GraphRow) -> str:
    return json.dumps(
        {
            "target_kind": row.target_kind,
            "target": row.target,
            "qualifiers": list(row.qualifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _cycle_audit_facts(
    facts: tuple[GraphFact, ...],
    audit_class: str,
) -> Iterator[GraphFact]:
    while True:
        found = False
        for fact in facts:
            if fact.audit_class == audit_class:
                found = True
                yield fact
        if not found:
            return


def iter_graph_records(
    tok,
    worlds_factory: Callable[[], Iterable[GraphWorld]],
) -> Iterator[RenderedRecord]:
    exposure = 0
    placements_by_length: Counter[int] = Counter()
    rule_text = (
        "Composition adds retrieved compose codes modulo four. "
        "Inverse traversal reverses edge direction. Equality is symmetric. "
        "Earlier dates have smaller ISO-8601 strings."
    )
    while True:
        saw_world = False
        for world in worlds_factory():
            saw_world = True
            peripheral_count = sum(
                fact.audit_class == "peripheral" for fact in world.facts
            )
            central_count = sum(
                fact.audit_class == "central" for fact in world.facts
            )
            if not peripheral_count or not central_count:
                raise ValueError(
                    "graph rendering requires peripheral and central facts"
                )

            batches = max(
                math.ceil(peripheral_count / 7),
                math.ceil(central_count / 2),
            )
            peripheral = _cycle_audit_facts(
                world.facts,
                "peripheral",
            )
            central = _cycle_audit_facts(world.facts, "central")
            for batch_index in range(batches):
                batch = tuple(next(peripheral) for _ in range(7)) + tuple(
                    next(central) for _ in range(2)
                )
                for fact in batch:
                    payload = _payload_text(fact.row)
                    payload_length = len(tok.encode(payload))
                    neutral = " the" * payload_length
                    payload_segment = TaggedSegment(
                        payload,
                        "payload",
                        fact.fact_id,
                    )
                    control_segment = TaggedSegment(
                        neutral,
                        "random_control",
                    )
                    placement = placements_by_length[payload_length]
                    placements_by_length[payload_length] += 1
                    matched_segments = (
                        (payload_segment, control_segment)
                        if placement % 2 == 0
                        else (control_segment, payload_segment)
                    )
                    yield RenderedRecord(
                        segments=(
                            TaggedSegment(
                                (
                                    f"Source {fact.row.source_id} relation "
                                    f"{fact.row.relation_id} returns "
                                ),
                                "plain",
                            ),
                            *matched_segments,
                        ),
                        schedule=ScheduleEntry(
                            component="graph",
                            record_id=fact.fact_id,
                            exposure=exposure,
                            curriculum_band=0,
                        ),
                    )
                    exposure += 1

                yield RenderedRecord(
                    segments=(TaggedSegment(rule_text, "rule"),),
                    schedule=ScheduleEntry(
                        component="graph",
                        record_id=(
                            f"rule-{world.world_id}-{batch_index}"
                        ),
                        exposure=exposure,
                        curriculum_band=0,
                    ),
                )
                exposure += 1
        if not saw_world:
            raise ValueError("worlds_factory produced no worlds")


def _answer_segments(answer: str) -> tuple[TaggedSegment, ...]:
    return (
        TaggedSegment("<|answer_state|>", "action"),
        TaggedSegment(answer, "provisional_answer"),
    )


def _return_segment_blocks(
    tok,
    row: GraphRow | None,
    fact_id: str | None,
) -> tuple[tuple[TaggedSegment, ...], tuple[TaggedSegment, ...]]:
    returned = tuple(serialize_return(row, fact_id))
    payload = next(
        (
            segment
            for segment in returned
            if segment.role == "payload"
        ),
        None,
    )
    if payload is None:
        return returned, ()
    control = (
        TaggedSegment(" the", "plain"),
        TaggedSegment(
            " the" * len(tok.encode(payload.text)),
            "random_control",
        ),
        TaggedSegment(" the", "plain"),
    )
    return returned, control


def _counterbalance_control_blocks(
    tok,
    segments: list[TaggedSegment],
    blocks: list[
        tuple[
            int,
            tuple[TaggedSegment, ...],
            tuple[TaggedSegment, ...],
        ]
    ],
    balance: Counter[tuple[int, int]],
    tie_offset: int,
) -> None:
    encoded_lengths = [len(tok.encode(segment.text)) for segment in segments]
    encoded_length_by_id = {
        id(segment): length
        for segment, length in zip(segments, encoded_lengths)
    }
    offsets = [0]
    for length in encoded_lengths:
        offsets.append(offsets[-1] + length)
    document_length = offsets[-1] + 1

    def role_range(
        block: tuple[TaggedSegment, ...],
        role: str,
        block_start: int,
    ) -> tuple[int, int]:
        local_start = 0
        for segment in block:
            length = encoded_length_by_id[id(segment)]
            if segment.role == role:
                return block_start + local_start, block_start + local_start + length
            local_start += length
        raise ValueError(f"block lacks role {role}")

    def key(start: int, end: int) -> tuple[int, int]:
        return (
            end - start,
            relative_position_bin(start, end, document_length),
        )

    for ordinal, (segment_index, returned, control) in enumerate(blocks):
        block_start = offsets[segment_index]
        returned_length = sum(
            encoded_length_by_id[id(segment)] for segment in returned
        )
        control_length = sum(
            encoded_length_by_id[id(segment)] for segment in control
        )
        if returned_length != control_length:
            raise ValueError("payload and random-control blocks must align")
        payload_key = key(*role_range(returned, "payload", block_start))
        control_key = key(
            *role_range(
                control,
                "random_control",
                block_start + returned_length,
            )
        )
        normal_score = (
            abs(balance[payload_key] + 1)
            + abs(balance[control_key] - 1)
        )
        reverse_score = (
            abs(balance[payload_key] - 1)
            + abs(balance[control_key] + 1)
        )
        reverse = reverse_score < normal_score or (
            reverse_score == normal_score
            and (tie_offset + ordinal) % 2 == 1
        )
        if reverse:
            block_size = len(returned) + len(control)
            segments[segment_index : segment_index + block_size] = [
                *control,
                *returned,
            ]
            balance[payload_key] -= 1
            balance[control_key] += 1
        else:
            balance[payload_key] += 1
            balance[control_key] -= 1


def iter_reasoning_records(
    tok,
    worlds_factory: Callable[[], Iterable[GraphWorld]],
    seed: int,
    max_hops: int,
) -> Iterator[RenderedRecord]:
    if max_hops not in CURRICULUM_HOPS:
        raise ValueError(f"max_hops must be one of {CURRICULUM_HOPS}")

    rng = random.Random(seed)
    exposure = 0
    control_position_balance: Counter[tuple[int, int]] = Counter()
    while True:
        saw_world = False
        for world in worlds_factory():
            saw_world = True
            fact_map = {fact.fact_id: fact for fact in world.facts}
            pairs = _generate_eval_pairs(
                world,
                n_pairs_per_task=8,
                seed=rng.randrange(1 << 30),
                path_hops=max_hops,
            )
            for pair in pairs:
                item = pair.original
                addresses = tuple(
                    GraphAddress(int(source), relation, direction)
                    for source, relation, direction in item.meta[
                        "gold_addresses"
                    ]
                )
                fact_ids = tuple(item.meta["gold_fact_ids"])
                if len(addresses) > max_hops:
                    continue

                segments: list[TaggedSegment] = [
                    TaggedSegment(item.prompt, "plain")
                ]
                matched_blocks: list[
                    tuple[
                        int,
                        tuple[TaggedSegment, ...],
                        tuple[TaggedSegment, ...],
                    ]
                ] = []
                for step in range(6):
                    if step < len(addresses):
                        address = addresses[step]
                        fact = fact_map[fact_ids[step]]
                        if fact.row.address != address:
                            raise ValueError(
                                "gold fact id does not match gold address"
                            )
                        source_slot = (
                            0
                            if item.task == "path_composition"
                            else step
                        )
                        action = GraphAction(
                            source_slot=source_slot,
                            relation_id=address.relation_id,
                            direction=address.direction,
                            read=True,
                            halt=False,
                        )
                        segments.append(
                            TaggedSegment(
                                tok.decode(serialize_action(action, tok)),
                                "action",
                            )
                        )
                        returned, control = _return_segment_blocks(
                            tok,
                            fact.row,
                            fact.fact_id,
                        )
                        block_start = len(segments)
                        segments.extend(returned)
                        segments.extend(control)
                        matched_blocks.append(
                            (block_start, returned, control)
                        )
                    else:
                        is_halt_step = step == len(addresses)
                        action = GraphAction(
                            source_slot=0,
                            relation_id="r0",
                            direction="out",
                            read=False,
                            halt=is_halt_step,
                        )
                        segments.append(
                            TaggedSegment(
                                tok.decode(serialize_action(action, tok)),
                                "action",
                            )
                        )
                        returned, _ = _return_segment_blocks(tok, None, None)
                        segments.extend(returned)
                    segments.extend(_answer_segments(item.answer))

                segments.append(
                    TaggedSegment(item.answer, "final_answer")
                )
                _counterbalance_control_blocks(
                    tok,
                    segments,
                    matched_blocks,
                    control_position_balance,
                    exposure,
                )
                yield RenderedRecord(
                    segments=tuple(segments),
                    schedule=ScheduleEntry(
                        component="reasoning",
                        record_id=item.qid,
                        exposure=exposure,
                        curriculum_band=max_hops,
                    ),
                )
                exposure += 1
        if not saw_world:
            raise ValueError("worlds_factory produced no worlds")
