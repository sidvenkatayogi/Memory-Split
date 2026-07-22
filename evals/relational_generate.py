"""Fixed-grammar decoding over the atomic relational graph store.

Only the action frame is constrained. The model chooses the source slot,
relation, direction, and READ/NOOP/HALT terminal. Prompt prefill batches contain
equal-length sequences only; no padding token is ever added to an eval prompt.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.graph_trace import serialize_action, serialize_return
from organizer.graph_store import AtomicGraphStore

N_GRAPH_STEPS = 6


@dataclass
class GraphDecodeState:
    slots: list[int | None]
    actions: list[GraphAction] = field(default_factory=list)
    rows: list[GraphRow | None] = field(default_factory=list)
    provisional_answers: list[str] = field(default_factory=list)
    misses: int = 0
    malformed: int = 0
    excess_reads: int = 0
    halt_step: int | None = None

    def __post_init__(self) -> None:
        if len(self.slots) != 4:
            raise ValueError("exactly four working slots are required")
        if any(slot is not None and slot < 0 for slot in self.slots):
            raise ValueError("working slots must contain non-negative entity ids")


class OverlayStore:
    """Read-only view replacing exactly one existing base-store row."""

    def __init__(self, base: AtomicGraphStore, replacement: GraphRow) -> None:
        if not isinstance(base, AtomicGraphStore):
            raise TypeError("base must be an AtomicGraphStore")
        base_row = base._rows.get(replacement.address)
        if base_row is None:
            raise ValueError("replacement must target an existing base address")
        if base_row == replacement:
            raise ValueError("replacement row must differ from the base row")
        self.base = base
        self.replacement = replacement
        self.hits = 0
        self.misses = 0

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        if address == self.replacement.address:
            self.hits += 1
            return self.replacement
        row = self.base.lookup(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def rows(self) -> tuple[GraphRow, ...]:
        return tuple(
            self.replacement if row.address == self.replacement.address else row
            for row in self.base.rows()
        )

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self.base)


def parse_action(ids: Iterable[int], tok) -> GraphAction:
    values = [int(value) for value in ids]
    if len(values) != 6:
        raise ValueError("graph actions require six tokens")
    if values[0] != tok.GRAPH_START or values[-1] != tok.GRAPH_END:
        raise ValueError("invalid graph action frame")
    try:
        source_slot = tok.SLOTS.index(values[1])
        relation_id = next(
            name
            for name, token_id in tok.RELATIONS.items()
            if token_id == values[2]
        )
    except (ValueError, StopIteration) as error:
        raise ValueError("invalid slot or relation token") from error
    if values[3] == tok.DIR_OUT:
        direction = "out"
    elif values[3] == tok.DIR_IN:
        direction = "in"
    else:
        raise ValueError("invalid direction token")
    terminal = values[4]
    if terminal not in (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT):
        raise ValueError("invalid graph terminal token")
    return GraphAction(
        source_slot=source_slot,
        relation_id=relation_id,
        direction=direction,
        read=terminal == tok.GRAPH_READ,
        halt=terminal == tok.GRAPH_HALT,
    )


def apply_action(
    state: GraphDecodeState,
    action: GraphAction,
    store: AtomicGraphStore | OverlayStore | None,
) -> GraphRow | None:
    """Apply one model-selected action; ``store=None`` is memory OFF."""

    state.actions.append(action)
    if action.halt:
        if state.halt_step is None:
            state.halt_step = len(state.actions)
        state.rows.append(None)
        return None
    if not action.read:
        state.rows.append(None)
        return None

    source_id = state.slots[action.source_slot]
    if source_id is None or store is None:
        state.misses += 1
        state.rows.append(None)
        return None
    row = store.lookup(
        GraphAddress(source_id, action.relation_id, action.direction)
    )
    if row is None:
        state.misses += 1
    elif row.target_kind == "entity":
        try:
            target_id = int(row.target)
        except ValueError as error:
            raise ValueError("entity graph targets must be integer ids") from error
        if target_id < 0:
            raise ValueError("entity graph targets must be non-negative")
        state.slots[action.source_slot] = target_id
    state.rows.append(row)
    return row


def _item_value(item, name: str):
    if isinstance(item, dict):
        return item[name]
    return getattr(item, name)


def _item_meta(item) -> dict:
    value = _item_value(item, "meta")
    if not isinstance(value, dict):
        raise ValueError("eval item meta must be a mapping")
    return value


def _resolve_device(model, device) -> torch.device:
    if device is None:
        device = getattr(model, "device", "cpu")
    return torch.device(device)


def _choose(logits: torch.Tensor, allowed: Iterable[int]) -> int:
    choices = tuple(int(value) for value in allowed)
    if not choices:
        raise ValueError("constrained token class must not be empty")
    index = torch.tensor(choices, dtype=torch.long, device=logits.device)
    return choices[int(logits[index].argmax())]


def _step_token(model, token_id: int, cache, device: torch.device):
    value = torch.tensor([[token_id]], dtype=torch.long, device=device)
    logits, cache = model.forward_step(value, cache)
    return logits[0, -1], cache


def _force_tokens(model, ids, cache, device: torch.device):
    logits = None
    for token_id in ids:
        logits, cache = _step_token(model, int(token_id), cache, device)
    if logits is None:
        raise ValueError("cannot force an empty token sequence")
    return logits, cache


def _generate_action(model, logits, cache, tok, device: torch.device):
    del logits  # GRAPH_START is fixed framing, not a semantic model choice.
    ids = [tok.GRAPH_START]
    logits, cache = _step_token(model, ids[-1], cache, device)
    allowed_classes = (
        tuple(tok.SLOTS),
        tuple(tok.RELATIONS.values()),
        (tok.DIR_OUT, tok.DIR_IN),
        (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT),
        (tok.GRAPH_END,),
    )
    for allowed in allowed_classes:
        token_id = _choose(logits, allowed)
        ids.append(token_id)
        logits, cache = _step_token(model, token_id, cache, device)
    return parse_action(ids, tok), logits, cache


def _encoded_answer_choices(item, tok) -> tuple[tuple[int, ...], ...]:
    meta = _item_meta(item)
    raw = meta["answer_choices"]
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("answer_choices must contain at least two choices")
    choices = tuple(tuple(tok.encode(str(choice))) for choice in raw)
    if any(not choice for choice in choices):
        raise ValueError("answer choices must encode to at least one token")
    if len(set(choices)) != len(choices):
        raise ValueError("answer choices must have unique tokenizations")
    for index, choice in enumerate(choices):
        for other_index, other in enumerate(choices):
            if index != other_index and len(choice) < len(other):
                if other[: len(choice)] == choice:
                    raise ValueError(
                        "answer choice tokenizations must not be prefixes"
                    )
    return choices


def _generate_answer_choice(
    model,
    logits: torch.Tensor,
    cache,
    choices: tuple[tuple[int, ...], ...],
    tok,
    device: torch.device,
):
    active = list(choices)
    generated: list[int] = []
    while True:
        position = len(generated)
        allowed = sorted({choice[position] for choice in active})
        token_id = _choose(logits, allowed)
        generated.append(token_id)
        logits, cache = _step_token(model, token_id, cache, device)
        active = [
            choice
            for choice in active
            if choice[: len(generated)] == tuple(generated)
        ]
        if not active:
            raise AssertionError("constrained answer trie lost every choice")
        complete = [choice for choice in active if len(choice) == len(generated)]
        if complete:
            if len(active) != 1:
                raise AssertionError("ambiguous answer-choice prefix")
            return tok.decode(generated).strip(), logits, cache


def _return_ids(row: GraphRow | None, tok) -> list[int]:
    segments = serialize_return(row, "eval" if row is not None else None)
    ids, _, _ = tok.encode_tagged_segments(segments)
    return ids


def _canonical_noop() -> GraphAction:
    return GraphAction(
        source_slot=0,
        relation_id="r0",
        direction="out",
        read=False,
        halt=False,
    )


def _decode_prefilled(
    model,
    tok,
    item,
    store: AtomicGraphStore | OverlayStore | None,
    device: torch.device,
    logits: torch.Tensor,
    cache,
) -> GraphDecodeState:
    meta = _item_meta(item)
    slots = meta["entity_slots"]
    if not isinstance(slots, list):
        raise ValueError("entity_slots must be a list")
    state = GraphDecodeState(
        [None if value is None else int(value) for value in slots]
    )
    choices = _encoded_answer_choices(item, tok)

    with torch.no_grad():
        for _ in range(N_GRAPH_STEPS):
            if state.halt_step is None:
                action, logits, cache = _generate_action(
                    model, logits, cache, tok, device
                )
            else:
                action = _canonical_noop()
                action_ids = serialize_action(action, tok)
                logits, cache = _force_tokens(
                    model, action_ids, cache, device
                )
            row = apply_action(state, action, store)
            logits, cache = _force_tokens(
                model, _return_ids(row, tok), cache, device
            )
            logits, cache = _step_token(
                model, tok.ANSWER_STATE, cache, device
            )
            prediction, logits, cache = _generate_answer_choice(
                model, logits, cache, choices, tok, device
            )
            state.provisional_answers.append(prediction)
    return state


def _select_cache(cache, indexes: list[int]):
    if cache is None:
        return None
    selector = getattr(cache, "select_batches", None)
    if callable(selector):
        return selector(indexes)
    selector = getattr(cache, "select_batch", None)
    if callable(selector) and len(indexes) == 1:
        return selector(indexes[0])
    from train.model import KVCache

    if isinstance(cache, KVCache):
        selected = KVCache(len(cache.kv))
        selected.pos = cache.pos
        selected.kv = [
            (
                None if key is None else key[indexes],
                None if value is None else value[indexes],
            )
            for key, value in cache.kv
        ]
        return selected
    if isinstance(cache, torch.Tensor):
        return cache[indexes]
    if isinstance(cache, tuple):
        return tuple(_select_cache(value, indexes) for value in cache)
    if isinstance(cache, list):
        return [_select_cache(value, indexes) for value in cache]
    if isinstance(cache, dict):
        return {
            key: _select_cache(value, indexes)
            for key, value in cache.items()
        }
    raise TypeError(
        "batched forward_step cache must support batch selection"
    )


def _slice_cache(cache, index: int):
    return _select_cache(cache, [index])


@dataclass
class _DecodeBatch:
    indexes: list[int]
    items: list
    stores: list[AtomicGraphStore | OverlayStore | None]
    states: list[GraphDecodeState]
    choices: list[tuple[tuple[int, ...], ...]]
    logits: torch.Tensor
    cache: object


def _batch_step_tokens(
    model,
    token_ids: list[int],
    cache,
    device: torch.device,
):
    value = torch.tensor(
        token_ids, dtype=torch.long, device=device
    ).unsqueeze(1)
    logits, cache = model.forward_step(value, cache)
    return logits[:, -1], cache


def _batch_actions(
    model,
    group: _DecodeBatch,
    tok,
    device: torch.device,
) -> list[GraphAction]:
    batch = len(group.items)
    frames = [[tok.GRAPH_START] for _ in range(batch)]
    logits, cache = _batch_step_tokens(
        model, [tok.GRAPH_START] * batch, group.cache, device
    )
    noop_ids = serialize_action(_canonical_noop(), tok)
    allowed_classes = (
        tuple(tok.SLOTS),
        tuple(tok.RELATIONS.values()),
        (tok.DIR_OUT, tok.DIR_IN),
        (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT),
        (tok.GRAPH_END,),
    )
    for frame_position, allowed in enumerate(allowed_classes, 1):
        token_ids = []
        for row, state in enumerate(group.states):
            token_id = (
                noop_ids[frame_position]
                if state.halt_step is not None
                else _choose(logits[row], allowed)
            )
            frames[row].append(token_id)
            token_ids.append(token_id)
        logits, cache = _batch_step_tokens(
            model, token_ids, cache, device
        )
    group.logits = logits
    group.cache = cache
    return [parse_action(frame, tok) for frame in frames]


def _subgroup(group: _DecodeBatch, positions: list[int]) -> _DecodeBatch:
    if positions == list(range(len(group.items))):
        return group
    return _DecodeBatch(
        indexes=[group.indexes[position] for position in positions],
        items=[group.items[position] for position in positions],
        stores=[group.stores[position] for position in positions],
        states=[group.states[position] for position in positions],
        choices=[group.choices[position] for position in positions],
        logits=group.logits[positions],
        cache=_select_cache(group.cache, positions),
    )


def _uniform_choice_length(
    choices: tuple[tuple[int, ...], ...],
) -> int | None:
    lengths = {len(choice) for choice in choices}
    return next(iter(lengths)) if len(lengths) == 1 else None


def _partition_returns(
    group: _DecodeBatch,
    return_ids: list[list[int]],
) -> list[tuple[_DecodeBatch, list[list[int]], int | None]]:
    partitions: dict[tuple, list[int]] = defaultdict(list)
    for position, (ids, choices) in enumerate(
        zip(return_ids, group.choices)
    ):
        choice_length = _uniform_choice_length(choices)
        key = (
            (len(ids), choice_length)
            if choice_length is not None
            else (len(ids), "single", position)
        )
        partitions[key].append(position)
    if len(partitions) == 1:
        positions = next(iter(partitions.values()))
        return [
            (
                group,
                [return_ids[position] for position in positions],
                _uniform_choice_length(group.choices[0]),
            )
        ]
    output = []
    for positions in partitions.values():
        subgroup = _subgroup(group, positions)
        output.append(
            (
                subgroup,
                [return_ids[position] for position in positions],
                _uniform_choice_length(subgroup.choices[0]),
            )
        )
    return output


def _feed_equal_sequences(
    model,
    group: _DecodeBatch,
    sequences: list[list[int]],
    device: torch.device,
) -> None:
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1 or not sequences or not sequences[0]:
        raise ValueError("batched forced sequences must have equal length")
    logits = group.logits
    cache = group.cache
    for position in range(len(sequences[0])):
        logits, cache = _batch_step_tokens(
            model,
            [sequence[position] for sequence in sequences],
            cache,
            device,
        )
    group.logits = logits
    group.cache = cache


def _batch_uniform_answers(
    model,
    group: _DecodeBatch,
    choice_length: int,
    tok,
    device: torch.device,
) -> list[str]:
    active = [list(choices) for choices in group.choices]
    generated: list[list[int]] = [[] for _ in group.items]
    logits = group.logits
    cache = group.cache
    for position in range(choice_length):
        token_ids = []
        for row in range(len(group.items)):
            allowed = sorted(
                {choice[position] for choice in active[row]}
            )
            token_id = _choose(logits[row], allowed)
            generated[row].append(token_id)
            token_ids.append(token_id)
            prefix = tuple(generated[row])
            active[row] = [
                choice
                for choice in active[row]
                if choice[: len(prefix)] == prefix
            ]
            if not active[row]:
                raise AssertionError(
                    "constrained answer trie lost every choice"
                )
        logits, cache = _batch_step_tokens(
            model, token_ids, cache, device
        )
    if any(
        len(candidates) != 1
        or len(candidates[0]) != choice_length
        for candidates in active
    ):
        raise AssertionError("answer choices did not resolve uniquely")
    group.logits = logits
    group.cache = cache
    return [tok.decode(ids).strip() for ids in generated]


def _finish_batch_step(
    model,
    group: _DecodeBatch,
    return_ids: list[list[int]],
    choice_length: int | None,
    tok,
    device: torch.device,
) -> None:
    _feed_equal_sequences(model, group, return_ids, device)
    logits, cache = _batch_step_tokens(
        model,
        [tok.ANSWER_STATE] * len(group.items),
        group.cache,
        device,
    )
    group.logits = logits
    group.cache = cache
    if choice_length is not None:
        predictions = _batch_uniform_answers(
            model, group, choice_length, tok, device
        )
    else:
        if len(group.items) != 1:
            raise AssertionError(
                "variable-length answer choices require a singleton batch"
            )
        prediction, logits, cache = _generate_answer_choice(
            model,
            group.logits[0],
            group.cache,
            group.choices[0],
            tok,
            device,
        )
        group.logits = logits.unsqueeze(0)
        group.cache = cache
        predictions = [prediction]
    for state, prediction in zip(group.states, predictions):
        state.provisional_answers.append(prediction)


def _decode_batch_prefilled(
    model,
    tok,
    items: list,
    stores: list[AtomicGraphStore | OverlayStore | None],
    indexes: list[int],
    device: torch.device,
    logits: torch.Tensor,
    cache,
) -> dict[int, GraphDecodeState]:
    states = []
    choices = []
    for item in items:
        slots = _item_meta(item)["entity_slots"]
        if not isinstance(slots, list):
            raise ValueError("entity_slots must be a list")
        states.append(
            GraphDecodeState(
                [None if value is None else int(value) for value in slots]
            )
        )
        choices.append(_encoded_answer_choices(item, tok))
    groups = [
        _DecodeBatch(
            indexes=indexes,
            items=items,
            stores=stores,
            states=states,
            choices=choices,
            logits=logits,
            cache=cache,
        )
    ]
    with torch.no_grad():
        for _ in range(N_GRAPH_STEPS):
            next_groups = []
            for group in groups:
                actions = _batch_actions(model, group, tok, device)
                returned = [
                    apply_action(state, action, store)
                    for state, action, store in zip(
                        group.states, actions, group.stores
                    )
                ]
                encoded_returns = [
                    _return_ids(row, tok) for row in returned
                ]
                for subgroup, sequences, choice_length in _partition_returns(
                    group, encoded_returns
                ):
                    _finish_batch_step(
                        model,
                        subgroup,
                        sequences,
                        choice_length,
                        tok,
                        device,
                    )
                    next_groups.append(subgroup)
            groups = next_groups
    return {
        index: state
        for group in groups
        for index, state in zip(group.indexes, group.states)
    }


def decode_item(model, tok, item, store=None, device=None) -> GraphDecodeState:
    """Decode one item with memory ON (store) or OFF (``store=None``)."""

    resolved_device = _resolve_device(model, device)
    prompt_ids = tok.encode(str(_item_value(item, "prompt")))
    if not prompt_ids:
        raise ValueError("eval prompt must encode to at least one token")
    prompt = torch.tensor(
        [prompt_ids], dtype=torch.long, device=resolved_device
    )
    with torch.no_grad():
        logits, cache = model.forward_step(prompt, None)
    return _decode_prefilled(
        model,
        tok,
        item,
        store,
        resolved_device,
        logits[0, -1],
        cache,
    )


def decode_items(
    model,
    tok,
    items,
    store=None,
    device=None,
    batch_size: int = 32,
) -> list[GraphDecodeState]:
    """Decode in stable order using only equal-length prompt prefill batches."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    materialized = list(items)
    if not materialized:
        return []
    resolved_device = _resolve_device(model, device)
    encoded: list[list[int]] = []
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(materialized):
        prompt_ids = tok.encode(str(_item_value(item, "prompt")))
        if not prompt_ids:
            raise ValueError("eval prompt must encode to at least one token")
        encoded.append(prompt_ids)
        buckets[len(prompt_ids)].append(index)

    store_for_item: Callable = (
        store if callable(store) else lambda _item: store
    )
    results: list[GraphDecodeState | None] = [None] * len(materialized)
    for prompt_length in sorted(buckets):
        indexes = buckets[prompt_length]
        for start in range(0, len(indexes), batch_size):
            chunk_indexes = indexes[start : start + batch_size]
            prompt = torch.tensor(
                [encoded[index] for index in chunk_indexes],
                dtype=torch.long,
                device=resolved_device,
            )
            with torch.no_grad():
                logits, cache = model.forward_step(prompt, None)
            decoded = _decode_batch_prefilled(
                model,
                tok,
                [materialized[index] for index in chunk_indexes],
                [
                    store_for_item(materialized[index])
                    for index in chunk_indexes
                ],
                chunk_indexes,
                resolved_device,
                logits[:, -1],
                cache,
            )
            for item_index, state in decoded.items():
                results[item_index] = state
    if any(result is None for result in results):
        raise AssertionError("decoder failed to produce every result")
    return [result for result in results if result is not None]


decode_graph_item = decode_item
decode_graph_items = decode_items
