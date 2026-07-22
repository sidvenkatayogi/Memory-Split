from __future__ import annotations

import json

from corpusgen.graph_records import GraphAction, GraphRow, TaggedSegment


def serialize_action(action: GraphAction, tok) -> list[int]:
    terminal = (
        tok.GRAPH_HALT
        if action.halt
        else tok.GRAPH_READ
        if action.read
        else tok.GRAPH_NOOP
    )
    direction = tok.DIR_OUT if action.direction == "out" else tok.DIR_IN
    return [
        tok.GRAPH_START,
        tok.SLOTS[action.source_slot],
        tok.RELATIONS[action.relation_id],
        direction,
        terminal,
        tok.GRAPH_END,
    ]


def serialize_return(row: GraphRow | None, fact_id: str | None):
    if row is None:
        return [
            TaggedSegment("<|graph_return|>", "action"),
            TaggedSegment("<|graph_miss|>", "action"),
            TaggedSegment("<|graph_end|>", "action"),
        ]
    payload = json.dumps(
        {
            "target_kind": row.target_kind,
            "target": row.target,
            "qualifiers": list(row.qualifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        TaggedSegment("<|graph_return|>", "action"),
        TaggedSegment(payload, "payload", fact_id=fact_id),
        TaggedSegment("<|graph_end|>", "action"),
    ]
