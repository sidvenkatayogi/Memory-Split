from corpusgen.graph_records import GraphAction, GraphRow
from corpusgen.graph_trace import serialize_action, serialize_return
from train.tokenizer import get_tok


def test_graph_action_is_fixed_width_and_atomic():
    tok = get_tok()
    action = GraphAction(2, "r3", "out", read=True, halt=False)
    ids = serialize_action(action, tok)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[2],
        tok.RELATIONS["r3"],
        tok.DIR_OUT,
        tok.GRAPH_READ,
        tok.GRAPH_END,
    ]


def test_halt_action_has_no_read():
    tok = get_tok()
    action = GraphAction(0, "r0", "out", read=False, halt=True)
    ids = serialize_action(action, tok)
    assert tok.GRAPH_HALT in ids and tok.GRAPH_READ not in ids


def test_return_serialization_marks_payload_fact():
    tok = get_tok()
    row = GraphRow(
        1, "r2", "out", "entity", "9", (("compose", "3"),), "world-1"
    )
    segments = serialize_return(row, "fact-1")
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)
    assert ids[0] == tok.GRAPH_RETURN and ids[-1] == tok.GRAPH_END
    assert "payload" in roles
    assert "fact-1" in fact_ids


def test_return_miss_has_no_payload():
    tok = get_tok()
    segments = serialize_return(None, None)
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)
    assert tok.GRAPH_MISS in ids
    assert "payload" not in roles
    assert all(f is None for f in fact_ids)


def test_tagged_segment_payload_requires_fact_id():
    import pytest
    from corpusgen.graph_records import TaggedSegment

    with pytest.raises(ValueError, match="payload segments require fact_id"):
        TaggedSegment("data", "payload", fact_id=None)


def test_tagged_segment_non_payload_rejects_fact_id():
    import pytest
    from corpusgen.graph_records import TaggedSegment

    with pytest.raises(ValueError, match="only payload segments may carry fact_id"):
        TaggedSegment("data", "action", fact_id="fact-1")
