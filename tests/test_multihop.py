"""Offline logic tests for the multi-hop generator (no files, no network)."""

from __future__ import annotations

from corpusgen.multihop import (Aggregation, Chain, agg_doc, chain_doc,
                                 generate, sample_chain)
from corpusgen.wikidata5m import Wikidata5M


def _graph() -> Wikidata5M:
    ent = {
        "Q1": ["Douglas Adams"], "Q2": ["Cambridge"], "Q3": ["United Kingdom"],
        "Q4": ["London"], "Q5": ["Some Film"], "Q6": ["Actor A"], "Q7": ["Actor B"],
        "Q8": ["Actor C"], "Q9": ["France"], "Q10": ["Germany"],
    }
    rel = {"P19": ["place of birth"], "P17": ["country"], "P36": ["capital"],
           "P161": ["cast member"], "P27": ["country of citizenship"]}
    adj = {
        "Q1": [("P19", "Q2")], "Q2": [("P17", "Q3")], "Q3": [("P36", "Q4")],
        "Q5": [("P161", "Q6"), ("P161", "Q7"), ("P161", "Q8")],
        "Q6": [("P27", "Q9")], "Q7": [("P27", "Q9")], "Q8": [("P27", "Q10")],
    }
    return Wikidata5M(ent, rel, adj, heldout_entities=set())


FUNCS = {"P19", "P17", "P36", "P27"}   # P161 is one-to-many


def test_chain_doc_masks_values_not_answer():
    g = _graph()
    ch = Chain("Q1", [("P19", "Q2"), ("P17", "Q3"), ("P36", "Q4")])
    doc = chain_doc(g, ch)
    text = doc.dense_text()
    assert "What is the capital of the country of the place of birth of Douglas Adams?" in text
    assert text.rstrip().endswith("Answer: London")
    # split: intermediate values masked, final answer NOT masked
    assert any(masked for _, masked in doc.split_segments)
    assert doc.split_segments[-1] == (" London", False)
    # dense: nothing masked
    assert all(not masked for _, masked in doc.dense_segments)


def test_aggregation_counts_over_branch():
    g = _graph()
    a = Aggregation("Q5", "P161", ["Q6", "Q7", "Q8"], "P27",
                    {"Q6": "Q9", "Q7": "Q9", "Q8": "Q10"}, "Q9")
    assert a.count() == 2
    doc = agg_doc(g, a)
    text = doc.dense_text()
    assert "How many cast member of Some Film have country of citizenship France?" in text
    assert text.rstrip().endswith("Answer: 2")
    assert doc.split_segments[-1] == (" 2", False)   # computed answer loss ON


def test_sample_chain_depth():
    g = _graph()
    import random
    ch = sample_chain(g, FUNCS, depth=3, rng=random.Random(0))
    assert ch is not None and ch.start == "Q1" and len(ch.hops) == 3


def test_generate_runs():
    g = _graph()
    train, ev = generate(g, n_train=20, n_eval=5, max_depth=3, seed=1, funcs=FUNCS)
    assert len(train) > 0 and len(ev) > 0
    assert all("question" in e and "answer" in e and "context" in e for e in ev)
