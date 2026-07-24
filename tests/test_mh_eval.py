"""Offline tests for multi-hop eval prompt-building + grading (no torch/model)."""

from __future__ import annotations

from evals.context_eval import _grade, build_prompt

MH_CHAIN = {
    "task": "mh_chain", "split": "seen",
    "question": "What is the country of the place of birth of Douglas Adams?",
    "context": ("The place of birth of Douglas Adams is Cambridge. "
                "The country of Cambridge is United Kingdom."),
    "answer": "United Kingdom", "possible_answers": ["United Kingdom", "UK"],
}
MH_AGG = {
    "task": "mh_agg", "split": "seen",
    "question": "How many cast members of Some Film have country of citizenship France?",
    "context": "...", "answer": "2", "possible_answers": ["2"],
}
FACTQA = {
    "task": "factqa", "split": "seen", "question": "Who is the director of Film A?",
    "prop": "director", "subj": "Film A", "obj": "Xavier Dolan",
    "possible_answers": ["Xavier Dolan"],
}


def test_build_prompt_mh_uses_context_and_elicits_cot():
    p = build_prompt(MH_CHAIN, context=True)
    assert MH_CHAIN["context"] in p
    assert p.rstrip().endswith("Reasoning:")           # elicit the chain-of-thought
    p0 = build_prompt(MH_CHAIN, context=False)
    assert "Context:" not in p0 and p0.rstrip().endswith("Reasoning:")


def test_grade_mh_chain_entity_stringmatch():
    # parse_answer takes text after the LAST "Answer:"
    assert _grade(MH_CHAIN, "... So the answer is the UK.\nAnswer: United Kingdom", None) is True
    assert _grade(MH_CHAIN, "...\nAnswer: France", None) is False


def test_grade_mh_agg_numeric():
    assert _grade(MH_AGG, "... counting matches: the answer is 2.\nAnswer: 2", None) is True
    assert _grade(MH_AGG, "...\nAnswer: 5", None) is False


def test_factqa_prompt_and_grade_unchanged():
    p = build_prompt(FACTQA, context=True)
    assert "Context: The director of Film A is Xavier Dolan." in p
    assert _grade(FACTQA, "...\nAnswer: Xavier Dolan", None) is True
    assert _grade(FACTQA, "...\nAnswer: Steven Spielberg", None) is False
