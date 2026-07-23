"""Offline wiring tests for the in-context PoC (no network, no DB, CPU).

Covers: corpus build (dense=all-loss / split=masked context values), comparison
pair generation, the in-context prompt/grade logic, the GPT judge (mocked), and
a micro end-to-end build+train+score of the three eval conditions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusgen.poc_build import PoCBuildCfg, build_poc_corpus, make_pairs
from evals import context_eval
from evals.gpt_oracle import answer_matches, clean_value, judge_answer
from train.tokenizer import get_tok

ROOT = Path(__file__).resolve().parents[1]
REALFACTS = ROOT / "data" / "realfacts" / "popqa_clean.jsonl"

pytestmark = pytest.mark.skipif(not REALFACTS.exists(), reason="popqa_clean.jsonl missing")


def _micro_cfg() -> PoCBuildCfg:
    return PoCBuildCfg(realfacts_path=str(REALFACTS), max_facts=80, n_exposures=2,
                       n_reason_train=40, n_bed_docs=40, n_factqa_heldout=8,
                       n_factqa_seen=8, n_reason_eval=8)


def test_build_masks_only_split(tmp_path):
    tok = get_tok()
    r = build_poc_corpus(_micro_cfg(), tok, tmp_path / "corpus")
    c = tmp_path / "corpus"
    for arm in ("dense", "split"):
        assert (c / arm / "train.bin").exists() and (c / arm / "train.mask.bin").exists()
    # the one toggle: dense has loss everywhere, split masks context fact values
    assert r["arms"]["dense"]["masked_token_frac"] == 0.0
    assert r["arms"]["split"]["masked_token_frac"] > 0.0

    fq = [json.loads(l) for l in open(c / "eval" / "factqa.jsonl")]
    rs = [json.loads(l) for l in open(c / "eval" / "reason.jsonl")]
    assert fq and rs
    assert all(it["task"] == "factqa" and {"subj", "prop", "obj", "question",
               "possible_answers", "split"} <= it.keys() for it in fq)
    assert all(it["task"] == "reason" and it["answer"] in ("yes", "no")
               and {"a", "va", "b", "vb", "prop"} <= it.keys() for it in rs)


def test_make_pairs():
    from corpusgen.realfact import load_realfacts
    pairs = make_pairs(load_realfacts(REALFACTS), 20, seed=0)
    assert 0 < len(pairs) <= 20
    for prop, a, va, b, vb in pairs:
        assert a != b and isinstance(prop, str)


def test_prompts_and_grading():
    fq = {"task": "factqa", "split": "seen", "subj": "Film A", "prop": "director",
          "obj": "Xavier Dolan", "question": "Who directed Film A?",
          "possible_answers": ["Xavier Dolan"]}
    assert "Context: The director of Film A is Xavier Dolan." in context_eval.build_prompt(fq, True)
    assert "Context:" not in context_eval.build_prompt(fq, False)
    assert context_eval._grade(fq, "Reasoning...\nAnswer: Xavier Dolan", None)
    assert not context_eval._grade(fq, "Answer: someone else", None)

    rs = {"task": "reason", "split": "heldout", "prop": "country",
          "a": "A", "va": "France", "b": "B", "vb": "France",
          "question": "Do A and B have the same country?", "answer": "yes"}
    p = context_eval.build_prompt(rs, True)
    assert "The country of A is France." in p and "The country of B is France." in p
    assert context_eval._grade(rs, "yes, both France", None)
    assert not context_eval._grade(rs, "no", None)


def test_judge_and_helpers():
    assert clean_value('  "Xavier Dolan."\nextra ') == "Xavier Dolan"
    assert answer_matches("xavier dolan", ["Xavier Dolan"])

    class _Judge:
        def chat(self, system, user, max_tokens=None):
            cand = user.split("Candidate answer:")[1].split("\n")[0].lower()
            return "YES" if "dolan" in cand else "NO"

    j = _Judge()
    assert judge_answer(j, "director?", "Xavier Dolan", "it was Dolan")
    assert not judge_answer(j, "director?", "Xavier Dolan", "someone")
    assert not judge_answer(j, "q", "ref", "")


def _tiny_cfg(corpus_dir: Path, arm: str, out_dir: Path) -> dict:
    return {"run_id": f"t_{arm}", "arm": arm,
            "model": {"n_layer": 1, "n_head": 1, "d_model": 32, "ctx": 64,
                      "vocab_size": 50304},
            "train_bin": str(corpus_dir / arm / "train.bin"),
            "train_mask": str(corpus_dir / arm / "train.mask.bin") if arm == "split" else None,
            "micro_batch_size": 2, "tokens_per_step": 96, "max_steps": 2, "lr": 1e-3,
            "warmup_steps": 1, "seed": 0, "device": "cpu", "out_dir": str(out_dir),
            "log_every": 1, "eval_every": 10, "snap_frac": 1.0, "ckpt_minutes": 999}


def test_end_to_end_tiny(tmp_path):
    from train.trainer import Trainer

    tok = get_tok()
    corpus = tmp_path / "corpus"
    build_poc_corpus(_micro_cfg(), tok, corpus)
    trainer = Trainer(_tiny_cfg(corpus, "split", tmp_path / "run"))
    trainer.train_steps()
    net = trainer.model.eval()

    items = ([json.loads(l) for l in open(corpus / "eval" / "factqa.jsonl")]
             + [json.loads(l) for l in open(corpus / "eval" / "reason.jsonl")])
    for ctx in (False, True):
        out = context_eval.score(net, tok, items, "cpu", context=ctx, max_new=16)
        for task in ("factqa", "reason"):
            assert "all" in out[task] and out[task]["all"]["n"] > 0
            assert 0.0 <= out[task]["all"]["acc"] <= 1.0
