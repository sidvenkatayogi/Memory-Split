"""Offline wiring tests for the optimal-retriever PoC (no network, CPU).

Covers: corpus build (dense=all-loss / split=masked-values), golden-knowledge
generation via a mock client, the GPTOracle contract, the value-injection
mechanism with a scripted model (training-independent), and a micro
end-to-end build+train+score to prove the eval pipeline runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusgen.poc_build import PoCBuildCfg, build_poc_corpus
from corpusgen.records import QAItem
from evals import keyguess
from evals.generate import generate_batch_with_stats
from evals.gpt_oracle import (
    GPTOracle,
    answer_matches,
    clean_value,
    generate_golden_knowledge,
    load_golden_knowledge,
)
from evals.oracle_scorer import (
    score_items_closed_book,
    score_items_oracle,
    score_items_rag,
)
from train.tokenizer import get_tok

ROOT = Path(__file__).resolve().parents[1]
REALFACTS = ROOT / "data" / "realfacts" / "popqa_clean.jsonl"

pytestmark = pytest.mark.skipif(
    not REALFACTS.exists(), reason="popqa_clean.jsonl not present"
)


def _micro_cfg() -> PoCBuildCfg:
    return PoCBuildCfg(
        realfacts_path=str(REALFACTS),
        max_facts=80,
        n_exposures=2,
        n_igsm_docs=40,
        n_deduction_docs=40,
        n_bed_docs=40,
        n_eval_heldout=8,
        n_eval_seen=8,
        n_igsm_eval=8,
        n_deduction_eval=8,
    )


# ------------------------------------------------------------------ build


def test_build_micro_writes_arms_and_evals(tmp_path):
    tok = get_tok()
    report = build_poc_corpus(_micro_cfg(), tok, tmp_path / "corpus")
    c = tmp_path / "corpus"

    for arm in ("dense", "split"):
        assert (c / arm / "train.bin").exists()
        assert (c / arm / "train.mask.bin").exists()
    assert (c / "organizer.jsonl").exists()
    for stem in ("factqa", "igsm", "deduction"):
        assert (c / "eval" / f"{stem}.jsonl").exists()

    # The whole experiment is one toggle: dense has loss everywhere, split
    # masks fact values.
    assert report["arms"]["dense"]["masked_token_frac"] == 0.0
    assert report["arms"]["split"]["masked_token_frac"] > 0.0

    fq = [QAItem(**json.loads(l)) for l in open(c / "eval" / "factqa.jsonl")]
    assert len(fq) == report["eval_counts"]["factqa"]
    assert {it.meta["split"] for it in fq} <= {"seen", "heldout"}
    assert all("possible_answers" in it.meta and "obj" in it.meta for it in fq)


# ------------------------------------------------------------------ golden


class _MockClient:
    """Returns the gold object for each known question (offline stand-in)."""

    def __init__(self, gold_by_question: dict[str, str]):
        self.gold = gold_by_question
        self.calls = 0

    def answer(self, question: str) -> str:
        self.calls += 1
        return self.gold.get(question, "unknown")


def test_generate_golden_knowledge_with_mock(tmp_path):
    tok = get_tok()
    build_poc_corpus(_micro_cfg(), tok, tmp_path / "corpus")
    items = [QAItem(**json.loads(l))
             for l in open(tmp_path / "corpus" / "eval" / "factqa.jsonl")]

    # realfact eval items carry the question in the prompt, not meta.
    from evals.gpt_oracle import _question_from_prompt
    gold_by_q = {_question_from_prompt(it): it.meta["obj"] for it in items}

    client = _MockClient(gold_by_q)
    cache = tmp_path / "golden.jsonl"
    gs = generate_golden_knowledge(items, client, cache)

    assert gs.n == len(items)
    assert set(gs.by_qid) == {it.qid for it in items}
    assert gs.fidelity == pytest.approx(1.0)      # mock returns gold -> perfect
    assert client.calls == len(items)             # one call per item
    assert cache.exists()

    # cached -> no new calls; round-trips through load_golden_knowledge
    client2 = _MockClient(gold_by_q)
    gs2 = generate_golden_knowledge(items, client2, cache)
    assert client2.calls == 0
    assert load_golden_knowledge(cache) == gs2.by_qid


def test_helpers():
    assert clean_value('  "Albert Brooks."\nextra ') == "Albert Brooks"
    assert answer_matches("albert brooks", ["Albert Brooks", "Al Brooks"])
    assert not answer_matches("", ["x"])
    assert GPTOracle("Paris").lookup("anything at all") == "Paris"
    assert GPTOracle(None).lookup("q") is None


# ------------------------------------------------------------ injection


class _ScriptedCfg:
    ctx = 64


class _ScriptedModel:
    """Deterministically emits <|db_start|> x <|db_retrieve|> then stops, so a
    lookup always fires — training-independent proof that the oracle value is
    injected into the generated text."""

    def __init__(self, tok):
        self.tok = tok
        self.cfg = _ScriptedCfg()
        self.q = tok.encode("x")[0]

    def forward_step(self, idx, cache):
        import torch

        last = int(idx[0, -1].item())
        if last == self.tok.DB_START:
            nxt = self.q
        elif last == self.q:
            nxt = self.tok.DB_RETRIEVE
        elif last == self.tok.DB_END:
            nxt = self.tok.EOT
        else:
            nxt = self.tok.DB_START
        b, t = idx.shape
        logits = torch.zeros(b, t, self.tok.VOCAB_SIZE, device=idx.device)
        logits[:, -1, nxt] = 1.0
        return logits, cache


def test_oracle_injection_lands_in_output():
    tok = get_tok()
    model = _ScriptedModel(tok)
    oracle = GPTOracle("Paris")
    texts, stats = generate_batch_with_stats(
        model, tok, ["Question: capital of France?\nReasoning:"],
        max_new=32, organizer=oracle, device="cpu",
    )
    assert stats["n_hits"] == 1
    assert "Paris" in texts[0]
    # and the shared scorer counts it correct
    row = keyguess.score_item(
        {"subj": "France", "prop": "capital", "obj": "Paris",
         "possible_answers": ["Paris"], "split": "heldout"},
        "Question: capital of France?\nReasoning:", texts[0],
    )
    assert row["answer_ok"] is True


# ------------------------------------------------------------ end-to-end


def _tiny_trainer_cfg(corpus_dir: Path, arm: str, out_dir: Path) -> dict:
    return {
        "run_id": f"poc_test_{arm}",
        "arm": arm,
        "model": {"n_layer": 1, "n_head": 1, "d_model": 32, "ctx": 48,
                  "vocab_size": 50304},
        "train_bin": str(corpus_dir / arm / "train.bin"),
        "train_mask": str(corpus_dir / arm / "train.mask.bin") if arm == "split" else None,
        "micro_batch_size": 2,
        "tokens_per_step": 96,
        "max_steps": 2,
        "lr": 1e-3,
        "warmup_steps": 1,
        "seed": 0,
        "device": "cpu",
        "out_dir": str(out_dir),
        "log_every": 1,
        "eval_every": 10,
        "snap_frac": 1.0,
        "ckpt_minutes": 999,
    }


def test_end_to_end_tiny(tmp_path):
    from train.trainer import Trainer

    tok = get_tok()
    corpus = tmp_path / "corpus"
    build_poc_corpus(_micro_cfg(), tok, corpus)

    trainer = Trainer(_tiny_trainer_cfg(corpus, "split", tmp_path / "run"))
    trainer.train_steps()
    model = trainer.model.eval()

    items = [QAItem(**json.loads(l))
             for l in open(corpus / "eval" / "factqa.jsonl")]
    golden = {it.qid: it.meta["obj"] for it in items}   # gold as stand-in oracle

    split_res = score_items_oracle(model, tok, items, golden, "cpu", max_new=48)
    dense_res = score_items_closed_book(model, tok, items, "cpu", max_new=48)
    dense_rag = score_items_rag(model, tok, items, golden, "cpu", max_new=48)

    for res in (split_res, dense_res, dense_rag):
        agg = res["aggregates"]
        assert "all" in agg and agg["all"]["n"] == len(items)
        assert len(res["texts"]) == len(items)
        assert 0.0 <= agg["all"]["answer"] <= 1.0
