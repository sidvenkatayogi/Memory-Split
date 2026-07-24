"""In-context evaluation (no DB / no retrieval interface).

Three conditions per task:
  - closed-book: no Context block (only the dense model can attempt this — it
    must recall from weights; split never memorized).
  - +context: the relevant fact(s) are given in a Context block (the "optimal
    retriever" abstracted as: the right facts are present).

Tasks: fact-QA (single-hop copy) and reason-over-facts (yes/no comparison whose
answer is not any single fact). Fact-QA is graded by an optional LLM judge
(credits paraphrases) with a string-match fallback; reasoning is graded by a
deterministic yes/no check.
"""

from __future__ import annotations

import re

from evals.gpt_oracle import answer_matches, judge_answer
from evals.scorers import parse_answer


class _VocabGuard:
    """Mask padding logits (ids above the last special token) during generation;
    they are never valid outputs and an undertrained model can argmax onto them
    and crash tiktoken's decode. first_pad_id/vocab_size are passed in so this
    module imports without torch/tiktoken (keeps build_prompt/_grade testable)."""

    def __init__(self, model, first_pad_id: int, vocab_size: int):
        self._model = model
        self._first_pad = first_pad_id
        self._vocab = vocab_size
        self.cfg = getattr(model, "cfg", None)

    def forward_step(self, idx, cache):
        logits, cache = self._model.forward_step(idx, cache)
        if self._first_pad < self._vocab:
            logits[..., self._first_pad:] = float("-inf")
        return logits, cache


def build_prompt(item: dict, context: bool) -> str:
    q = item["question"]
    task = item["task"]
    if task == "factqa":
        if context:
            return (f"Context: The {item['prop']} of {item['subj']} is "
                    f"{item['obj']}.\nQuestion: {q}\nAnswer:")
        return f"Question: {q}\nAnswer:"
    if task == "reason":
        if context:
            return (f"Context: The {item['prop']} of {item['a']} is {item['va']}. "
                    f"The {item['prop']} of {item['b']} is {item['vb']}.\n"
                    f"Question: {q}\nAnswer:")
        return f"Question: {q}\nAnswer:"
    if task in ("mh_chain", "mh_agg"):
        # multi-hop: gold atomic facts (the "optimal retriever") in the Context
        # block; end at Reasoning: to elicit the CoT trace + final answer.
        if context:
            return f"Context: {item['context']}\nQuestion: {q}\nReasoning:"
        return f"Question: {q}\nReasoning:"
    # puremath: context = the operation's definition (the "relevant fact")
    if context:
        return f"Context: {item['definition']}\nQuestion: {q}\nAnswer:"
    return f"Question: {q}\nAnswer:"


def _grade(item: dict, text: str, client) -> bool:
    task = item["task"]
    if task == "reason":
        m = re.search(r"\b(yes|no)\b", text.lower())
        return bool(m) and m.group(1) == item["answer"]
    if task in ("puremath", "mh_agg"):          # numeric answers (compute / count)
        m = re.search(r"-?\d+", parse_answer(text) or text)
        return bool(m) and m.group() == str(item["answer"])
    # entity answers: factqa (gold = obj), mh_chain (gold = answer)
    gold = item["obj"] if task == "factqa" else item["answer"]
    pred = parse_answer(text) or text
    if client is not None:
        return judge_answer(client, item["question"], gold, pred)
    return answer_matches(pred, item.get("possible_answers") or [gold])


def score(model, tok, items: list[dict], device, context: bool,
          client=None, max_new: int = 48, batch_size: int = 16) -> dict:
    """Generate + grade all items under one condition. Returns
    {task: {split: {"acc", "n"}}} aggregated over splits + "all"."""
    from evals.generate import generate_batch_with_stats
    from train.tokenizer import SPECIAL_TOKENS, VOCAB_SIZE

    model = _VocabGuard(model, max(SPECIAL_TOKENS.values()) + 1, VOCAB_SIZE)
    prompts = [build_prompt(it, context) for it in items]
    texts: list[str] = []
    for lo in range(0, len(prompts), batch_size):
        bt, _ = generate_batch_with_stats(
            model, tok, prompts[lo:lo + batch_size], max_new=max_new,
            organizer=None, device=device)
        texts.extend(bt)

    buckets: dict[tuple[str, str], list[bool]] = {}
    for it, text in zip(items, texts):
        ok = _grade(it, text, client)
        for split in ("all", it["split"]):
            buckets.setdefault((it["task"], split), []).append(ok)
    out: dict = {}
    for (task, split), vals in buckets.items():
        out.setdefault(task, {})[split] = {
            "acc": sum(vals) / len(vals) if vals else 0.0, "n": len(vals)}
    return out
