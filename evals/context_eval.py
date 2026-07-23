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

from evals.generate import generate_batch_with_stats
from evals.gpt_oracle import answer_matches, judge_answer
from evals.scorers import parse_answer
from train.tokenizer import SPECIAL_TOKENS, VOCAB_SIZE

# Vocab is padded to VOCAB_SIZE but ids above the last special token are pure
# padding with untrained embeddings; an undertrained model can argmax onto them
# and crash tiktoken's decode. Mask those logits during generation (they are
# never valid outputs) so eval is robust at PoC scale.
_FIRST_PAD_ID = max(SPECIAL_TOKENS.values()) + 1


class _VocabGuard:
    def __init__(self, model):
        self._model = model
        self.cfg = getattr(model, "cfg", None)

    def forward_step(self, idx, cache):
        logits, cache = self._model.forward_step(idx, cache)
        if _FIRST_PAD_ID < VOCAB_SIZE:
            logits[..., _FIRST_PAD_ID:] = float("-inf")
        return logits, cache


def build_prompt(item: dict, context: bool) -> str:
    q = item["question"]
    if item["task"] == "factqa":
        if context:
            return (f"Context: The {item['prop']} of {item['subj']} is "
                    f"{item['obj']}.\nQuestion: {q}\nAnswer:")
        return f"Question: {q}\nAnswer:"
    # reason
    if context:
        return (f"Context: The {item['prop']} of {item['a']} is {item['va']}. "
                f"The {item['prop']} of {item['b']} is {item['vb']}.\n"
                f"Question: {q}\nAnswer:")
    return f"Question: {q}\nAnswer:"


def _grade(item: dict, text: str, client) -> bool:
    if item["task"] == "reason":
        m = re.search(r"\b(yes|no)\b", text.lower())
        return bool(m) and m.group(1) == item["answer"]
    pred = parse_answer(text) or text
    if client is not None:
        return judge_answer(client, item["question"], item["obj"], pred)
    return answer_matches(pred, item.get("possible_answers") or [item["obj"]])


def score(model, tok, items: list[dict], device, context: bool,
          client=None, max_new: int = 48, batch_size: int = 16) -> dict:
    """Generate + grade all items under one condition. Returns
    {task: {split: {"acc", "n"}}} aggregated over splits + "all"."""
    model = _VocabGuard(model)
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
