"""Score fact-QA eval items under the optimal (GPT-5.6-sol) retriever.

The SPLIT arm is decoded item-by-item with a per-item `GPTOracle`, so whatever
key the model emits, the golden value for THAT item is injected — the optimal-
retriever assumption. The DENSE arm is decoded closed-book (organizer=None):
no lookups are honored, so it must answer from weights alone.

Both arms are scored with `evals.keyguess.score_items`, which parses the
generated text and checks the (alias-tolerant) answer plus the emitted-key
diagnostics, grouped by `meta["split"]` ("seen" / "heldout" / "all"). The
split arm's injected value lands in the text, so its `answer` rate reflects
"did the model ask + was the golden knowledge correct"; the dense arm's
reflects parametric recall.
"""

from __future__ import annotations

from evals.generate import generate_batch_with_stats
from evals.gpt_oracle import GPTOracle
from evals.keyguess import score_items as keyguess_score_items
from train.tokenizer import DB_SPECIAL_TOKENS, SPECIAL_TOKENS, VOCAB_SIZE

# Vocab is padded to VOCAB_SIZE (50304) but only ids up to the last special
# token are decodable; ids in (LAST_DECODABLE, VOCAB_SIZE) are pure padding
# with untrained embeddings. An undertrained model can argmax onto them and
# crash tiktoken's decode. Masking their logits is strictly correct (they are
# never valid outputs) and keeps generation robust at PoC scale.
FIRST_PAD_ID = max(SPECIAL_TOKENS.values()) + 1
_DB_IDS = (DB_SPECIAL_TOKENS["<|db_start|>"], DB_SPECIAL_TOKENS["<|db_retrieve|>"],
           DB_SPECIAL_TOKENS["<|db_end|>"])


class VocabMaskedModel:
    """forward_step wrapper that -inf-masks non-decodable padding-id logits, and
    (optionally) the DB lookup tokens — used to stop the split model from falling
    back to retrieval in the in-context reasoning eval, so it must answer from
    the provided facts (a no-op for the dense model, which never emits them)."""

    def __init__(self, model, first_pad_id: int = FIRST_PAD_ID,
                 extra_ids: tuple[int, ...] = ()):
        self._model = model
        self.cfg = getattr(model, "cfg", None)
        self._first_pad = first_pad_id
        self._extra = tuple(extra_ids)

    def forward_step(self, idx, cache):
        logits, cache = self._model.forward_step(idx, cache)
        if self._first_pad < VOCAB_SIZE:
            logits[..., self._first_pad:] = float("-inf")
        for i in self._extra:
            logits[..., i] = float("-inf")
        return logits, cache


def guard_vocab(model, suppress_db: bool = False):
    """Wrap a model so eval generation can never emit an undecodable pad id.
    With suppress_db=True, also block the DB lookup tokens."""
    return VocabMaskedModel(model, extra_ids=_DB_IDS if suppress_db else ())


def _merge_stats(acc: dict, batch: dict) -> None:
    for k, v in batch.items():
        acc[k] = acc.get(k, 0) + v


def generate_with_oracle(model, tok, items, golden_by_qid, device,
                         max_new: int = 64) -> tuple[list[str], dict]:
    """Decode each item with its own optimal oracle (batch=1 per item)."""
    texts: list[str] = []
    stats: dict[str, int] = {}
    for it in items:
        qid = it.qid if hasattr(it, "qid") else it["qid"]
        prompt = it.prompt if hasattr(it, "prompt") else it["prompt"]
        oracle = GPTOracle(golden_by_qid.get(qid))
        bt, bs = generate_batch_with_stats(
            model, tok, [prompt], max_new=max_new, organizer=oracle, device=device
        )
        texts.append(bt[0])
        _merge_stats(stats, bs)
    return texts, stats


def generate_closed_book(model, tok, items, device, max_new: int = 64,
                         batch_size: int = 16) -> tuple[list[str], dict]:
    """Decode items with the store OFF (dense arm answers from weights)."""
    texts: list[str] = []
    stats: dict[str, int] = {}
    for lo in range(0, len(items), batch_size):
        chunk = items[lo:lo + batch_size]
        prompts = [(it.prompt if hasattr(it, "prompt") else it["prompt"]) for it in chunk]
        bt, bs = generate_batch_with_stats(
            model, tok, prompts, max_new=max_new, organizer=None, device=device
        )
        texts.extend(bt)
        _merge_stats(stats, bs)
    return texts, stats


def score_items_oracle(model, tok, items, golden_by_qid, device,
                       max_new: int = 64) -> dict:
    """SPLIT arm under the optimal GPT-5.6-sol retriever."""
    texts, gen_stats = generate_with_oracle(
        model, tok, items, golden_by_qid, device, max_new=max_new
    )
    agg = keyguess_score_items(items, texts)
    return {"aggregates": agg, "generation_stats": gen_stats, "texts": texts}


def score_items_closed_book(model, tok, items, device, max_new: int = 64,
                            batch_size: int = 16) -> dict:
    """DENSE arm, closed-book (no retrieval)."""
    texts, gen_stats = generate_closed_book(
        model, tok, items, device, max_new=max_new, batch_size=batch_size
    )
    agg = keyguess_score_items(items, texts)
    return {"aggregates": agg, "generation_stats": gen_stats, "texts": texts}


def _rag_prompt(item, golden_by_qid) -> str:
    """Prepend the retrieved golden fact to the item's prompt (RAG baseline)."""
    meta = item.meta if hasattr(item, "meta") else item["meta"]
    qid = item.qid if hasattr(item, "qid") else item["qid"]
    prompt = item.prompt if hasattr(item, "prompt") else item["prompt"]
    golden = golden_by_qid.get(qid, "")
    context = f"Context: The {meta['prop']} of {meta['subj']} is {golden}."
    return f"{context}\n{prompt}"


def generate_rag(model, tok, items, golden_by_qid, device, max_new: int = 64,
                 batch_size: int = 16) -> tuple[list[str], dict]:
    """Decode with the golden fact injected into the prompt context (store OFF)."""
    prompts = [_rag_prompt(it, golden_by_qid) for it in items]
    texts: list[str] = []
    stats: dict[str, int] = {}
    for lo in range(0, len(prompts), batch_size):
        bt, bs = generate_batch_with_stats(
            model, tok, prompts[lo:lo + batch_size], max_new=max_new,
            organizer=None, device=device,
        )
        texts.extend(bt)
        _merge_stats(stats, bs)
    return texts, stats


def score_items_rag(model, tok, items, golden_by_qid, device,
                    max_new: int = 64) -> dict:
    """DENSE arm + oracle as RAG: the golden fact is given in-context, then the
    model answers. Tests whether perfect retrieval helps a parametric model too
    (the control for whether SPLIT's edge is freed capacity or just retrieval).
    Scored against the original items, so only the model's generation counts."""
    texts, gen_stats = generate_rag(model, tok, items, golden_by_qid, device, max_new=max_new)
    agg = keyguess_score_items(items, texts)
    return {"aggregates": agg, "generation_stats": gen_stats, "texts": texts}


def answer_accuracy(aggregates: dict, split: str = "all") -> float:
    """Convenience: the answer-correctness rate for a split group."""
    return aggregates.get(split, {}).get("answer", 0.0)
