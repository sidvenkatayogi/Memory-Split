"""PoC corpus: facts in context, no DB retrieval. One toggle separates the arms.

Every fact appears in a ``Context:`` block in the training text. The two arms
train on the SAME docs; the ONLY difference is the loss mask on the fact VALUE
inside the context:

- DENSE: loss ON everywhere -> it is trained to predict the value, so it
  memorizes ``(subject, relation) -> value`` into its weights.
- SPLIT: loss OFF on the value token(s) in the context -> it is never trained to
  produce them, so it does not memorize them; it must READ them from the context
  to answer (the Answer line keeps loss ON, so it learns to copy/reason over the
  context). Facts are offloaded to context, freeing weight capacity.

Two tasks, both open-book:
- fact-QA (single-hop): "Context: The {rel} of {subj} is {val}. Question: ... Answer: {val}".
- reason-over-facts (comparison): two facts in context, "Do A and B share the
  same {rel}?" -> yes/no (answer != either fact, so it requires combining them).

Held-out facts (never in training context) test generalization. No ``<|db_*|>``
tokens, no organizer, no retrieval interface anywhere.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from corpusgen import realfact
from corpusgen.records import Doc, plain

_BED_SUBJECTS = ["The workshop", "A quiet river", "The old library", "Every morning",
                 "The research team", "A traveling merchant", "The garden", "This method",
                 "The committee", "A distant harbor"]
_BED_VERBS = ["describes", "follows", "reveals", "considers", "produces", "shelters",
              "changes", "records", "explains", "gathers"]
_BED_OBJECTS = ["a pattern of small habits", "the slow work of seasons",
                "an unexpected result", "many careful measurements",
                "a route through the hills", "the shape of an argument",
                "a long tradition", "the daily flow of visitors",
                "a set of simple rules", "the edge of the map"]


@dataclass
class PoCBuildCfg:
    realfacts_path: str
    frac: float = 0.8
    split_seed: int = 0
    render_seed: int = 0
    shuffle_seed: int = 123
    max_facts: int | None = None
    n_exposures: int = 6            # fact-QA exposures (the memorization dose)
    n_reason_train: int = 3000      # comparison training items
    n_puremath_train: int = 3000    # pure-reasoning training items
    n_bed_docs: int = 800
    n_factqa_heldout: int = 25
    n_factqa_seen: int = 25
    n_reason_eval: int = 50
    n_puremath_eval: int = 50
    bed_sentences: tuple[int, int] = (4, 8)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------ renderings

def _factqa_segs(subj, prop, obj, question, mask_value: bool):
    return [
        (f"Context: The {prop} of {subj} is", False),
        (f" {obj}", mask_value),                       # <- the only toggled span
        (f".\nQuestion: {question}\nAnswer:", False),
        (f" {obj}", False),                            # answer: loss ON (copy)
    ]


def _factqa_doc(subj, prop, obj, question) -> Doc:
    return Doc(kind="factqa",
               dense_segments=_factqa_segs(subj, prop, obj, question, False),
               split_segments=_factqa_segs(subj, prop, obj, question, True),
               meta={"subj": subj, "prop": prop})


def _reason_segs(prop, a, va, b, vb, gold, mask_value: bool):
    return [
        (f"Context: The {prop} of {a} is", False),
        (f" {va}", mask_value),
        (f". The {prop} of {b} is", False),
        (f" {vb}", mask_value),
        (f".\nQuestion: Do {a} and {b} have the same {prop}?\nAnswer:", False),
        (f" {gold}", False),
    ]


def _reason_doc(prop, a, va, b, vb, gold) -> Doc:
    return Doc(kind="reason",
               dense_segments=_reason_segs(prop, a, va, b, vb, gold, False),
               split_segments=_reason_segs(prop, a, va, b, vb, gold, True),
               meta={"prop": prop})


# pure reasoning: compute a named operation; the operation's DEFINITION is the
# "relevant fact" that can be provided in context (a closer starting point).
_OPS = [
    ("mod", "mod means the remainder after dividing the first number by the second.",
     lambda a, b: a % b),
    ("plus", "plus means the sum of the two numbers.", lambda a, b: a + b),
    ("minus", "minus means the absolute difference of the two numbers.",
     lambda a, b: abs(a - b)),
]
_OP_BY_NAME = {name: (definition, fn) for name, definition, fn in _OPS}


def _puremath_segs(question, definition, answer, mask_value: bool):
    return [
        ("Context:", False),
        (f" {definition}", mask_value),                # split masks the DEFINITION
        (f"\nQuestion: {question}\nAnswer:", False),
        (f" {answer}", False),                         # the computed answer (reasoning)
    ]


def _puremath_doc(question, definition, answer) -> Doc:
    return Doc(kind="puremath",
               dense_segments=_puremath_segs(question, definition, answer, False),
               split_segments=_puremath_segs(question, definition, answer, True),
               meta={})


def _puremath_pool(lo, hi):
    pool = []
    for name, definition, fn in _OPS:
        for a in range(lo, hi + 1):
            for b in range(lo, hi + 1):
                pool.append({"op": name, "a": a, "b": b, "definition": definition,
                             "question": f"What is {a} {name} {b}?",
                             "answer": str(fn(a, b))})
    return pool


def gen_puremath(n_train, n_eval, seed, lo=2, hi=12):
    """Disjoint train/eval over (op, a, b) combos. Eval combos are HELD OUT of
    training; training samples with replacement (repeats = exposures), so the
    model learns the operation and eval tests generalization to unseen operands."""
    rng = random.Random(f"{seed}:puremath")
    pool = _puremath_pool(lo, hi)
    rng.shuffle(pool)
    n_eval = min(n_eval, len(pool) // 4)
    eval_items = pool[:n_eval]
    train_pool = pool[n_eval:]
    train_items = [rng.choice(train_pool) for _ in range(n_train)] if train_pool else []
    return train_items, eval_items


def _toy_bed_docs(n, seed, sent_range) -> list[Doc]:
    rng = random.Random(f"{seed}:bed")
    docs = []
    for _ in range(n):
        text = " ".join(
            f"{rng.choice(_BED_SUBJECTS)} {rng.choice(_BED_VERBS)} {rng.choice(_BED_OBJECTS)}."
            for _ in range(rng.randint(*sent_range))
        )
        docs.append(Doc(kind="bed", dense_segments=[plain(text)],
                        split_segments=[plain(text)], meta={}))
    return docs


# ------------------------------------------------------------ comparison pairs

def make_pairs(facts, n, seed) -> list[tuple]:
    """Balanced same/different entity pairs sharing a relation.
    Each pair is (prop, subjA, objA, subjB, objB); gold = yes iff objA == objB."""
    rng = random.Random(f"{seed}:pairs")
    by_prop: dict[str, list[tuple[str, str]]] = {}
    for f in facts:
        by_prop.setdefault(f.prop, []).append((f.subj, f.obj))
    matching, different = [], []
    for prop, entries in by_prop.items():
        by_obj: dict[str, list[str]] = {}
        for s, o in entries:
            subs = by_obj.setdefault(o, [])
            if s not in subs:
                subs.append(s)
        for obj, subs in by_obj.items():
            for k in range(min(3, len(subs) - 1)):
                matching.append((prop, subs[k], obj, subs[k + 1], obj))
        objs = list(by_obj)
        if len(objs) >= 2:
            for _ in range(min(80, len(objs) * 3)):
                o1, o2 = rng.sample(objs, 2)
                different.append((prop, rng.choice(by_obj[o1]), o1,
                                  rng.choice(by_obj[o2]), o2))
    rng.shuffle(matching)
    rng.shuffle(different)
    half = min(n // 2, len(matching))
    pairs = matching[:half] + different[: n - half]
    rng.shuffle(pairs)
    return pairs[:n]


# ------------------------------------------------------------ build

def _encode_arm(tok, docs, arm) -> tuple[np.ndarray, np.ndarray]:
    id_bufs, mask_bufs = [], []
    for doc in docs:
        segs = doc.dense_segments if arm == "dense" else doc.split_segments
        ids, mask = tok.encode_segments(segs, add_eot=True)
        id_bufs.append(np.asarray(ids, dtype=np.uint16))
        mask_bufs.append(np.asarray(mask, dtype=np.uint8))
    ids = np.concatenate(id_bufs) if id_bufs else np.zeros(0, dtype=np.uint16)
    mask = np.concatenate(mask_bufs) if mask_bufs else np.zeros(0, dtype=np.uint8)
    return ids, mask


def _write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_poc_corpus(cfg: PoCBuildCfg, tok, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = realfact.load_realfacts(cfg.realfacts_path)
    if cfg.max_facts is not None and cfg.max_facts < len(facts):
        facts = sorted(facts, key=lambda f: (f.prop, f.subj))
        facts = random.Random(f"{cfg.split_seed}:cap").sample(facts, cfg.max_facts)
    seen, heldout = realfact.split_by_relation(facts, cfg.frac, seed=cfg.split_seed)

    # ---- training docs (all open-book; split masks the context value) ----
    factqa_docs = [
        _factqa_doc(f.subj, f.prop, f.obj, f.question)
        for _ in range(cfg.n_exposures) for f in seen
    ]
    train_pairs = make_pairs(seen, cfg.n_reason_train, cfg.render_seed)
    reason_docs = [_reason_doc(p, a, va, b, vb, "yes" if va == vb else "no")
                   for (p, a, va, b, vb) in train_pairs]
    puremath_train, puremath_eval_raw = gen_puremath(
        cfg.n_puremath_train, cfg.n_puremath_eval, cfg.render_seed)
    puremath_docs = [_puremath_doc(it["question"], it["definition"], it["answer"])
                     for it in puremath_train]
    bed_docs = _toy_bed_docs(cfg.n_bed_docs, cfg.render_seed, cfg.bed_sentences)

    docs = factqa_docs + reason_docs + puremath_docs + bed_docs
    random.Random(cfg.shuffle_seed).shuffle(docs)

    arm_reports = {}
    for arm in ("dense", "split"):
        ids, mask = _encode_arm(tok, docs, arm)
        (out_dir / arm).mkdir(parents=True, exist_ok=True)
        ids.tofile(out_dir / arm / "train.bin")
        mask.tofile(out_dir / arm / "train.mask.bin")
        arm_reports[arm] = {"n_tokens": int(ids.size),
                            "masked_token_frac": float((mask == 0).mean()) if ids.size else 0.0}

    # ---- eval sets ----
    rng = random.Random(f"{cfg.split_seed}:eval")

    def factqa_items(pool, label, k):
        picks = rng.sample(pool, min(k, len(pool)))
        return [{"qid": f"fq-{label}-{i}", "task": "factqa", "split": label,
                 "subj": f.subj, "prop": f.prop, "obj": f.obj,
                 "question": f.question, "possible_answers": list(f.possible_answers)}
                for i, f in enumerate(picks)]

    factqa_eval = (factqa_items(heldout, "heldout", cfg.n_factqa_heldout)
                   + factqa_items(seen, "seen", cfg.n_factqa_seen))

    reason_eval = []
    for i, (p, a, va, b, vb) in enumerate(make_pairs(heldout, cfg.n_reason_eval,
                                                      cfg.render_seed + 1)):
        reason_eval.append({"qid": f"rs-heldout-{i}", "task": "reason",
                            "split": "heldout", "prop": p,
                            "a": a, "va": va, "b": b, "vb": vb,
                            "question": f"Do {a} and {b} have the same {p}?",
                            "answer": "yes" if va == vb else "no"})

    puremath_eval = [{"qid": f"pm-{i}", "task": "puremath", "split": "heldout",
                      "op": it["op"], "question": it["question"],
                      "definition": it["definition"], "answer": it["answer"]}
                     for i, it in enumerate(puremath_eval_raw)]

    _write_jsonl(factqa_eval, out_dir / "eval" / "factqa.jsonl")
    _write_jsonl(reason_eval, out_dir / "eval" / "reason.jsonl")
    _write_jsonl(puremath_eval, out_dir / "eval" / "puremath.jsonl")

    report = {
        "cfg": cfg.to_dict(),
        "n_seen": len(seen), "n_heldout": len(heldout),
        "component_docs": {"factqa": len(factqa_docs), "reason": len(reason_docs),
                           "puremath": len(puremath_docs), "bed": len(bed_docs)},
        "arms": arm_reports,
        "eval_counts": {"factqa": len(factqa_eval), "reason": len(reason_eval),
                        "puremath": len(puremath_eval)},
    }
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report
