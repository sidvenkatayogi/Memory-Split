"""Multi-hop reasoning corpus over the Wikidata5M graph (user's v2 task).

Two question families, both grounded in the graph so gold is deterministic:

1. CHAIN (single answer): follow a path of FUNCTIONAL relations of depth k
   (k <= max_depth), e.g. "What is the country of the place of birth of X?".
   The answer is one entity.

2. AGGREGATION (over a branch): for a one-to-many relation R, count how many of
   X's R-neighbours satisfy a functional filter A == V, e.g. "How many cast
   members of X have country of citizenship France?". The answer is a NUMBER
   computed over retrieved values -- the "compute over values" that makes it
   reasoning, not retrieval-chaining, and the user's aggregate-over-branch way
   of handling one-to-many hops.

Rendering (retrieve-then-reason), matching the facts-in-context PoC:

    Context:   the atomic facts the chain needs (value spans MASKED for split)
    Question:  the natural-language question
    Reasoning: step-by-step trace; each looked-up value is MASKED for split
    Answer:    final answer -- loss ON for BOTH arms (a copy / a computed
               number), so split must reason from the (masked) retrieved values.

DENSE renders every span loss ON (memorises + reasons). SPLIT masks the
retrieved fact VALUES (never the plan, the queries, or the final answer).
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from corpusgen.records import Doc, Segment
from corpusgen.wikidata5m import Wikidata5M

# A chunk is (text, is_value): is_value spans are masked in the SPLIT rendering.
Chunk = tuple[str, bool]


def _render(chunks: list[Chunk]) -> tuple[list[Segment], list[Segment]]:
    dense = [(t, False) for t, _ in chunks]
    split = [(t, is_val) for t, is_val in chunks]
    return dense, split


# --------------------------------------------------------------- chain sampling

@dataclass
class Chain:
    start: str                       # Qid
    hops: list[tuple[str, str]]      # [(Pid, obj Qid), ...] in order

    def answer_qid(self) -> str:
        return self.hops[-1][1]


def sample_chain(g: Wikidata5M, funcs: set[str], depth: int,
                 rng: random.Random, tries: int = 40) -> Chain | None:
    """A path of `depth` functional hops where every hop resolves uniquely."""
    subjects = list(g.adj)
    if not subjects:
        return None
    for _ in range(tries):
        start = rng.choice(subjects)
        cur, hops, ok = start, [], True
        for _ in range(depth):
            uniq = [(r, o) for (r, o) in g.adj.get(cur, ())
                    if r in funcs and len(g.objects(cur, r)) == 1]
            if not uniq:
                ok = False
                break
            r, o = rng.choice(uniq)
            hops.append((r, o))
            cur = o
        if ok and len(hops) == depth:
            return Chain(start, hops)
    return None


# ---------------------------------------------------------- aggregation sampling

@dataclass
class Aggregation:
    x: str                           # subject Qid
    rel: str                         # one-to-many Pid
    objects: list[str]               # R-neighbours (Qids)
    attr: str                        # functional filter Pid
    attr_val: dict[str, str]         # obj Qid -> attr value Qid (only where present)
    value: str                       # filter value Qid V

    def count(self) -> int:
        return sum(1 for o in self.objects if self.attr_val.get(o) == self.value)


def sample_aggregation(g: Wikidata5M, funcs: set[str], rng: random.Random,
                       tries: int = 80, min_branch: int = 2) -> Aggregation | None:
    subjects = list(g.adj)
    for _ in range(tries):
        x = rng.choice(subjects)
        rels: dict[str, list[str]] = defaultdict(list)
        for r, o in g.adj.get(x, ()):
            rels[r].append(o)
        multi = [(r, os) for r, os in rels.items() if len(os) >= min_branch]
        if not multi:
            continue
        rel, objects = rng.choice(multi)
        # functional filter A present (uniquely) on >=2 of the branch objects
        acount: dict[str, int] = defaultdict(int)
        aval: dict[str, dict[str, str]] = defaultdict(dict)
        for o in objects:
            for r2, o2 in g.adj.get(o, ()):
                if r2 in funcs and len(g.objects(o, r2)) == 1:
                    acount[r2] += 1
                    aval[r2][o] = o2
        cand = [a for a in acount if acount[a] >= 2]
        if not cand:
            continue
        attr = rng.choice(cand)
        vals = dict(aval[attr])
        value = Counter(vals.values()).most_common(1)[0][0]
        return Aggregation(x, rel, list(objects), attr, vals, value)
    return None


# ----------------------------------------------------------------- rendering

def _chain_texts(g: Wikidata5M, ch: Chain) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (question, steps) where steps = [(rel_label, subj_label, obj_label)]."""
    nested = g.label(ch.start)
    for r, _ in ch.hops:
        nested = f"the {g.rel_label(r)} of {nested}"
    question = f"What is {nested}?"
    steps, prev = [], g.label(ch.start)
    for r, o in ch.hops:
        steps.append((g.rel_label(r), prev, g.label(o)))
        prev = g.label(o)
    return question, steps


def chain_doc(g: Wikidata5M, ch: Chain) -> Doc:
    question, steps = _chain_texts(g, ch)
    answer = steps[-1][2]
    chunks: list[Chunk] = [("Context:", False)]
    for rel, subj, obj in steps:
        chunks += [(f" The {rel} of {subj} is", False), (f" {obj}", True), (".", False)]
    chunks.append((f"\nQuestion: {question}\nReasoning:", False))
    for rel, subj, obj in steps:
        chunks += [(f" The {rel} of {subj} is", False), (f" {obj}", True), (".", False)]
    chunks += [(f" So the answer is {answer}.\nAnswer:", False), (f" {answer}", False)]
    dense, split = _render(chunks)
    return Doc(kind="mh_chain", dense_segments=dense, split_segments=split,
               meta={"depth": len(ch.hops), "start": ch.start})


def _agg_texts(g: Wikidata5M, a: Aggregation):
    xl, rl, al, vl = g.label(a.x), g.rel_label(a.rel), g.rel_label(a.attr), g.label(a.value)
    objs = [(o, g.label(o), g.label(a.attr_val[o])) for o in a.objects if o in a.attr_val]
    question = f"How many {rl} of {xl} have {al} {vl}?"
    return xl, rl, al, vl, objs, question


def agg_doc(g: Wikidata5M, a: Aggregation) -> Doc:
    xl, rl, al, vl, objs, question = _agg_texts(g, a)
    count = a.count()
    chunks: list[Chunk] = [("Context:", False)]
    for _, ol, av in objs:
        chunks += [(f" The {rl} of {xl} is", False), (f" {ol}", True),
                   (f". The {al} of {ol} is", False), (f" {av}", True), (".", False)]
    chunks.append((f"\nQuestion: {question}\nReasoning: The {rl} of {xl} are", False))
    chunks.append((" " + ", ".join(ol for _, ol, _ in objs), True))
    chunks.append((".", False))
    for _, ol, av in objs:
        chunks += [(f" The {al} of {ol} is", False), (f" {av}", True), (".", False)]
    chunks += [(f" Counting those with {al} {vl}: the answer is {count}.\nAnswer:", False),
               (f" {count}", False)]
    dense, split = _render(chunks)
    return Doc(kind="mh_agg", dense_segments=dense, split_segments=split,
               meta={"rel": a.rel, "attr": a.attr, "count": count})


# ------------------------------------------------------------ eval item helpers

def _chain_context(g: Wikidata5M, ch: Chain) -> str:
    _, steps = _chain_texts(g, ch)
    return " ".join(f"The {rel} of {subj} is {obj}." for rel, subj, obj in steps)


def _agg_context(g: Wikidata5M, a: Aggregation) -> str:
    xl, rl, al, vl, objs, _ = _agg_texts(g, a)
    parts = [f"The {rl} of {xl} is {ol}. The {al} of {ol} is {av}." for _, ol, av in objs]
    return " ".join(parts)


# --------------------------------------------------------------------- generate

def generate(g: Wikidata5M, n_train: int, n_eval: int, max_depth: int = 3,
             seed: int = 0, agg_frac: float = 0.3,
             funcs: set[str] | None = None) -> tuple[list[Doc], list[dict]]:
    """Return (train_docs, eval_items). Eval items carry question, gold answer,
    the atomic facts (for +context / oracle conditions), and a seen/heldout
    label (heldout = chain rooted at an inductive held-out entity)."""
    rng = random.Random(f"{seed}:mh")
    if funcs is None:
        funcs = g.functional_relations()
    train_docs: list[Doc] = []
    for _ in range(n_train):
        if rng.random() < agg_frac:
            a = sample_aggregation(g, funcs, rng)
            if a is not None:
                train_docs.append(agg_doc(g, a))
                continue
        ch = sample_chain(g, funcs, rng.randint(1, max_depth), rng)
        if ch is not None:
            train_docs.append(chain_doc(g, ch))

    eval_items: list[dict] = []
    seen_n = 0
    for i in range(n_eval * 3):  # oversample; keep first n_eval that build
        if len(eval_items) >= n_eval:
            break
        if rng.random() < agg_frac:
            a = sample_aggregation(g, funcs, rng)
            if a is None:
                continue
            _, _, _, _, _, q = _agg_texts(g, a)
            eval_items.append({
                "qid": f"mh-agg-{i}", "task": "mh_agg", "split": "seen",
                "question": q, "answer": str(a.count()), "possible_answers": [str(a.count())],
                "context": _agg_context(g, a), "depth": 2,
            })
        else:
            ch = sample_chain(g, funcs, rng.randint(1, max_depth), rng)
            if ch is None:
                continue
            q, steps = _chain_texts(g, ch)
            aq = ch.answer_qid()
            eval_items.append({
                "qid": f"mh-chain-{i}", "task": "mh_chain",
                "split": "heldout" if ch.start in g.heldout_entities else "seen",
                "question": q, "answer": steps[-1][2],
                "possible_answers": g.entity_aliases(aq),
                "context": _chain_context(g, ch), "depth": len(ch.hops),
            })
    return train_docs, eval_items
