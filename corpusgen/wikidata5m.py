"""Wikidata5M -> fact records + graph, for the Memory-Split v2 experiment.

This is the user's OWN v2 fact substrate (replaces PopQA). It reads the offline
Wikidata5M (KEPLER) files and produces:
  - an entity/relation ALIAS index (many surface forms per canonical id),
  - the triple GRAPH (adjacency: subj -> [(rel, obj)]), used for multi-hop,
  - stratified DOSE subsets (N entities) for the dose-response ladder,
  - single-hop FACT RECORDS in the pipeline's {subj,prop,obj,question,
    possible_answers} schema (drop-in for realfact.load_realfacts).

Wikidata5M triples are entity->entity (no literals), so every relation is
traversable for multi-hop. "Functional" relations (a single obj per
(subj, rel)) are detected empirically from the loaded graph.

Expected files (default names; override via Wikidata5MPaths):
  triples:   wikidata5m_transductive_train.txt   ("head<TAB>rel<TAB>tail" / line)
  inductive: wikidata5m_inductive_test.txt        (held-out entities, optional)
  aliases:   wikidata5m_alias/wikidata5m_entity.txt    ("Qid<TAB>name<TAB>alias...")
             wikidata5m_alias/wikidata5m_relation.txt   ("Pid<TAB>name<TAB>alias...")

Nothing here is trained on the store directly; the graph is the fact source the
corpus generators (single-hop + multi-hop) render from.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- io

@dataclass
class Wikidata5MPaths:
    root: str
    triples: str = "wikidata5m_transductive_train.txt"
    inductive: str = "wikidata5m_inductive_test.txt"
    entity_alias: str = "wikidata5m_alias/wikidata5m_entity.txt"
    relation_alias: str = "wikidata5m_alias/wikidata5m_relation.txt"

    def p(self, name: str) -> Path:
        return Path(self.root) / name


def _read_alias_file(path: Path) -> dict[str, list[str]]:
    """`id<TAB>name<TAB>alias2<TAB>...` -> {id: [name, alias2, ...]}.

    The first field after the id is treated as the canonical label; the rest are
    aliases (used for possible_answers and phrasing variety).
    """
    out: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            qid, names = parts[0], [p for p in parts[1:] if p]
            if names:
                out[qid] = names
    return out


def _read_entities(path: Path) -> set[str]:
    ents: set[str] = set()
    if not path.exists():
        return ents
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for tok in line.rstrip("\n").split("\t"):
                if tok.startswith("Q"):
                    ents.add(tok)
    return ents


# ------------------------------------------------------------------------ graph

@dataclass
class Wikidata5M:
    """Loaded graph + alias index. All ids are canonical Q/P strings."""

    entity_names: dict[str, list[str]]          # Qid -> [label, alias, ...]
    rel_names: dict[str, list[str]]             # Pid -> [label, alias, ...]
    adj: dict[str, list[tuple[str, str]]]       # subj Qid -> [(Pid, obj Qid), ...]
    heldout_entities: set[str] = field(default_factory=set)

    # -- construction -------------------------------------------------------

    @classmethod
    def load(cls, paths: Wikidata5MPaths, max_triples: int | None = None) -> "Wikidata5M":
        entity_names = _read_alias_file(paths.p(paths.entity_alias))
        rel_names = _read_alias_file(paths.p(paths.relation_alias))
        adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        n = 0
        with open(paths.p(paths.triples), encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                # keep only triples whose ids we can name (needed to render text)
                if h in entity_names and t in entity_names and r in rel_names:
                    adj[h].append((r, t))
                    n += 1
                    if max_triples is not None and n >= max_triples:
                        break
        heldout = _read_entities(paths.p(paths.inductive))
        return cls(entity_names, rel_names, dict(adj), heldout)

    # -- accessors ----------------------------------------------------------

    def label(self, qid: str) -> str:
        names = self.entity_names.get(qid)
        return names[0] if names else qid

    def rel_label(self, pid: str) -> str:
        names = self.rel_names.get(pid)
        return names[0] if names else pid

    def entity_aliases(self, qid: str) -> list[str]:
        return self.entity_names.get(qid, [qid])

    def n_entities(self) -> int:
        return len(self.adj)

    def n_triples(self) -> int:
        return sum(len(v) for v in self.adj.values())

    def objects(self, subj: str, rel: str) -> list[str]:
        return [o for (r, o) in self.adj.get(subj, ()) if r == rel]

    # -- relation typing ----------------------------------------------------

    def functional_relations(self, min_support: int = 20, thresh: float = 0.9) -> set[str]:
        """Pids that behave functionally: for >= `thresh` of subjects that have
        the relation, there is exactly ONE object. `min_support` guards against
        deciding on too few examples. Used to pick clean single-answer hops.
        """
        single = defaultdict(int)
        total = defaultdict(int)
        for subj, edges in self.adj.items():
            counts: dict[str, int] = defaultdict(int)
            for r, _ in edges:
                counts[r] += 1
            for r, c in counts.items():
                total[r] += 1
                if c == 1:
                    single[r] += 1
        out = set()
        for r, tot in total.items():
            if tot >= min_support and single[r] / tot >= thresh:
                out.add(r)
        return out

    # -- dose subsetting ----------------------------------------------------

    def dose_subset(self, n_entities: int, seed: int = 0,
                    keep_frac_heldout: float = 0.1) -> "Wikidata5M":
        """Return a subgraph induced by a sample of `n_entities` subjects, plus
        the objects they point to (so every kept edge stays resolvable).

        Sampling is stratified by the subject's *primary relation* so a small
        dose keeps the relation mix (not accidentally all one type). A slice of
        the inductive held-out entities is preserved for the generalization test.
        """
        rng = random.Random(f"{seed}:dose:{n_entities}")
        subjects = [s for s in self.adj if s not in self.heldout_entities]
        # stratify by the subject's first relation
        by_rel: dict[str, list[str]] = defaultdict(list)
        for s in subjects:
            by_rel[self.adj[s][0][0]].append(s)
        for lst in by_rel.values():
            rng.shuffle(lst)
        # proportional allocation across relation buckets
        picked: list[str] = []
        rels = sorted(by_rel)
        i = 0
        while len(picked) < min(n_entities, len(subjects)):
            bucket = by_rel[rels[i % len(rels)]]
            if bucket:
                picked.append(bucket.pop())
            i += 1
            if all(not by_rel[r] for r in rels):
                break
        picked_set = set(picked)

        sub_adj: dict[str, list[tuple[str, str]]] = {}
        objs_needed: set[str] = set()
        for s in picked_set:
            edges = self.adj[s]
            sub_adj[s] = list(edges)
            objs_needed.update(o for _, o in edges)
        # ensure object nodes exist (as leaves if not themselves sampled) so
        # multi-hop traversal can still resolve one more hop where present
        for o in objs_needed:
            if o not in sub_adj and o in self.adj:
                sub_adj.setdefault(o, list(self.adj[o]))

        keep_held = {s for s in picked_set if s in self.heldout_entities}
        return Wikidata5M(self.entity_names, self.rel_names, sub_adj, keep_held)

    # -- single-hop fact records (drop-in for realfact JSONL) ---------------

    def to_fact_records(self, functional: set[str] | None = None) -> list[dict]:
        """Emit {subj,prop,obj,question,possible_answers} rows, one per triple.

        `functional` (a set of Pids, e.g. from `functional_relations()`)
        restricts to single-clean-answer relations. Pass None to emit all.
        Non-functional relations still live in the graph for multi-hop
        aggregation questions; they are just noisy as *single-hop* QA gold.
        """
        rows: list[dict] = []
        for subj, edges in self.adj.items():
            s_name = self.label(subj)
            for rel, obj in edges:
                if functional is not None and rel not in functional:
                    continue
                r_name = self.rel_label(rel)
                rows.append({
                    "subj": s_name,
                    "prop": r_name,
                    "obj": self.label(obj),
                    "question": f"What is the {r_name} of {s_name}?",
                    "possible_answers": self.entity_aliases(obj),
                })
        return rows
