# Selective Recursive Graph Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate SRGM training and evaluation pipeline that tests whether selectively removing direct arbitrary-fact targets improves reusable relational reasoning under a matched Dense/Split comparison.

**Architecture:** Keep the existing MemorySplit pipeline intact and add focused SRGM modules beside it. One immutable typed graph provides exact atomic edge reads; one shared token stream is paired with condition-specific target-weight sidecars; a fixed six-step two-block recurrent model is identical across conditions; and equal-information graph evaluation carries the claim.

**Tech Stack:** Python 3.12, PyTorch >=2.6, NumPy >=1.26, tiktoken >=0.7, PyYAML >=6.0, pytest >=8.0, AWS EC2 `p5.48xlarge` with eight H100 80GB GPUs.

## Global Constraints

- Treat `docs/superpowers/specs/2026-07-21-selective-recursive-graph-memory-design.md` as authoritative.
- Preserve the legacy `GPT`, flat `Organizer`, corpus builder, and old evaluation commands until SRGM passes its end-to-end smoke test.
- Use vocabulary size 50,304 and context length 1,024; allocate graph tokens only from padded IDs 50,261–50,291.
- An atomic graph read returns exactly one row or `MISS`; it never expands a neighborhood, ranks candidates, performs message passing, composes a path, or returns an answer.
- Use four working-node slots and exactly six recurrent steps in every condition; `HALT` changes later reads to no-ops but never reduces realized compute.
- Keep graph bytes, model parameters, initial state dict, raw token IDs, record order, packing boundaries, optimizer settings, and recurrent supervision identical within each paired seed.
- Normalize every weighted next-token loss by the original number of raw target positions, not the number of active targets.
- Use the fixed 45% natural / 30% graph / 25% reasoning mixture and the fixed 20% / 30% / 50% hop curriculum.
- Use three paired training seeds for every claim-bearing condition.
- Do not begin protected runs until shared-schedule identity, mask coverage, scorer batch invariance, selector gates, task learnability, and Dense-above-chance recognition all pass.
- Give both claim-bearing twins the same fresh graph and memory-ON interface for the primary endpoint.
- Report the graph separately in rows and bytes; no per-fact trainable embedding is permitted.
- Run implementation in an isolated worktree created with `superpowers:using-git-worktrees`; do not mix generated corpora or checkpoints with legacy outputs.

---

## File Structure

### New core modules

- `corpusgen/graph_records.py` — immutable graph, tagged segment, action, trace, and selector-feature contracts.
- `organizer/graph_store.py` — exact functional atomic graph store and deterministic snapshot serialization.
- `corpusgen/graph_trace.py` — graph action/return token serialization.
- `corpusgen/srgm_worlds.py` — streaming graph-world, reasoning-oracle, and counterfactual-pair generation.
- `corpusgen/srgm_selector.py` — 225-parameter selector, hard-concrete training objective, and frozen artifact.
- `corpusgen/srgm_build.py` — one shared schedule/token stream plus Dense, Split, and random-target weight sidecars.
- `train/srgm_data.py` — weighted packed batches with deep-supervision masks.
- `train/srgm_loss.py` — all-token-normalized final and recurrent losses.
- `train/srgm_model.py` — exact 30M/160M/360M/1B recurrent model presets and cache.
- `train/provenance.py` — content hashes and strict checkpoint/corpus compatibility.
- `train/srgm_trainer.py` — SRGM optimizer, logging, checkpoint, and resume loop.
- `evals/srgm_generate.py` — model-generated atomic graph actions and memory ON/OFF decoding.
- `evals/srgm_metrics.py` — pair consistency, path metrics, recognition, language, and guardrails.
- `evals/srgm_controls.py` — shuffled/relevant/irrelevant/gold-path/isomorphism controls.
- `evals/srgm_stats.py` — seed-level interaction, confidence intervals, and frozen verdict.
- `evals/srgm_figures.py` — dose-response and recursive-depth figures.

### New commands and operations

- `scripts/calibrate_srgm_selector.py`
- `scripts/build_srgm_corpus.py`
- `scripts/run_srgm_train.py`
- `scripts/run_srgm_evals.py`
- `scripts/analyze_srgm.py`
- `scripts/make_srgm_manifest.py`
- `scripts/check_srgm_throughput.py`
- `scripts/srgm_smoke_test.py`
- `cluster/aws/run_srgm_manifest.py`
- `cluster/aws/SRGM.md`

### Existing files modified

- `train/tokenizer.py:27-72` — reserve graph-control and relation IDs while retaining old token IDs.
- `README.md` — add only an SRGM spec/plan/runbook pointer after the smoke test passes.

### New tests

- `tests/test_graph_store.py`
- `tests/test_graph_trace.py`
- `tests/test_srgm_worlds.py`
- `tests/test_srgm_selector.py`
- `tests/test_srgm_build.py`
- `tests/test_srgm_data.py`
- `tests/test_srgm_loss.py`
- `tests/test_srgm_model.py`
- `tests/test_srgm_provenance.py`
- `tests/test_srgm_trainer.py`
- `tests/test_srgm_generate.py`
- `tests/test_srgm_metrics.py`
- `tests/test_srgm_controls.py`
- `tests/test_srgm_stats.py`
- `tests/test_srgm_manifest.py`
- `tests/test_srgm_smoke.py`

---

### Task 1: Typed Graph Contracts and Atomic Store

**Files:**
- Create: `corpusgen/graph_records.py`
- Create: `organizer/graph_store.py`
- Create: `tests/test_graph_store.py`

**Interfaces:**
- Consumes: only Python standard-library dataclasses, JSON, hashing, and paths.
- Produces: `GraphAddress`, `GraphRow`, `TaggedSegment`, `ScheduleEntry`,
  `RenderedRecord`, `SelectorFeatures`, `GraphAction`, and
  `AtomicGraphStore`, used by every later task.

- [ ] **Step 1: Write the failing atomic-store tests**

```python
# tests/test_graph_store.py
import pytest

from corpusgen.graph_records import GraphAddress, GraphRow
from organizer.graph_store import AtomicGraphStore


def row(source=1, relation="r0", target="2"):
    return GraphRow(
        source_id=source,
        relation_id=relation,
        direction="out",
        target_kind="entity",
        target=target,
        qualifiers=(("compose", "1"),),
        provenance_id="world-0",
    )


def test_atomic_lookup_returns_one_exact_row():
    store = AtomicGraphStore([row()])
    assert store.lookup(GraphAddress(1, "r0", "out")) == row()
    assert store.hits == 1 and store.misses == 0


def test_duplicate_functional_address_is_rejected():
    store = AtomicGraphStore([row()])
    with pytest.raises(ValueError, match="duplicate graph address"):
        store.add(row(target="3"))


def test_missing_address_returns_none():
    store = AtomicGraphStore([row()])
    assert store.lookup(GraphAddress(2, "r0", "out")) is None
    assert store.hits == 0 and store.misses == 1


def test_snapshot_round_trip_is_sorted_and_hash_stable(tmp_path):
    first = row(source=2, relation="r1", target="5")
    second = row(source=1, relation="r0", target="2")
    store = AtomicGraphStore([first, second])
    path = tmp_path / "graph.jsonl"
    store.save(path)
    loaded = AtomicGraphStore.load(path)
    assert loaded.rows() == (second, first)
    assert loaded.snapshot_sha256() == store.snapshot_sha256()
```

- [ ] **Step 2: Run the store tests and confirm the missing-module failure**

Run: `.venv/bin/python -m pytest tests/test_graph_store.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'corpusgen.graph_records'`.

- [ ] **Step 3: Add the immutable graph contracts**

```python
# corpusgen/graph_records.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["out", "in"]
TargetKind = Literal["entity", "literal"]
SegmentRole = Literal[
    "plain", "payload", "rule", "action", "provisional_answer", "final_answer"
]


@dataclass(frozen=True, order=True)
class GraphAddress:
    source_id: int
    relation_id: str
    direction: Direction

    def __post_init__(self) -> None:
        if self.source_id < 0:
            raise ValueError("source_id must be non-negative")
        if not self.relation_id:
            raise ValueError("relation_id must be non-empty")
        if self.direction not in ("out", "in"):
            raise ValueError(f"invalid direction: {self.direction}")


@dataclass(frozen=True, order=True)
class GraphRow:
    source_id: int
    relation_id: str
    direction: Direction
    target_kind: TargetKind
    target: str
    qualifiers: tuple[tuple[str, str], ...] = ()
    provenance_id: str = ""

    def __post_init__(self) -> None:
        GraphAddress(self.source_id, self.relation_id, self.direction)
        if self.target_kind not in ("entity", "literal"):
            raise ValueError(f"invalid target_kind: {self.target_kind}")
        if not self.target:
            raise ValueError("target must be non-empty")

    @property
    def address(self) -> GraphAddress:
        return GraphAddress(self.source_id, self.relation_id, self.direction)

    def as_json(self) -> dict:
        return {
            "source_id": self.source_id,
            "relation_id": self.relation_id,
            "direction": self.direction,
            "target_kind": self.target_kind,
            "target": self.target,
            "qualifiers": [list(q) for q in self.qualifiers],
            "provenance_id": self.provenance_id,
        }

    @classmethod
    def from_json(cls, value: dict) -> "GraphRow":
        return cls(
            source_id=int(value["source_id"]),
            relation_id=str(value["relation_id"]),
            direction=value["direction"],
            target_kind=value["target_kind"],
            target=str(value["target"]),
            qualifiers=tuple((str(k), str(v)) for k, v in value["qualifiers"]),
            provenance_id=str(value["provenance_id"]),
        )


@dataclass(frozen=True)
class TaggedSegment:
    text: str
    role: SegmentRole
    fact_id: str | None = None

    def __post_init__(self) -> None:
        if self.role == "payload" and self.fact_id is None:
            raise ValueError("payload segments require fact_id")
        if self.role != "payload" and self.fact_id is not None:
            raise ValueError("only payload segments may carry fact_id")


@dataclass(frozen=True)
class ScheduleEntry:
    component: str
    record_id: str
    exposure: int
    curriculum_band: int


@dataclass(frozen=True)
class RenderedRecord:
    segments: tuple[TaggedSegment, ...]
    schedule: ScheduleEntry


@dataclass(frozen=True)
class SelectorFeatures:
    log_exposure: float
    payload_entropy: float
    payload_tokens: float
    expected_queries: float
    path_centrality: float

    def vector(self) -> tuple[float, float, float, float, float]:
        return (
            self.log_exposure,
            self.payload_entropy,
            self.payload_tokens,
            self.expected_queries,
            self.path_centrality,
        )


@dataclass(frozen=True)
class GraphAction:
    source_slot: int
    relation_id: str
    direction: Direction
    read: bool
    halt: bool

    def __post_init__(self) -> None:
        if self.source_slot not in range(4):
            raise ValueError("source_slot must be in [0, 3]")
        if self.halt and self.read:
            raise ValueError("HALT cannot also read")
```

- [ ] **Step 4: Implement deterministic functional storage**

```python
# organizer/graph_store.py
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from corpusgen.graph_records import GraphAddress, GraphRow


class AtomicGraphStore:
    def __init__(self, rows: Iterable[GraphRow] = ()) -> None:
        self._rows: dict[GraphAddress, GraphRow] = {}
        self.hits = 0
        self.misses = 0
        for row in rows:
            self.add(row)

    def add(self, row: GraphRow) -> None:
        if row.address in self._rows:
            raise ValueError(f"duplicate graph address: {row.address}")
        self._rows[row.address] = row

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        row = self._rows.get(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def rows(self) -> tuple[GraphRow, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def canonical_bytes(self) -> bytes:
        lines = [
            json.dumps(row.as_json(), sort_keys=True, separators=(",", ":"))
            for row in self.rows()
        ]
        return ("\n".join(lines) + ("\n" if lines else "")).encode()

    def snapshot_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.canonical_bytes())

    @classmethod
    def load(cls, path: str | Path) -> "AtomicGraphStore":
        rows = [
            GraphRow.from_json(json.loads(line))
            for line in Path(path).read_text().splitlines()
            if line
        ]
        return cls(rows)

    def __len__(self) -> int:
        return len(self._rows)
```

- [ ] **Step 5: Run the focused tests**

Run: `.venv/bin/python -m pytest tests/test_graph_store.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit the graph contracts**

```bash
git add corpusgen/graph_records.py organizer/graph_store.py tests/test_graph_store.py
git commit -m "feat: add atomic relational graph store"
```

---

### Task 2: Graph-Control Tokens and Trace Serialization

**Files:**
- Modify: `train/tokenizer.py:27-72`
- Create: `corpusgen/graph_trace.py`
- Create: `tests/test_graph_trace.py`
- Modify: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: `GraphAction`, `GraphRow`, `TaggedSegment` from Task 1.
- Produces: fixed token IDs, `encode_tagged_segments()`, `serialize_action()`, and `serialize_return()` for the builder and decoder.

- [ ] **Step 1: Write failing token and trace tests**

```python
# tests/test_graph_trace.py
from corpusgen.graph_records import GraphAction, GraphRow
from corpusgen.graph_trace import serialize_action, serialize_return
from train.tokenizer import get_tok


def test_graph_action_is_fixed_width_and_atomic():
    tok = get_tok()
    action = GraphAction(2, "r3", "out", read=True, halt=False)
    ids = serialize_action(action, tok)
    assert ids == [
        tok.GRAPH_START,
        tok.SLOTS[2],
        tok.RELATIONS["r3"],
        tok.DIR_OUT,
        tok.GRAPH_READ,
        tok.GRAPH_END,
    ]


def test_halt_action_has_no_read():
    tok = get_tok()
    action = GraphAction(0, "r0", "out", read=False, halt=True)
    ids = serialize_action(action, tok)
    assert tok.GRAPH_HALT in ids and tok.GRAPH_READ not in ids


def test_return_serialization_marks_payload_fact():
    tok = get_tok()
    row = GraphRow(
        1, "r2", "out", "entity", "9", (("compose", "3"),), "world-1"
    )
    segments = serialize_return(row, "fact-1")
    ids, roles, fact_ids = tok.encode_tagged_segments(segments)
    assert ids[0] == tok.GRAPH_RETURN and ids[-1] == tok.GRAPH_END
    assert "payload" in roles
    assert "fact-1" in fact_ids
```

Add to `tests/test_tokenizer.py`:

```python
def test_graph_special_token_ids_are_reserved_and_atomic():
    tok = get_tok()
    assert tok.GRAPH_START == 50261
    assert tok.GRAPH_MISS == 50275
    assert tok.RELATIONS["r0"] == 50276
    assert tok.RELATIONS["r15"] == 50291
    for text, token_id in tok.graph_special_tokens.items():
        assert tok.encode(text) == [token_id]
```

- [ ] **Step 2: Run the tests and confirm missing graph-token attributes**

Run: `.venv/bin/python -m pytest tests/test_graph_trace.py tests/test_tokenizer.py -q`

Expected: FAIL with missing `corpusgen.graph_trace` or `Tok.GRAPH_START`.

- [ ] **Step 3: Reserve graph IDs without changing the padded vocabulary**

Replace the special-token declaration in `train/tokenizer.py` with:

```python
DB_SPECIAL_TOKENS = {
    "<|db_start|>": 50257,
    "<|db_retrieve|>": 50258,
    "<|db_end|>": 50259,
    "<|eot|>": 50260,
}
GRAPH_SPECIAL_TOKENS = {
    "<|graph_start|>": 50261,
    "<|graph_read|>": 50262,
    "<|graph_return|>": 50263,
    "<|graph_end|>": 50264,
    "<|graph_halt|>": 50265,
    "<|graph_noop|>": 50266,
    "<|slot_0|>": 50267,
    "<|slot_1|>": 50268,
    "<|slot_2|>": 50269,
    "<|slot_3|>": 50270,
    "<|dir_out|>": 50271,
    "<|dir_in|>": 50272,
    "<|graph_step|>": 50273,
    "<|answer_state|>": 50274,
    "<|graph_miss|>": 50275,
    **{f"<|rel_{i}|>": 50276 + i for i in range(16)},
}
SPECIAL_TOKENS = {**DB_SPECIAL_TOKENS, **GRAPH_SPECIAL_TOKENS}
VOCAB_SIZE = 50304
```

Add these assignments to `Tok.__init__`:

```python
self.graph_special_tokens = dict(GRAPH_SPECIAL_TOKENS)
self.GRAPH_START = GRAPH_SPECIAL_TOKENS["<|graph_start|>"]
self.GRAPH_READ = GRAPH_SPECIAL_TOKENS["<|graph_read|>"]
self.GRAPH_RETURN = GRAPH_SPECIAL_TOKENS["<|graph_return|>"]
self.GRAPH_END = GRAPH_SPECIAL_TOKENS["<|graph_end|>"]
self.GRAPH_HALT = GRAPH_SPECIAL_TOKENS["<|graph_halt|>"]
self.GRAPH_NOOP = GRAPH_SPECIAL_TOKENS["<|graph_noop|>"]
self.GRAPH_STEP = GRAPH_SPECIAL_TOKENS["<|graph_step|>"]
self.ANSWER_STATE = GRAPH_SPECIAL_TOKENS["<|answer_state|>"]
self.GRAPH_MISS = GRAPH_SPECIAL_TOKENS["<|graph_miss|>"]
self.DIR_OUT = GRAPH_SPECIAL_TOKENS["<|dir_out|>"]
self.DIR_IN = GRAPH_SPECIAL_TOKENS["<|dir_in|>"]
self.SLOTS = tuple(GRAPH_SPECIAL_TOKENS[f"<|slot_{i}|>"] for i in range(4))
self.RELATIONS = {
    f"r{i}": GRAPH_SPECIAL_TOKENS[f"<|rel_{i}|>"] for i in range(16)
}
```

Add this method to `Tok`:

```python
def encode_tagged_segments(self, segments):
    ids: list[int] = []
    roles: list[str] = []
    fact_ids: list[str | None] = []
    for segment in segments:
        segment_ids = self._enc.encode(segment.text, allowed_special="all")
        ids.extend(segment_ids)
        roles.extend([segment.role] * len(segment_ids))
        fact_ids.extend([segment.fact_id] * len(segment_ids))
    return ids, roles, fact_ids
```

- [ ] **Step 4: Implement fixed-width action and tagged-return serialization**

```python
# corpusgen/graph_trace.py
from __future__ import annotations

import json

from corpusgen.graph_records import GraphAction, GraphRow, TaggedSegment


def serialize_action(action: GraphAction, tok) -> list[int]:
    terminal = (
        tok.GRAPH_HALT
        if action.halt
        else tok.GRAPH_READ
        if action.read
        else tok.GRAPH_NOOP
    )
    direction = tok.DIR_OUT if action.direction == "out" else tok.DIR_IN
    return [
        tok.GRAPH_START,
        tok.SLOTS[action.source_slot],
        tok.RELATIONS[action.relation_id],
        direction,
        terminal,
        tok.GRAPH_END,
    ]


def serialize_return(row: GraphRow | None, fact_id: str | None):
    if row is None:
        return [
            TaggedSegment("<|graph_return|>", "action"),
            TaggedSegment("<|graph_miss|>", "action"),
            TaggedSegment("<|graph_end|>", "action"),
        ]
    payload = json.dumps(
        {
            "target_kind": row.target_kind,
            "target": row.target,
            "qualifiers": list(row.qualifiers),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return [
        TaggedSegment("<|graph_return|>", "action"),
        TaggedSegment(payload, "payload", fact_id=fact_id),
        TaggedSegment("<|graph_end|>", "action"),
    ]
```

- [ ] **Step 5: Run focused and legacy tokenizer tests**

Run: `.venv/bin/python -m pytest tests/test_graph_trace.py tests/test_tokenizer.py -q`

Expected: all tests PASS and the legacy IDs 50,257–50,260 remain unchanged.

- [ ] **Step 6: Commit graph tokenization**

```bash
git add train/tokenizer.py corpusgen/graph_trace.py tests/test_graph_trace.py tests/test_tokenizer.py
git commit -m "feat: add graph action token protocol"
```

---

### Task 3: Streaming Graph Worlds and Counterfactual Oracles

**Files:**
- Create: `corpusgen/srgm_worlds.py`
- Create: `tests/test_srgm_worlds.py`

**Interfaces:**
- Consumes: graph contracts and fixed relation vocabulary from Tasks 1–2.
- Produces: `WorldConfig`, `GraphFact`, `GraphWorld`, `CounterfactualPair`,
  `iter_worlds()`, `generate_eval_pairs()`, and streaming renderers.

- [ ] **Step 1: Write failing world and oracle tests**

```python
# tests/test_srgm_worlds.py
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
)


def test_world_has_six_functional_facts_per_entity():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    assert len(world.facts) == 64 * 6
    addresses = [fact.row.address for fact in world.facts]
    assert len(addresses) == len(set(addresses))


def test_world_generation_is_seed_deterministic():
    cfg = WorldConfig(n_entities=64, seed=11)
    assert generate_world(3, cfg) == generate_world(3, cfg)


def test_all_counterfactual_pairs_flip_and_keep_sizes():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    pairs = generate_eval_pairs(world, n_pairs_per_task=20, seed=17)
    assert {pair.task for pair in pairs} == {
        "path_composition", "date_ordering", "balanced_equality"
    }
    for pair in pairs:
        assert pair.original.answer != pair.counterfactual.answer
        assert pair.original.meta["pair_id"] == pair.counterfactual.meta["pair_id"]
        assert pair.original.meta["graph_rows"] == pair.counterfactual.meta["graph_rows"]


def test_protected_answers_are_derived_not_payload_copies():
    world = generate_world(0, WorldConfig(n_entities=64, seed=7))
    payloads = {fact.row.target for fact in world.facts}
    pairs = generate_eval_pairs(world, n_pairs_per_task=10, seed=19)
    for pair in pairs:
        assert pair.original.answer not in payloads
        assert pair.counterfactual.answer not in payloads
```

- [ ] **Step 2: Run tests and confirm the missing generator**

Run: `.venv/bin/python -m pytest tests/test_srgm_worlds.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'corpusgen.srgm_worlds'`.

- [ ] **Step 3: Implement deterministic six-fact worlds**

```python
# corpusgen/srgm_worlds.py
from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass

from corpusgen.graph_records import (
    GraphAction,
    GraphAddress,
    GraphRow,
    RenderedRecord,
    ScheduleEntry,
    SelectorFeatures,
    TaggedSegment,
)
from corpusgen.graph_trace import serialize_action, serialize_return
from corpusgen.records import QAItem


@dataclass(frozen=True)
class WorldConfig:
    n_entities: int = 64
    seed: int = 0
    relation_count: int = 4
    entity_id_offset: int = 0


@dataclass(frozen=True)
class GraphFact:
    fact_id: str
    row: GraphRow
    features: SelectorFeatures
    audit_class: str


@dataclass(frozen=True)
class GraphWorld:
    world_id: int
    entity_names: tuple[str, ...]
    facts: tuple[GraphFact, ...]


@dataclass(frozen=True)
class CounterfactualPair:
    task: str
    original: QAItem
    counterfactual: QAItem
    changed_row: GraphRow


def _features(exposure: int, entropy: float, queries: float, centrality: float):
    return SelectorFeatures(
        log_exposure=math.log1p(exposure),
        payload_entropy=entropy,
        payload_tokens=1.0,
        expected_queries=queries,
        path_centrality=centrality,
    )


def generate_world(world_id: int, cfg: WorldConfig) -> GraphWorld:
    if cfg.n_entities < 16:
        raise ValueError("n_entities must be at least 16")
    rng = random.Random((cfg.seed << 32) ^ world_id)
    names = [f"entity-{world_id}-{i}-{rng.getrandbits(32):08x}" for i in range(cfg.n_entities)]
    facts: list[GraphFact] = []
    for relation_index in range(cfg.relation_count):
        targets = list(range(cfg.n_entities))
        rng.shuffle(targets)
        for source_id, target_id in enumerate(targets):
            global_source = cfg.entity_id_offset + source_id
            global_target = cfg.entity_id_offset + target_id
            compose = rng.randrange(4)
            row = GraphRow(
                source_id=global_source,
                relation_id=f"r{relation_index}",
                direction="out",
                target_kind="entity",
                target=str(global_target),
                qualifiers=(("compose", str(compose)),),
                provenance_id=f"world-{world_id}",
            )
            centrality = 1.0 if source_id < max(1, cfg.n_entities // 4) else 0.1
            audit = "central" if centrality == 1.0 else "peripheral"
            facts.append(
                GraphFact(
                    fact_id=str(global_source * 6 + relation_index),
                    row=row,
                    features=_features(4, math.log2(cfg.n_entities), 1.0, centrality),
                    audit_class=audit,
                )
            )
    for source_id in range(cfg.n_entities):
        global_source = cfg.entity_id_offset + source_id
        for relation_id, value, entropy in (
            ("r4", f"{1930 + rng.randrange(76):04d}-{1 + rng.randrange(12):02d}-{1 + rng.randrange(28):02d}", 14.7),
            ("r5", f"category-{rng.randrange(64)}", 6.0),
        ):
            row = GraphRow(
                source_id=global_source,
                relation_id=relation_id,
                direction="out",
                target_kind="literal",
                target=value,
                provenance_id=f"world-{world_id}",
            )
            facts.append(
                GraphFact(
                    fact_id=str(
                        global_source * 6 + (4 if relation_id == "r4" else 5)
                    ),
                    row=row,
                    features=_features(2, entropy, 0.25, 0.05),
                    audit_class="peripheral",
                )
            )
    return GraphWorld(world_id, tuple(names), tuple(facts))


def iter_worlds(n_entities, world_size, seed, world_id_offset=0):
    for ordinal, start in enumerate(range(0, n_entities, world_size)):
        size = min(world_size, n_entities - start)
        yield generate_world(
            world_id_offset + ordinal,
            WorldConfig(
                n_entities=size,
                seed=seed,
                entity_id_offset=start,
            ),
        )
```

- [ ] **Step 4: Add the three deterministic reasoning oracles**

Add to `corpusgen/srgm_worlds.py`:

```python
def _row_map(world: GraphWorld):
    return {fact.row.address: fact for fact in world.facts}


def _follow(world: GraphWorld, start: int, relations: tuple[str, ...]):
    rows = _row_map(world)
    current = start
    used = []
    compose = 0
    for relation in relations:
        fact = rows[next(a for a in rows if a.source_id == current and a.relation_id == relation)]
        used.append(fact)
        compose = (compose + int(dict(fact.row.qualifiers)["compose"])) % 4
        current = int(fact.row.target)
    return current, compose, tuple(used)


def _item(
    qid, task, prompt, answer, pair_id, rows, variant,
    entity_slots, gold_facts, answer_choices, changed_row=None
):
    return QAItem(
        qid=qid,
        task=task,
        prompt=prompt,
        answer=answer,
        meta={
            "pair_id": pair_id,
            "template": task,
            "graph_rows": rows,
            "variant": variant,
            "changed_row": None if changed_row is None else changed_row.as_json(),
            "entity_slots": list(entity_slots),
            "gold_addresses": [
                [
                    fact.row.address.source_id,
                    fact.row.address.relation_id,
                    fact.row.address.direction,
                ]
                for fact in gold_facts
            ],
            "gold_fact_ids": [fact.fact_id for fact in gold_facts],
            "answer_choices": list(answer_choices),
        },
    )


def _replace(row, *, target=None, compose=None):
    qualifiers = row.qualifiers
    if compose is not None:
        qualifiers = tuple(
            (key, str(compose) if key == "compose" else value)
            for key, value in qualifiers
        )
    return GraphRow(
        row.source_id, row.relation_id, row.direction, row.target_kind,
        row.target if target is None else target, qualifiers, row.provenance_id
    )


def generate_eval_pairs(
    world: GraphWorld, n_pairs_per_task: int, seed: int
) -> list[CounterfactualPair]:
    rng = random.Random((seed << 32) ^ world.world_id)
    rows = _row_map(world)
    pairs: list[CounterfactualPair] = []
    for task in ("path_composition", "date_ordering", "balanced_equality"):
        for index in range(n_pairs_per_task):
            pair_id = f"{world.world_id}-{task}-{index}"
            entity_ids = range(
                min(fact.row.source_id for fact in world.facts),
                min(fact.row.source_id for fact in world.facts) + len(world.entity_names),
            )
            a, b = rng.sample(list(entity_ids), 2)
            if task == "path_composition":
                relations = tuple(f"r{rng.randrange(4)}" for _ in range(rng.randint(2, 4)))
                _, compose, used = _follow(world, a, relations)
                answer = f"r{compose}"
                first = used[0].row
                old_code = int(dict(first.qualifiers)["compose"])
                changed = _replace(first, compose=(old_code + 1) % 4)
                flipped = f"r{(compose + 1) % 4}"
                gold_facts = used
                entity_slots = (a, None, None, None)
                answer_choices = tuple(f"r{i}" for i in range(4))
                prompt = f"Start at slot 0 and follow {' '.join(relations)}. Return the composed relation."
            elif task == "date_ordering":
                fact_a = rows[next(x for x in rows if x.source_id == a and x.relation_id == "r4")]
                fact_b = rows[next(x for x in rows if x.source_id == b and x.relation_id == "r4")]
                date_a, date_b = fact_a.row.target, fact_b.row.target
                answer = "<|slot_0|>" if date_a < date_b else "<|slot_1|>"
                flipped = "<|slot_1|>" if answer == "<|slot_0|>" else "<|slot_0|>"
                changed = _replace(
                    fact_a.row,
                    target="2099-12-31" if answer == "<|slot_0|>" else "1900-01-01",
                )
                gold_facts = (fact_a, fact_b)
                entity_slots = (a, b, None, None)
                answer_choices = ("<|slot_0|>", "<|slot_1|>")
                prompt = "Read the dates reached from slots 0 and 1. Return the earlier slot."
            else:
                fact_a = rows[next(x for x in rows if x.source_id == a and x.relation_id == "r5")]
                fact_b = rows[next(x for x in rows if x.source_id == b and x.relation_id == "r5")]
                cat_a, cat_b = fact_a.row.target, fact_b.row.target
                answer = "yes" if cat_a == cat_b else "no"
                flipped = "no" if answer == "yes" else "yes"
                replacement = (
                    f"category-{(int(cat_b.split('-')[1]) + 1) % 64}"
                    if answer == "yes" else cat_b
                )
                changed = _replace(fact_a.row, target=replacement)
                gold_facts = (fact_a, fact_b)
                entity_slots = (a, b, None, None)
                answer_choices = ("yes", "no")
                prompt = "Read the categories reached from slots 0 and 1. Are they equal?"
            first_id = min(fact.row.source_id for fact in world.facts)
            prompt = (
                f"Slot 0 refers to {world.entity_names[a - first_id]}. "
                f"Slot 1 refers to {world.entity_names[b - first_id]}. "
                + prompt
            )
            original = _item(
                f"{pair_id}-o", task, prompt, answer, pair_id,
                len(world.facts), "original", entity_slots, gold_facts,
                answer_choices
            )
            counterfactual = _item(
                f"{pair_id}-c", task, prompt, flipped, pair_id,
                len(world.facts), "counterfactual", entity_slots, gold_facts,
                answer_choices, changed
            )
            pairs.append(CounterfactualPair(task, original, counterfactual, changed))
    return pairs
```

- [ ] **Step 5: Add streaming bed, graph, and curriculum reasoning records**

Add to `corpusgen/srgm_worlds.py`:

```python
def iter_bed_records(bed_iter):
    for index, text in enumerate(bed_iter):
        yield RenderedRecord(
            (TaggedSegment(text, "plain"),),
            ScheduleEntry("bed", f"bed-{index}", index, 0),
        )


def iter_graph_records(tok, worlds_factory):
    exposure = 0
    rule_text = (
        "Composition adds retrieved compose codes modulo four. "
        "Inverse traversal reverses edge direction. Equality is symmetric. "
        "Earlier dates have smaller ISO-8601 strings."
    )
    while True:
        for world in worlds_factory():
            peripheral = [fact for fact in world.facts if fact.audit_class == "peripheral"]
            central = [fact for fact in world.facts if fact.audit_class == "central"]
            p_index = c_index = 0
            while p_index < len(peripheral) or c_index < len(central):
                batch = []
                for _ in range(7):
                    batch.append(peripheral[p_index % len(peripheral)])
                    p_index += 1
                for _ in range(2):
                    batch.append(central[c_index % len(central)])
                    c_index += 1
                for fact in batch:
                    payload = json.dumps(
                        {
                            "target_kind": fact.row.target_kind,
                            "target": fact.row.target,
                            "qualifiers": list(fact.row.qualifiers),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    neutral = " the" * len(tok.encode(payload))
                    yield RenderedRecord(
                        (
                            TaggedSegment(
                                f"Source {fact.row.source_id} relation "
                                f"{fact.row.relation_id} returns ",
                                "plain",
                            ),
                            TaggedSegment(payload, "payload", fact.fact_id),
                            TaggedSegment(neutral, "plain"),
                        ),
                        ScheduleEntry("graph", fact.fact_id, exposure, 0),
                    )
                    exposure += 1
                yield RenderedRecord(
                    (TaggedSegment(rule_text, "rule"),),
                    ScheduleEntry("graph", f"rule-{world.world_id}", exposure, 0),
                )
                exposure += 1


def _answer_segments(answer):
    return (
        TaggedSegment("<|answer_state|>", "action"),
        TaggedSegment(answer, "provisional_answer"),
    )


def _return_segments_with_control(tok, row, fact_id):
    segments = list(serialize_return(row, fact_id))
    payload = next(
        (segment for segment in segments if segment.role == "payload"),
        None,
    )
    if payload is not None:
        segments.append(TaggedSegment(" the" * len(tok.encode(payload.text)), "plain"))
    return segments


def iter_reasoning_records(tok, worlds_factory, seed, max_hops):
    rng = random.Random(seed)
    exposure = 0
    while True:
        for world in worlds_factory():
            fact_map = {fact.fact_id: fact for fact in world.facts}
            pairs = generate_eval_pairs(world, 8, rng.randrange(1 << 30))
            for pair in pairs:
                item = pair.original
                addresses = [
                    GraphAddress(int(source), relation, direction)
                    for source, relation, direction in item.meta["gold_addresses"]
                ]
                fact_ids = list(item.meta["gold_fact_ids"])
                if item.task == "path_composition" and len(addresses) > max_hops:
                    addresses = addresses[:max_hops]
                    fact_ids = fact_ids[:max_hops]
                    compose = sum(
                        int(dict(fact_map[fact_id].row.qualifiers)["compose"])
                        for fact_id in fact_ids
                    ) % 4
                    answer = f"r{compose}"
                elif len(addresses) > max_hops:
                    continue
                else:
                    answer = item.answer
                segments = [TaggedSegment(item.prompt, "plain")]
                for step in range(6):
                    if step < len(addresses):
                        address = addresses[step]
                        slot = 0 if item.task == "path_composition" else step
                        action = GraphAction(
                            slot, address.relation_id, address.direction, True, False
                        )
                        action_text = tok.decode(serialize_action(action, tok))
                        segments.append(TaggedSegment(action_text, "action"))
                        segments.extend(
                            _return_segments_with_control(
                                tok,
                                fact_map[fact_ids[step]].row, fact_ids[step]
                            )
                        )
                    else:
                        halt = GraphAction(0, "r0", "out", False, True)
                        segments.append(
                            TaggedSegment(
                                tok.decode(serialize_action(halt, tok)), "action"
                            )
                        )
                        segments.extend(
                            _return_segments_with_control(tok, None, None)
                        )
                    segments.extend(_answer_segments(answer))
                segments.append(TaggedSegment(answer, "final_answer"))
                yield RenderedRecord(
                    tuple(segments),
                    ScheduleEntry("reasoning", item.qid, exposure, max_hops),
                )
                exposure += 1
```

- [ ] **Step 6: Run generator tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_worlds.py -q`

Expected: `4 passed`.

- [ ] **Step 7: Commit world generation**

```bash
git add corpusgen/srgm_worlds.py tests/test_srgm_worlds.py
git commit -m "feat: generate relational graph worlds"
```

---

### Task 4: Learned Selector and Frozen Artifact

**Files:**
- Create: `corpusgen/srgm_selector.py`
- Create: `scripts/measure_srgm_selector_costs.py`
- Create: `scripts/calibrate_srgm_selector.py`
- Create: `tests/test_srgm_selector.py`

**Interfaces:**
- Consumes: `SelectorFeatures` and disjoint `GraphFact` calibration rows from Task 3.
- Produces: `SelectorMLP`, `SelectorCosts`, `SelectorArtifact`, `selector_loss()`, `route_fact()`, and a hashed frozen artifact consumed by the corpus builder and model.

- [ ] **Step 1: Write failing selector tests**

```python
# tests/test_srgm_selector.py
import torch

from corpusgen.graph_records import SelectorFeatures
from corpusgen.srgm_selector import (
    SelectorArtifact,
    SelectorCosts,
    SelectorMLP,
    route_fact,
    selector_loss,
)


def test_selector_has_exactly_225_parameters():
    assert sum(p.numel() for p in SelectorMLP().parameters()) == 225


def test_selector_loss_prefers_cheaper_route():
    logits = torch.tensor([-8.0, 8.0])
    internal = torch.tensor([0.1, 4.0])
    reads = torch.tensor([8.0, 0.1])
    hops = torch.tensor([4.0, 0.1])
    loss = selector_loss(logits, internal, reads, hops, SelectorCosts())
    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_artifact_round_trip_and_hash(tmp_path):
    artifact = SelectorArtifact.from_model(
        SelectorMLP(), mean=(0, 0, 0, 0, 0), std=(1, 1, 1, 1, 1),
        costs=SelectorCosts(), threshold=0.5, training_data_sha256="a" * 64
    )
    path = tmp_path / "selector.pt"
    artifact.save(path)
    loaded = SelectorArtifact.load(path)
    assert loaded.sha256() == artifact.sha256()


def test_route_is_deterministic_and_name_free():
    model = SelectorMLP()
    features = SelectorFeatures(1.0, 4.0, 2.0, 0.1, 0.1)
    first = route_fact(model, features, (0,) * 5, (1,) * 5, 0.5)
    second = route_fact(model, features, (0,) * 5, (1,) * 5, 0.5)
    assert first == second
```

- [ ] **Step 2: Run tests and confirm the missing selector**

Run: `.venv/bin/python -m pytest tests/test_srgm_selector.py -q`

Expected: FAIL with missing `corpusgen.srgm_selector`.

- [ ] **Step 3: Implement the 5→32→1 gate and cost objective**

```python
# corpusgen/srgm_selector.py
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from corpusgen.graph_records import SelectorFeatures


class SelectorMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass(frozen=True)
class SelectorCosts:
    write: float = 1.0
    read: float = 1.0
    hop: float = 1.0


def hard_concrete(logits: torch.Tensor, temperature: float, training: bool):
    if not training:
        return (torch.sigmoid(logits) >= 0.5).to(logits.dtype)
    u = torch.rand_like(logits).clamp_(1e-6, 1 - 1e-6)
    soft = torch.sigmoid((torch.log(u) - torch.log1p(-u) + logits) / temperature)
    stretched = (soft * 1.2 - 0.1).clamp(0, 1)
    hard = (stretched >= 0.5).to(stretched.dtype)
    return hard.detach() - stretched.detach() + stretched


def selector_loss(logits, internal_cost, expected_reads, expected_hops, costs):
    gate = hard_concrete(logits, temperature=0.67, training=True)
    external = costs.write + costs.read * expected_reads + costs.hop * expected_hops
    return ((1.0 - gate) * internal_cost + gate * external).mean()


def _normalized(features, mean, std):
    vector = torch.tensor(features.vector(), dtype=torch.float32)
    return (vector - torch.tensor(mean)) / torch.tensor(std).clamp_min(1e-6)


@torch.no_grad()
def route_fact(model, features, mean, std, threshold):
    score = torch.sigmoid(model(_normalized(features, mean, std).unsqueeze(0)))[0]
    return bool(score.item() >= threshold)


@dataclass(frozen=True)
class SelectorArtifact:
    state_dict: dict
    mean: tuple[float, ...]
    std: tuple[float, ...]
    costs: dict
    threshold: float
    training_data_sha256: str

    @classmethod
    def from_model(cls, model, mean, std, costs, threshold, training_data_sha256):
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        return cls(state, tuple(mean), tuple(std), asdict(costs), threshold, training_data_sha256)

    def canonical_bytes(self):
        metadata = json.dumps(
            {
                "mean": self.mean,
                "std": self.std,
                "costs": self.costs,
                "threshold": self.threshold,
                "training_data_sha256": self.training_data_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        chunks = [metadata]
        for key in sorted(self.state_dict):
            tensor = self.state_dict[key].detach().cpu().contiguous()
            header = json.dumps(
                {"key": key, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            chunks.extend((header, tensor.numpy().tobytes()))
        return b"\0".join(chunks)

    def sha256(self):
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict,
                "mean": self.mean,
                "std": self.std,
                "costs": self.costs,
                "threshold": self.threshold,
                "training_data_sha256": self.training_data_sha256,
            },
            path,
        )

    @classmethod
    def load(cls, path):
        value = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**value)
```

- [ ] **Step 4: Add the bounded calibration command**

First create the disjoint cost-measurement command:

```python
# scripts/measure_srgm_selector_costs.py
#!/usr/bin/env python
import argparse
import hashlib

import torch
import torch.nn.functional as F

from corpusgen.graph_records import SelectorFeatures
from corpusgen.srgm_worlds import WorldConfig, generate_world
from train.model import GPT, PRESETS
from train.tokenizer import get_tok


def measure_costs(n_worlds, seed, exposures):
    torch.manual_seed(seed)
    tok = get_tok()
    model = GPT(PRESETS["toy"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    worlds = [
        generate_world(
            world_id=1_000_000 + index,
            cfg=WorldConfig(
                n_entities=64,
                seed=seed,
                entity_id_offset=index * 64,
            ),
        )
        for index in range(n_worlds)
    ]
    features = []
    internal_cost = []
    expected_reads = []
    expected_hops = []
    audit_class = []
    digest = hashlib.sha256()
    for world in worlds:
        for fact in world.facts:
            text = (
                f"Source {fact.row.source_id} relation "
                f"{fact.row.relation_id} returns {fact.row.target}"
            )
            ids = torch.tensor([tok.encode(text)], dtype=torch.long)
            payload_start = len(tok.encode(text.rsplit(" ", 1)[0]))
            cumulative = 0.0
            for _ in range(exposures):
                logits, _ = model(ids[:, :-1])
                target = ids[:, 1:]
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target.reshape(-1),
                    reduction="none",
                )
                cumulative += losses[payload_start - 1:].sum().item()
                loss = losses.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            features.append(fact.features.vector())
            internal_cost.append(cumulative)
            expected_reads.append(fact.features.expected_queries)
            expected_hops.append(max(1.0, fact.features.path_centrality))
            audit_class.append(fact.audit_class)
            digest.update((fact.fact_id + text).encode())
    for index, text in enumerate((
        "Composition adds edge codes modulo four.",
        "Inverse traversal reverses edge direction.",
        "Equality is symmetric.",
        "Earlier ISO-8601 dates have smaller strings.",
    )):
        ids = torch.tensor([tok.encode(text)], dtype=torch.long)
        cumulative = 0.0
        for _ in range(exposures):
            logits, _ = model(ids[:, :-1])
            target = ids[:, 1:]
            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
                reduction="none",
            )
            cumulative += losses.sum().item()
            optimizer.zero_grad(set_to_none=True)
            losses.mean().backward()
            optimizer.step()
        rule_features = SelectorFeatures(
            log_exposure=4.0,
            payload_entropy=0.1,
            payload_tokens=float(target.numel()),
            expected_queries=32.0,
            path_centrality=1.0,
        )
        features.append(rule_features.vector())
        internal_cost.append(cumulative)
        expected_reads.append(rule_features.expected_queries)
        expected_hops.append(6.0)
        audit_class.append("structural")
        digest.update(f"rule-{index}:{text}".encode())
    return {
        "features": torch.tensor(features, dtype=torch.float32),
        "internal_cost": torch.tensor(internal_cost, dtype=torch.float32),
        "expected_reads": torch.tensor(expected_reads, dtype=torch.float32),
        "expected_hops": torch.tensor(expected_hops, dtype=torch.float32),
        "audit_class": audit_class,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--exposures", type=int, default=4)
    args = parser.parse_args()
    torch.save(
        measure_costs(args.worlds, args.seed, args.exposures),
        args.out,
    )


if __name__ == "__main__":
    main()
```

The calibration worlds use IDs and seeds disjoint from every protected build;
their model is discarded after writing the cost bundle.

```python
# scripts/calibrate_srgm_selector.py
#!/usr/bin/env python
import argparse
import itertools
import json
from pathlib import Path

import torch

from corpusgen.srgm_selector import (
    SelectorArtifact,
    SelectorCosts,
    SelectorMLP,
    selector_loss,
)


def train_selector(features, internal, reads, hops, costs, steps, seed):
    torch.manual_seed(seed)
    mean = features.mean(0)
    std = features.std(0).clamp_min(1e-6)
    x = (features - mean) / std
    model = SelectorMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = selector_loss(model(x), internal, reads, hops, costs)
        loss.backward()
        opt.step()
    return model, tuple(mean.tolist()), tuple(std.tolist()), costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    data = torch.load(args.calibration, map_location="cpu", weights_only=False)
    audit = data["audit_class"]
    peripheral = [index for index, value in enumerate(audit) if value == "peripheral"]
    retained = [
        index for index, value in enumerate(audit)
        if value in ("central", "structural")
    ]
    candidates = []
    for write, read, hop in itertools.product(
        (0.25, 0.5, 1.0, 2.0, 4.0),
        (0.25, 1.0, 4.0),
        (0.25, 1.0, 4.0),
    ):
        trial_costs = SelectorCosts(write, read, hop)
        model, mean, std, _ = train_selector(
            data["features"], data["internal_cost"], data["expected_reads"],
            data["expected_hops"], trial_costs, args.steps, args.seed
        )
        normalized = (
            data["features"] - torch.tensor(mean)
        ) / torch.tensor(std).clamp_min(1e-6)
        with torch.no_grad():
            routed = torch.sigmoid(model(normalized)) >= 0.5
        rate = routed.float().mean().item()
        peripheral_external = routed[peripheral].float().mean().item()
        retained_internal = (~routed[retained]).float().mean().item()
        if 0.40 <= rate <= 0.60:
            external = (
                trial_costs.write
                + trial_costs.read * data["expected_reads"]
                + trial_costs.hop * data["expected_hops"]
            )
            objective = torch.where(
                routed, external, data["internal_cost"]
            ).mean().item()
            candidates.append((
                objective, model, mean, std, trial_costs, routed,
                rate, peripheral_external, retained_internal
            ))
    if not candidates:
        raise SystemExit("no selector cost tuple met the 40–60% route-rate gate")
    (
        _, model, mean, std, costs, routed, externalization_rate,
        peripheral_external, retained_internal
    ) = min(candidates, key=lambda value: value[0])
    if peripheral_external < 0.80 or retained_internal < 0.80:
        raise SystemExit(
            "lowest-cost selector failed blinded audit gates: "
            f"peripheral_external={peripheral_external:.4f}, "
            f"retained_internal={retained_internal:.4f}"
        )
    artifact = SelectorArtifact.from_model(
        model, mean, std, costs, 0.5, data["sha256"]
    )
    artifact.save(args.out)
    report = {
        "artifact_sha256": artifact.sha256(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "externalization_rate": externalization_rate,
        "peripheral_external_rate": peripheral_external,
        "retained_internal_rate": retained_internal,
    }
    Path(str(args.out) + ".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run selector tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_selector.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit selector calibration**

```bash
git add corpusgen/srgm_selector.py scripts/measure_srgm_selector_costs.py scripts/calibrate_srgm_selector.py tests/test_srgm_selector.py
git commit -m "feat: learn selective graph routing"
```

---

### Task 5: Shared Record Schedule and Weight Sidecars

**Files:**
- Create: `corpusgen/srgm_build.py`
- Create: `scripts/build_srgm_corpus.py`
- Create: `tests/test_srgm_build.py`

**Interfaces:**
- Consumes: graph worlds, tagged segments, trace codec, and frozen selector artifact from Tasks 1–4.
- Produces: one `train.bin`, one `train.deep.bin`, condition-specific `dense.weights.bin`, `split.weights.bin`, and `random.weights.bin`, one graph snapshot, one schedule, eval JSONLs, and a hash-rich `report.json`.

- [ ] **Step 1: Write failing shared-schedule and mask-ledger tests**

```python
# tests/test_srgm_build.py
import json

import numpy as np
import pytest

from corpusgen.srgm_build import SRGMBuildCfg, build_srgm_corpus
from corpusgen.srgm_selector import SelectorArtifact, SelectorCosts, SelectorMLP
from train.tokenizer import get_tok


@pytest.fixture
def selector_artifact(monkeypatch):
    monkeypatch.setattr(
        "corpusgen.srgm_build.route_fact",
        lambda model, features, mean, std, threshold:
            0.075 <= features.path_centrality <= 0.5,
    )
    return SelectorArtifact.from_model(
        SelectorMLP(), (0,) * 5, (1,) * 5, SelectorCosts(), 0.5, "0" * 64
    )


def bed():
    while True:
        yield "A neutral bed document contains ordinary language and no graph facts."


def test_build_writes_one_shared_token_stream(tmp_path, selector_artifact):
    cfg = SRGMBuildCfg(
        n_entities=64, total_tokens=60_000, seed=11,
        n_eval_pairs_per_task=20, world_size=32
    )
    report = build_srgm_corpus(cfg, get_tok(), bed(), selector_artifact, tmp_path)
    assert (tmp_path / "train.bin").exists()
    assert not (tmp_path / "dense" / "train.bin").exists()
    token_count = len(np.memmap(tmp_path / "train.bin", dtype=np.uint16, mode="r"))
    for condition in ("dense", "split", "random"):
        weights = np.memmap(
            tmp_path / f"{condition}.weights.bin", dtype=np.uint8, mode="r"
        )
        assert len(weights) == token_count
    assert report["checks"]["shared_schedule_identity"]


def test_split_masks_only_routed_payloads(tmp_path, selector_artifact):
    cfg = SRGMBuildCfg(
        n_entities=64, total_tokens=60_000, seed=13,
        n_eval_pairs_per_task=10, world_size=32
    )
    report = build_srgm_corpus(cfg, get_tok(), bed(), selector_artifact, tmp_path)
    ledger = json.loads((tmp_path / "mask_ledger.json").read_text())
    assert ledger["split"]["unmasked_external_payload_tokens"] == 0
    assert ledger["split"]["masked_rule_tokens"] == 0
    assert report["checks"]["exact_mask_coverage"]


def test_random_mask_matches_mass_and_span_histogram(tmp_path, selector_artifact):
    cfg = SRGMBuildCfg(
        n_entities=64, total_tokens=60_000, seed=17,
        n_eval_pairs_per_task=10, world_size=32
    )
    report = build_srgm_corpus(cfg, get_tok(), bed(), selector_artifact, tmp_path)
    assert report["checks"]["random_mask_mass_within_1pct"]
    assert report["checks"]["random_span_histogram_within_1pct"]


def test_eval_counts_and_counterfactual_pairs_are_exact(tmp_path, selector_artifact):
    cfg = SRGMBuildCfg(
        n_entities=64, total_tokens=60_000, seed=19,
        n_eval_pairs_per_task=20, world_size=32
    )
    report = build_srgm_corpus(cfg, get_tok(), bed(), selector_artifact, tmp_path)
    assert report["eval_counts"] == {
        "path_composition": 40,
        "date_ordering": 40,
        "balanced_equality": 40,
    }
```

- [ ] **Step 2: Run tests and confirm the missing SRGM builder**

Run: `.venv/bin/python -m pytest tests/test_srgm_build.py -q`

Expected: FAIL with missing `corpusgen.srgm_build`.

- [ ] **Step 3: Define the dedicated build configuration and schedule**

```python
# corpusgen/srgm_build.py
from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np

from corpusgen.graph_records import (
    GraphAction,
    GraphAddress,
    RenderedRecord,
    ScheduleEntry,
    TaggedSegment,
)
from corpusgen.graph_trace import serialize_action, serialize_return
from corpusgen.srgm_selector import SelectorArtifact, SelectorMLP, route_fact
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
    iter_bed_records,
    iter_graph_records,
    iter_reasoning_records,
    iter_worlds,
)
from organizer.graph_store import AtomicGraphStore

Condition = Literal["dense", "split", "random"]
COMPONENT_SHARES = {"bed": 0.45, "graph": 0.30, "reasoning": 0.25}
REASONING_SHARES = {
    "path_composition": 0.10,
    "date_ordering": 0.075,
    "balanced_equality": 0.075,
}


@dataclass(frozen=True)
class SRGMBuildCfg:
    n_entities: int
    total_tokens: int
    seed: int
    n_eval_pairs_per_task: int = 10_000
    world_size: int = 64
    context: int = 1024


@dataclass(frozen=True)
class EncodedSpan:
    start: int
    stop: int
    role: str
    fact_id: str | None
    position_decile: int

    @property
    def length(self) -> int:
        return self.stop - self.start


def schedule_sha256(entries: Iterable[ScheduleEntry]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        line = json.dumps(asdict(entry), sort_keys=True, separators=(",", ":"))
        digest.update(line.encode() + b"\n")
    return digest.hexdigest()


class ScheduleWriter:
    def __init__(self, path: Path):
        self.handle = open(path, "w")
        self.digest = hashlib.sha256()
        self.count = 0

    def add(self, entry: ScheduleEntry):
        line = json.dumps(asdict(entry), sort_keys=True, separators=(",", ":"))
        self.handle.write(line + "\n")
        self.digest.update(line.encode() + b"\n")
        self.count += 1

    def close(self):
        self.handle.close()
        return self.digest.hexdigest(), self.count
```

- [ ] **Step 4: Encode once and derive condition weights**

Add to `corpusgen/srgm_build.py`:

```python
def encode_tagged(tok, segments: list[TaggedSegment]):
    ids: list[int] = []
    raw_spans: list[tuple[int, int, str, str | None]] = []
    for segment in segments:
        segment_ids = tok.encode(segment.text)
        start = len(ids)
        ids.extend(segment_ids)
        raw_spans.append((start, len(ids), segment.role, segment.fact_id))
    ids.append(tok.EOT)
    raw_spans.append((len(ids) - 1, len(ids), "plain", None))
    spans = [
        EncodedSpan(
            start, stop, role, fact_id,
            min(9, 10 * start // max(1, len(ids)))
        )
        for start, stop, role, fact_id in raw_spans
    ]
    return np.asarray(ids, dtype=np.uint16), spans


def _candidate_plain_spans(spans: list[EncodedSpan]):
    protected = {"rule", "action", "provisional_answer", "final_answer", "payload"}
    return [span for span in spans if span.role not in protected and span.length > 0]


def derive_weights(
    n_tokens: int,
    spans: list[EncodedSpan],
    routes,
    condition: Condition,
    rng: random.Random,
) -> np.ndarray:
    weights = np.ones(n_tokens, dtype=np.uint8)
    selected = [
        span for span in spans
        if span.role == "payload" and span.fact_id in routes
    ]
    if condition == "split":
        for span in selected:
            weights[span.start:span.stop] = 0
    elif condition == "random":
        candidates = _candidate_plain_spans(spans)
        by_length: dict[tuple[int, int], list[EncodedSpan]] = {}
        for span in candidates:
            by_length.setdefault((span.length, span.position_decile), []).append(span)
        for span in selected:
            matches = by_length.get((span.length, span.position_decile), [])
            if not matches:
                raise ValueError(
                    f"no random-control span for "
                    f"length={span.length}, decile={span.position_decile}"
                )
            chosen = matches[rng.randrange(len(matches))]
            matches.remove(chosen)
            weights[chosen.start:chosen.stop] = 0
    return weights


class SharedWriter:
    def __init__(self, out: Path):
        self.token_file = open(out / "train.bin", "wb")
        self.deep_file = open(out / "train.deep.bin", "wb")
        self.weight_files = {
            condition: open(out / f"{condition}.weights.bin", "wb")
            for condition in ("dense", "split", "random")
        }
        self.total_tokens = 0
        self.masked_tokens = {condition: 0 for condition in self.weight_files}
        self.span_hist = {condition: {} for condition in self.weight_files}
        self.external_payload_tokens = 0
        self.unmasked_external_payload_tokens = 0
        self.masked_rule_tokens = 0
        self.component_tokens = {name: 0 for name in COMPONENT_SHARES}

    def add(self, ids, spans, routes, rng, component):
        ids.tofile(self.token_file)
        deep = _deep_supervision_weights(spans, len(ids))
        deep.tofile(self.deep_file)
        for condition, handle in self.weight_files.items():
            weights = derive_weights(
                len(ids), spans, routes, condition, rng
            )
            weights.tofile(handle)
            masked = int((weights == 0).sum())
            self.masked_tokens[condition] += masked
            for span in spans:
                if (weights[span.start:span.stop] == 0).all():
                    key = str(span.length)
                    self.span_hist[condition][key] = (
                        self.span_hist[condition].get(key, 0) + 1
                    )
        split = derive_weights(
            len(ids), spans, routes, "split", random.Random(0)
        )
        for span in spans:
            if span.role == "payload" and span.fact_id in routes:
                self.external_payload_tokens += span.length
                self.unmasked_external_payload_tokens += int(
                    split[span.start:span.stop].sum()
                )
            if span.role == "rule":
                self.masked_rule_tokens += int(
                    (split[span.start:span.stop] == 0).sum()
                )
        self.total_tokens += len(ids)
        self.component_tokens[component] += len(ids)

    def close(self):
        self.token_file.close()
        self.deep_file.close()
        for handle in self.weight_files.values():
            handle.close()

    def ledger(self):
        return {
            "split": {
                "external_payload_tokens": self.external_payload_tokens,
                "unmasked_external_payload_tokens": self.unmasked_external_payload_tokens,
                "masked_rule_tokens": self.masked_rule_tokens,
                "masked_tokens": self.masked_tokens["split"],
                "span_histogram": self.span_hist["split"],
            },
            "random": {
                "masked_tokens": self.masked_tokens["random"],
                "span_histogram": self.span_hist["random"],
            },
            "component_tokens": self.component_tokens,
            "total_tokens": self.total_tokens,
        }


class RouteIndex:
    def __init__(self, path, n_facts, mode="r+"):
        self.values = np.memmap(path, dtype=np.uint8, mode=mode, shape=(n_facts,))

    @classmethod
    def create(cls, path, n_facts):
        value = cls(path, n_facts, mode="w+")
        value.values[:] = 0
        return value

    def set(self, fact_id, external):
        self.values[int(fact_id)] = int(bool(external))

    def __contains__(self, fact_id):
        return bool(self.values[int(fact_id)])

    def flush(self):
        self.values.flush()

    def count(self):
        return int(self.values.sum())


class GraphSnapshotWriter:
    def __init__(self, path):
        self.handle = open(path, "w")
        self.digest = hashlib.sha256()
        self.count = 0

    def add(self, row):
        line = json.dumps(row.as_json(), sort_keys=True, separators=(",", ":"))
        self.handle.write(line + "\n")
        self.digest.update(line.encode() + b"\n")
        self.count += 1

    def close(self):
        self.handle.close()
        return self.digest.hexdigest(), self.count
```

For each payload, the graph renderer emits a neutral plain span made from the
single-token string `" the"` repeated to the payload token length and placed
in the same position decile. This guarantees a `(length, decile)` candidate
without copying the fact value.

- [ ] **Step 5: Implement one-pass stream assembly**

Add the build entry point to `corpusgen/srgm_build.py`:

```python
def build_srgm_corpus(
    cfg: SRGMBuildCfg,
    tok,
    bed_iter: Iterator[str],
    selector_artifact: SelectorArtifact,
    out_dir: str | Path,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selector = SelectorMLP()
    selector.load_state_dict(selector_artifact.state_dict)
    selector.eval()
    writer = SharedWriter(out)
    schedule_writer = ScheduleWriter(out / "schedule.jsonl")
    routes = RouteIndex.create(out / "routes.bin", cfg.n_entities * 6)
    graph_writer = GraphSnapshotWriter(out / "graph.jsonl")
    external_probe_facts = []
    internal_probe_facts = []
    for world in iter_worlds(cfg.n_entities, cfg.world_size, cfg.seed):
        for fact in world.facts:
            graph_writer.add(fact.row)
            external = route_fact(
                selector, fact.features, selector_artifact.mean,
                selector_artifact.std, selector_artifact.threshold
            )
            if external:
                routes.set(fact.fact_id, True)
                if fact.audit_class == "peripheral" and len(external_probe_facts) < 2_000:
                    external_probe_facts.append(fact)
            elif len(internal_probe_facts) < 2_000:
                internal_probe_facts.append(fact)
    routes.flush()
    graph_hash, graph_count = graph_writer.close()
    records = _render_component_records(cfg, tok, bed_iter)
    for record in _shared_largest_deficit_schedule(
        records, COMPONENT_SHARES, cfg.total_tokens, tok
    ):
        ids, spans = encode_tagged(tok, record.segments)
        rng = random.Random((cfg.seed << 32) ^ schedule_writer.count)
        writer.add(ids, spans, routes, rng, record.schedule.component)
        schedule_writer.add(record.schedule)
    writer.close()
    schedule_hash, schedule_count = schedule_writer.close()
    eval_counts = _write_fresh_eval_sets(
        cfg, out, selector, selector_artifact
    )
    eval_counts.update(
        _write_probe_sets(out, external_probe_facts, internal_probe_facts, cfg.seed)
    )
    eval_counts["shared_text"] = _write_shared_text(out, bed_iter, 1_000)
    ledger = writer.ledger()
    (out / "mask_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    report = _build_report(
        cfg, out, schedule_hash, schedule_count, eval_counts,
        routes.count(), graph_hash, graph_count, ledger,
        selector_artifact.sha256()
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
```

Add these concrete support types and output helpers:

```python
def _deep_supervision_weights(spans, n_tokens):
    weights = np.zeros(n_tokens, dtype=np.uint8)
    supervised = {"action", "provisional_answer", "final_answer"}
    for span in spans:
        if span.role in supervised:
            weights[span.start:span.stop] = 1
    return weights


def _write_graph(rows, path):
    AtomicGraphStore(rows).save(path)


def _write_fresh_eval_sets(cfg, out, selector, selector_artifact):
    eval_dir = out / "eval"
    eval_dir.mkdir(exist_ok=True)
    fresh = generate_world(
        9_000_000 + cfg.seed,
        WorldConfig(
            n_entities=512,
            seed=cfg.seed + 1_000_000,
            entity_id_offset=cfg.n_entities + 1_000_000,
        ),
    )
    AtomicGraphStore(fact.row for fact in fresh.facts).save(
        eval_dir / "fresh_graph.jsonl"
    )
    fact_map = {fact.fact_id: fact for fact in fresh.facts}
    candidates = generate_eval_pairs(
        fresh, cfg.n_eval_pairs_per_task * 20, cfg.seed + 2_000_000
    )
    selected = []
    counts = {task: 0 for task in (
        "path_composition", "date_ordering", "balanced_equality"
    )}
    for pair in candidates:
        fact_ids = pair.original.meta["gold_fact_ids"]
        routed = all(
            route_fact(
                selector, fact_map[fact_id].features,
                selector_artifact.mean, selector_artifact.std,
                selector_artifact.threshold
            )
            for fact_id in fact_ids
        )
        if routed and counts[pair.task] < cfg.n_eval_pairs_per_task:
            selected.append(pair)
            counts[pair.task] += 1
    if any(value != cfg.n_eval_pairs_per_task for value in counts.values()):
        raise ValueError(f"insufficient externally routed fresh pairs: {counts}")
    pairs = selected
    counts = {}
    for task in ("path_composition", "date_ordering", "balanced_equality"):
        items = []
        for pair in pairs:
            if pair.task == task:
                items.extend((pair.original, pair.counterfactual))
        path = eval_dir / f"{task}.jsonl"
        with open(path, "w") as handle:
            for item in items:
                handle.write(json.dumps(asdict(item), sort_keys=True) + "\n")
        counts[task] = len(items)
    return counts


def _write_probe_sets(out, external_facts, internal_facts, seed):
    rng = random.Random(seed + 3_000_000)
    eval_dir = out / "eval"
    pool = [fact.row.target for fact in external_facts + internal_facts]
    recognition = []
    recall = []
    for fact in external_facts:
        distractors = [value for value in pool if value != fact.row.target]
        choices = rng.sample(distractors, 3) + [fact.row.target]
        rng.shuffle(choices)
        recognition.append({
            "qid": f"recognition-{fact.fact_id}",
            "prompt": (
                f"Source {fact.row.source_id} relation "
                f"{fact.row.relation_id} returns"
            ),
            "choices": choices,
            "answer_index": choices.index(fact.row.target),
            "fact_id": fact.fact_id,
            "route": "external",
        })
        recall.append({
            "qid": f"recall-{fact.fact_id}",
            "source_id": fact.row.source_id,
            "relation_id": fact.row.relation_id,
            "direction": fact.row.direction,
            "answer": fact.row.target,
            "route": "external",
        })
    for fact in internal_facts:
        recall.append({
            "qid": f"recall-{fact.fact_id}",
            "source_id": fact.row.source_id,
            "relation_id": fact.row.relation_id,
            "direction": fact.row.direction,
            "answer": fact.row.target,
            "route": "internal",
        })
    for name, rows in (("recognition", recognition), ("recall", recall)):
        with open(eval_dir / f"{name}.jsonl", "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    return {"recognition": len(recognition), "recall": len(recall)}


def _write_shared_text(out, bed_iter, n_docs):
    path = out / "eval" / "shared_text.jsonl"
    with open(path, "w") as handle:
        for index in range(n_docs):
            handle.write(json.dumps({"id": index, "text": next(bed_iter)}) + "\n")
    return n_docs


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hist_distance(first, second):
    keys = set(first) | set(second)
    denominator = max(1, sum(first.values()))
    return sum(abs(first.get(key, 0) - second.get(key, 0)) for key in keys) / denominator


def _build_report(
    cfg, out, schedule_hash, schedule_count, eval_counts,
    external_fact_count, graph_sha256, graph_count, ledger, selector_sha256
):
    split_masked = ledger["split"]["masked_tokens"]
    random_masked = ledger["random"]["masked_tokens"]
    mass_error = abs(split_masked - random_masked) / max(1, split_masked)
    hist_error = _hist_distance(
        ledger["split"]["span_histogram"],
        ledger["random"]["span_histogram"],
    )
    component_shares = {
        name: count / ledger["total_tokens"]
        for name, count in ledger["component_tokens"].items()
    }
    share_error = max(
        abs(component_shares[name] - COMPONENT_SHARES[name])
        for name in COMPONENT_SHARES
    )
    return {
        "cfg": asdict(cfg),
        "schedule_sha256": schedule_hash,
        "schedule_count": schedule_count,
        "corpus_sha256": _file_sha256(out / "train.bin"),
        "graph_sha256": graph_sha256,
        "graph_rows": graph_count,
        "selector_sha256": selector_sha256,
        "external_fact_count": external_fact_count,
        "eval_counts": eval_counts,
        "mask_ledger": ledger,
        "component_shares": component_shares,
        "checks": {
            "shared_schedule_identity": True,
            "exact_mask_coverage": (
                ledger["split"]["unmasked_external_payload_tokens"] == 0
                and ledger["split"]["masked_rule_tokens"] == 0
            ),
            "random_mask_mass_within_1pct": mass_error <= 0.01,
            "random_span_histogram_within_1pct": hist_error <= 0.01,
            "mixture_within_1pct": share_error <= 0.01,
        },
    }
```

Use these exact iterator contracts for the remaining two helpers:

```python
def _render_component_records(cfg, tok, bed_iter):
    def worlds_factory():
        return iter_worlds(cfg.n_entities, cfg.world_size, cfg.seed)

    return {
        "bed": iter_bed_records(bed_iter),
        "graph": iter_graph_records(tok, worlds_factory),
        "reasoning": {
            1: iter_reasoning_records(tok, worlds_factory, cfg.seed + 101, max_hops=1),
            2: iter_reasoning_records(tok, worlds_factory, cfg.seed + 102, max_hops=2),
            4: iter_reasoning_records(tok, worlds_factory, cfg.seed + 104, max_hops=4),
        },
    }


def _shared_largest_deficit_schedule(records, shares, total_tokens, tok):
    emitted = {component: 0 for component in shares}
    total = 0
    while total < total_tokens:
        component = max(
            sorted(shares),
            key=lambda name: shares[name] - emitted[name] / max(1, total_tokens),
        )
        fraction = total / total_tokens
        band = 1 if fraction < 0.20 else 2 if fraction < 0.50 else 4
        source = records["reasoning"][band] if component == "reasoning" else records[component]
        record = next(source)
        length = len(tok.encode("".join(segment.text for segment in record.segments))) + 1
        emitted[component] += length
        total += length
        yield RenderedRecord(
            record.segments,
            replace(record.schedule, curriculum_band=band),
        )
```

`iter_bed_records()`, `iter_graph_records()`, and `iter_reasoning_records()` are streaming
generators in `corpusgen/srgm_worlds.py`. Their exact outputs are:

- bed: one `plain` segment and a `ScheduleEntry("bed", ...)`;
- graph: statement/action framing, one tagged payload, an adjacent same-length
  neutral control span, and `ScheduleEntry("graph", ...)`; and
- reasoning: question, gold atomic action/return sequence, six provisional
  answer states with post-halt no-ops, a derived final answer, and
  `ScheduleEntry("reasoning", ...)`.

Add unit tests for each generator before wiring it into
`_render_component_records()`.

- [ ] **Step 6: Add the bounded corpus command**

```python
# scripts/build_srgm_corpus.py
#!/usr/bin/env python
import argparse
import json
import shutil
from pathlib import Path

from corpusgen.srgm_build import SRGMBuildCfg, build_srgm_corpus
from corpusgen.srgm_selector import SelectorArtifact
from train.tokenizer import get_tok


LOADS = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000,
         "n1p8m": 1_800_000, "n5m": 5_000_000}


def bed_jsonl(path):
    while True:
        with open(path) as handle:
            for line in handle:
                row = json.loads(line)
                yield row["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--load", required=True, choices=LOADS)
    parser.add_argument("--total-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bed-file", required=True)
    args = parser.parse_args()
    cfg = SRGMBuildCfg(LOADS[args.load], args.total_tokens, args.seed)
    out_dir = Path(args.out_root) / f"{args.load}_ds{args.seed}"
    build_srgm_corpus(
        cfg, get_tok(), bed_jsonl(args.bed_file), SelectorArtifact.load(args.selector),
        out_dir
    )
    shutil.copyfile(str(args.selector) + ".json", out_dir / "selector_audit.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run builder tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_build.py -q`

Expected: `4 passed`.

- [ ] **Step 8: Commit the paired corpus builder**

```bash
git add corpusgen/srgm_build.py scripts/build_srgm_corpus.py tests/test_srgm_build.py
git commit -m "feat: build paired SRGM corpora"
```

---

### Task 6: Weighted Packed Data and All-Token-Normalized Loss

**Files:**
- Create: `train/srgm_data.py`
- Create: `train/srgm_loss.py`
- Create: `tests/test_srgm_data.py`
- Create: `tests/test_srgm_loss.py`

**Interfaces:**
- Consumes: shared token, condition-weight, and deep-weight files from Task 5.
- Produces: `SRGMBatch`, `WeightedPackedShards`, `weighted_causal_loss()`, and `recurrent_supervised_loss()`.

- [ ] **Step 1: Write failing weighted-loss tests**

```python
# tests/test_srgm_loss.py
import torch
import torch.nn.functional as F

from train.srgm_loss import recurrent_supervised_loss, weighted_causal_loss


def test_weighted_loss_divides_by_all_positions():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 7)
    targets = torch.tensor([[1, 2, 3, 4]])
    weights = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    per_token = F.cross_entropy(
        logits.view(-1, 7), targets.view(-1), reduction="none"
    ).view_as(targets)
    expected = (per_token * weights).sum() / weights.numel()
    assert torch.allclose(weighted_causal_loss(logits, targets, weights), expected)


def test_zero_weights_produce_differentiable_zero():
    logits = torch.randn(1, 3, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, 3]])
    loss = weighted_causal_loss(logits, targets, torch.zeros_like(targets).float())
    loss.backward()
    assert loss.item() == 0.0
    assert logits.grad is not None


def test_recurrent_loss_supervises_deep_positions_only():
    torch.manual_seed(1)
    steps = tuple(torch.randn(1, 3, 5) for _ in range(3))
    targets = torch.tensor([[1, 2, 3]])
    weights = torch.ones_like(targets).float()
    deep = torch.tensor([[0.0, 1.0, 0.0]])
    loss = recurrent_supervised_loss(steps, targets, weights, deep)
    final = weighted_causal_loss(steps[-1], targets, weights)
    assert loss > final
```

```python
# tests/test_srgm_data.py
import numpy as np
import torch

from train.srgm_data import WeightedPackedShards


def test_weighted_batch_aligns_next_token_sidecars(tmp_path):
    tokens = np.arange(30, dtype=np.uint16)
    weights = np.ones(30, dtype=np.uint8)
    deep = np.zeros(30, dtype=np.uint8)
    weights[5] = 0
    deep[7] = 1
    tokens.tofile(tmp_path / "train.bin")
    weights.tofile(tmp_path / "weights.bin")
    deep.tofile(tmp_path / "deep.bin")
    data = WeightedPackedShards(
        tmp_path / "train.bin", tmp_path / "weights.bin", tmp_path / "deep.bin",
        ctx=8, batch_size=2
    )
    batch = data.next_batch()
    assert batch.x.shape == batch.targets.shape == batch.weights.shape == (2, 8)
    assert batch.weights.dtype == torch.float32
    assert (batch.targets == batch.x + 1).all()
```

- [ ] **Step 2: Run the tests and confirm missing modules**

Run: `.venv/bin/python -m pytest tests/test_srgm_data.py tests/test_srgm_loss.py -q`

Expected: FAIL with missing `train.srgm_data` and `train.srgm_loss`.

- [ ] **Step 3: Implement raw-position-normalized losses**

```python
# train/srgm_loss.py
from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_causal_loss(logits, targets, weights):
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    return (per_token * weights).sum() / targets.numel()


def recurrent_supervised_loss(step_logits, targets, weights, deep_weights):
    final_loss = weighted_causal_loss(step_logits[-1], targets, weights)
    if len(step_logits) == 1:
        return final_loss
    intermediate = [
        weighted_causal_loss(logits, targets, deep_weights)
        for logits in step_logits[:-1]
    ]
    return final_loss + torch.stack(intermediate).mean()
```

- [ ] **Step 4: Implement the weighted packed loader**

```python
# train/srgm_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SRGMBatch:
    x: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor
    deep_weights: torch.Tensor


class WeightedPackedShards:
    def __init__(self, bin_path, weights_path, deep_path, ctx, batch_size,
                 device="cpu", start_cursor=0):
        self.tokens = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.weights = np.memmap(weights_path, dtype=np.uint8, mode="r")
        self.deep = np.memmap(deep_path, dtype=np.uint8, mode="r")
        if not (len(self.tokens) == len(self.weights) == len(self.deep)):
            raise ValueError("token/weight/deep lengths differ")
        self.ctx = ctx
        self.batch_size = batch_size
        self.device = device
        self.cursor = start_cursor
        self.epoch = 0
        if len(self.tokens) <= batch_size * (ctx + 1):
            raise ValueError("corpus smaller than one batch")

    def next_batch(self):
        span = self.batch_size * (self.ctx + 1)
        if self.cursor + span >= len(self.tokens):
            self.cursor = 0
            self.epoch += 1
        start = self.cursor
        self.cursor += self.batch_size * self.ctx
        shape = (self.batch_size, self.ctx + 1)
        token = np.asarray(self.tokens[start:start + span]).astype(np.int64).reshape(shape)
        weight = np.asarray(self.weights[start:start + span]).reshape(shape)
        deep = np.asarray(self.deep[start:start + span]).reshape(shape)
        x = torch.from_numpy(token[:, :-1].copy())
        targets = torch.from_numpy(token[:, 1:].copy())
        weights = torch.from_numpy(weight[:, 1:].copy()).float()
        deep_weights = torch.from_numpy(deep[:, 1:].copy()).float()
        tensors = [x, targets, weights, deep_weights]
        if self.device == "cuda":
            tensors = [value.pin_memory().to("cuda", non_blocking=True) for value in tensors]
        elif self.device != "cpu":
            tensors = [value.to(self.device) for value in tensors]
        return SRGMBatch(*tensors)

    def state_dict(self):
        return {"cursor": self.cursor, "epoch": self.epoch}

    def load_state_dict(self, state):
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
```

- [ ] **Step 5: Run weighted-data tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_data.py tests/test_srgm_loss.py -q`

Expected: `4 passed`.

- [ ] **Step 6: Commit weighted training primitives**

```bash
git add train/srgm_data.py train/srgm_loss.py tests/test_srgm_data.py tests/test_srgm_loss.py
git commit -m "feat: add all-token normalized SRGM loss"
```

---

### Task 7: Fixed Six-Step Recurrent Model

**Files:**
- Create: `train/srgm_model.py`
- Create: `tests/test_srgm_model.py`

**Interfaces:**
- Consumes: `Block`, `RMSNorm`, RoPE helpers, selector, and weighted losses.
- Produces: `SRGMConfig`, `SRGM_PRESETS`, `SRGMOutput`, `SRGMCache`, `SRGM.forward()`, and `SRGM.forward_step()`.

- [ ] **Step 1: Write failing architecture and cache tests**

```python
# tests/test_srgm_model.py
import torch

from train.srgm_model import SRGM, SRGMConfig, SRGM_PRESETS


def tiny():
    return SRGM(SRGMConfig(
        n_backbone_layer=2, n_ponder_layer=2, n_ponder_steps=3,
        n_head=2, d_model=64, ctx=64
    ))


def test_exact_protected_parameter_counts():
    assert SRGM(SRGM_PRESETS["srgm30m"]).num_params() == 30_576_097
    assert SRGM(SRGM_PRESETS["srgm160m"]).num_params() == 162_221_025
    assert SRGM(SRGM_PRESETS["srgm360m"]).num_params() == 356_033_761
    assert SRGM(SRGM_PRESETS["srgm1b"]).num_params() == 1_030_667_233


def test_forward_returns_one_logit_tensor_per_step():
    model = tiny()
    x = torch.randint(0, 100, (2, 12))
    out = model(x)
    assert len(out.step_logits) == 3
    assert out.logits.shape == (2, 12, 50304)


def test_ponder_parameters_are_shared_across_steps():
    model = tiny()
    assert len(model.ponder) == 2
    parameter_ids = {id(p) for p in model.ponder.parameters()}
    assert len(parameter_ids) == len(list(model.ponder.parameters()))


def test_weighted_training_loss_is_finite():
    model = tiny()
    x = torch.randint(0, 100, (2, 12))
    targets = torch.randint(0, 100, (2, 12))
    weights = torch.ones_like(targets).float()
    deep = torch.zeros_like(weights)
    deep[:, 4:8] = 1
    out = model(x, targets, weights, deep)
    assert torch.isfinite(out.loss)


def test_incremental_cache_matches_full_forward():
    torch.manual_seed(4)
    model = tiny().eval()
    x = torch.randint(0, 100, (1, 10))
    with torch.no_grad():
        full = model(x).logits
        logits, cache = model.forward_step(x[:, :6], None)
        observed = [logits[:, -1]]
        for index in range(6, 10):
            logits, cache = model.forward_step(x[:, index:index + 1], cache)
            observed.append(logits[:, -1])
    for offset, position in enumerate(range(5, 10)):
        assert torch.allclose(full[:, position], observed[offset], atol=3e-4)
```

- [ ] **Step 2: Run tests and confirm missing recurrent model**

Run: `.venv/bin/python -m pytest tests/test_srgm_model.py -q`

Expected: FAIL with missing `train.srgm_model`.

- [ ] **Step 3: Define exact presets, output, and cache**

```python
# train/srgm_model.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from corpusgen.srgm_selector import SelectorMLP
from train.model import Block, RMSNorm, _rope_cos_sin
from train.srgm_loss import recurrent_supervised_loss


@dataclass(frozen=True)
class SRGMConfig:
    n_backbone_layer: int
    n_ponder_layer: int
    n_ponder_steps: int
    n_head: int
    d_model: int
    vocab_size: int = 50304
    ctx: int = 1024
    rope_base: float = 10000.0

    @property
    def head_dim(self):
        if self.d_model % self.n_head:
            raise ValueError("d_model must divide n_head")
        return self.d_model // self.n_head


SRGM_PRESETS = {
    "srgm30m": SRGMConfig(4, 2, 6, 4, 256),
    "srgm160m": SRGMConfig(10, 2, 6, 12, 768),
    "srgm360m": SRGMConfig(18, 2, 6, 16, 1024),
    "srgm1b": SRGMConfig(20, 2, 6, 14, 1792),
}


@dataclass(frozen=True)
class SRGMOutput:
    logits: torch.Tensor
    step_logits: tuple[torch.Tensor, ...]
    loss: torch.Tensor | None


class SRGMCache:
    def __init__(self, cfg):
        self.backbone = [(None, None) for _ in range(cfg.n_backbone_layer)]
        self.ponder = [
            [(None, None) for _ in range(cfg.n_ponder_layer)]
            for _ in range(cfg.n_ponder_steps)
        ]
        self.pos = 0
```

- [ ] **Step 4: Implement two-state fixed recurrence and weighted loss**

Add to `train/srgm_model.py`:

```python
class SRGM(nn.Module):
    def __init__(self, cfg: SRGMConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = nn.ModuleList([Block(cfg) for _ in range(cfg.n_backbone_layer)])
        self.ponder = nn.ModuleList([Block(cfg) for _ in range(cfg.n_ponder_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.selector = SelectorMLP()
        for parameter in self.selector.parameters():
            parameter.requires_grad_(False)
        cos, sin = _rope_cos_sin(cfg.head_dim, cfg.ctx, cfg.rope_base, "cpu")
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        self.apply(self._init)

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _recur(self, base, cos, sin):
        z = torch.zeros_like(base)
        y = base
        logits = []
        for _ in range(self.cfg.n_ponder_steps):
            z, _ = self.ponder[0](base + y + z, cos, sin)
            y, _ = self.ponder[1](y + z, cos, sin)
            logits.append(self.lm_head(self.ln_f(y)))
        return tuple(logits)

    def forward(self, idx, targets=None, weights=None, deep_weights=None):
        _, length = idx.shape
        if length > self.cfg.ctx:
            raise ValueError("sequence exceeds context")
        cos, sin = self.rope_cos[:length], self.rope_sin[:length]
        base = self.wte(idx)
        for block in self.backbone:
            base, _ = block(base, cos, sin)
        step_logits = self._recur(base, cos, sin)
        loss = None
        if targets is not None:
            if weights is None or deep_weights is None:
                raise ValueError("weighted training requires both sidecars")
            loss = recurrent_supervised_loss(
                step_logits, targets, weights, deep_weights
            )
        return SRGMOutput(step_logits[-1], step_logits, loss)

    def num_params(self):
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self):
        return self.wte.weight.device
```

Load the frozen selector state after model initialization so `self.apply()` does
not overwrite it.

- [ ] **Step 5: Implement a cache slot per effective recurrent layer**

Add `forward_step()` to `SRGM`:

```python
@torch.no_grad()
def forward_step(self, idx, cache):
    if cache is None:
        cache = SRGMCache(self.cfg)
    position = cache.pos
    length = idx.size(1)
    cos = self.rope_cos[position:position + length]
    sin = self.rope_sin[position:position + length]
    base = self.wte(idx)
    for layer, block in enumerate(self.backbone):
        base, value = block(base, cos, sin, kv=cache.backbone[layer])
        cache.backbone[layer] = value
    z = torch.zeros_like(base)
    y = base
    for step in range(self.cfg.n_ponder_steps):
        z, value = self.ponder[0](
            base + y + z, cos, sin, kv=cache.ponder[step][0]
        )
        cache.ponder[step][0] = value
        y, value = self.ponder[1](
            y + z, cos, sin, kv=cache.ponder[step][1]
        )
        cache.ponder[step][1] = value
    cache.pos += length
    return self.lm_head(self.ln_f(y)), cache
```

Move the method into the class body and preserve the no-grad decorator.

- [ ] **Step 6: Run model tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_model.py -q`

Expected: `5 passed`.

- [ ] **Step 7: Run legacy model tests**

Run: `.venv/bin/python -m pytest tests/test_model.py -q`

Expected: legacy tests remain PASS.

- [ ] **Step 8: Commit the recurrent model**

```bash
git add train/srgm_model.py tests/test_srgm_model.py
git commit -m "feat: add fixed-depth recursive graph model"
```

---

### Task 8: Strict Provenance and SRGM Trainer

**Files:**
- Create: `train/provenance.py`
- Create: `train/srgm_trainer.py`
- Create: `scripts/run_srgm_train.py`
- Create: `tests/test_srgm_provenance.py`
- Create: `tests/test_srgm_trainer.py`

**Interfaces:**
- Consumes: weighted batches and recurrent model from Tasks 6–7.
- Produces: `RunProvenance`, `collect_provenance()`, `assert_compatible()`, `SRGMTrainer`, and resumable training logs used by operations and evaluation.

- [ ] **Step 1: Write failing provenance and resume tests**

```python
# tests/test_srgm_provenance.py
import pytest

from train.provenance import RunProvenance, assert_compatible


def provenance(graph="a"):
    return RunProvenance(
        git_revision="1" * 40, corpus_sha256="2" * 64,
        schedule_sha256="3" * 64, tokenizer_sha256="4" * 64,
        graph_sha256=graph * 64, selector_sha256="6" * 64,
        evaluator_revision="7" * 40
    )


def test_provenance_mismatch_names_field():
    with pytest.raises(ValueError, match="graph_sha256"):
        assert_compatible(provenance("a"), provenance("b"))


def test_equal_provenance_passes():
    assert_compatible(provenance(), provenance())
```

```python
# tests/test_srgm_trainer.py
import torch

from train.srgm_trainer import SRGMTrainer


def test_paired_conditions_start_from_same_state_hash(tiny_srgm_cfgs):
    dense = SRGMTrainer(tiny_srgm_cfgs["dense"])
    split = SRGMTrainer(tiny_srgm_cfgs["split"])
    assert dense.initial_state_sha256 == split.initial_state_sha256


def test_resume_restores_next_batch_and_loss(tmp_path, tiny_srgm_cfg):
    trainer = SRGMTrainer(tiny_srgm_cfg)
    trainer.train_steps(2)
    trainer.save_ckpt()
    expected_batch = trainer.data.next_batch()
    resumed = SRGMTrainer(tiny_srgm_cfg)
    resumed.load_ckpt()
    actual_batch = resumed.data.next_batch()
    assert torch.equal(actual_batch.x, expected_batch.x)
    assert resumed.step == 2
```

- [ ] **Step 2: Run tests and confirm missing trainer modules**

Run: `.venv/bin/python -m pytest tests/test_srgm_provenance.py tests/test_srgm_trainer.py -q`

Expected: FAIL with missing `train.provenance` and `train.srgm_trainer`.

- [ ] **Step 3: Implement canonical provenance checks**

```python
# train/provenance.py
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunProvenance:
    git_revision: str
    corpus_sha256: str
    schedule_sha256: str
    tokenizer_sha256: str
    graph_sha256: str
    selector_sha256: str
    evaluator_revision: str


def assert_compatible(expected: RunProvenance, actual: RunProvenance):
    for field, expected_value in asdict(expected).items():
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            raise ValueError(
                f"provenance mismatch for {field}: "
                f"{expected_value} != {actual_value}"
            )


def current_git_revision():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def collect_provenance(cfg):
    report = json.loads(Path(cfg["report_json"]).read_text())
    return RunProvenance(
        git_revision=current_git_revision(),
        corpus_sha256=sha256_file(cfg["train_bin"]),
        schedule_sha256=report["schedule_sha256"],
        tokenizer_sha256=cfg["tokenizer_sha256"],
        graph_sha256=report["graph_sha256"],
        selector_sha256=report["selector_sha256"],
        evaluator_revision=cfg["evaluator_revision"],
    )
```

- [ ] **Step 4: Implement the dedicated trainer**

```python
# train/srgm_trainer.py
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from train.provenance import assert_compatible, collect_provenance
from train.srgm_data import WeightedPackedShards
from train.srgm_model import SRGM, SRGM_PRESETS
from train.trainer import cosine_lr, pick_device


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(str(tuple(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class SRGMTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = pick_device(cfg.get("device", "auto"))
        torch.manual_seed(cfg["init_seed"])
        if self.device == "cuda":
            torch.cuda.manual_seed_all(cfg["init_seed"])
        model_cfg = SRGM_PRESETS[cfg["model"]]
        self.model = SRGM(model_cfg)
        selector = torch.load(cfg["selector"], map_location="cpu", weights_only=False)
        self.model.selector.load_state_dict(selector["state_dict"])
        self.initial_state_sha256 = state_sha256(self.model.state_dict())
        self.model.to(self.device)
        self.data = WeightedPackedShards(
            cfg["train_bin"], cfg["train_weights"], cfg["train_deep"],
            model_cfg.ctx, cfg["micro_batch_size"], self.device
        )
        self.provenance = collect_provenance(cfg)
        self.step = 0
        self.max_steps = cfg["total_tokens"] // cfg["tokens_per_step"]
        self.accum = cfg["tokens_per_step"] // (
            cfg["micro_batch_size"] * model_cfg.ctx
        )
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            target = decay if parameter.requires_grad and parameter.dim() >= 2 else no_decay
            if parameter.requires_grad:
                target.append(parameter)
        self.opt = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8,
            fused=self.device == "cuda"
        )
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.out_dir / "ckpt.pt"
        self.log_path = self.out_dir / "log.jsonl"
        import yaml
        (self.out_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False)
        )

    def _autocast(self):
        if self.device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def train_steps(self, n_steps=None):
        target = min(self.max_steps, self.step + n_steps) if n_steps else self.max_steps
        while self.step < target:
            lr = cosine_lr(self.step, self.cfg["lr"], 300, self.max_steps)
            for group in self.opt.param_groups:
                group["lr"] = lr
            self.opt.zero_grad(set_to_none=True)
            losses = []
            start = time.time()
            for _ in range(self.accum):
                batch = self.data.next_batch()
                with self._autocast():
                    output = self.model(
                        batch.x, batch.targets, batch.weights, batch.deep_weights
                    )
                (output.loss / self.accum).backward()
                losses.append(output.loss.item())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.step += 1
            row = {
                "step": self.step,
                "loss": sum(losses) / len(losses),
                "lr": lr,
                "tok_s": self.cfg["tokens_per_step"] / max(time.time() - start, 1e-9),
                "peak_allocated_bytes": (
                    torch.cuda.max_memory_allocated() if self.device == "cuda" else 0
                ),
            }
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        self.save_ckpt()

    def save_ckpt(self):
        state = {
            "model": self.model.state_dict(), "opt": self.opt.state_dict(),
            "data": self.data.state_dict(), "step": self.step,
            "rng_torch": torch.get_rng_state(), "provenance": self.provenance,
            "rng_cuda": (
                torch.cuda.get_rng_state_all() if self.device == "cuda" else None
            ),
            "cfg": self.cfg,
        }
        temporary = self.ckpt_path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, self.ckpt_path)

    def load_ckpt(self):
        state = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        assert_compatible(self.provenance, state["provenance"])
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        self.data.load_state_dict(state["data"])
        self.step = state["step"]
        torch.set_rng_state(state["rng_torch"].cpu())
        if self.device == "cuda" and state["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
```

- [ ] **Step 5: Add the run command**

```python
# scripts/run_srgm_train.py
#!/usr/bin/env python
import argparse

import yaml

from train.srgm_trainer import SRGMTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    trainer = SRGMTrainer(cfg)
    if args.resume == "auto" and trainer.ckpt_path.exists():
        trainer.load_ckpt()
    trainer.train_steps(args.steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run trainer and provenance tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_provenance.py tests/test_srgm_trainer.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit strict training**

```bash
git add train/provenance.py train/srgm_trainer.py scripts/run_srgm_train.py tests/test_srgm_provenance.py tests/test_srgm_trainer.py
git commit -m "feat: train SRGM with strict provenance"
```

---

### Task 9: Model-Generated Atomic Graph Decoding

**Files:**
- Create: `evals/srgm_generate.py`
- Create: `tests/test_srgm_generate.py`

**Interfaces:**
- Consumes: `SRGM.forward_step()`, graph token grammar, `GraphAction`, `QAItem`, and `AtomicGraphStore`.
- Produces: `GraphDecodeState`, `parse_action()`, `apply_action()`, `decode_graph_item()`, and length-bucketed `decode_graph_items()`.

- [ ] **Step 1: Write failing action and memory-mode tests**

```python
# tests/test_srgm_generate.py
from corpusgen.graph_records import GraphAddress, GraphAction, GraphRow
from corpusgen.records import QAItem
from evals.srgm_generate import (
    GraphDecodeState,
    apply_action,
    parse_action,
)
from organizer.graph_store import AtomicGraphStore
from train.tokenizer import get_tok


def test_parse_fixed_action_tokens():
    tok = get_tok()
    ids = [
        tok.GRAPH_START, tok.SLOTS[1], tok.RELATIONS["r2"],
        tok.DIR_OUT, tok.GRAPH_READ, tok.GRAPH_END
    ]
    assert parse_action(ids, tok) == GraphAction(1, "r2", "out", True, False)


def test_entity_read_updates_selected_slot_in_place():
    row = GraphRow(7, "r2", "out", "entity", "9", (), "world-0")
    store = AtomicGraphStore([row])
    state = GraphDecodeState(slots=[1, 7, None, None])
    result = apply_action(
        state, GraphAction(1, "r2", "out", True, False), store
    )
    assert result == row
    assert state.slots == [1, 9, None, None]


def test_literal_read_leaves_entity_slot_unchanged():
    row = GraphRow(7, "r4", "out", "literal", "1950-01-01", (), "world-0")
    state = GraphDecodeState(slots=[7, None, None, None])
    result = apply_action(
        state, GraphAction(0, "r4", "out", True, False),
        AtomicGraphStore([row])
    )
    assert result == row
    assert state.slots[0] == 7


def test_memory_off_returns_miss_without_updating_slot():
    state = GraphDecodeState(slots=[7, None, None, None])
    result = apply_action(
        state, GraphAction(0, "r0", "out", True, False), None
    )
    assert result is None
    assert state.slots[0] == 7
    assert state.misses == 1
```

- [ ] **Step 2: Run tests and confirm the missing decoder**

Run: `.venv/bin/python -m pytest tests/test_srgm_generate.py -q`

Expected: FAIL with missing `evals.srgm_generate`.

- [ ] **Step 3: Implement strict action parsing and slot semantics**

```python
# evals/srgm_generate.py
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from corpusgen.graph_records import GraphAction, GraphAddress, GraphRow
from corpusgen.graph_trace import serialize_return


@dataclass
class GraphDecodeState:
    slots: list[int | None]
    actions: list[GraphAction] = field(default_factory=list)
    rows: list[GraphRow | None] = field(default_factory=list)
    provisional_answers: list[str] = field(default_factory=list)
    misses: int = 0
    excess_reads: int = 0
    halt_step: int | None = None

    def __post_init__(self):
        if len(self.slots) != 4:
            raise ValueError("exactly four working slots are required")


def parse_action(ids, tok):
    if len(ids) != 6:
        raise ValueError("graph actions require six tokens")
    if ids[0] != tok.GRAPH_START or ids[-1] != tok.GRAPH_END:
        raise ValueError("invalid graph action frame")
    try:
        source_slot = tok.SLOTS.index(ids[1])
        relation = next(name for name, value in tok.RELATIONS.items() if value == ids[2])
    except (ValueError, StopIteration) as error:
        raise ValueError("invalid slot or relation token") from error
    direction = "out" if ids[3] == tok.DIR_OUT else "in" if ids[3] == tok.DIR_IN else None
    if direction is None:
        raise ValueError("invalid direction token")
    terminal = ids[4]
    if terminal not in (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT):
        raise ValueError("invalid graph terminal token")
    return GraphAction(
        source_slot=source_slot,
        relation_id=relation,
        direction=direction,
        read=terminal == tok.GRAPH_READ,
        halt=terminal == tok.GRAPH_HALT,
    )


def apply_action(state, action, store):
    state.actions.append(action)
    if action.halt:
        if state.halt_step is None:
            state.halt_step = len(state.actions)
        state.rows.append(None)
        return None
    if not action.read:
        state.rows.append(None)
        return None
    source_id = state.slots[action.source_slot]
    if source_id is None or store is None:
        state.misses += 1
        state.rows.append(None)
        return None
    row = store.lookup(GraphAddress(source_id, action.relation_id, action.direction))
    if row is None:
        state.misses += 1
    elif row.target_kind == "entity":
        state.slots[action.source_slot] = int(row.target)
    state.rows.append(row)
    return row
```

- [ ] **Step 4: Add constrained six-step decoding**

Add to `evals/srgm_generate.py`:

```python
def _choose(logits, allowed):
    index = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    return allowed[int(logits[index].argmax())]


def _step_token(model, token_id, cache, device):
    value = torch.tensor([[token_id]], dtype=torch.long, device=device)
    logits, cache = model.forward_step(value, cache)
    return logits[0, -1], cache


def _generate_action(model, logits, cache, tok, device):
    ids = [tok.GRAPH_START]
    logits, cache = _step_token(model, ids[-1], cache, device)
    for allowed in (
        list(tok.SLOTS),
        list(tok.RELATIONS.values()),
        [tok.DIR_OUT, tok.DIR_IN],
        [tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT],
        [tok.GRAPH_END],
    ):
        token_id = _choose(logits, allowed)
        ids.append(token_id)
        logits, cache = _step_token(model, token_id, cache, device)
    return parse_action(ids, tok), logits, cache


def _force_tokens(model, ids, cache, device):
    logits = None
    for token_id in ids:
        logits, cache = _step_token(model, token_id, cache, device)
    return logits, cache


def decode_graph_item(model, tok, item, store, device):
    prompt_ids = tok.encode(item.prompt)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache = model.forward_step(prompt, None)
    state = GraphDecodeState(slots=list(item.meta["entity_slots"]))
    encoded_choices = [tok.encode(choice) for choice in item.meta["answer_choices"]]
    if any(len(ids) != 1 for ids in encoded_choices):
        raise ValueError("every answer choice must encode to one token")
    answer_ids = [ids[0] for ids in encoded_choices]
    for step in range(6):
        if state.halt_step is None:
            action, logits, cache = _generate_action(
                model, logits[0, -1], cache, tok, device
            )
            row = apply_action(state, action, store)
        else:
            action = GraphAction(0, "r0", "out", False, True)
            state.actions.append(action)
            state.rows.append(None)
            halt_ids = [
                tok.GRAPH_START, tok.SLOTS[0], tok.RELATIONS["r0"],
                tok.DIR_OUT, tok.GRAPH_HALT, tok.GRAPH_END
            ]
            logits, cache = _force_tokens(model, halt_ids, cache, device)
            row = None
        return_segments = serialize_return(
            row, item.meta["gold_fact_ids"][step] if row is not None else None
        )
        return_ids, _, _ = tok.encode_tagged_segments(return_segments)
        logits, cache = _force_tokens(model, return_ids, cache, device)
        logits, cache = _step_token(model, tok.ANSWER_STATE, cache, device)
        prediction_id = _choose(logits, answer_ids)
        state.provisional_answers.append(tok.decode([prediction_id]).strip())
        logits, cache = _step_token(model, prediction_id, cache, device)
    return state


def decode_graph_items(model, tok, items, store_for_item, device):
    buckets: dict[int, list] = {}
    for item in items:
        buckets.setdefault(len(tok.encode(item.prompt)), []).append(item)
    rows = []
    for length in sorted(buckets):
        for item in buckets[length]:
            rows.append(decode_graph_item(
                model, tok, item, store_for_item(item), device
            ))
    return rows
```

The first `forward_step(prompt, None)` call yields batched logits; the focused
implementation must pass `logits[0, -1]` consistently and add a regression test
that batch-size 1 and equal-length batched decoding produce identical rows.
Never place unequal prompt lengths in one cache prefill.

- [ ] **Step 5: Add six-step, grammar, and batch-invariance tests**

Extend `tests/test_srgm_generate.py` with scripted model fixtures that assert:

```python
def test_halt_still_records_six_steps(scripted_decode):
    state = scripted_decode(first_action="halt")
    assert len(state.actions) == 6
    assert state.halt_step == 1
    assert all(not action.read for action in state.actions[1:])


def test_equal_length_batch_matches_single_item_decode(scripted_batch_decode):
    singles, batched = scripted_batch_decode()
    assert batched == singles
```

- [ ] **Step 6: Run decoder tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_generate.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit graph decoding**

```bash
git add evals/srgm_generate.py tests/test_srgm_generate.py
git commit -m "feat: decode atomic graph actions"
```

---

### Task 10: Equal-Information Metrics and Causal Controls

**Files:**
- Create: `evals/srgm_metrics.py`
- Create: `evals/srgm_controls.py`
- Create: `scripts/run_srgm_evals.py`
- Create: `tests/test_srgm_metrics.py`
- Create: `tests/test_srgm_controls.py`

**Interfaces:**
- Consumes: decoded states, original/counterfactual QA rows, graph snapshots, and checkpoints.
- Produces: exact per-stratum scores, path diagnostics, Wilson burden/leakage bounds, eight causal interventions, and one evaluation summary per memory mode.

- [ ] **Step 1: Write failing metric tests**

```python
# tests/test_srgm_metrics.py
import pytest

from evals.srgm_metrics import (
    counterfactual_pair_accuracy,
    path_metrics,
    wilson_interval,
)


def result(qid, pair_id, correct, actions=()):
    return {
        "qid": qid, "correct": correct, "pred": "yes", "answer": "yes",
        "actions": list(actions), "meta": {"pair_id": pair_id, "world_id": "w0"}
    }


def test_pair_accuracy_requires_both_variants():
    rows = [
        result("p0-o", "p0", True), result("p0-c", "p0", True),
        result("p1-o", "p1", True), result("p1-c", "p1", False),
    ]
    assert counterfactual_pair_accuracy(rows) == pytest.approx(0.5)


def test_pair_metric_rejects_missing_variant():
    with pytest.raises(ValueError, match="exactly two variants"):
        counterfactual_pair_accuracy([result("p0-o", "p0", True)])


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(35, 100)
    assert low < 0.35 < high


def test_path_metrics_report_exact_and_per_hop():
    rows = [{
        "actions": ["a", "b", "c"], "gold_actions": ["a", "b", "x"],
        "correct_referents": [True, True, False], "misses": 1,
        "excess_reads": 2, "halt_step": 4
    }]
    out = path_metrics(rows)
    assert out["full_path_exact"] == 0.0
    assert out["per_hop_accuracy"] == pytest.approx(2 / 3)
    assert out["correct_referent_rate"] == pytest.approx(2 / 3)
```

- [ ] **Step 2: Write failing control-store tests**

```python
# tests/test_srgm_controls.py
from corpusgen.graph_records import GraphAddress, GraphRow
from evals.srgm_controls import (
    IrrelevantSwapStore,
    RelevantSwapStore,
    ShuffledStore,
)
from organizer.graph_store import AtomicGraphStore


def base_store():
    return AtomicGraphStore([
        GraphRow(0, "r0", "out", "entity", "1", (), "w"),
        GraphRow(1, "r0", "out", "entity", "2", (), "w"),
        GraphRow(2, "r0", "out", "entity", "3", (), "w"),
    ])


def test_shuffled_store_preserves_addresses_not_targets():
    store = ShuffledStore(base_store(), seed=7)
    assert len(store) == 3
    assert store.lookup(GraphAddress(0, "r0", "out")).target != "1"


def test_relevant_swap_changes_only_named_address():
    address = GraphAddress(0, "r0", "out")
    store = RelevantSwapStore(base_store(), address, replacement_target="3")
    assert store.lookup(address).target == "3"
    assert store.lookup(GraphAddress(1, "r0", "out")).target == "2"
```

- [ ] **Step 3: Run tests and confirm missing metric/control modules**

Run: `.venv/bin/python -m pytest tests/test_srgm_metrics.py tests/test_srgm_controls.py -q`

Expected: FAIL with missing SRGM evaluation modules.

- [ ] **Step 4: Implement pair, path, and Wilson metrics**

```python
# evals/srgm_metrics.py
from __future__ import annotations

import math
from collections import defaultdict


def counterfactual_pair_accuracy(rows):
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["meta"]["pair_id"]].append(row)
    if any(len(values) != 2 for values in by_pair.values()):
        raise ValueError("every pair must contain exactly two variants")
    return sum(all(value["correct"] for value in values) for values in by_pair.values()) / len(by_pair)


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def path_metrics(rows):
    hop_correct = 0
    hop_total = 0
    referent_correct = 0
    referent_total = 0
    exact = 0
    for row in rows:
        actions = row["actions"]
        gold = row["gold_actions"]
        exact += actions == gold
        hop_total += len(gold)
        hop_correct += sum(a == b for a, b in zip(actions, gold))
        referents = row["correct_referents"]
        referent_total += len(referents)
        referent_correct += sum(referents)
    return {
        "full_path_exact": exact / len(rows),
        "per_hop_accuracy": hop_correct / hop_total,
        "correct_referent_rate": referent_correct / referent_total,
        "miss_rate": sum(row["misses"] for row in rows) / hop_total,
        "mean_excess_reads": sum(row["excess_reads"] for row in rows) / len(rows),
        "mean_halt_step": sum(row["halt_step"] for row in rows) / len(rows),
    }


def assert_expected_counts(rows_by_task, n_pairs=10_000):
    expected = n_pairs * 2
    for task in ("path_composition", "date_ordering", "balanced_equality"):
        actual = len(rows_by_task.get(task, ()))
        if actual != expected:
            raise ValueError(f"{task}: expected {expected} rows, got {actual}")
```

Add recognition scoring through the existing
`evals.natural.loglikelihood_choice_scores()`, exact recall, selector rates,
mask-ledger checks, and shared-text bits-per-byte to this module. Return every
guardrail as `{value, threshold, passed, n}` rather than a bare Boolean.

- [ ] **Step 5: Implement deterministic store interventions**

```python
# evals/srgm_controls.py
from __future__ import annotations

import random

from corpusgen.graph_records import GraphRow
from organizer.graph_store import AtomicGraphStore


class OverlayStore:
    def __init__(self, base, replacement):
        self.base = base
        self.replacement = replacement

    def lookup(self, address):
        if self.replacement is not None and address == self.replacement.address:
            return self.replacement
        return self.base.lookup(address)


class ShuffledStore(AtomicGraphStore):
    def __init__(self, base, seed):
        rows = list(base.rows())
        targets = [(row.target_kind, row.target, row.qualifiers) for row in rows]
        if len(targets) < 2:
            raise ValueError("shuffled control requires at least two rows")
        offset = 1 + random.Random(seed).randrange(len(targets) - 1)
        targets = targets[offset:] + targets[:offset]
        changed = [
            GraphRow(
                row.source_id, row.relation_id, row.direction,
                target_kind, target, qualifiers, row.provenance_id
            )
            for row, (target_kind, target, qualifiers) in zip(rows, targets)
        ]
        super().__init__(changed)


class RelevantSwapStore(AtomicGraphStore):
    def __init__(self, base, address, replacement_target):
        changed = []
        for row in base.rows():
            target = replacement_target if row.address == address else row.target
            changed.append(GraphRow(
                row.source_id, row.relation_id, row.direction, row.target_kind,
                target, row.qualifiers, row.provenance_id
            ))
        super().__init__(changed)


class IrrelevantSwapStore(RelevantSwapStore):
    pass


def relabel_store(base, permutation):
    return AtomicGraphStore([
        GraphRow(
            permutation[row.source_id], row.relation_id, row.direction,
            row.target_kind,
            str(permutation[int(row.target)]) if row.target_kind == "entity" else row.target,
            row.qualifiers, row.provenance_id
        )
        for row in base.rows()
    ])
```

Gold-path replay is implemented as an evaluator mode that force-feeds the
recorded `gold_addresses` and returned rows while still asking the model for
the provisional answer after each step.

- [ ] **Step 6: Add the complete evaluation command**

```python
# scripts/run_srgm_evals.py
#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
import yaml

from corpusgen.graph_records import GraphRow
from evals.srgm_controls import OverlayStore
from evals.srgm_generate import decode_graph_items
from evals.srgm_metrics import (
    assert_expected_counts,
    counterfactual_pair_accuracy,
    path_metrics,
)
from organizer.graph_store import AtomicGraphStore
from train.srgm_model import SRGM, SRGM_PRESETS
from train.tokenizer import get_tok


def load_items(path):
    from corpusgen.records import QAItem
    return [QAItem(**json.loads(line)) for line in Path(path).read_text().splitlines()]


def load_model(run, device):
    cfg = yaml.safe_load(open(run / "config.yaml"))
    model = SRGM(SRGM_PRESETS[cfg["model"]])
    state = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--memory", choices=("on", "off"), required=True)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    run = Path(args.run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(run, device)
    tok = get_tok()
    data = Path(cfg["data_dir"])
    store = AtomicGraphStore.load(data / "eval" / "fresh_graph.jsonl")
    def store_for_item(item):
        if args.memory == "off":
            return None
        changed = item.meta.get("changed_row")
        replacement = GraphRow.from_json(changed) if changed is not None else None
        return OverlayStore(store, replacement)

    rows_by_task = {}
    for task in ("path_composition", "date_ordering", "balanced_equality"):
        items = load_items(data / "eval" / f"{task}.jsonl")
        states = decode_graph_items(
            model, tok, items,
            store_for_item, device
        )
        rows_by_task[task] = _states_to_rows(items, states)
    assert_expected_counts(rows_by_task)
    summary = {
        task: {
            "counterfactual_pair_accuracy": counterfactual_pair_accuracy(rows),
            "path": path_metrics(rows),
        }
        for task, rows in rows_by_task.items()
    }
    _run_guardrails_and_controls(summary, model, tok, cfg, data, args.controls, device)
    out = run / "evals" / f"memory_{args.memory}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
```

Add these conversions and orchestration functions above `main()`:

```python
def _action_json(action):
    return [
        action.source_slot, action.relation_id, action.direction,
        action.read, action.halt
    ]


def _states_to_rows(items, states):
    rows = []
    for item, state in zip(items, states):
        gold_addresses = [
            [int(source), relation, direction]
            for source, relation, direction in item.meta["gold_addresses"]
        ]
        correct_referents = []
        for row, address in zip(state.rows, gold_addresses):
            correct_referents.append(
                row is not None
                and [row.source_id, row.relation_id, row.direction] == address
            )
        prediction = state.provisional_answers[-1]
        rows.append({
            "qid": item.qid,
            "task": item.task,
            "correct": prediction.strip().lower() == item.answer.strip().lower(),
            "pred": prediction,
            "answer": item.answer,
            "actions": [_action_json(action) for action in state.actions],
            "gold_actions": gold_addresses,
            "correct_referents": correct_referents,
            "misses": state.misses,
            "excess_reads": max(
                0,
                sum(action.read for action in state.actions) - len(gold_addresses),
            ),
            "halt_step": state.halt_step or 6,
            "n_steps": len(state.actions),
            "meta": item.meta,
        })
    return rows


def _run_guardrails_and_controls(
    summary, model, tok, cfg, data, controls, device
):
    report = json.loads((data / "report.json").read_text())
    recognition = recognition_guardrail(
        model, tok, data / "eval" / "recognition.jsonl", None, device,
        require_lower_above=0.30 if cfg["condition"] == "dense" else None,
        require_upper_below=0.30 if cfg["condition"] == "split" else None,
    )
    summary["guardrails"] = {
        "mask": mask_ledger_guardrail(report["mask_ledger"]),
        "selector": selector_guardrail(data / "selector_audit.json"),
        "burden_or_leakage": recognition,
        "factual_job": recall_guardrail(model, tok, data, device),
        "language": language_bpb_guardrail(model, tok, data, device),
    }
    if controls:
        summary["controls"] = run_control_suite(
            model=model,
            tok=tok,
            data_dir=data,
            device=device,
            expected_names=(
                "memory_off", "shuffled_store", "relevant_edge_swap",
                "irrelevant_edge_swap", "gold_path_replay", "entity_renaming",
                "no_query", "recursion_depth",
            ),
        )
```

Import the six named guardrail helpers from `evals.srgm_metrics` and
`run_control_suite` from `evals.srgm_controls`. Each helper returns
`{"value": float, "threshold": float, "passed": bool, "n": int}` and raises
on missing data.

- [ ] **Step 7: Run metric and control tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_metrics.py tests/test_srgm_controls.py -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit equal-information evaluation**

```bash
git add evals/srgm_metrics.py evals/srgm_controls.py scripts/run_srgm_evals.py tests/test_srgm_metrics.py tests/test_srgm_controls.py
git commit -m "feat: evaluate SRGM causal controls"
```

---

### Task 11: Seed-Level Statistics, Verdict, and Figures

**Files:**
- Create: `evals/srgm_stats.py`
- Create: `evals/srgm_figures.py`
- Create: `scripts/analyze_srgm.py`
- Create: `tests/test_srgm_stats.py`

**Interfaces:**
- Consumes: per-run memory-ON/OFF summaries and exact per-item rows from Task 10.
- Produces: `pooled_sigma()`, `load_interaction()`, `paired_t_interval()`, `VerdictInputs`, `decide_verdict()`, `analysis.json`, `summary.md`, and figures.

- [ ] **Step 1: Write failing formula and verdict tests**

```python
# tests/test_srgm_stats.py
import pytest

from evals.srgm_stats import (
    VerdictInputs,
    decide_verdict,
    load_interaction,
    paired_t_interval,
    pooled_sigma,
)


def test_pooled_sigma_uses_six_degrees_of_freedom():
    values = {
        50_000: [0.00, 0.01, -0.01],
        200_000: [0.02, 0.03, 0.01],
        800_000: [0.05, 0.06, 0.04],
    }
    assert pooled_sigma(values) == pytest.approx(0.01)


def test_paired_t_interval_for_three_seeds():
    mean, low, high = paired_t_interval([0.02, 0.03, 0.04])
    assert mean == pytest.approx(0.03)
    assert low < mean < high


def test_load_interaction_recovers_positive_slope():
    values = {
        50_000: [0.00, 0.01, 0.00],
        200_000: [0.02, 0.03, 0.02],
        800_000: [0.05, 0.06, 0.05],
    }
    out = load_interaction(values)
    assert out["slope"] > 0 and out["ci_lo"] > 0


def test_validate_requires_every_guardrail():
    inputs = VerdictInputs(
        delta_360=(0.08, 0.09, 0.07),
        pooled_sigma=0.01,
        interaction_ci=(0.01, 0.04),
        stratum_means=(0.07, 0.08, 0.09),
        split_minus_random=(0.03, 0.02, 0.04),
        guardrails={"selector": True, "burden": True, "leakage": True},
    )
    assert decide_verdict(inputs) == "validated"
    failed = VerdictInputs(**{**inputs.__dict__, "guardrails": {"selector": False}})
    assert decide_verdict(failed) == "invalid"
```

- [ ] **Step 2: Run tests and confirm the missing statistics module**

Run: `.venv/bin/python -m pytest tests/test_srgm_stats.py -q`

Expected: FAIL with missing `evals.srgm_stats`.

- [ ] **Step 3: Implement the frozen seed formulas**

```python
# evals/srgm_stats.py
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

T_975 = {2: 4.302652729911275, 5: 2.570581835636314}


def pooled_sigma(by_load):
    numerator = 0.0
    for values in by_load.values():
        if len(values) != 3:
            raise ValueError("every load requires three paired seeds")
        mean = float(np.mean(values))
        numerator += sum((value - mean) ** 2 for value in values)
    return math.sqrt(numerator / 6)


def paired_t_interval(values):
    if len(values) != 3:
        raise ValueError("paired t interval requires three seeds")
    mean = float(np.mean(values))
    standard = float(np.std(values, ddof=1))
    radius = T_975[2] * standard / math.sqrt(3)
    return mean, mean - radius, mean + radius


def load_interaction(by_load):
    rows = []
    for load, values in sorted(by_load.items()):
        for seed, value in enumerate(values):
            rows.append((math.log(load), seed, value))
    design = np.array([
        [1.0, log_load, float(seed == 1), float(seed == 2)]
        for log_load, seed, _ in rows
    ])
    target = np.array([value for _, _, value in rows])
    beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    degrees = len(target) - design.shape[1]
    sigma2 = float(residual @ residual / degrees)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    slope = float(beta[1])
    radius = T_975[degrees] * math.sqrt(float(covariance[1, 1]))
    return {"slope": slope, "ci_lo": slope - radius, "ci_hi": slope + radius}


@dataclass(frozen=True)
class VerdictInputs:
    delta_360: tuple[float, float, float]
    pooled_sigma: float
    interaction_ci: tuple[float, float]
    stratum_means: tuple[float, float, float]
    split_minus_random: tuple[float, float, float]
    guardrails: dict[str, bool]


def decide_verdict(value):
    if not value.guardrails or not all(value.guardrails.values()):
        return "invalid"
    mean_360, _, upper_360 = paired_t_interval(list(value.delta_360))
    margin = max(0.02, 2 * value.pooled_sigma)
    validates = (
        mean_360 > margin
        and all(delta > 0 for delta in value.delta_360)
        and all(delta > 0 for delta in value.stratum_means)
        and value.interaction_ci[0] > 0
        and all(delta > 0.01 for delta in value.split_minus_random)
    )
    if validates:
        return "validated"
    rejects = (
        upper_360 < 0.02
        and value.interaction_ci[1] <= 0
        and max(value.split_minus_random) <= 0.01
    )
    return "rejected" if rejects else "inconclusive"
```

- [ ] **Step 4: Add analysis and figures**

```python
# scripts/analyze_srgm.py
#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from evals.srgm_stats import VerdictInputs, decide_verdict, load_interaction, pooled_sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = _load_complete_runs(Path(args.runs_root))
    _require_expected_run_matrix(runs)
    by_load = _paired_160m_deltas(runs)
    sigma = pooled_sigma(by_load)
    interaction = load_interaction(by_load)
    inputs = VerdictInputs(
        delta_360=tuple(_paired_360m_deltas(runs)),
        pooled_sigma=sigma,
        interaction_ci=(interaction["ci_lo"], interaction["ci_hi"]),
        stratum_means=tuple(_360m_stratum_means(runs)),
        split_minus_random=tuple(_split_minus_random(runs)),
        guardrails=_collect_guardrails(runs),
    )
    result = {
        "verdict": decide_verdict(inputs),
        "pooled_sigma": sigma,
        "interaction": interaction,
        "inputs": inputs.__dict__,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    _write_summary(result, out / "summary.md")
    _write_figures(runs, out)


if __name__ == "__main__":
    main()
```

Add these exact helpers above `main()`:

```python
TASKS = ("path_composition", "date_ordering", "balanced_equality")
LOADS = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}


def _load_complete_runs(root):
    runs = {}
    for directory in sorted(root.iterdir()):
        config_path = directory / "config.yaml"
        on_path = directory / "evals" / "memory_on" / "summary.json"
        off_path = directory / "evals" / "memory_off" / "summary.json"
        if not config_path.exists():
            continue
        if not on_path.exists() or not off_path.exists():
            raise ValueError(f"incomplete memory modes: {directory.name}")
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        key = (cfg["model"], cfg["condition"], cfg["load"], cfg["init_seed"])
        if key in runs:
            raise ValueError(f"duplicate run key: {key}")
        runs[key] = {
            "cfg": cfg,
            "on": json.loads(on_path.read_text()),
            "off": json.loads(off_path.read_text()),
        }
    return runs


def _require_expected_run_matrix(runs):
    expected = set()
    for load in ("n50k", "n200k", "n800k"):
        for condition in ("dense", "split"):
            for seed in range(3):
                expected.add(("srgm160m", condition, load, seed))
    for seed in range(3):
        expected.add(("srgm160m", "random", "n800k", seed))
    for condition in ("dense", "split"):
        for seed in range(3):
            expected.add(("srgm360m", condition, "n1p8m", seed))
    if set(runs) != expected:
        raise ValueError(
            f"run matrix mismatch; missing={sorted(expected - set(runs))}, "
            f"extra={sorted(set(runs) - expected)}"
        )


def _composite(run):
    return sum(
        run["on"][task]["counterfactual_pair_accuracy"] for task in TASKS
    ) / len(TASKS)


def _paired_160m_deltas(runs):
    return {
        LOADS[load]: [
            _composite(runs[("srgm160m", "split", load, seed)])
            - _composite(runs[("srgm160m", "dense", load, seed)])
            for seed in range(3)
        ]
        for load in LOADS
    }


def _paired_360m_deltas(runs):
    return [
        _composite(runs[("srgm360m", "split", "n1p8m", seed)])
        - _composite(runs[("srgm360m", "dense", "n1p8m", seed)])
        for seed in range(3)
    ]


def _360m_stratum_means(runs):
    return [
        sum(
            runs[("srgm360m", "split", "n1p8m", seed)]["on"][task][
                "counterfactual_pair_accuracy"
            ]
            - runs[("srgm360m", "dense", "n1p8m", seed)]["on"][task][
                "counterfactual_pair_accuracy"
            ]
            for seed in range(3)
        ) / 3
        for task in TASKS
    ]


def _split_minus_random(runs):
    return [
        _composite(runs[("srgm160m", "split", "n800k", seed)])
        - _composite(runs[("srgm160m", "random", "n800k", seed)])
        for seed in range(3)
    ]


def _collect_guardrails(runs):
    combined = {}
    for key, run in runs.items():
        for mode in ("on", "off"):
            for name, value in run[mode]["guardrails"].items():
                combined[f"{key}:{mode}:{name}"] = bool(value["passed"])
    return combined


def _write_summary(result, path):
    lines = [
        "# SRGM protected analysis",
        "",
        f"- Verdict: **{result['verdict']}**",
        f"- Pooled seed sigma: {result['pooled_sigma']:.6f}",
        f"- Load interaction: {result['interaction']}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_figures(runs, out):
    from evals.srgm_figures import dose_response_figure, halt_depth_figure
    dose_response_figure(runs, out / "dose_response.png")
    halt_depth_figure(runs, out / "halt_depth.png")
```

Create `evals/srgm_figures.py`:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _composite(run):
    tasks = ("path_composition", "date_ordering", "balanced_equality")
    return sum(
        run["on"][task]["counterfactual_pair_accuracy"] for task in tasks
    ) / len(tasks)


def dose_response_figure(runs, path):
    fig, axis = plt.subplots(figsize=(6, 4))
    loads = {"n50k": 50_000, "n200k": 200_000, "n800k": 800_000}
    for condition, marker in (("dense", "o"), ("split", "s")):
        for load, count in loads.items():
            values = [
                _composite(runs[("srgm160m", condition, load, seed)])
                for seed in range(3)
            ]
            axis.scatter([count] * 3, values, marker=marker, label=condition if load == "n50k" else None)
    axis.set_xscale("log")
    axis.set_xlabel("Distinct entities")
    axis.set_ylabel("Counterfactual-consistent composite")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def halt_depth_figure(runs, path):
    fig, axis = plt.subplots(figsize=(6, 4))
    for condition, marker in (("dense", "o"), ("split", "s")):
        run = runs[("srgm360m", condition, "n1p8m", 0)]
        values = run["on"]["controls"]["recursion_depth"]["mean_halt_by_gold_hop"]
        hops = sorted(int(value) for value in values)
        axis.plot(hops, [values[str(hop)] for hop in hops], marker=marker, label=condition)
    axis.set_xlabel("Gold path length")
    axis.set_ylabel("Mean predicted halt step")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
```

- [ ] **Step 5: Run statistics tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_stats.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit frozen analysis**

```bash
git add evals/srgm_stats.py evals/srgm_figures.py scripts/analyze_srgm.py tests/test_srgm_stats.py
git commit -m "feat: add SRGM decision analysis"
```

---

### Task 12: Protected Manifests and AWS P5 Execution

**Files:**
- Create: `scripts/make_srgm_manifest.py`
- Create: `scripts/check_srgm_throughput.py`
- Create: `cluster/aws/run_srgm_manifest.py`
- Create: `cluster/aws/SRGM.md`
- Create: `tests/test_srgm_manifest.py`

**Interfaces:**
- Consumes: corpus paths, selector hash, model presets, trainer command, and frozen battery sizes.
- Produces: exact build/train manifests, eight independent H100 workers, a 200-step throughput gate, and reproducible launch instructions.

- [ ] **Step 1: Write failing manifest matrix tests**

```python
# tests/test_srgm_manifest.py
from pathlib import Path

from scripts.make_srgm_manifest import make_jobs


def test_initial_matrix_has_27_runs_and_three_seeds(tmp_path):
    jobs = make_jobs("initial", tmp_path / "data", tmp_path / "runs", "selector.pt")
    assert len(jobs) == 27
    assert {job["init_seed"] for job in jobs} == {0, 1, 2}
    assert sum(job["model"] == "srgm160m" for job in jobs) == 21
    assert sum(job["model"] == "srgm360m" for job in jobs) == 6


def test_exact_token_budgets_and_conditions(tmp_path):
    jobs = make_jobs("initial", tmp_path / "data", tmp_path / "runs", "selector.pt")
    for job in jobs:
        expected = 1_599_602_688 if job["model"] == "srgm160m" else 3_599_761_408
        assert job["total_tokens"] == expected
        assert job["tokens_per_step"] == 524_288
        condition = job["condition"]
        assert job["train_weights"].endswith(f"{condition}.weights.bin")


def test_each_pair_shares_data_seed_and_corpus(tmp_path):
    jobs = make_jobs("initial", tmp_path / "data", tmp_path / "runs", "selector.pt")
    paired = {}
    for job in jobs:
        if job["condition"] in ("dense", "split"):
            key = (job["model"], job["load"], job["init_seed"])
            paired.setdefault(key, []).append(job)
    for values in paired.values():
        assert len(values) == 2
        assert values[0]["data_seed"] == values[1]["data_seed"]
        assert values[0]["train_bin"] == values[1]["train_bin"]


def test_later_1b_matrix_has_six_runs(tmp_path):
    jobs = make_jobs("replicate1b", tmp_path / "data", tmp_path / "runs", "selector.pt")
    assert len(jobs) == 6
    assert all(job["total_tokens"] == 9_999_745_024 for job in jobs)
```

- [ ] **Step 2: Run tests and confirm the missing manifest command**

Run: `.venv/bin/python -m pytest tests/test_srgm_manifest.py -q`

Expected: FAIL with missing `scripts.make_srgm_manifest`.

- [ ] **Step 3: Implement the exact run matrix**

```python
# scripts/make_srgm_manifest.py
#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import yaml

SCALE = {
    "srgm30m": (299_892_736, 8, 1.5e-3),
    "srgm160m": (1_599_602_688, 16, 1.5e-3),
    "srgm360m": (3_599_761_408, 8, 1.0e-3),
    "srgm1b": (9_999_745_024, 4, 6.0e-4),
}
LOAD_ENTITIES = {
    "n50k": 50_000, "n200k": 200_000, "n800k": 800_000,
    "n1p8m": 1_800_000, "n5m": 5_000_000,
}


def _source_sha256(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _job(model, condition, load, seed, data_root, out_root, selector):
    total, micro_batch, lr = SCALE[model]
    data_seed = 10_000 + seed
    data_dir = Path(data_root) / f"{load}_ds{data_seed}"
    run_id = f"{model}_{condition}_{load}_s{seed}"
    return {
        "run_id": run_id,
        "model": model,
        "condition": condition,
        "load": load,
        "n_entities": LOAD_ENTITIES[load],
        "init_seed": seed,
        "data_seed": data_seed,
        "train_bin": str(data_dir / "train.bin"),
        "train_weights": str(data_dir / f"{condition}.weights.bin"),
        "train_deep": str(data_dir / "train.deep.bin"),
        "report_json": str(data_dir / "report.json"),
        "selector": str(selector),
        "data_dir": str(data_dir),
        "total_tokens": total,
        "tokens_per_step": 524_288,
        "micro_batch_size": micro_batch,
        "lr": lr,
        "device": "auto",
        "out_dir": str(Path(out_root) / run_id),
        "tokenizer_sha256": _source_sha256(
            [
                "train/tokenizer.py",
                ".tiktoken_cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
                ".tiktoken_cache/6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
            ]
        ),
        "evaluator_revision": _git_revision(),
    }


def make_jobs(stage, data_root, out_root, selector):
    jobs = []
    if stage == "dev":
        for condition in ("dense", "split"):
            jobs.append(_job("srgm30m", condition, "n50k", 0, data_root, out_root, selector))
    elif stage == "initial":
        for load in ("n50k", "n200k", "n800k"):
            for condition in ("dense", "split"):
                for seed in range(3):
                    jobs.append(_job("srgm160m", condition, load, seed, data_root, out_root, selector))
        for seed in range(3):
            jobs.append(_job("srgm160m", "random", "n800k", seed, data_root, out_root, selector))
        for condition in ("dense", "split"):
            for seed in range(3):
                jobs.append(_job("srgm360m", condition, "n1p8m", seed, data_root, out_root, selector))
    elif stage == "replicate1b":
        for condition in ("dense", "split"):
            for seed in range(3):
                jobs.append(_job("srgm1b", condition, "n5m", seed, data_root, out_root, selector))
    else:
        raise ValueError(f"unknown stage: {stage}")
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dev", "initial", "replicate1b"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("protected manifests require a clean worktree")
    jobs = make_jobs(args.stage, args.data_root, args.out_root, args.selector)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    config_dir = manifest.parent / f"{args.stage}_configs"
    config_dir.mkdir(exist_ok=True)
    with open(manifest, "w") as handle:
        for job in jobs:
            path = config_dir / f"{job['run_id']}.yaml"
            path.write_text(yaml.safe_dump(job, sort_keys=False))
            handle.write(str(path) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the 200-step throughput gate**

```python
# scripts/check_srgm_throughput.py
#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def check(log_paths):
    rows = []
    for path in log_paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines())
    measured = [row for row in rows if 50 < row["step"] <= 200]
    if not measured:
        raise ValueError("no measured steps in (50, 200]")
    throughput = sum(row["tok_s"] for row in measured) / len(measured)
    peak = max(row["peak_allocated_bytes"] for row in measured)
    return {
        "mean_tok_s": throughput,
        "peak_allocated_bytes": peak,
        "throughput_pass": throughput >= 60_000,
        "memory_pass": peak <= 72 * 1024 ** 3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    args = parser.parse_args()
    result = check(args.logs)
    print(json.dumps(result, indent=2))
    if not result["throughput_pass"] or not result["memory_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement one queue worker per H100**

```python
# cluster/aws/run_srgm_manifest.py
#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
from pathlib import Path


def load_manifest(path):
    return [
        line.strip() for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def command(config, steps=None):
    value = [".venv/bin/python", "scripts/run_srgm_train.py",
             "--config", config, "--resume", "auto"]
    if steps is not None:
        value.extend(["--steps", str(steps)])
    return value


def run_manifest(configs, gpu_count, dry_run=False, steps=None):
    work = queue.Queue()
    for config in configs:
        work.put(config)
    failures = []

    def worker(gpu):
        while True:
            try:
                config = work.get_nowait()
            except queue.Empty:
                return
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = command(config, steps)
            if dry_run:
                print(f"gpu={gpu} {' '.join(cmd)}")
            else:
                result = subprocess.run(cmd, env=env, check=False)
                if result.returncode:
                    failures.append((config, result.returncode))
            work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=False)
        for gpu in range(gpu_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"failed runs: {failures}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    run_manifest(load_manifest(args.manifest), args.gpus, args.dry_run, args.steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Document the P5 launch sequence**

Create `cluster/aws/SRGM.md` with these exact commands:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
test "$(nvidia-smi -L | wc -l | tr -d ' ')" = "8"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | rg -c 'H100')" = "8"
.venv/bin/python -m pytest tests -q
for seed in 10000 10001 10002; do
  for load in n50k n200k n800k; do
    .venv/bin/python scripts/build_srgm_corpus.py \
      --out-root /local/srgm-data --selector /local/srgm-data/selector.pt \
      --load "$load" --total-tokens 1599602688 --seed "$seed" \
      --bed-file /local/srgm-data/fineweb-edu.jsonl
  done
  .venv/bin/python scripts/build_srgm_corpus.py \
    --out-root /local/srgm-data --selector /local/srgm-data/selector.pt \
    --load n1p8m --total-tokens 3599761408 --seed "$seed" \
    --bed-file /local/srgm-data/fineweb-edu.jsonl
done
.venv/bin/python scripts/make_srgm_manifest.py \
  --stage initial --data-root /local/srgm-data --out-root /local/srgm-runs \
  --selector /local/srgm-data/selector.pt \
  --manifest /local/srgm-runs/initial.tsv
rg 'srgm360m' /local/srgm-runs/initial.tsv \
  > /local/srgm-runs/throughput360.tsv
.venv/bin/python cluster/aws/run_srgm_manifest.py \
  /local/srgm-runs/throughput360.tsv --gpus 8 --steps 200
.venv/bin/python scripts/check_srgm_throughput.py \
  /local/srgm-runs/srgm360m_*/log.jsonl
.venv/bin/python cluster/aws/run_srgm_manifest.py \
  /local/srgm-runs/initial.tsv --gpus 8
```

Append this operational policy:

```markdown
## Protected-run policy

- Use an On-Demand `p5.48xlarge` or an EC2 Capacity Block; Spot is prohibited.
- Stage every corpus and selector artifact on local NVMe before billable
  training starts, then verify hashes against every generated config.
- Sync each atomic checkpoint to durable object storage after creation.
- Treat any missing third paired seed, failed provenance check, or failed
  throughput gate as non-claim-bearing. Do not substitute a different seed.
```

- [ ] **Step 7: Run manifest and launcher dry-run tests**

Run: `.venv/bin/python -m pytest tests/test_srgm_manifest.py -q`

Expected: `4 passed`.

Run:

```bash
: > /tmp/empty-srgm-manifest
.venv/bin/python cluster/aws/run_srgm_manifest.py \
  /tmp/empty-srgm-manifest --gpus 8 --dry-run
```

Expected: exit 0 after creating `/tmp/empty-srgm-manifest` as an empty file.

- [ ] **Step 8: Commit protected operations**

```bash
git add scripts/make_srgm_manifest.py scripts/check_srgm_throughput.py cluster/aws/run_srgm_manifest.py cluster/aws/SRGM.md tests/test_srgm_manifest.py
git commit -m "feat: orchestrate SRGM P5 battery"
```

---

### Task 13: End-to-End Gate, Documentation, and Full Verification

**Files:**
- Create: `scripts/srgm_smoke_test.py`
- Create: `tests/test_srgm_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: every preceding task.
- Produces: a disposable toy corpus, paired two-step checkpoints, memory-factorial evaluation, launch-gate report, and user-facing command entry point.

- [ ] **Step 1: Write the failing end-to-end smoke test**

```python
# tests/test_srgm_smoke.py
from scripts.srgm_smoke_test import run_smoke


def test_srgm_pipeline_end_to_end(tmp_path):
    report = run_smoke(tmp_path, train_steps=2, device="cpu")
    assert report["shared_schedule_identity"]
    assert report["initial_state_hash_equal"]
    assert report["dense_steps"] == report["split_steps"] == 2
    assert report["memory_modes"] == ["off", "on"]
    assert report["six_step_decode"]
    assert report["counterfactual_pairs_complete"]
```

- [ ] **Step 2: Run the smoke test and confirm the missing command**

Run: `.venv/bin/python -m pytest tests/test_srgm_smoke.py -q`

Expected: FAIL with missing `scripts.srgm_smoke_test`.

- [ ] **Step 3: Implement the disposable paired pipeline**

```python
# scripts/srgm_smoke_test.py
#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from corpusgen.srgm_build import SRGMBuildCfg, build_srgm_corpus
from corpusgen.srgm_selector import SelectorArtifact, SelectorCosts, SelectorMLP
from scripts.run_srgm_evals import evaluate_run
from train.srgm_trainer import SRGMTrainer
from train.tokenizer import get_tok


def _bed():
    yield from itertools.cycle([
        "The controlled smoke corpus contains ordinary language.",
        "A second neutral sentence checks stable token packing.",
    ])


def _make_smoke_selector_artifact():
    model = SelectorMLP()
    for parameter in model.parameters():
        parameter.data.zero_()
    return SelectorArtifact.from_model(
        model=model,
        mean=(0.0, 0.0, 0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0, 1.0, 1.0),
        costs=SelectorCosts(),
        threshold=0.5,
        training_data_sha256="0" * 64,
    )


def _config(root, condition, selector, device, train_steps):
    data = root / "data"
    return {
        "run_id": f"smoke-{condition}",
        "model": "srgm30m",
        "condition": condition,
        "load": "smoke",
        "n_entities": 32,
        "init_seed": 7,
        "data_seed": 17,
        "train_bin": str(data / "train.bin"),
        "train_weights": str(data / f"{condition}.weights.bin"),
        "train_deep": str(data / "train.deep.bin"),
        "report_json": str(data / "report.json"),
        "selector": str(selector),
        "data_dir": str(data),
        "total_tokens": train_steps * 2048,
        "tokens_per_step": 2048,
        "micro_batch_size": 2,
        "lr": 1e-3,
        "device": device,
        "out_dir": str(root / "runs" / condition),
        "tokenizer_sha256": "smoke",
        "evaluator_revision": "smoke",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/tmp/memorysplit-srgm-smoke")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    report = run_smoke(args.out, args.steps, args.device)
    print(json.dumps(report, indent=2))
    if not all(value is True or value == ["off", "on"] or value == args.steps
               for value in report.values()):
        raise SystemExit(1)


def run_smoke(root, train_steps=2, device="cpu"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selector = root / "selector.pt"
    artifact = _make_smoke_selector_artifact()
    artifact.save(selector)
    data = root / "data"
    build = build_srgm_corpus(
        SRGMBuildCfg(32, 40_000, 17, n_eval_pairs_per_task=4, world_size=16),
        get_tok(), _bed(), artifact, data
    )
    trainers = {}
    for condition in ("dense", "split"):
        trainer = SRGMTrainer(_config(root, condition, selector, device, train_steps))
        trainer.max_steps = train_steps
        trainer.train_steps()
        trainers[condition] = trainer
    evals = {
        condition: {
            mode: evaluate_run(Path(trainer.out_dir), mode, limit_pairs=4)
            for mode in ("off", "on")
        }
        for condition, trainer in trainers.items()
    }
    return {
        "shared_schedule_identity": build["checks"]["shared_schedule_identity"],
        "initial_state_hash_equal": (
            trainers["dense"].initial_state_sha256
            == trainers["split"].initial_state_sha256
        ),
        "dense_steps": trainers["dense"].step,
        "split_steps": trainers["split"].step,
        "memory_modes": ["off", "on"],
        "six_step_decode": all(
            row["n_steps"] == 6
            for condition in evals.values() for mode in condition.values()
            for row in mode["rows"]
        ),
        "counterfactual_pairs_complete": all(
            mode["pairs_complete"]
            for condition in evals.values() for mode in condition.values()
        ),
    }


if __name__ == "__main__":
    main()
```

Refactor `scripts/run_srgm_evals.py` so the CLI calls this exact public
interface:

```python
def evaluate_run(run: Path, memory_mode: str, limit_pairs: int | None = None):
    if memory_mode not in ("off", "on"):
        raise ValueError("memory_mode must be off or on")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(run, device)
    tok = get_tok()
    data = Path(cfg["data_dir"])
    store = (
        AtomicGraphStore.load(data / "eval" / "fresh_graph.jsonl")
        if memory_mode == "on" else None
    )
    rows = []
    pairs_complete = True
    for task in ("path_composition", "date_ordering", "balanced_equality"):
        items = load_items(data / "eval" / f"{task}.jsonl")
        if limit_pairs is not None:
            items = items[: limit_pairs * 2]
        def item_store(item):
            if store is None:
                return None
            changed = item.meta.get("changed_row")
            replacement = (
                GraphRow.from_json(changed) if changed is not None else None
            )
            return OverlayStore(store, replacement)
        states = decode_graph_items(
            model, tok, items, item_store, device
        )
        task_rows = _states_to_rows(items, states)
        pairs_complete = pairs_complete and len(task_rows) % 2 == 0
        rows.extend(task_rows)
    return {"rows": rows, "pairs_complete": pairs_complete}
```

- [ ] **Step 4: Run the focused smoke test**

Run: `.venv/bin/python -m pytest tests/test_srgm_smoke.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Add the SRGM entry point to README**

Append this concise section without rewriting historical text:

````markdown
## Selective Recursive Graph Memory follow-up

The approved relational follow-up is specified in
`docs/superpowers/specs/2026-07-21-selective-recursive-graph-memory-design.md`.
Its implementation plan and AWS launch sequence are in
`docs/superpowers/plans/2026-07-21-selective-recursive-graph-memory.md` and
`cluster/aws/SRGM.md`. Run the offline gate with:

```bash
.venv/bin/python scripts/srgm_smoke_test.py --device cpu
```
````

- [ ] **Step 6: Run every focused SRGM test**

Run:

```bash
.venv/bin/python -m pytest \
  tests/test_graph_store.py \
  tests/test_graph_trace.py \
  tests/test_srgm_worlds.py \
  tests/test_srgm_selector.py \
  tests/test_srgm_build.py \
  tests/test_srgm_data.py \
  tests/test_srgm_loss.py \
  tests/test_srgm_model.py \
  tests/test_srgm_provenance.py \
  tests/test_srgm_trainer.py \
  tests/test_srgm_generate.py \
  tests/test_srgm_metrics.py \
  tests/test_srgm_controls.py \
  tests/test_srgm_stats.py \
  tests/test_srgm_manifest.py \
  tests/test_srgm_smoke.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 7: Run the complete offline suite**

Run: `.venv/bin/python -m pytest tests -q`

Expected: all tests PASS with no new deselections.

- [ ] **Step 8: Run the command-line smoke gate**

Run: `.venv/bin/python scripts/srgm_smoke_test.py --device cpu`

Expected: exit 0 and a JSON report with every gate set to `true`.

- [ ] **Step 9: Verify no protected output can overwrite legacy artifacts**

Run:

```bash
before="$(git status --porcelain)"
.venv/bin/python scripts/make_srgm_manifest.py \
  --stage initial --data-root /tmp/srgm-data --out-root /tmp/srgm-runs \
  --selector /tmp/selector.pt --manifest /tmp/srgm-initial.tsv
test "$(wc -l < /tmp/srgm-initial.tsv | tr -d ' ')" = "27"
test "$(git status --porcelain)" = "$before"
```

Expected: 27 SRGM configs under `/tmp`; repository status is unchanged.

- [ ] **Step 10: Commit the integrated gate and documentation**

```bash
git add scripts/srgm_smoke_test.py tests/test_srgm_smoke.py README.md
git commit -m "docs: add SRGM execution gate"
```

---

## Final Execution Gate

Before any protected corpus build or GPU reservation:

1. Confirm all task commits are present and the worktree is clean.
2. Freeze and hash the selector artifact.
3. Build all three data seeds per load and verify every report gate.
4. Run the 30M paired development gate; both arms must exceed 75% in every
   primary stratum.
5. Run the 160M and 360M 200-step P5 probes; require 360M throughput at or
   above 60,000 raw tokens/s/GPU and peak allocation at or below 72GB.
6. Verify checkpoint/resume reproduces the next loss within `1e-5`.
7. Only then generate the protected 27-run manifest.

## Execution Handoff

Plan implementation uses one of two workflows:

1. **Subagent-Driven (recommended):** dispatch a fresh implementation subagent
   for each task, followed by specification and code-quality review.
2. **Inline Execution:** execute tasks in this session with
   `superpowers:executing-plans`, in reviewable batches.
