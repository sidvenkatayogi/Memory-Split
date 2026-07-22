# Relational MemorySplit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable Dense-versus-Split experiment in which every claim-bearing model is the existing standard GPT and the only A/B training difference is direct target weight on cost-routed relational fact payloads.

**Architecture:** Reuse `train.model.GPT`, `train.trainer.Trainer`, and the existing tokenizer/optimizer/checkpoint path. Add an exact atomic graph, six serialized autoregressive graph-action steps, one shared corpus with three target-weight sidecars, equal-information evaluation, and platform-neutral manifests launched by FarmShare Slurm or an AWS H100 worker queue.

**Tech Stack:** Python 3.12, PyTorch >=2.6, NumPy >=1.26, tiktoken >=0.7, PyYAML >=6.0, pytest >=8.0, FarmShare L40S, AWS EC2 `p5.48xlarge`.

## Global Constraints

- The approved spec is `docs/superpowers/specs/2026-07-21-selective-recursive-graph-memory-design.md`.
- Dense, Split, and random-mask conditions instantiate `train.model.GPT`; do not create another model or trainer class.
- Use vocabulary size 50,304, context 1,024, and graph IDs 50,261–50,291.
- One graph read returns exactly one row or `MISS`; the store performs no search, ranking, path composition, or answer computation.
- Every reasoning record has six action slots and six provisional answer slots; `HALT` is followed by no-op slots.
- All paired conditions share exact token IDs, record order, packing, model parameters, initialization, optimizer, and decode budget.
- Dense weights every target by one. Split zeros only direct cost-routed payload targets. Random-mask preserves factual loss and masks equal-mass matched non-factual spans.
- Weighted loss divides by all original target positions.
- Use the fixed 45% natural / 30% graph / 25% reasoning mixture.
- Initial protected matrix: 15 FarmShare 160M jobs plus 6 AWS-preferred 360M jobs, all with three seeds.
- No protected run starts until local smoke, 29M learnability, route, burden, leakage, pairing, provenance, and platform preflight gates pass.

## Current Implemented Foundation

These commits are complete and retained:

- `db7e3d4` — immutable graph contracts and atomic store.
- `14ddc50` — graph-control token protocol and serializers.
- `f6ddd3a` — streaming graph worlds and counterfactual oracles.

Task 1 below finishes the two review fixes preserved in
`stash@{0}: wip: task 3 review fixes before simplification`.

---

## File Map

### Retained/new relational files

- `corpusgen/graph_records.py`
- `organizer/graph_store.py`
- `corpusgen/graph_trace.py`
- `corpusgen/srgm_worlds.py`
- `corpusgen/relational_build.py`
- `evals/relational_generate.py`
- `evals/relational_metrics.py`
- `evals/relational_stats.py`

### Existing training files modified

- `train/model.py` — optional target weights and standard 360M preset.
- `train/data.py` — optional weight sidecar with a backward-compatible weighted batch method.
- `train/trainer.py` — select weighted versus legacy batches without changing optimizer/model family.

### Commands and platform files

- `scripts/build_relational_corpus.py`
- `scripts/run_relational_evals.py`
- `scripts/analyze_relational.py`
- `scripts/relational_smoke_test.py`
- `scripts/package_relational_run.py`
- `scripts/make_relational_manifest.py`
- `scripts/platform_preflight.py`
- `cluster/slurm/relational_train.sbatch`
- `cluster/aws/run_relational_manifest.py`
- `cluster/RELATIONAL-RUNBOOK.md`

---

### Task 1: Finish and Re-Review Streaming Graph Worlds

**Files:**
- Modify: `corpusgen/srgm_worlds.py`
- Modify: `tests/test_srgm_worlds.py`

**Interfaces:**
- Produces globally unique functional addresses, deterministic rebalanced world chunks, evidence-changing counterfactual pairs, and bounded 1/2/4-hop record streams.

- [ ] **Step 1: Restore the interrupted review fixes**

```bash
stash_ref="$(
  git stash list --format='%gd %s' |
  awk '/wip: task 3 review fixes before simplification/{print $1; exit}'
)"
test -n "$stash_ref"
git stash apply "$stash_ref"
```

Expected: only `corpusgen/srgm_worlds.py` and `tests/test_srgm_worlds.py` are modified.

- [ ] **Step 2: Verify the global-address regression test**

The test must construct two public `generate_world()` calls with different
`world_id` values and no explicit offset:

```python
def test_public_world_ids_have_disjoint_addresses():
    first = generate_world(3, WorldConfig(n_entities=32, seed=7))
    second = generate_world(4, WorldConfig(n_entities=32, seed=7))
    store = AtomicGraphStore(
        [fact.row for fact in first.facts]
        + [fact.row for fact in second.facts]
    )
    assert len(store) == 2 * 32 * 6
```

- [ ] **Step 3: Verify deterministic tail rebalancing**

```python
@pytest.mark.parametrize("n_entities,expected", [
    (65, [49, 16]),
    (70, [54, 16]),
    (127, [63, 64]),
])
def test_world_stream_never_emits_undersized_tail(n_entities, expected):
    worlds = list(iter_worlds(n_entities, world_size=64, seed=11))
    assert [len(world.entity_names) for world in worlds] == expected
    assert sum(len(world.entity_names) for world in worlds) == n_entities
    assert all(len(world.entity_names) >= 16 for world in worlds)
```

- [ ] **Step 4: Verify every reasoning family and fixed trace shape**

Consume records until all three tasks are observed. Assert for each:

```python
assert len(record.meta["actions"]) == 6
assert len(record.meta["provisional_answers"]) == 6
halt_index = next(i for i, action in enumerate(record.meta["actions"]) if action.halt)
assert all(action.halt is False and action.read is False
           for action in record.meta["actions"][halt_index + 1:])
for payload, neutral in record.meta["payload_control_pairs"]:
    assert len(tok.encode(payload)) == len(tok.encode(neutral))
```

- [ ] **Step 5: Run focused and full tests**

Run:

```bash
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python \
  -m pytest tests/test_srgm_worlds.py -q
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python \
  -m pytest tests -q
```

Expected: focused and full suites PASS with pristine output.

- [ ] **Step 6: Commit and re-review**

```bash
git add corpusgen/srgm_worlds.py tests/test_srgm_worlds.py
git commit -m "fix: enforce relational world invariants"
```

---

### Task 2: Frozen Cost Policy and Shared Corpus Builder

**Files:**
- Create: `corpusgen/relational_build.py`
- Create: `scripts/build_relational_corpus.py`
- Create: `tests/test_relational_build.py`

**Interfaces:**
- Consumes: streaming worlds/records and tokenizer.
- Produces: `RoutePolicy`, one `train.bin`, `dense.weights.bin`, `split.weights.bin`, `random.weights.bin`, graph/policy/schedule manifests, and held-out eval sets.

- [ ] **Step 1: Write failing route-policy tests**

```python
from corpusgen.relational_build import FactCost, calibrate_write_cost


def test_write_cost_is_selected_without_semantic_labels():
    facts = [
        FactCost("a", entropy=12, exposures=1, expected_reads=1, expected_hops=1),
        FactCost("b", entropy=8, exposures=1, expected_reads=1, expected_hops=1),
        FactCost("c", entropy=1, exposures=16, expected_reads=20, expected_hops=2),
        FactCost("d", entropy=1, exposures=16, expected_reads=20, expected_hops=2),
    ]
    policy = calibrate_write_cost(facts)
    assert 0.40 <= policy.route_rate(facts) <= 0.60
    assert policy.is_external(facts[0])
    assert not policy.is_external(facts[-1])
```

- [ ] **Step 2: Implement the deterministic policy**

```python
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class FactCost:
    fact_id: str
    entropy: float
    exposures: int
    expected_reads: float
    expected_hops: float


@dataclass(frozen=True)
class RoutePolicy:
    write_cost: float
    read_cost: float = 0.25
    hop_cost: float = 0.25

    def is_external(self, fact: FactCost) -> bool:
        predict = fact.entropy / max(fact.exposures, 1)
        external = (
            self.write_cost
            + self.read_cost * fact.expected_reads
            + self.hop_cost * fact.expected_hops
        )
        return predict > external

    def route_rate(self, facts) -> float:
        return sum(self.is_external(fact) for fact in facts) / len(facts)

    def sha256(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


def calibrate_write_cost(facts) -> RoutePolicy:
    candidates = [RoutePolicy(value) for value in (0.25, 0.5, 1, 2, 4, 8)]
    valid = [
        policy for policy in candidates
        if 0.40 <= policy.route_rate(facts) <= 0.60
    ]
    if not valid:
        raise ValueError("no write cost yields a 40–60% route rate")
    return min(valid, key=lambda policy: (
        abs(policy.route_rate(facts) - 0.50),
        policy.write_cost,
    ))
```

- [ ] **Step 3: Write failing shared-stream tests**

Build a 64-entity, 60k-token fixture and assert:

```python
tokens = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
dense = np.memmap(out / "dense.weights.bin", dtype=np.uint8, mode="r")
split = np.memmap(out / "split.weights.bin", dtype=np.uint8, mode="r")
random = np.memmap(out / "random.weights.bin", dtype=np.uint8, mode="r")
assert len(tokens) == len(dense) == len(split) == len(random)
assert (dense == 1).all()
assert report["checks"]["external_payload_coverage"]
assert report["checks"]["random_mass_within_1pct"]
assert report["checks"]["random_span_histogram_within_1pct"]
assert report["checks"]["mixture_within_1pct"]
assert report["checks"]["schedule_hash_stable"]
```

- [ ] **Step 4: Implement one-pass streaming output**

`build_relational_corpus()` must:

1. stream graph worlds and write `graph.jsonl`;
2. compute and freeze `route-policy.json`;
3. choose the component once per record from the 45/30/25 deficit schedule;
4. encode each record once;
5. write token IDs and all three weight arrays in the same loop;
6. emit a mask ledger and SHA-256 manifests; and
7. write original/counterfactual fresh-graph eval JSONL.

Use a writer with this public method:

```python
def add(
    self,
    component: str,
    token_ids: np.ndarray,
    spans: list[EncodedSpan],
    policy: RoutePolicy,
    rng: random.Random,
) -> None:
    self.token_file.write(token_ids.tobytes())
    for condition in ("dense", "split", "random"):
        weights = derive_weights(condition, spans, policy, rng)
        self.weight_files[condition].write(weights.tobytes())
```

Never retain all worlds, records, or token chunks in memory.

- [ ] **Step 5: Add the corpus command**

```bash
.venv/bin/python scripts/build_relational_corpus.py \
  --out DATA_ROOT/n50k_ds10000 \
  --entities 50000 \
  --tokens 1599602688 \
  --data-seed 10000 \
  --bed-jsonl DATA_ROOT/fineweb-edu.jsonl
```

All paths in generated manifests are relative to the corpus root.

- [ ] **Step 6: Run and commit**

```bash
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python \
  -m pytest tests/test_relational_build.py -q
git add corpusgen/relational_build.py scripts/build_relational_corpus.py \
  tests/test_relational_build.py
git commit -m "feat: build paired relational corpora"
```

---

### Task 3: Weighted Loss in the Existing GPT/Trainer

**Files:**
- Modify: `train/model.py`
- Modify: `train/data.py`
- Modify: `train/trainer.py`
- Modify: `tests/test_model.py`
- Modify: `tests/test_data.py`
- Modify: `tests/test_trainer.py`

**Interfaces:**
- Adds optional target weights while preserving every legacy call and checkpoint.
- Adds standard `d360m`; no new model/trainer class.

- [ ] **Step 1: Write failing all-position loss test**

```python
def test_target_weights_normalize_by_all_positions():
    torch.manual_seed(0)
    model = tiny()
    x = torch.randint(0, 100, (1, 8))
    targets = torch.randint(0, 100, (1, 8))
    weights = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.float32)
    logits, loss = model(x, targets, target_weights=weights)
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    assert torch.allclose(loss, (per_token * weights).sum() / targets.numel())
```

- [ ] **Step 2: Extend `GPT.forward()` without changing architecture**

```python
def forward(self, idx, targets=None, target_weights=None):
    # existing embedding and block forward remains unchanged
    loss = None
    if targets is not None:
        flat = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(targets)
        if target_weights is None:
            valid = targets.ne(-100)
            loss = flat[valid].mean()
        else:
            if target_weights.shape != targets.shape:
                raise ValueError("target_weights shape must match targets")
            valid_weights = target_weights * targets.ne(-100)
            loss = (flat * valid_weights).sum() / targets.numel()
    return logits, loss
```

- [ ] **Step 3: Add standard 360M preset**

```python
PRESETS["d360m"] = GPTConfig(n_layer=20, n_head=16, d_model=1024, ctx=1024)
```

Test parameter count:

```python
assert GPT(PRESETS["d360m"]).num_params() == 356_033_536
```

Use an analytical parameter-count helper or meta-device construction so the
unit test does not allocate the full model on CPU.

- [ ] **Step 4: Add backward-compatible weighted batches**

Keep `PackedShards.next_batch()` unchanged. Add:

```python
def next_weighted_batch(self):
    x, targets = self.next_batch()
    if self.target_weights is None:
        return x, targets, torch.ones_like(targets, dtype=torch.float32)
    weights = self._aligned_next_token_weights_for_last_batch()
    return x, targets, weights
```

The constructor accepts optional `weights_path`; legacy mask behavior remains
unchanged.

- [ ] **Step 5: Branch only the trainer batch call**

```python
if self.cfg.get("train_weights"):
    x, y, weights = self.data.next_weighted_batch()
    _, loss = self.model(x, y, target_weights=weights)
else:
    x, y = self.data.next_batch()
    _, loss = self.model(x, y)
```

Do not change AdamW, learning-rate schedule, gradient accumulation, snapshots,
or legacy checkpoint keys.

- [ ] **Step 6: Verify compatibility and commit**

```bash
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python -m pytest \
  tests/test_model.py tests/test_data.py tests/test_trainer.py -q
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python -m pytest tests -q
git add train/model.py train/data.py train/trainer.py \
  tests/test_model.py tests/test_data.py tests/test_trainer.py
git commit -m "feat: train GPT with target weights"
```

---

### Task 4: Atomic Graph Evaluation and Frozen Statistics

**Files:**
- Create: `evals/relational_generate.py`
- Create: `evals/relational_metrics.py`
- Create: `evals/relational_stats.py`
- Create: `scripts/run_relational_evals.py`
- Create: `scripts/analyze_relational.py`
- Create: `tests/test_relational_generate.py`
- Create: `tests/test_relational_metrics.py`
- Create: `tests/test_relational_stats.py`

**Interfaces:**
- Consumes standard `GPT.forward_step()`, graph-action tokens, eval JSONL, and atomic stores.
- Produces memory ON/OFF rows, counterfactual pair scores, guardrails, difference-in-differences, and verdict.

- [ ] **Step 1: Write action-state tests**

```python
def test_entity_and_literal_slot_updates():
    entity = GraphRow(7, "r0", "out", "entity", "9", (), "w")
    literal = GraphRow(9, "r4", "out", "literal", "1950-01-01", (), "w")
    state = GraphDecodeState([7, None, None, None])
    assert apply_action(state, GraphAction(0, "r0", "out", True, False),
                        AtomicGraphStore([entity, literal])) == entity
    assert state.slots[0] == 9
    assert apply_action(state, GraphAction(0, "r4", "out", True, False),
                        AtomicGraphStore([entity, literal])) == literal
    assert state.slots[0] == 9


def test_memory_off_is_miss_and_halt_keeps_six_slots(scripted_model, tok):
    item = fixture_item()
    result = decode_item(scripted_model, tok, item, store=None)
    assert result.misses >= 1
    assert len(result.actions) == 6
    halt = next(i for i, action in enumerate(result.actions) if action.halt)
    assert all(not action.read for action in result.actions[halt + 1:])


def test_counterfactual_overlay_changes_only_one_row(base_store):
    replacement = changed_supporting_row()
    overlay = OverlayStore(base_store, replacement)
    assert overlay.lookup(replacement.address) == replacement
    other = next(row for row in base_store.rows()
                 if row.address != replacement.address)
    assert overlay.lookup(other.address) == other


def test_equal_length_batch_matches_single_decode(scripted_model, tok):
    items = [fixture_item("a"), fixture_item("b")]
    singles = [decode_item(scripted_model, tok, item, fixture_store())
               for item in items]
    batched = decode_items(scripted_model, tok, items, fixture_store())
    assert batched == singles
```

- [ ] **Step 2: Implement constrained fixed grammar**

At each of six steps constrain only syntactic token classes:

```python
ALLOWED = (
    tok.SLOTS,
    tuple(tok.RELATIONS.values()),
    (tok.DIR_OUT, tok.DIR_IN),
    (tok.GRAPH_READ, tok.GRAPH_NOOP, tok.GRAPH_HALT),
    (tok.GRAPH_END,),
)
```

The model still chooses slot, relation, direction, and terminal. On entity
reads, replace the selected slot with the returned target. On literal reads,
leave the slot unchanged. Counterfactual rows overlay exactly one address.

- [ ] **Step 3: Implement exact metrics**

```python
def counterfactual_pair_accuracy(rows) -> float:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["pair_id"], []).append(row)
    if any(len(pair) != 2 for pair in grouped.values()):
        raise ValueError("every pair requires original and counterfactual rows")
    return sum(all(row["correct"] for row in pair)
               for pair in grouped.values()) / len(grouped)


def path_metrics(rows) -> dict:
    gold = sum(len(row["gold_actions"]) for row in rows)
    hop_correct = sum(
        predicted == expected
        for row in rows
        for predicted, expected in zip(row["actions"], row["gold_actions"])
    )
    return {
        "full_path_exact": sum(
            row["actions"] == row["gold_actions"] for row in rows
        ) / len(rows),
        "per_hop_accuracy": hop_correct / gold,
        "miss_rate": sum(row["misses"] for row in rows) / gold,
    }


def wilson_interval(successes, total):
    if total <= 0:
        raise ValueError("total must be positive")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * (
        p * (1 - p) / total + z * z / (4 * total * total)
    ) ** 0.5 / denominator
    return center - radius, center + radius


def recognition_accuracy(score_choices, items):
    correct = 0
    for item in items:
        scores = score_choices(item["prompt"], item["choices"])
        correct += max(range(len(scores)), key=scores.__getitem__) == item["answer_index"]
    low, high = wilson_interval(correct, len(items))
    return {"accuracy": correct / len(items), "ci_lo": low,
            "ci_hi": high, "n": len(items)}


def shared_text_bpb(total_nll_nats, total_utf8_bytes):
    if total_utf8_bytes <= 0:
        raise ValueError("held-out text must contain bytes")
    return total_nll_nats / (total_utf8_bytes * math.log(2))
```

Raise on duplicate/missing pair variants or unexpected counts.

- [ ] **Step 4: Implement seed-level verdict**

Use three paired seeds. Compute:

```python
dose_effect = [
    (split_high[s] - dense_high[s]) - (split_low[s] - dense_low[s])
    for s in range(3)
]
```

Implement paired t intervals, pooled seed sigma, stratum means, random-mask
contrast, and the exact validate/reject/invalid rules from spec §7.4.

- [ ] **Step 5: Run and commit**

```bash
/Users/stephenzhang/Documents/MemorySplit/.venv/bin/python -m pytest \
  tests/test_relational_generate.py \
  tests/test_relational_metrics.py \
  tests/test_relational_stats.py -q
git add evals/relational_generate.py evals/relational_metrics.py \
  evals/relational_stats.py scripts/run_relational_evals.py \
  scripts/analyze_relational.py tests/test_relational_generate.py \
  tests/test_relational_metrics.py tests/test_relational_stats.py
git commit -m "feat: evaluate relational memory split"
```

---

### Task 5: Local Smoke and Portable Bundle

**Files:**
- Create: `scripts/relational_smoke_test.py`
- Create: `scripts/package_relational_run.py`
- Create: `tests/test_relational_smoke.py`
- Create: `tests/test_relational_bundle.py`

**Interfaces:**
- Produces a two-step local run and a deterministic uploadable tarball with no absolute runtime paths.

- [ ] **Step 1: Write failing smoke test**

```python
def test_local_pipeline(tmp_path):
    report = run_smoke(tmp_path, steps=2, device="cpu")
    assert report == {
        "shared_stream": True,
        "dense_steps": 2,
        "split_steps": 2,
        "resume_exact": True,
        "memory_modes": ["off", "on"],
        "pairs_complete": True,
    }
```

- [ ] **Step 2: Implement the smoke command**

The command builds 32 entities and 40k tokens, trains Dense and Split from the
same state dict for two steps, resumes one checkpoint for one repeated batch,
and evaluates four pairs per task in both memory modes.

```bash
.venv/bin/python scripts/relational_smoke_test.py \
  --device cpu --out /tmp/relational-smoke
```

- [ ] **Step 3: Write failing bundle test**

```python
def test_bundle_is_relative_and_hash_complete(tmp_path):
    archive = package_run(tmp_path / "bundle.tar.gz")
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
        assert "manifest.json" in names
        assert "route-policy.json" in names
        assert "configs/160m.tsv" in names
        assert "configs/360m.tsv" in names
        for member in bundle.getmembers():
            assert not member.name.startswith("/")
            assert ".." not in Path(member.name).parts
```

- [ ] **Step 4: Package exact portable members**

`manifest.json` records member SHA-256, Git revision, expected run counts
`{"160m": 15, "360m": 6}`, and required environment variables
`["DATA_ROOT", "OUT_ROOT"]`.

Reject a dirty worktree or config containing `/Users/`, `/scratch/`, `s3://`,
or an absolute path.

- [ ] **Step 5: Run and commit**

```bash
.venv/bin/python -m pytest \
  tests/test_relational_smoke.py tests/test_relational_bundle.py -q
.venv/bin/python scripts/relational_smoke_test.py --device cpu
.venv/bin/python scripts/package_relational_run.py \
  --out artifacts/relational-run.tar.gz
git add scripts/relational_smoke_test.py scripts/package_relational_run.py \
  tests/test_relational_smoke.py tests/test_relational_bundle.py
git commit -m "feat: package portable relational tests"
```

---

### Task 6: FarmShare and AWS Launch Readiness

**Files:**
- Create: `scripts/make_relational_manifest.py`
- Create: `scripts/platform_preflight.py`
- Create: `cluster/slurm/relational_train.sbatch`
- Create: `cluster/aws/run_relational_manifest.py`
- Create: `cluster/RELATIONAL-RUNBOOK.md`
- Create: `tests/test_relational_manifest.py`
- Create: `tests/test_platform_preflight.py`

**Interfaces:**
- Produces the same relative YAML configs for both platforms; only launchers differ.

- [ ] **Step 1: Test exact run matrices**

```python
def test_160m_manifest_has_15_runs():
    jobs = make_jobs("160m")
    assert len(jobs) == 15
    assert {job["seed"] for job in jobs} == {0, 1, 2}


def test_360m_manifest_has_6_runs():
    jobs = make_jobs("360m")
    assert len(jobs) == 6
    assert all(job["model"] == "d360m" for job in jobs)


def test_configs_use_environment_roots_only():
    for job in make_jobs("160m") + make_jobs("360m"):
        rendered = yaml.safe_dump(job)
        assert "/Users/" not in rendered
        assert "/scratch/" not in rendered
```

- [ ] **Step 2: Implement relative configs**

Each config stores:

```yaml
data_rel: n800k_ds10000
out_rel: d160m_split_n800k_s0
```

The launcher resolves them against `DATA_ROOT` and `OUT_ROOT` immediately
before invoking `scripts/run_train.py`.

- [ ] **Step 3: Implement platform preflight**

Local mode checks dependencies, smoke bundle, and hashes.

FarmShare mode additionally checks:

- at least one CUDA L40S;
- writable scratch with 500GB free;
- Slurm commands available; and
- checkpoint/resume on a 2-step fixture.

AWS mode additionally checks:

- exactly eight H100 GPUs;
- 72GB usable memory per GPU;
- local NVMe with 1TB free;
- 200-step 360M throughput at or above 60k raw tokens/s/GPU after step 50;
- exact next-loss resume within `1e-5`; and
- On-Demand/Capacity-Block declaration, never Spot.

- [ ] **Step 4: Implement launchers**

FarmShare:

```bash
sbatch --export=ALL,CONFIG_REL="$config" \
  cluster/slurm/relational_train.sbatch
```

AWS:

```bash
.venv/bin/python cluster/aws/run_relational_manifest.py \
  configs/360m.tsv --gpus 8 --steps 200
.venv/bin/python scripts/platform_preflight.py \
  --platform aws --runs-root "$OUT_ROOT"
.venv/bin/python cluster/aws/run_relational_manifest.py \
  configs/360m.tsv --gpus 8
```

Use one worker queue per visible GPU. A failed config is recorded and makes the
launcher exit nonzero after remaining independent jobs checkpoint.

- [ ] **Step 5: Write the runbook**

The runbook has three explicit sections:

1. **Local required:** unit suite, smoke, bundle.
2. **FarmShare required:** 29M pilot and all 15 160M jobs.
3. **AWS preferred:** six concurrent 360M jobs.

It includes corpus staging, hash checks, manifest generation, evaluation,
analysis, checkpoint sync, and the rule that missing third seeds are never
substituted.

- [ ] **Step 6: Final verification and commit**

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/relational_smoke_test.py --device cpu
.venv/bin/python scripts/package_relational_run.py \
  --out artifacts/relational-run.tar.gz
.venv/bin/python scripts/platform_preflight.py \
  --platform local --bundle artifacts/relational-run.tar.gz
git add scripts/make_relational_manifest.py scripts/platform_preflight.py \
  cluster/slurm/relational_train.sbatch \
  cluster/aws/run_relational_manifest.py cluster/RELATIONAL-RUNBOOK.md \
  tests/test_relational_manifest.py tests/test_platform_preflight.py
git commit -m "feat: ready relational runs for cloud and cluster"
```

---

## Final Execution Gate

Before protected data or GPU jobs:

1. All tests pass locally.
2. The two-step smoke report is entirely green.
3. The portable bundle verifies all member hashes and relative paths.
4. The frozen route policy is 40–60% external and passes 80% tail/structure
   guardrails.
5. The 29M learnability pair exceeds 75% in all three reasoning strata.
6. FarmShare preflight passes before the 15-job 160M manifest.
7. AWS preflight and 200-step throughput pass before the six-job 360M
   manifest.

## Execution Handoff

Continue with Subagent-Driven Development:

1. Task 1 resumes the preserved world-invariant fixes.
2. Every task receives implementation and task-scoped review.
3. After Task 6, run one frontier whole-branch review and full fresh
   verification before choosing merge/push/keep/discard.
