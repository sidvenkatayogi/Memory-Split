# Memory Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 2, 3, and 5 are independent of each other and are dispatched to parallel subagents after Task 1 lands.

**Goal:** Build and run the fact-load dose-response experiment from
`docs/superpowers/specs/2026-07-17-memory-split-design.md` (read it first;
it is authoritative for scientific choices).

**Architecture:** One Python package tree (corpusgen / organizer / train /
evals), pure-Python seeded generators, memmap token+mask shards, a minimal
GPT trainer, generative evals with lookup interception, paired statistics.
Slurm scaffolding for Stanford FarmShare adapted from `ShrinkMuon/cluster/`.

**Tech stack:** Python 3.12, PyTorch >= 2.6, tiktoken (GPT-2 BPE), numpy,
HF `datasets` (FineWeb-Edu bed + natural benchmarks), pytest, matplotlib.

## Global constraints

- Determinism everywhere: every generator takes an explicit integer seed;
  same seed => byte-identical output. Use `random.Random(seed)` /
  `numpy.random.Generator(PCG64(seed))`; never global RNG.
- Loss-mask convention: `mask == 1` means loss ON; `mask == 0` means loss
  OFF (fact values). In tensors, masked-off targets become label `-100`.
- Special tokens (exact ids): `<|db_start|>`=50257, `<|db_retrieve|>`=50258,
  `<|db_end|>`=50259, `<|eot|>`=50260; vocab padded to 50304.
- Lookup wrapping (exact): dense `... majored in Communications at ...` =>
  split `... majored in <|db_start|>Kai Nakamura, major<|db_retrieve|> Communications<|db_end|> at ...`
  — query text `f"{name}, {relation}"`, value segment `f" {value}"`
  (leading space), only the value segment masked; special tokens and query
  get loss.
- Answer format for every generative task: prompt ends with `Reasoning:`;
  training doc contains CoT then `\nAnswer: {answer}<|eot|>`; scorers parse
  the text after the LAST `Answer:`.
- Tests must run offline and in < 1 min (`python -m pytest tests -q`);
  anything needing network or a GPU is behind `-m slow` or a script.
- Token shares (dense rendering): bed 62%, bios 23%, reasoning 12%
  (igsm 7% + deduction 5%), factqa 3%. Both arms match per-component token
  budgets within 1% (split therefore gets fewer bio exposures).
- File formats: docs as JSONL of segment lists; token shards as uint16
  `.bin` + parallel uint8 `.mask.bin`; eval items as JSONL; run outputs
  under `outputs/{run_id}/` with `config.yaml`, `ckpt.pt`, `snapshots/`,
  `log.jsonl`, `evals/*.jsonl`.

---

### Task 1: Shared core — records, tokenizer, organizer (owner: lead)

**Files:**
- Create: `corpusgen/__init__.py`, `corpusgen/records.py`
- Create: `train/__init__.py`, `train/tokenizer.py`
- Create: `organizer/__init__.py`, `organizer/store.py`
- Test: `tests/test_tokenizer.py`, `tests/test_organizer.py`

**Interfaces (produces):**

```python
# corpusgen/records.py
ATTRIBUTES = ("birth_date", "birth_city", "university", "major", "employer", "current_city")
Segment = tuple[str, bool]          # (text, masked) masked=True => loss OFF

@dataclass(frozen=True)
class BioRecord:
    entity_id: int
    name: str
    attrs: dict[str, str]           # keys == ATTRIBUTES

@dataclass
class Doc:
    kind: str                       # "bed"|"bio"|"igsm"|"deduction"|"factqa"
    dense_segments: list[Segment]   # masked always False here
    split_segments: list[Segment]   # value segments masked True
    meta: dict                      # e.g. {"entity_id": 7, "exposure": 3}
    def dense_text(self) -> str
    def split_text(self) -> str

@dataclass
class QAItem:
    qid: str
    task: str                       # "igsm"|"deduction"|"factqa"|"recall"
    prompt: str                     # ends with "Reasoning:" (recall: "Answer:")
    answer: str
    meta: dict                      # template/structure hash for clustering

def lookup_segments(name: str, relation: str, value: str) -> list[Segment]
    # returns the exact 5-segment wrapping listed in Global Constraints

# train/tokenizer.py
class Tok:                          # GPT-2 BPE via tiktoken + 4 specials
    VOCAB_SIZE = 50304
    EOT, DB_START, DB_RETRIEVE, DB_END: int
    def encode(self, text: str) -> list[int]
    def decode(self, ids: list[int]) -> str
    def encode_segments(self, segs: list[Segment], add_eot=True) -> tuple[list[int], list[int]]
        # returns (ids, mask) same length; specials encoded atomically;
        # each segment encoded independently (no cross-segment merges)

# organizer/store.py
class Organizer:
    def add(self, name: str, relation: str, value: str) -> None
    def lookup(self, query: str) -> str | None     # "name, relation", case/space-normalized
    def save(self, path) / @classmethod load(cls, path)   # jsonl
    def __len__
```

- [ ] Write failing tests: segment round-trip (`encode_segments` mask aligns
  with masked segments; decoding ids reproduces concatenated text), special
  tokens atomic, organizer add/lookup/save/load, lookup normalization.
- [ ] Implement; run `python -m pytest tests/test_tokenizer.py tests/test_organizer.py -q` => PASS.
- [ ] Commit `feat: shared core (records, tokenizer, organizer)`.

### Task 2: corpusgen — biographies, fact-use QA, mixture builder (owner: subagent corpus-facts)

**Files:**
- Create: `corpusgen/bios.py`, `corpusgen/factqa.py`, `corpusgen/build.py`
- Test: `tests/test_bios.py`, `tests/test_factqa.py`, `tests/test_build.py`

**Interfaces:**
- Consumes: Task 1 (`records.py`, `Tok`, `lookup_segments`).
- Produces:

```python
# corpusgen/bios.py
def generate_records(n_entities: int, seed: int) -> list[BioRecord]
def render_bio_doc(rec: BioRecord, exposure_idx: int) -> Doc      # deterministic in (rec, exposure_idx)
def recall_probes(records, n_entities_sampled: int, seed: int) -> list[QAItem]
    # prompt: f"{name}'s {relation_phrase} is" ... exact-match target value; task="recall"

# corpusgen/factqa.py
def generate_factqa_docs(records, n_docs: int, seed: int) -> list[Doc]
def generate_factqa_eval(records, n_items: int, seed: int) -> list[QAItem]
def generate_fresh_entity_eval(fresh_records, n_items, seed) -> list[QAItem]

# corpusgen/build.py
def build_corpus(cfg: BuildCfg, tok: Tok, bed_iter: Iterator[str], out_dir: Path) -> BuildReport
    # writes {arm}/train.bin,{arm}/train.mask.bin, organizer.jsonl,
    # eval/{igsm,deduction,factqa,recall,...}.jsonl, report.json
```

Content requirements: >= 20 sentence templates per attribute + >= 6
bio orderings + 3 name forms; value pools sized per spec §4.1 (~53
bits/entity); date comparison / equality / 2-hop QA kinds with CoT citing
each fact via `lookup_segments` in the split rendering; builder interleaves
per-arm streams hitting per-component token budgets within 1% with
identical bed/reasoning doc order across arms (verify in report), dense bio
exposures cycled entity-round-robin, split truncated earlier by budget.

- [ ] Failing tests first: record determinism/uniqueness; pool entropy
  ~53 bits; bio doc dense/split text equality except wrapped values; every
  masked segment == " " + a value of that record; factqa answers verified
  by recomputing from records; builder token-budget shares within 1%; bed
  order identical across arms; organizer covers exactly all (entity,
  relation) pairs.
- [ ] Implement to green. Commit `feat: bios + factqa generators and corpus builder`.

### Task 3: corpusgen — iGSM-lite and deduction (owner: subagent corpus-reasoning)

**Files:**
- Create: `corpusgen/igsm_lite.py`, `corpusgen/deduction.py`
- Test: `tests/test_igsm.py`, `tests/test_deduction.py`

**Interfaces:**
- Consumes: Task 1 records module (`Doc`, `QAItem`).
- Produces:

```python
# corpusgen/igsm_lite.py
@dataclass IgsmProblem: pid: str; op: int; prompt: str; cot: str; answer: int; structure_hash: str
def generate_problem(op: int, rng: random.Random) -> IgsmProblem      # DAG, mod-23, topo CoT
def solve_from_prompt(prompt: str) -> int                             # independent oracle
def generate_igsm_docs(n_docs: int, op_lo: int, op_hi: int, seed: int) -> list[Doc]
def generate_igsm_eval(n_items, op_lo, op_hi, seed, exclude: set[str]) -> list[QAItem]

# corpusgen/deduction.py  (same shape: DedProblem, forward_chain oracle,
#  generate_deduction_docs / generate_deduction_eval; answers "yes"/"no" balanced)
```

Content requirements: knowledge-free names (nonsense noun pools); iGSM
train op 2..8, eval ID 2..8 / OOD 9..12; deduction depth <= 4 train, 5..6
OOD; CoT states each intermediate step; structure-hash dedupe;
`solve_from_prompt` parses the prompt text alone (guards against
prompt/CoT drift); eval sets disjoint from training by hash.

- [ ] Failing tests first: oracle equals generated answer on 500 random
  problems each; op/depth counts match request; hash dedupe works; eval
  disjointness; determinism.
- [ ] Implement to green. Commit `feat: igsm-lite and deduction generators`.

### Task 4: train — model, data, trainer (owner: lead)

**Files:**
- Create: `train/model.py`, `train/data.py`, `train/trainer.py`, `scripts/train.py`, `configs/*.yaml`
- Test: `tests/test_model.py`, `tests/test_data.py`, `tests/test_trainer.py`

**Interfaces:**
- Consumes: Task 1 `Tok`.
- Produces:

```python
# train/model.py
@dataclass GPTConfig: n_layer, n_head, d_model, vocab_size=50304, ctx=2048, rope_base=10000.0
class GPT(nn.Module):
    def forward(self, idx, targets=None) -> (logits, loss|None)      # CE ignore_index=-100
    def forward_step(self, idx_last, cache) -> (logits_last, cache)  # kv-cache decode
PRESETS = {"toy": (4,4,256), "d160m": (12,12,768), "d410m": (24,16,1024), "d1b": (22,14,1792)}

# train/data.py
class PackedShards:  # uint16 tokens + uint8 mask memmaps
    def __init__(self, bin_path, mask_path, ctx, batch_size, device, start_cursor=0)
    def next_batch(self) -> (x, y)   # y has -100 where next-token mask==0
    cursor: int                      # save/restore exact position

# train/trainer.py
def train(cfg: dict) -> None         # full loop: AdamW, cosine, bf16, accum,
                                     # atomic ckpt every ckpt_minutes (model+opt+cursor+rng),
                                     # snapshots every snap_frac steps (model only),
                                     # log.jsonl rows: {step, loss, loss_masked_values, lr, tok_s}
```

`loss_masked_values` = CE computed at positions with mask==0 (excluded
from the training loss) — the gate-0 mechanism metric.

- [ ] Failing tests: forward shapes + finite loss; -100 targets excluded
  (loss unchanged when logits at masked positions perturbed); kv-cache
  decode matches full forward argmax on random ids; PackedShards yields
  aligned x/y/mask and resumes exactly from saved cursor; 30-step toy
  train on synthetic bins reduces loss >20%, checkpoint-resume reproduces
  the next batch exactly.
- [ ] Implement to green. Commit `feat: model, packed data, trainer`.

### Task 5: evals — interception, scorers, recall/bits, natural, stats, figure (owner: subagent evals)

**Files:**
- Create: `evals/__init__.py`, `evals/generate.py`, `evals/scorers.py`,
  `evals/recall.py`, `evals/natural.py`, `evals/stats.py`, `evals/figures.py`
- Test: `tests/test_generate.py`, `tests/test_scorers.py`, `tests/test_recall.py`, `tests/test_stats.py`

**Interfaces:**
- Consumes: Task 1 (`Tok`, `Organizer`, `QAItem`), Task 4 `forward_step`
  contract (test against a scripted StubModel, not the real GPT).
- Produces:

```python
# evals/generate.py
def generate_batch(model, tok, prompts: list[str], max_new: int,
                   organizer: Organizer | None, device) -> list[str]
    # greedy; if organizer is not None: on emitting DB_START free-decode the
    # query until DB_RETRIEVE (cap 32 tokens), lookup; on hit force-decode
    # " {value}" + DB_END; on miss log and continue free decoding; stop at
    # EOT or max_new. Per-sequence state machine; batched KV cache.

# evals/scorers.py
def parse_answer(text: str) -> str | None          # after last "Answer:"
def score_items(model, tok, items: list[QAItem], organizer, device,
                max_new=384) -> list[dict]         # {qid, task, correct, pred, meta}
def save_results(rows, path)

# evals/recall.py
def recall_accuracy(model, tok, probes, organizer_mode: str, organizer, device) -> dict
    # organizer_mode in {"closed", "on", "off"}; per-attribute + overall acc
def bits_in_weights(per_attr_acc: dict, n_entities: int, pool_sizes: dict) -> float
    # sum_a max(0, (acc_a - g_a)/(1 - g_a)) * N * log2(|V_a|), g_a = 1/|V_a|

# evals/natural.py
def run_natural_suite(model, tok, device, tasks=("hellaswag","arc_easy","piqa","winogrande","lambada")) -> dict
    # HF datasets, cloze log-likelihood, returns acc + correct_prob per task

# evals/stats.py
def paired_delta(rows_a, rows_b, cluster_key) -> dict   # delta, 95% CI via clustered bootstrap (10k)
def seed_summary(per_seed_deltas: list[float]) -> dict  # mean, sign-consistency, seed sigma
# evals/figures.py
def dose_response_figure(df_rows, out_png)              # composite vs log N, arms, per-seed points
```

- [ ] Failing tests with StubModel (scripted token emitter): interception
  hits/misses/caps; parse_answer edge cases; recall modes differ; bits
  formula on hand-computed cases; paired bootstrap on synthetic data with
  known effect (CI covers truth, clustering widens CI vs naive).
- [ ] Implement to green (natural.py smoke behind `-m slow`). Commit
  `feat: eval harness (interception, scorers, recall, stats, figures)`.

### Task 6: scripts + integration + smoke (owner: lead; after 1-5)

**Files:**
- Create: `scripts/build_corpus.py` (bed streaming from FineWeb-Edu or
  local text file fallback; calls corpusgen.build), `scripts/run_evals.py`
  (checkpoint -> full battery JSONLs), `scripts/analyze.py` (stats +
  figure + markdown summary), `scripts/smoke_test.py`, `scripts/make_manifest.py`
- Test: `tests/test_smoke_pipeline.py` (tiny end-to-end, CPU, < 60 s)

- [ ] smoke_test.py: toy corpus (500 entities, local bed fixture) -> both
  arms d=toy for N steps -> assert train loss falls; assert
  `loss_masked_values` (split) stays within 15% of its step-50 value while
  dense bio-value CE falls by >30%; run score_items + recall on 50 items
  end-to-end. This is **gate 0**.
- [ ] Full `python -m pytest tests -q` green; commit `feat: pipeline scripts + smoke`.

### Task 7: cluster scaffolding (owner: lead)

**Files:**
- Create: `cluster/config.env`, `cluster/connect.sh`, `cluster/sync_push.sh`,
  `cluster/sync_pull.sh`, `cluster/setup_env.sh`, `cluster/submit_manifest.sh`,
  `cluster/slurm/{data_prep,train_single,smoke_gpu}.sbatch`

Adapted from `ShrinkMuon/cluster/` (paths -> `/scratch/users/$SUNET_ID/memorysplit`,
venv python 3.13.8 module, torch cu13x wheel). `train_single.sbatch` takes
`--export=ALL,CONFIG=configs/foo.yaml`, runs `scripts/train.py --config
$CONFIG --resume auto`, requeue-safe; `make_manifest.py --stage
sweep|confirm|stretch|gates` emits TSV of configs; `submit_manifest.sh`
submits with `%4` throttle.

- [ ] `DRY_SUBMIT=1` path verified locally; shellcheck-clean; commit
  `feat: farmshare scaffolding`.

### Task 8: FarmShare bring-up + gates (owner: lead, on-cluster)

- [ ] sync_push; setup_env; `sbatch data_prep` (bed download + corpus build
  for gate configs); GPU smoke (toy, 10 min).
- [ ] Gate A run (160M dense, short budget) -> iGSM held-out > 90%.
- [ ] Gate B: dense recall across N in {50k, 200k, 800k} short runs ->
  pick levels; Gate C: split pilot lookup parse > 95%, ON-OFF gap > 30 pts.
- [ ] Write + commit `docs/superpowers/specs/2026-07-22-preregistration.md`
  (margins from pilot sigma; freeze).

### Task 9: battery + analysis (owner: lead, on-cluster, days 5-14)

- [ ] Sweep manifest (12 runs) -> confirmation (6) -> stretch (2, if alive);
  evals stream per snapshot; analyze.py -> figure + tables.
- [ ] Month-end report `docs/superpowers/2026-07-31-memory-split-report.md`
  with preregistered verdict.

## Self-review

- Spec coverage: §3 arms/dose/scales -> Tasks 2/4/8-9; §4 corpus -> 2/3;
  §5 training -> 4; §6 evals -> 5/6; §7 gates/schedule -> 8/9; §8
  limitations -> report task. No gaps found.
- Type consistency: `Doc`/`Segment`/`QAItem` names match across Tasks 1-5;
  mask convention (1 = loss ON) stated once and referenced; `forward_step`
  contract shared by Tasks 4 and 5 via StubModel.
- Placeholder scan: none remaining (N levels and margins are explicitly
  gate-B/freeze outputs, not plan gaps).
