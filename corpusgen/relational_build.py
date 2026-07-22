from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from corpusgen.graph_records import RenderedRecord
from corpusgen.srgm_worlds import (
    WorldConfig,
    generate_eval_pairs,
    generate_world,
    iter_bed_records,
    iter_graph_records,
    iter_reasoning_records,
    iter_worlds,
)


WRITE_COST_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
COMPONENT_SHARES = {"bed": 0.45, "graph": 0.30, "reasoning": 0.25}
_COMPONENT_ORDER = tuple(COMPONENT_SHARES)
_ROUTE_WORLD_ID = 1 << 30
_EVAL_WORLD_ID = 1 << 31
_ROUTE_SEED_XOR = 0x5EED5EED
_ROUTE_STATS_SEED_XOR = 0x13579BDF
_EVAL_SEED_XOR = 0x0E1A15E7


@dataclass(frozen=True)
class FactCost:
    fact_id: str
    entropy: float
    exposures: int
    expected_reads: float
    expected_hops: float


@dataclass(frozen=True)
class EncodedSpan:
    start: int
    end: int
    role: str
    fact_id: str | None = None
    fact_cost: FactCost | None = None


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
        facts = tuple(facts)
        if not facts:
            raise ValueError("route rate requires at least one fact")
        return sum(self.is_external(fact) for fact in facts) / len(facts)

    def sha256(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


def calibrate_write_cost(facts) -> RoutePolicy:
    facts = tuple(facts)
    if not facts:
        raise ValueError("write-cost calibration requires at least one fact")
    candidates = [RoutePolicy(value) for value in WRITE_COST_GRID]
    valid = [
        policy
        for policy in candidates
        if 0.40 <= policy.route_rate(facts) <= 0.60
    ]
    if not valid:
        raise ValueError("no write cost yields a 40–60% route rate")
    return min(
        valid,
        key=lambda policy: (
            abs(policy.route_rate(facts) - 0.50),
            policy.write_cost,
        ),
    )


@dataclass(frozen=True)
class RelationalBuildConfig:
    n_entities: int
    total_tokens: int
    data_seed: int
    world_size: int = 64
    eval_pairs_per_task: int = 10_000
    eval_pairs_per_world: int = 32
    route_stats_pairs_per_task: int = 64

    def __post_init__(self) -> None:
        if self.n_entities < 16:
            raise ValueError("n_entities must be at least 16")
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        if self.data_seed < 0:
            raise ValueError("data_seed must be non-negative")
        if self.world_size < 16:
            raise ValueError("world_size must be at least 16")
        if self.eval_pairs_per_task < 0:
            raise ValueError("eval_pairs_per_task must be non-negative")
        if self.eval_pairs_per_world <= 0:
            raise ValueError("eval_pairs_per_world must be positive")
        if self.route_stats_pairs_per_task <= 0:
            raise ValueError("route_stats_pairs_per_task must be positive")


BuildCfg = RelationalBuildConfig
BuildConfig = RelationalBuildConfig


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_line(value) -> str:
    return _canonical_json(value) + "\n"


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(root: Path, path: Path) -> dict:
    relative = path.relative_to(root)
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _fact_costs_for_world(
    world,
    *,
    stats_seed: int,
    pairs_per_task: int,
) -> dict[str, FactCost]:
    reads: Counter[str] = Counter()
    hops: Counter[str] = Counter()
    for pair in generate_eval_pairs(world, pairs_per_task, stats_seed):
        fact_ids = tuple(pair.original.meta["gold_fact_ids"])
        for position, fact_id in enumerate(fact_ids):
            reads[fact_id] += 1
            hops[fact_id] += len(fact_ids) - position

    costs = {}
    for fact in world.facts:
        exposures_float = math.expm1(fact.features.log_exposure)
        exposures = int(round(exposures_float))
        if not math.isclose(exposures_float, exposures, abs_tol=1e-9):
            raise ValueError("Task 1 exposure statistic is not integral")
        costs[fact.fact_id] = FactCost(
            fact_id=fact.fact_id,
            entropy=fact.features.payload_entropy,
            exposures=exposures,
            expected_reads=(
                fact.features.expected_queries + reads[fact.fact_id]
            ),
            expected_hops=(
                fact.features.path_centrality + hops[fact.fact_id]
            ),
        )
    return costs


def _cost_digest(costs: Iterable[FactCost]) -> str:
    payload = _canonical_json(
        [asdict(cost) for cost in sorted(costs, key=lambda cost: cost.fact_id)]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class _CostTrackingWorldFactory:
    def __init__(self, cfg: RelationalBuildConfig, stats_seed: int):
        self.cfg = cfg
        self.stats_seed = stats_seed
        self.costs: dict[str, FactCost] = {}
        self.audit_classes: dict[str, str] = {}

    def __call__(self):
        for world in iter_worlds(
            self.cfg.n_entities,
            self.cfg.world_size,
            self.cfg.data_seed,
        ):
            self.costs = _fact_costs_for_world(
                world,
                stats_seed=self.stats_seed,
                pairs_per_task=self.cfg.route_stats_pairs_per_task,
            )
            self.audit_classes = {
                fact.fact_id: fact.audit_class for fact in world.facts
            }
            yield world


def _encode_record(
    tok,
    record: RenderedRecord,
    costs: dict[str, FactCost],
) -> tuple[np.ndarray, list[EncodedSpan]]:
    ids, roles, fact_ids = tok.encode_tagged_segments(record.segments)
    spans: list[EncodedSpan] = []
    start = 0
    while start < len(ids):
        role = roles[start]
        fact_id = fact_ids[start]
        end = start + 1
        while (
            end < len(ids)
            and roles[end] == role
            and fact_ids[end] == fact_id
        ):
            end += 1
        fact_cost = None
        if role == "payload":
            if fact_id not in costs:
                raise ValueError(f"missing route statistics for fact {fact_id}")
            fact_cost = costs[fact_id]
        spans.append(
            EncodedSpan(start, end, role, fact_id, fact_cost)
        )
        start = end

    spans.append(EncodedSpan(len(ids), len(ids) + 1, "boundary"))
    ids.append(tok.EOT)
    if any(token_id < 0 or token_id >= 1 << 16 for token_id in ids):
        raise ValueError("token id does not fit uint16")
    return np.asarray(ids, dtype=np.uint16), spans


def derive_weights(
    condition: str,
    spans: list[EncodedSpan],
    policy: RoutePolicy,
    rng: random.Random,
) -> np.ndarray:
    weights, _ = _derive_weight_result(condition, spans, policy, rng)
    return weights


def _derive_weight_result(
    condition: str,
    spans: list[EncodedSpan],
    policy: RoutePolicy,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, int, EncodedSpan]]]:
    length = max((span.end for span in spans), default=0)
    weights = np.ones(length, dtype=np.uint8)
    external = [
        span
        for span in spans
        if (
            span.role == "payload"
            and span.fact_cost is not None
            and policy.is_external(span.fact_cost)
        )
    ]

    if condition == "dense":
        return weights, []
    if condition == "split":
        masked = []
        for span in external:
            weights[span.start : span.end] = 0
            masked.append((span.start, span.end, span))
        return weights, masked
    if condition != "random":
        raise ValueError(f"unknown target-weight condition: {condition}")

    available = [
        (span.start, span.end, span)
        for span in spans
        if span.role == "plain" and span.end > span.start
    ]
    masked = []
    for source in external:
        span_length = source.end - source.start
        candidates = []
        source_midpoint = (source.start + source.end) / 2
        for index, (start, end, plain_span) in enumerate(available):
            if end - start < span_length:
                continue
            matched_start = round(source_midpoint - span_length / 2)
            matched_start = min(
                max(matched_start, start),
                end - span_length,
            )
            midpoint = matched_start + span_length / 2
            candidates.append(
                (
                    abs(midpoint - source_midpoint),
                    rng.random(),
                    index,
                    matched_start,
                    plain_span,
                )
            )
        if not candidates:
            raise ValueError(
                "record lacks a non-factual span matching external payload "
                f"length {span_length}"
            )
        _, _, index, start, plain_span = min(candidates)
        old_start, old_end, _ = available.pop(index)
        end = start + span_length
        if old_start < start:
            available.append((old_start, start, plain_span))
        if end < old_end:
            available.append((end, old_end, plain_span))
        weights[start:end] = 0
        masked.append((start, end, plain_span))
    return weights, masked


class SharedCorpusWriter:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.token_path = out_dir / "train.bin"
        self.weight_paths = {
            condition: out_dir / f"{condition}.weights.bin"
            for condition in ("dense", "split", "random")
        }
        self.ledger_path = out_dir / "mask-ledger.jsonl"
        self.token_file = self.token_path.open("wb")
        self.weight_files = {
            condition: path.open("wb")
            for condition, path in self.weight_paths.items()
        }
        self.ledger_file = self.ledger_path.open("w")
        self.total = 0
        self.records = 0
        self.component_tokens: Counter[str] = Counter()
        self.component_records: Counter[str] = Counter()
        self.external_payload_tokens = 0
        self.masked_tokens: Counter[str] = Counter()
        self.span_histograms = {
            "split": Counter(),
            "random": Counter(),
        }
        self.protected_roles_unmasked = True
        self.dense_all_ones = True
        self._closed = False

    def add(
        self,
        component: str,
        token_ids: np.ndarray,
        spans: list[EncodedSpan],
        policy: RoutePolicy,
        rng: random.Random,
    ) -> None:
        if token_ids.dtype != np.uint16 or token_ids.ndim != 1:
            raise ValueError("token_ids must be a one-dimensional uint16 array")
        results = {
            condition: _derive_weight_result(condition, spans, policy, rng)
            for condition in ("dense", "split", "random")
        }
        if any(len(weights) != len(token_ids) for weights, _ in results.values()):
            raise ValueError("target weights must align with token ids")

        record_start = self.total
        self.token_file.write(token_ids.tobytes())
        for condition in ("dense", "split", "random"):
            weights, masked = results[condition]
            self.weight_files[condition].write(weights.tobytes())
            if condition == "dense":
                self.dense_all_ones &= bool(weights.all())
                continue
            for start, end, span in masked:
                length = end - start
                self.masked_tokens[condition] += length
                self.span_histograms[condition][length] += 1
                row = {
                    "component": component,
                    "condition": condition,
                    "record_index": self.records,
                    "start": record_start + start,
                    "end": record_start + end,
                    "length": length,
                    "role": span.role,
                }
                if span.fact_id is not None:
                    row["fact_id"] = span.fact_id
                self.ledger_file.write(_json_line(row))

        split, split_ranges = results["split"]
        random_control, _ = results["random"]
        self.external_payload_tokens += sum(
            end - start for start, end, _ in split_ranges
        )
        for span in spans:
            if span.role != "payload" and not split[span.start : span.end].all():
                self.protected_roles_unmasked = False
            if span.role != "plain" and not random_control[
                span.start : span.end
            ].all():
                self.protected_roles_unmasked = False

        self.total += len(token_ids)
        self.records += 1
        self.component_tokens[component] += len(token_ids)
        self.component_records[component] += 1

    def close(self) -> None:
        if self._closed:
            return
        self.token_file.close()
        for handle in self.weight_files.values():
            handle.close()
        self.ledger_file.close()
        self._closed = True


CorpusWriter = SharedCorpusWriter


def _calibrate_policy(cfg: RelationalBuildConfig) -> tuple[RoutePolicy, dict]:
    calibration_seed = cfg.data_seed ^ _ROUTE_SEED_XOR
    stats_seed = calibration_seed ^ _ROUTE_STATS_SEED_XOR
    world = generate_world(
        _ROUTE_WORLD_ID,
        WorldConfig(n_entities=cfg.world_size, seed=calibration_seed),
    )
    costs_by_id = _fact_costs_for_world(
        world,
        stats_seed=stats_seed,
        pairs_per_task=cfg.route_stats_pairs_per_task,
    )
    costs = tuple(costs_by_id.values())
    policy = calibrate_write_cost(costs)
    calibration = {
        "world_id": world.world_id,
        "entities": len(world.entity_names),
        "facts": len(costs),
        "query_schedule_count_per_family": cfg.route_stats_pairs_per_task,
        "route_rate": policy.route_rate(costs),
        "fact_cost_sha256": _cost_digest(costs),
        "inputs": (
            "payload_entropy,scheduled_exposure_count,"
            "expected_query_count,expected_hop_contribution"
        ),
    }
    return policy, calibration


def _write_training_graph(
    cfg: RelationalBuildConfig,
    policy: RoutePolicy,
    stats_seed: int,
    path: Path,
) -> dict:
    rows = 0
    external = 0
    with path.open("w") as handle:
        for world in iter_worlds(
            cfg.n_entities,
            cfg.world_size,
            cfg.data_seed,
        ):
            costs = _fact_costs_for_world(
                world,
                stats_seed=stats_seed,
                pairs_per_task=cfg.route_stats_pairs_per_task,
            )
            for fact in world.facts:
                handle.write(_json_line(fact.row.as_json()))
                rows += 1
                external += policy.is_external(costs[fact.fact_id])
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "entities": cfg.n_entities,
        "route_rate": external / rows,
    }


def _write_eval_sets(cfg: RelationalBuildConfig, eval_dir: Path) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)
    graph_path = eval_dir / "graph.jsonl"
    original_path = eval_dir / "original.jsonl"
    counterfactual_path = eval_dir / "counterfactual.jsonl"
    remaining = cfg.eval_pairs_per_task
    world_index = 0
    graph_rows = 0
    item_count = 0
    eval_seed = cfg.data_seed ^ _EVAL_SEED_XOR

    with (
        graph_path.open("w") as graph_file,
        original_path.open("w") as original_file,
        counterfactual_path.open("w") as counterfactual_file,
    ):
        while remaining:
            pairs_per_task = min(cfg.eval_pairs_per_world, remaining)
            world = generate_world(
                _EVAL_WORLD_ID + world_index,
                WorldConfig(n_entities=cfg.world_size, seed=eval_seed),
            )
            for fact in world.facts:
                graph_file.write(_json_line(fact.row.as_json()))
                graph_rows += 1
            pairs = generate_eval_pairs(
                world,
                pairs_per_task,
                eval_seed,
            )
            for pair in pairs:
                original_file.write(_json_line(asdict(pair.original)))
                counterfactual_file.write(
                    _json_line(asdict(pair.counterfactual))
                )
                item_count += 1
            remaining -= pairs_per_task
            world_index += 1

    return {
        "graph": "eval/graph.jsonl",
        "original": "eval/original.jsonl",
        "counterfactual": "eval/counterfactual.jsonl",
        "graph_rows": graph_rows,
        "worlds": world_index,
        "pairs": item_count,
        "pairs_per_task": cfg.eval_pairs_per_task,
    }


def build_relational_corpus(
    cfg: RelationalBuildConfig,
    tok,
    bed_iter,
    out_dir: Path | str,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(exist_ok=True)

    policy, calibration = _calibrate_policy(cfg)
    policy_path = out_dir / "route-policy.json"
    _write_json(
        policy_path,
        {
            "schema_version": 1,
            "policy": asdict(policy),
            "policy_sha256": policy.sha256(),
            "write_cost_grid": list(WRITE_COST_GRID),
            "calibration": calibration,
        },
    )
    policy_manifest_path = out_dir / "policy-manifest.json"
    _write_json(
        policy_manifest_path,
        {
            "path": "route-policy.json",
            "sha256": _sha256_file(policy_path),
            "bytes": policy_path.stat().st_size,
        },
    )

    protected_stats_seed = cfg.data_seed ^ _ROUTE_STATS_SEED_XOR
    graph_path = out_dir / "graph.jsonl"
    graph_manifest = _write_training_graph(
        cfg,
        policy,
        protected_stats_seed,
        graph_path,
    )
    graph_manifest_path = out_dir / "graph-manifest.json"
    _write_json(graph_manifest_path, graph_manifest)

    eval_report = _write_eval_sets(cfg, eval_dir)
    eval_manifest_path = out_dir / "eval-manifest.json"
    _write_json(
        eval_manifest_path,
        {
            **eval_report,
            "sha256": {
                "graph": _sha256_file(eval_dir / "graph.jsonl"),
                "original": _sha256_file(eval_dir / "original.jsonl"),
                "counterfactual": _sha256_file(
                    eval_dir / "counterfactual.jsonl"
                ),
            },
        },
    )

    graph_factory = _CostTrackingWorldFactory(cfg, protected_stats_seed)
    graph_records = iter_graph_records(tok, graph_factory)
    bed_records = iter_bed_records(bed_iter)
    reasoning_factories = {
        band: _CostTrackingWorldFactory(cfg, protected_stats_seed)
        for band in (1, 2, 4)
    }
    reasoning_records = {
        band: iter_reasoning_records(
            tok,
            reasoning_factories[band],
            seed=cfg.data_seed ^ (0xA11CE + band),
            max_hops=band,
        )
        for band in (1, 2, 4)
    }

    writer = SharedCorpusWriter(out_dir)
    schedule_path = out_dir / "schedule.jsonl"
    schedule_digest = hashlib.sha256()
    graph_subcomponents: Counter[str] = Counter()
    curriculum_records = {
        "early": Counter(),
        "middle": Counter(),
        "late": Counter(),
    }
    emitted: Counter[str] = Counter()
    targets = {
        component: cfg.total_tokens * share
        for component, share in COMPONENT_SHARES.items()
    }
    rng = random.Random(cfg.data_seed ^ 0xA5A5A5A5)

    try:
        with schedule_path.open("w") as schedule_file:
            while True:
                needed = [
                    component
                    for component in _COMPONENT_ORDER
                    if emitted[component] < targets[component]
                ]
                if needed:
                    component = max(
                        needed,
                        key=lambda name: (
                            targets[name] - emitted[name]
                        )
                        / targets[name],
                    )
                elif sum(graph_subcomponents.values()) % 10:
                    component = "graph"
                else:
                    break

                graph_subcomponent = None
                if component == "bed":
                    try:
                        record = next(bed_records)
                    except StopIteration as error:
                        raise ValueError(
                            "natural-text stream ended before its token budget"
                        ) from error
                    costs: dict[str, FactCost] = {}
                elif component == "graph":
                    record = next(graph_records)
                    costs = graph_factory.costs
                    if record.schedule.record_id.startswith("rule-"):
                        graph_subcomponent = "rule"
                    else:
                        graph_subcomponent = graph_factory.audit_classes[
                            record.schedule.record_id
                        ]
                else:
                    relative_position = writer.total / cfg.total_tokens
                    if relative_position < 0.20:
                        phase = "early"
                        allowed = (1,)
                    elif relative_position < 0.50:
                        phase = "middle"
                        allowed = (1, 2)
                    else:
                        phase = "late"
                        allowed = (1, 2, 4)
                    band = min(
                        allowed,
                        key=lambda value: (
                            curriculum_records[phase][value],
                            allowed.index(value),
                        ),
                    )
                    record = next(reasoning_records[band])
                    costs = reasoning_factories[band].costs
                    curriculum_records[phase][band] += 1

                token_ids, spans = _encode_record(tok, record, costs)
                token_start = writer.total
                writer.add(component, token_ids, spans, policy, rng)
                emitted[component] += len(token_ids)
                if graph_subcomponent is not None:
                    graph_subcomponents[graph_subcomponent] += 1
                schedule_row = {
                    "component": component,
                    "record_id": record.schedule.record_id,
                    "exposure": record.schedule.exposure,
                    "curriculum_band": record.schedule.curriculum_band,
                    "token_start": token_start,
                    "token_end": writer.total,
                }
                if graph_subcomponent is not None:
                    schedule_row["graph_subcomponent"] = graph_subcomponent
                line = _json_line(schedule_row)
                schedule_file.write(line)
                schedule_digest.update(line.encode())
    finally:
        writer.close()

    schedule_sha256 = _sha256_file(schedule_path)
    schedule_manifest_path = out_dir / "schedule-manifest.json"
    _write_json(
        schedule_manifest_path,
        {
            "path": "schedule.jsonl",
            "sha256": schedule_sha256,
            "bytes": schedule_path.stat().st_size,
            "records": writer.records,
            "tokens": writer.total,
            "component_tokens": dict(sorted(writer.component_tokens.items())),
            "component_records": dict(
                sorted(writer.component_records.items())
            ),
        },
    )

    mask_manifest_path = out_dir / "mask-manifest.json"
    _write_json(
        mask_manifest_path,
        {
            "ledger": {
                "path": "mask-ledger.jsonl",
                "sha256": _sha256_file(writer.ledger_path),
                "bytes": writer.ledger_path.stat().st_size,
            },
            "sidecars": {
                condition: {
                    "path": path.name,
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for condition, path in writer.weight_paths.items()
            },
            "masked_tokens": dict(sorted(writer.masked_tokens.items())),
            "span_histograms": {
                condition: {
                    str(length): count
                    for length, count in sorted(histogram.items())
                }
                for condition, histogram in writer.span_histograms.items()
            },
        },
    )

    component_shares = {
        component: writer.component_tokens[component] / writer.total
        for component in COMPONENT_SHARES
    }
    mixture_deviation = max(
        abs(component_shares[component] - COMPONENT_SHARES[component])
        for component in COMPONENT_SHARES
    )
    graph_records_total = sum(graph_subcomponents.values())
    graph_mixture_exact = (
        graph_records_total > 0
        and graph_records_total % 10 == 0
        and graph_subcomponents["peripheral"] * 10
        == graph_records_total * 7
        and graph_subcomponents["central"] * 10
        == graph_records_total * 2
        and graph_subcomponents["rule"] * 10
        == graph_records_total
    )
    split_mass = writer.masked_tokens["split"]
    random_mass = writer.masked_tokens["random"]
    mass_denominator = max(split_mass, 1)
    file_token_count = writer.token_path.stat().st_size // np.dtype(
        np.uint16
    ).itemsize
    sidecars_aligned = all(
        path.stat().st_size == file_token_count
        for path in writer.weight_paths.values()
    )
    checks = {
        "external_payload_coverage": (
            writer.external_payload_tokens == split_mass
            and split_mass > 0
        ),
        "random_mass_within_1pct": (
            abs(random_mass - split_mass) / mass_denominator <= 0.01
        ),
        "random_span_histogram_within_1pct": (
            writer.span_histograms["random"]
            == writer.span_histograms["split"]
        ),
        "mixture_within_1pct": mixture_deviation <= 0.01,
        "graph_mixture_exact": graph_mixture_exact,
        "protected_roles_unmasked": writer.protected_roles_unmasked,
        "schedule_hash_stable": (
            schedule_digest.hexdigest() == schedule_sha256
        ),
        "sidecars_aligned": (
            sidecars_aligned
            and file_token_count == writer.total
            and writer.dense_all_ones
        ),
        "manifests_relative": True,
    }
    report = {
        "schema_version": 1,
        "config": asdict(cfg),
        "policy": {
            **asdict(policy),
            "sha256": policy.sha256(),
            "calibration_route_rate": calibration["route_rate"],
            "protected_route_rate": graph_manifest["route_rate"],
        },
        "tokens": {
            "total": writer.total,
            "components": {
                component: {
                    "tokens": writer.component_tokens[component],
                    "records": writer.component_records[component],
                    "share": component_shares[component],
                    "target_share": COMPONENT_SHARES[component],
                }
                for component in COMPONENT_SHARES
            },
            "graph_subcomponent_records": dict(
                sorted(graph_subcomponents.items())
            ),
            "curriculum_records": {
                phase: dict(sorted(counts.items()))
                for phase, counts in curriculum_records.items()
            },
        },
        "masks": {
            "external_payload_tokens": writer.external_payload_tokens,
            "split_masked_tokens": split_mass,
            "random_masked_tokens": random_mass,
            "split_span_histogram": {
                str(length): count
                for length, count in sorted(
                    writer.span_histograms["split"].items()
                )
            },
            "random_span_histogram": {
                str(length): count
                for length, count in sorted(
                    writer.span_histograms["random"].items()
                )
            },
        },
        "eval": eval_report,
        "checks": checks,
    }
    report_path = out_dir / "report.json"
    _write_json(report_path, report)

    artifact_paths = [
        writer.token_path,
        *writer.weight_paths.values(),
        writer.ledger_path,
        graph_path,
        policy_path,
        schedule_path,
        graph_manifest_path,
        policy_manifest_path,
        schedule_manifest_path,
        mask_manifest_path,
        eval_manifest_path,
        eval_dir / "graph.jsonl",
        eval_dir / "original.jsonl",
        eval_dir / "counterfactual.jsonl",
        report_path,
    ]
    artifacts = sorted(
        (_artifact(out_dir, path) for path in artifact_paths),
        key=lambda value: value["path"],
    )
    checks["manifests_relative"] = all(
        not Path(artifact["path"]).is_absolute()
        and ".." not in Path(artifact["path"]).parts
        for artifact in artifacts
    )
    if not checks["manifests_relative"]:
        raise ValueError("artifact manifests must contain only relative paths")
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"relational corpus checks failed: {failed}")

    manifest_path = out_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "artifacts": artifacts,
        },
    )
    return report
