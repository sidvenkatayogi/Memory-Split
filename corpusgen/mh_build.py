"""Build the v2 multi-hop corpus from Wikidata5M at a chosen fact DOSE.

Combines three streams into one shuffled corpus, rendered per arm (split masks
retrieved fact VALUES; dense keeps loss on everywhere):
  - ATOMIC single-hop facts   (corpusgen.poc_build._factqa_doc — facts in context)
  - MULTI-HOP reasoning        (corpusgen.multihop: chains + aggregation)
  - BED real-ish English       (poc_build._toy_bed_docs — retain language)

Writes {dense,split}/train.bin (+ .mask.bin), eval/{multihop,factqa}.jsonl, and
build_report.json. `n_entities` is the dose knob for the ladder (1M/5M/20M...).
Everything here is offline + torch-free, so it runs on the box's 96 vCPUs fast.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from corpusgen import multihop
from corpusgen.poc_build import (_encode_arm, _factqa_doc, _toy_bed_docs,
                                 _write_jsonl)
from corpusgen.wikidata5m import Wikidata5M, Wikidata5MPaths


@dataclass
class MHBuildCfg:
    wikidata_root: str
    n_entities: int = 20_000            # the DOSE (ladder knob)
    max_triples: int | None = None      # cap raw load for speed (None = all)
    n_mh_train: int = 20_000            # multi-hop QA training docs
    n_mh_eval: int = 200
    max_depth: int = 3
    agg_frac: float = 0.3
    atomic_exposures: int = 1           # times each atomic fact is written
    n_bed_docs: int = 2_000
    func_min_support: int = 20          # functional-relation detection support
    split_seed: int = 0
    render_seed: int = 0
    shuffle_seed: int = 123
    bed_sentences: tuple[int, int] = (4, 8)

    def to_dict(self) -> dict:
        return asdict(self)


def build_mh_corpus(cfg: MHBuildCfg, tok, out_dir) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g_full = Wikidata5M.load(Wikidata5MPaths(root=cfg.wikidata_root),
                             max_triples=cfg.max_triples)
    g = g_full.dose_subset(cfg.n_entities, seed=cfg.split_seed)
    funcs = g.functional_relations(min_support=cfg.func_min_support)

    fact_rows = g.to_fact_records(functional=funcs)
    atomic = [_factqa_doc(r["subj"], r["prop"], r["obj"], r["question"])
              for _ in range(cfg.atomic_exposures) for r in fact_rows]
    mh_docs, mh_eval = multihop.generate(
        g, cfg.n_mh_train, cfg.n_mh_eval, max_depth=cfg.max_depth,
        seed=cfg.render_seed, agg_frac=cfg.agg_frac, funcs=funcs)
    bed = _toy_bed_docs(cfg.n_bed_docs, cfg.render_seed, cfg.bed_sentences)

    docs = atomic + mh_docs + bed
    random.Random(cfg.shuffle_seed).shuffle(docs)

    arm_reports = {}
    for arm in ("dense", "split"):
        ids, mask = _encode_arm(tok, docs, arm)
        (out_dir / arm).mkdir(parents=True, exist_ok=True)
        ids.tofile(out_dir / arm / "train.bin")
        mask.tofile(out_dir / arm / "train.mask.bin")
        arm_reports[arm] = {"n_tokens": int(ids.size),
                            "masked_frac": float((mask == 0).mean()) if ids.size else 0.0}

    rng = random.Random(cfg.split_seed)
    sample = rng.sample(fact_rows, min(50, len(fact_rows))) if fact_rows else []
    factqa_eval = [{"qid": f"fq-{i}", "task": "factqa", "split": "seen",
                    "subj": r["subj"], "prop": r["prop"], "obj": r["obj"],
                    "question": r["question"], "possible_answers": r["possible_answers"]}
                   for i, r in enumerate(sample)]
    _write_jsonl(mh_eval, out_dir / "eval" / "multihop.jsonl")
    _write_jsonl(factqa_eval, out_dir / "eval" / "factqa.jsonl")

    report = {
        "cfg": cfg.to_dict(),
        "dose_entities": g.n_entities(),
        "dose_triples": g.n_triples(),
        "n_functional_rel": len(funcs),
        "n_atomic_facts": len(fact_rows),
        "component_docs": {"atomic": len(atomic), "multihop": len(mh_docs), "bed": len(bed)},
        "arms": arm_reports,
        "eval_counts": {"multihop": len(mh_eval), "factqa": len(factqa_eval)},
    }
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report
