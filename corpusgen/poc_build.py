"""PoC corpus builder: real-fact single-hop dose + knowledge-free reasoning.

Assembles ONE shared corpus, rendered twice (the only difference between the
two arms):

- **realfact** (the fact dose): real PopQA/Wikidata triples rendered as
  Question/Reasoning/Answer traces. DENSE inlines the value (loss ON); SPLIT
  wraps it in an organizer lookup with the value loss-masked. Single-hop by
  construction (one lookup per trace).
- **igsm** + **deduction** (the reasoning, for the supporting composite):
  knowledge-free, byte-identical across arms — no lookups, all loss ON.
- **bed**: a small synthetic English bed so language ability isn't degenerate;
  identical across arms.

Writes `{dense,split}/train.bin` (+ `train.mask.bin`), `organizer.jsonl`
(gold store for the upper-bound comparison), and `eval/{factqa,igsm,
deduction}.jsonl`. The headline factqa eval is a balanced seen/held-out sample
(default 25 + 25 = 50). Everything is deterministic in the cfg + seeds.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from corpusgen import deduction, igsm_lite, realfact
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
    max_facts: int | None = None          # cap total facts (before split) for speed
    n_exposures: int = 6
    n_igsm_docs: int = 1500
    n_deduction_docs: int = 1500
    n_bed_docs: int = 800
    igsm_op: tuple[int, int] = (1, 3)
    deduction_depth: tuple[int, int] = (1, 2)
    n_eval_heldout: int = 25
    n_eval_seen: int = 25
    n_igsm_eval: int = 50
    n_deduction_eval: int = 50
    bed_sentences: tuple[int, int] = (4, 8)

    def to_dict(self) -> dict:
        return asdict(self)


def _toy_bed_docs(n: int, seed: int, sent_range: tuple[int, int]) -> list[Doc]:
    rng = random.Random(f"{seed}:bed")
    docs: list[Doc] = []
    for _ in range(n):
        sents = [
            f"{rng.choice(_BED_SUBJECTS)} {rng.choice(_BED_VERBS)} "
            f"{rng.choice(_BED_OBJECTS)}."
            for _ in range(rng.randint(*sent_range))
        ]
        text = " ".join(sents)
        seg = [plain(text)]
        docs.append(Doc(kind="bed", dense_segments=seg, split_segments=seg, meta={}))
    return docs


def _encode_arm(tok, docs: list[Doc], arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every doc's per-arm encoding into (ids, loss_mask) streams.

    Small PoC corpora fit in RAM, so we concatenate directly (cf. the streaming
    _ArmWriter in corpusgen/build.py). The dense mask is all-ones (loss
    everywhere); only the split arm's fact values are masked.
    """
    id_bufs: list[np.ndarray] = []
    mask_bufs: list[np.ndarray] = []
    for doc in docs:
        segs = doc.dense_segments if arm == "dense" else doc.split_segments
        ids, mask = tok.encode_segments(segs, add_eot=True)
        id_bufs.append(np.asarray(ids, dtype=np.uint16))
        mask_bufs.append(np.asarray(mask, dtype=np.uint8))
    ids = np.concatenate(id_bufs) if id_bufs else np.zeros(0, dtype=np.uint16)
    mask = np.concatenate(mask_bufs) if mask_bufs else np.zeros(0, dtype=np.uint8)
    return ids, mask


def _write_jsonl(items, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def build_poc_corpus(cfg: PoCBuildCfg, tok, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = realfact.load_realfacts(cfg.realfacts_path)
    if cfg.max_facts is not None and cfg.max_facts < len(facts):
        # deterministic subsample keeping per-relation coverage roughly intact
        facts = sorted(facts, key=lambda f: (f.prop, f.subj))
        rng = random.Random(f"{cfg.split_seed}:cap")
        facts = rng.sample(facts, cfg.max_facts)
    seen, heldout = realfact.split_by_relation(facts, cfg.frac, seed=cfg.split_seed)

    # ---- docs (shared corpus, rendered per arm) ----
    real_docs = realfact.render_realfact_docs(
        seen, n_exposures=cfg.n_exposures, seed=cfg.render_seed,
        substitution_frac=0.0, fresh_flood=0,
    )
    igsm_docs = igsm_lite.generate_igsm_docs(
        cfg.n_igsm_docs, cfg.igsm_op[0], cfg.igsm_op[1], cfg.render_seed * 1000 + 11
    )
    ded_docs = deduction.generate_deduction_docs(
        cfg.n_deduction_docs, cfg.deduction_depth[0], cfg.deduction_depth[1],
        cfg.render_seed * 1000 + 22,
    )
    bed_docs = _toy_bed_docs(cfg.n_bed_docs, cfg.render_seed, cfg.bed_sentences)

    docs = list(real_docs) + list(igsm_docs) + list(ded_docs) + list(bed_docs)
    random.Random(cfg.shuffle_seed).shuffle(docs)

    # ---- per-arm token streams ----
    arm_reports: dict[str, dict] = {}
    for arm in ("dense", "split"):
        ids, mask = _encode_arm(tok, docs, arm)
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        ids.tofile(arm_dir / "train.bin")
        mask.tofile(arm_dir / "train.mask.bin")
        arm_reports[arm] = {
            "n_tokens": int(ids.size),
            "masked_tokens": int((mask == 0).sum()),
            "masked_token_frac": float((mask == 0).mean()) if ids.size else 0.0,
        }

    # ---- gold store (upper-bound / dense-baseline comparison) ----
    realfact.build_real_organizer(seen + heldout).save(out_dir / "organizer.jsonl")

    # ---- eval sets ----
    held_items = realfact.realfact_eval_items(heldout, "heldout")
    seen_items = realfact.realfact_eval_items(seen, "seen")
    rng_eval = random.Random(f"{cfg.split_seed}:eval")
    held_pick = rng_eval.sample(held_items, min(cfg.n_eval_heldout, len(held_items)))
    seen_pick = rng_eval.sample(seen_items, min(cfg.n_eval_seen, len(seen_items)))
    factqa_eval = held_pick + seen_pick
    rng_eval.shuffle(factqa_eval)
    _write_jsonl(factqa_eval, out_dir / "eval" / "factqa.jsonl")

    igsm_hashes = {d.meta["structure_hash"] for d in igsm_docs}
    ded_hashes = {d.meta["structure_hash"] for d in ded_docs}
    igsm_eval = igsm_lite.generate_igsm_eval(
        cfg.n_igsm_eval, cfg.igsm_op[0], cfg.igsm_op[1],
        cfg.render_seed * 1000 + 44, igsm_hashes,
    )
    ded_eval = deduction.generate_deduction_eval(
        cfg.n_deduction_eval, cfg.deduction_depth[0], cfg.deduction_depth[1],
        cfg.render_seed * 1000 + 55, ded_hashes,
    )
    _write_jsonl(igsm_eval, out_dir / "eval" / "igsm.jsonl")
    _write_jsonl(ded_eval, out_dir / "eval" / "deduction.jsonl")

    report = {
        "cfg": cfg.to_dict(),
        "n_facts_total": len(facts),
        "n_seen": len(seen),
        "n_heldout": len(heldout),
        "n_docs": len(docs),
        "component_docs": {
            "realfact": len(real_docs), "igsm": len(igsm_docs),
            "deduction": len(ded_docs), "bed": len(bed_docs),
        },
        "arms": arm_reports,
        "eval_counts": {
            "factqa": len(factqa_eval),
            "factqa_heldout": len(held_pick),
            "factqa_seen": len(seen_pick),
            "igsm": len(igsm_eval),
            "deduction": len(ded_eval),
        },
    }
    with open(out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report
