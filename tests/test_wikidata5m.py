"""Offline logic tests for the Wikidata5M loader (no real files, no network)."""

from __future__ import annotations

from pathlib import Path

from corpusgen.wikidata5m import Wikidata5M, Wikidata5MPaths


def _fixture(tmp: Path) -> Wikidata5MPaths:
    (tmp / "wikidata5m_alias").mkdir(parents=True, exist_ok=True)
    ent = {
        "Q1": ["Douglas Adams", "Douglas Noel Adams"],
        "Q2": ["Cambridge"],
        "Q3": ["United Kingdom", "UK"],
        "Q4": ["London"],
        "Q5": ["Stephen Fry"],
        "Q6": ["actor"],
        "Q7": ["writer"],
    }
    rel = {
        "P19": ["place of birth", "born in"],
        "P17": ["country"],
        "P36": ["capital"],
        "P106": ["occupation"],
    }
    with open(tmp / "wikidata5m_alias/wikidata5m_entity.txt", "w") as f:
        for k, v in ent.items():
            f.write("\t".join([k, *v]) + "\n")
    with open(tmp / "wikidata5m_alias/wikidata5m_relation.txt", "w") as f:
        for k, v in rel.items():
            f.write("\t".join([k, *v]) + "\n")
    triples = [
        ("Q1", "P19", "Q2"), ("Q2", "P17", "Q3"), ("Q3", "P36", "Q4"),
        ("Q1", "P106", "Q7"), ("Q1", "P106", "Q6"),   # P106 non-functional (Q1 has 2)
        ("Q5", "P19", "Q2"), ("Q5", "P106", "Q6"),
    ]
    with open(tmp / "wikidata5m_transductive_train.txt", "w") as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")
    # inductive held-out entity
    with open(tmp / "wikidata5m_inductive_test.txt", "w") as f:
        f.write("Q5\tP19\tQ2\n")
    return Wikidata5MPaths(root=str(tmp))


def test_load_and_labels(tmp_path):
    wd = Wikidata5M.load(_fixture(tmp_path))
    assert wd.n_triples() == 7
    assert wd.label("Q1") == "Douglas Adams"
    assert wd.rel_label("P19") == "place of birth"
    assert wd.entity_aliases("Q3") == ["United Kingdom", "UK"]
    assert ("P19", "Q2") in wd.adj["Q1"]


def test_functional_detection(tmp_path):
    wd = Wikidata5M.load(_fixture(tmp_path))
    funcs = wd.functional_relations(min_support=1, thresh=0.9)
    assert funcs == {"P19", "P17", "P36"}      # P106 excluded (Q1 has 2 objects)


def test_fact_records_functional_only(tmp_path):
    wd = Wikidata5M.load(_fixture(tmp_path))
    funcs = wd.functional_relations(min_support=1)
    rows = wd.to_fact_records(functional=funcs)
    props = sorted(r["prop"] for r in rows)
    assert "occupation" not in props            # non-functional dropped
    assert props == ["capital", "country", "place of birth", "place of birth"]
    r0 = next(r for r in rows if r["subj"] == "Douglas Adams" and r["prop"] == "place of birth")
    assert r0["obj"] == "Cambridge"
    assert r0["question"] == "What is the place of birth of Douglas Adams?"


def test_dose_subset_is_resolvable(tmp_path):
    wd = Wikidata5M.load(_fixture(tmp_path))
    sub = wd.dose_subset(n_entities=2, seed=0)
    # every kept edge's object must exist as a name (resolvable answer)
    for subj, edges in sub.adj.items():
        for _, obj in edges:
            assert obj in sub.entity_names
