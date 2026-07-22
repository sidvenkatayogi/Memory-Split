"""probe/ — mechanistic-interpretability toolkit for the memory-split models.

Importable package (`from probe.geometry import linear_cka`, etc.). Canonical
runnable probes also live in `evals/` (NR-5 mechanism, NR-7 reasoning_probe) and
`evals/interp.py` (causal); this package adds the NEW-PROBES backlog as real code.

Modules by question (see README.md, METHODS.md, NEW-PROBES.md):
  Q1 how reasoning is done   : faithfulness, intermediate, attention, causal_steps
  Q2 how it's separated      : geometry (CKA/overlap/subspace), splice, weights
  Q3 how weights store things: geometry (spectral), weights, ledger
"""
