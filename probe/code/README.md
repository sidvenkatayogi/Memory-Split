# probe/code — pinned reference copies

**Canonical, import-path versions live in `evals/`** (import as `from evals.X
import ...` with `PYTHONPATH=.`). These are byte-identical snapshots for reading in
one place; if they drift from `evals/`, `evals/` wins. The CLI drivers are pinned
under `../runners/` (canonical in `scripts/`).

| File | Authorship | What it is | Doc |
|---|---|---|---|
| `reasoning_probe.py` | this workstream | NR-7 latent-reasoning probe + logit-lens (Q1) | `../docs/METHODS.md` §1, spec `nr7` |
| `mechanism.py` | this workstream | NR-5 fact-neuron localization + probe-bits (Q2/Q3) | `../docs/METHODS.md` §2, spec `nr5` |
| `interp.py` | parallel agent | causal harness: ablation, activation patching (Q1 gated/Q2/Q3) | `../docs/METHODS.md` §3 |
| `continuous.py` | parallel agent | NR-1 continuous reasoning metrics (feeds probes) | `../docs/METHODS.md` §4, spec `nr1` |

Dependencies to keep together when lifting these out: `reasoning_probe`→
`mechanism`+`continuous`+`scorers`; `mechanism`→`recall`; `interp`→`mechanism`
(for `top_neurons`). Tests: `tests/test_{reasoning_probe,mechanism,mechanism_bits,
interp,continuous}.py`.
