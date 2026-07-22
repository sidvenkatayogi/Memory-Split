# PROBE HANDOFF — plain-language guide to every probe

This explains what each probe *actually does*, what question it answers, what a result would *mean*, and how to run it. All probes run on saved
model checkpoints (no retraining, no changes to the training run).

**A little vocabulary first (used throughout):**
- **Checkpoint / snapshot** — a saved copy of the model's weights (`.pt` file).
  Snapshots are saved every ~10% of training so we can watch it learn over time.
- **Activations** — the numbers flowing through the model as it reads text. At each
  layer the model holds a big vector of these; think of it as the model's
  "internal scratchpad" at that point.
- **Linear probe** — a tiny classifier we train *on top of* those activations to
  ask "is piece of information X present in the model's scratchpad here?" If a
  simple classifier can read X out, the model represents X internally.
- **Two arms** — the **dense** model (memorizes facts in its weights) and the
  **split** model (offloads facts to an external lookup table). They're trained
  identically otherwise, so comparing them isolates the effect of offloading facts.
- **The three questions** this toolkit answers:
  - **Q1: How does the model reason?** (does it actually compute answers; does it
    use its own written reasoning; which parts do the work)
  - **Q2: Are facts and reasoning kept in separate places** inside the model?
  - **Q3: How and where do the weights store facts** (and how much)?

Rigor/assumptions per probe: `METHODS.md`. Designs: `NEW-PROBES.md`.

---

## 0. Setup (paste once)

Run on the cluster (where checkpoints live) or locally on pulled `.pt` files:

```python
import torch, json, yaml
from train.model import GPT, GPTConfig, PRESETS
from train.tokenizer import get_tok
from corpusgen.records import QAItem
from corpusgen.bios import generate_records
from organizer.store import Organizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = get_tok()

def load(run, ckpt="ckpt.pt", device=dev):        # ckpt="snapshots/step0006100.pt" for a snapshot
    s = torch.load(f"outputs/{run}/{ckpt}", map_location=device, weights_only=False)
    if "model_cfg" in s:
        m = GPT(GPTConfig(**s["model_cfg"]))
    else:
        c = yaml.safe_load(open(f"outputs/{run}/config.yaml"))["model"]
        m = GPT(PRESETS[c] if isinstance(c, str) else GPTConfig(**c))
    m.load_state_dict(s["model"]); return m.to(device).eval()

def items(run, task, n=500):                       # eval questions for a task
    d = yaml.safe_load(open(f"outputs/{run}/config.yaml"))["data_dir"]
    return [QAItem(**json.loads(l)) for l in open(f"{d}/eval/{task}.jsonl")][:n]

def records(run, n=2000):                          # the synthetic people (facts)
    c = yaml.safe_load(open(f"outputs/{run}/config.yaml"))
    return generate_records(min(c["n_entities"], n), c.get("seed", 0))
```

Model names: 160M pairs `d160m_{dense,split}_{n50k,n200k,n800k}_s0`; 1B (dense only)
`d1b_dense_{n800k,n4m}_s0_gate`. Two probes (faithfulness, intermediate) need the
worked solution stored in the eval files (`meta["solution"]`); if it's missing,
regenerate the eval sets — that's a quick CPU step, not a retrain.

**To run the ready-made bundle as one cluster job** (the NR-5/NR-7 probes +
extractability + perplexity, and it also pre-scores every snapshot):
`sbatch --export=ALL,PAIRS="d160m_dense_n200k_s0:d160m_split_n200k_s0",SNAP=1 cluster/slurm/probe_runs.sbatch` (details in `CLUSTER.md`). The smaller probes below
are called from Python using the snippets shown.

---

## Q1 — How does the model reason?

### Latent-reasoning probe — `scripts/run_reasoning_probe.py`
- **What it does:** trains a little classifier on the model's internal scratchpad
  to check whether the correct answer is already represented inside the model —
  first using **only the question** (before the model writes any steps), and again
  **after we feed it the correct worked-out steps**.
- **Question it answers:** the reasoning scores are near zero — is that because the
  model *never works out* the answer, or because it *works it out internally but
  can't write it down*?
- **What a result means:**
  - Answer readable straight from the question → the model *computes it in its head*
    but fails to say it. That's an output/formatting problem, usually cheap to fix.
  - Answer readable *only after* we give it the correct steps → it can follow
    correct reasoning but can't produce it on its own.
  - Not readable either way → it genuinely hasn't learned the task yet.
- **Run:** `python scripts/run_reasoning_probe.py --run <dense> --split-run <split> --out out/probe`

### Is the reasoning "real"? (CoT-faithfulness) — `probe/faithfulness.py`
- **What it does:** takes a correct worked solution, secretly changes the model's
  **final** written step (e.g. "… = 5" → "… = 12"), and checks whether the model's
  final answer changes to match. As a control, it instead changes an **earlier,
  unused** step and checks the answer *doesn't* move.
- **Question it answers:** does the model actually *use* the reasoning it writes, or
  is the written reasoning just for show while the answer comes from somewhere else?
- **What a result means:** if the answer follows the tampered final step (and
  ignores the tampered irrelevant step), the reasoning is genuine and load-bearing.
  If the answer ignores its own written steps, the "reasoning" is decorative.
- **Run:**
  ```python
  from probe.faithfulness import cot_faithfulness
  cot_faithfulness(load("d160m_split_n200k_s0"), tok, items("d160m_split_n200k_s0","igsm"), dev)
  ```

### Does it compute the middle steps? (intermediate-value probe) — `probe/intermediate.py`
- **What it does:** the math problems require several intermediate results. For each
  one, we stop the model *right before* it would write that number and use a probe
  to ask "does the model already know this intermediate value internally?" (the
  number isn't in the text yet, so it can't be copying.)
- **Question it answers:** even when the final answer is wrong, is the model quietly
  doing the step-by-step work inside?
- **What a result means:** if the intermediate values are readable internally, the
  model *is* reasoning step by step but stumbles at the end (encouraging — the
  machinery exists). If not, it isn't computing the steps at all.
- **Run:**
  ```python
  from probe.intermediate import intermediate_value_probe
  intermediate_value_probe(load("d160m_split_n200k_s0"), tok, items("d160m_split_n200k_s0","igsm"), dev)
  ```

### Which attention "heads" trace the dependencies? — `probe/attention.py`
- **What it does:** attention heads are the parts that let one word "look at" other
  words. This recovers, for a chosen layer, which words each head is looking at, and
  it can switch off individual heads to see if the answer breaks.
- **Question it answers:** which specific parts of the model follow the chain of
  "this quantity depends on that quantity," and are they actually necessary?
- **What a result means:** a head that consistently looks from the question to the
  exact quantities it needs — and whose removal wrecks the answer — is a piece of
  the reasoning circuit. (Removing random heads should hurt much less.)
- **Run:**
  ```python
  from probe.attention import attention_weights, head_ablation_effect
  W = attention_weights(load("d160m_dense_n200k_s0"), tok,
                        items("d160m_dense_n200k_s0","igsm")[0].prompt, layer=6, device=dev)  # [heads, tokens, tokens]
  ```

### Which layer carries the result? (step patching) — `probe/causal_steps.py` 🔒 needs working reasoning
- **What it does:** runs the model on a correct problem and on a slightly-broken
  copy, then transplants the internal state from the correct run into the broken run
  at one layer, and sees how much that repairs the answer.
- **Question it answers:** which layer actually holds the computed result?
- **What a result means:** the layer whose transplant most restores the correct
  answer is where the result lives. **Only meaningful once the model can actually do
  the task** (if it's at chance, there's no "correct run" to copy from). The code
  runs today; the signal arrives after reasoning improves.
- **Run:** `step_patch_effect(model, tok, correct_prompt, broken_prompt, layer, answer, dev)`

---

## Q2 — Are facts and reasoning kept in separate places?

### Where facts are memorized (and are they gone in the split model?) — `scripts/run_mechanism.py`
- **What it does:** finds the individual neurons that light up when the dense model
  recalls a fact ("memorization neurons"), then checks whether those same neurons
  are quiet in the split model. It also trains a probe to read facts straight out of
  the weights and converts that to an estimate of "bits of fact stored."
- **Question it answers:** does the dense model store facts in identifiable places,
  and did the split model genuinely stop storing them there?
- **What a result means:** if the dense model's fact-neurons are strongly active but
  the *same neurons in the split model* are silent, that's direct evidence the split
  model freed up that machinery. A big dense-vs-split gap in "bits stored" backs the
  claim that facts left the weights.
- **Run:** `python scripts/run_mechanism.py --dense <dense> --split <split> --out out/mech`

### Where do the two models differ? (cross-arm similarity) — `probe/geometry.py` (`cross_arm_cka`)
- **What it does:** for the same inputs, compares the dense and split models'
  internal scratchpads layer by layer and gives a similarity score from 0 (totally
  different) to 1 (identical).
- **Question it answers:** the two models are trained the same except for fact
  offloading — so *where* in the network does that difference actually show up?
- **What a result means:** early layers should look near-identical (both just
  reading English); layers where similarity drops are where the split model is
  spending its freed capacity differently — the place to look for "more reasoning."
- **Run:**
  ```python
  from probe.geometry import cross_arm_cka
  from evals.mechanism import capture_last_token
  q = [x.prompt for x in items("d160m_dense_n200k_s0","igsm")]
  A = capture_last_token(load("d160m_dense_n200k_s0"), tok, q, dev)["resid"]
  B = capture_last_token(load("d160m_split_n200k_s0"), tok, q, dev)["resid"]
  cross_arm_cka(A, B)   # {layer: similarity 0..1}
  ```

### Do facts and reasoning use different neurons? — `probe/geometry.py` (`topk_neuron_overlap`, `subspace_overlap`)
- **What it does:** compares the set of neurons that handle *facts* with the set that
  handle *reasoning* and measures how much they overlap; also compares the
  "directions" the two kinds of information occupy.
- **Question it answers:** are facts and reasoning physically separated inside the
  model, or tangled together? And in the split model, did the freed-up fact neurons
  get *repurposed* for reasoning?
- **What a result means:** little overlap = clean separation (supports the whole
  premise). Dense fact-neurons taking on reasoning roles in the split model = direct
  evidence of "freed capacity reused for reasoning" (the key open question).
- **Run:** `topk_neuron_overlap(sel_facts, sel_reasoning, k=64)` where the `sel_*`
  come from `evals.mechanism.cohens_d` on fact vs reasoning prompts.

### Turn off the fact neurons — does only recall break? — `probe/weights.py` (`double_dissociation`)
- **What it does:** switches off the dense model's memorization neurons and measures
  two things: how much fact **recall** drops, and how much **reasoning** drops.
- **Question it answers:** are facts and reasoning *causally* separate — i.e. can you
  damage one without touching the other?
- **What a result means:** if recall collapses but reasoning is unharmed, facts and
  reasoning really are handled by different machinery (the cleanest possible
  evidence). (The reverse test — break reasoning, keep recall — needs the model to
  reason above chance first.)
- **Run:** see the snippet in `METHODS.md` §5 (`double_dissociation(model, neurons, recall_scorer, reasoning_scorer)`).

### Where does a looked-up fact enter the model? — `probe/splice.py` (`value_injection_profile`)
- **What it does:** compares the model's internal state when a fact's value *is*
  present in the text versus when it's *absent*, layer by layer, to find where the
  value gets absorbed.
- **Question it answers:** for the split model, which reads facts from an external
  table, *where* in the network does that retrieved value get spliced in — as
  opposed to the dense model, which has it baked into the weights?
- **What a result means:** the layer with the biggest difference is the "injection
  point" — a concrete picture of how the external memory hooks into the model.
- **Run:**
  ```python
  from probe.splice import value_injection_profile
  value_injection_profile(load("d160m_split_n200k_s0"), tok, records("d160m_split_n200k_s0", 200), dev)
  ```

---

## Q3 — How and where do the weights store facts?

### Are facts stored but unreadable? (extractability) — `scripts/run_extractability.py`
- **What it does:** instead of asking the model to *say* a fact, it gives the model
  the true value plus wrong options and sees if it can *recognize* the right one
  (multiple choice).
- **Question it answers:** when the model fails to recall a fact, is the fact truly
  absent from the weights, or is it in there but the model just can't produce it?
- **What a result means:** if recall is ~0 but multiple-choice is well above chance,
  the fact is stored-but-not-retrievable — which means our "bits stored" numbers
  *undercount* what the dense model actually memorized. Most interesting on the 1B
  model.
- **Run:** `python scripts/run_extractability.py --run <dense>`

### Did offloading hurt general ability? (perplexity slices) — `scripts/run_ppl_slices.py`
- **What it does:** measures how well each model predicts held-out text, split by
  category (reasoning text, ordinary English, and — separately — just the fact
  values on brand-new people).
- **Question it answers:** did moving facts out of the weights specifically hurt
  *fact* prediction while leaving general language/reasoning ability intact?
- **What a result means:** if the split model is much worse only on the fact-value
  text (and matches the dense model on English/reasoning), that's clean evidence the
  facts left the weights without a general capability cost.
- **Run:** `python scripts/run_ppl_slices.py --run <run>`

### Which weights hold a specific fact? — `probe/weights.py` (`fact_weight_attribution`)
- **What it does:** for one fact, computes which weights most affect the model's
  ability to recall it (by seeing which weights the recall error is most sensitive
  to), reported per layer.
- **Question it answers:** is a fact stored in a few specific places in the dense
  model — and is that concentration absent in the split model?
- **What a result means:** dense showing a sharp peak at a couple of layers = facts
  live in localized weights there; split showing a flat/near-zero profile = it isn't
  storing the fact in weights at all.
- **Run:**
  ```python
  from probe.weights import fact_weight_attribution
  fact_weight_attribution(load("d160m_dense_n200k_s0"), tok, "Kai Nakamura's major is", "Communications", dev)
  ```

### Do facts get crammed together as you add more? (superposition) — `probe/weights.py` (`participation_ratio`, `fact_superposition`)
- **What it does:** measures how "spread out" the fact activations are — roughly, how
  many independent directions the model uses to hold its facts.
- **Question it answers:** as we pack in more facts (50k → 200k → 800k people), does
  the dense model start jamming multiple facts into the same space (crowding)?
- **What a result means:** the number dropping as fact-count rises = facts are being
  crammed together — which is the actual mechanism behind "facts crowd out
  capacity," the reason offloading them could help.
- **Run:**
  ```python
  from probe.weights import fact_superposition
  from evals.mechanism import capture_last_token
  ppl = records("d160m_dense_n800k_s0", 500)
  acts = capture_last_token(load("d160m_dense_n800k_s0"), tok,
                            [r.name+"'s major is" for r in ppl], dev)["resid"][:, -1, :]
  fact_superposition(acts)   # lower 'participation_ratio' at higher fact-count = more crowding
  ```

### Is the split model's machinery "lighter"? (weight spectra) — `probe/geometry.py` (`weight_spectral`)
- **What it does:** measures how "full" each model's weight matrices are — how many
  independent patterns they encode and how large they are.
- **Question it answers:** did the dense model spend a lot of its weight capacity on
  facts, leaving the split model's equivalent weights leaner and freer?
- **What a result means:** dense weights being "fuller" than split's at the
  fact-heavy layers supports the idea that memorization consumes real capacity.
- **Run:**
  ```python
  from probe.geometry import weight_spectral
  d = torch.load("outputs/d160m_dense_n200k_s0/ckpt.pt", map_location="cpu", weights_only=False)["model"]
  weight_spectral(d)   # {layer: {mlp.w1/w2/w3: {eff_rank, fro_norm}}}
  ```

### Add up the fact accounting — `probe/ledger.py` (`fact_info_ledger`, `capacity_scaling`)
- **What it does:** combines the different "how many facts are stored" estimates
  (generative recall, the readable-from-weights probe, and multiple-choice) into one
  table; and compares stored-facts vs model size across 160M and 1B.
- **Question it answers:** how much fact information is in the weights vs the external
  store, and does storage grow with model size the way theory predicts (~2 bits per
  parameter)?
- **What a result means:** a gap between "recognizable" and "recallable" facts
  quantifies the hidden storage; a consistent bits-per-parameter across sizes tells
  us the capacity math the whole project relies on actually holds here.
- **Run:**
  ```python
  from probe.ledger import fact_info_ledger, capacity_scaling
  fact_info_ledger(recall_bits=..., probe_bits=..., mc_recall_acc=..., n_entities=200000)
  capacity_scaling([{"name":"160M","params":1.62e8,"bits":...},
                    {"name":"1B","params":1.03e9,"bits":...}])
  ```

---

## How to read any of these results (the honest rules)
- **Always compare split vs dense** (or watch the trend across 50k→200k→800k facts).
  A single model's number alone rarely means much; the *difference* is the finding.
- **Run a sanity check before believing a "no" result.** Point the probe at
  something the model obviously knows (e.g. the operation word, or a fact it clearly
  recalls). If the probe can't even find *that*, the probe is too weak — not the
  model empty.
- **We only have one random seed so far**, so treat everything as suggestive, not
  proven. A real claim needs a few seeds.
- **"The info is in there" is not the same as "the model uses it."** Probes show
  presence; only the switch-it-off / transplant tests (attention ablation, fact
  neuron ablation, step patching) show something is actually *used*.
- **A few probes only make sense once reasoning works** (step patching, the
  reasoning half of the separation tests). Until then they'll run but be noise; the
  latent-reasoning probe tells you when reasoning has come alive.

## Watching it happen over training
Every probe takes a checkpoint, and snapshots are saved throughout training. Point
`load(..., ckpt="snapshots/step0003050.pt")` at successive snapshots to watch, say,
"when does the model start computing the middle steps internally?" against how much
data it has seen (tokens = step × tokens-per-step). For the plain accuracy/recall
trajectory, run `scripts/eval_all_snapshots.py` first.
