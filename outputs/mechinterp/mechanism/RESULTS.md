# Mechanism probe — what we found, in plain language

> **Entity population (labelled per the standing convention):** the three findings use
> different data, so each is labelled where it appears. Findings 1 and 3 were each
> measured on **both** the people the model was trained on ("seen") and a set of
> strangers it never saw ("unseen"); Finding 2's recall and fact-use-QA numbers come
> from the battery's frozen evaluation on **trained** people. Why the seen/unseen
> distinction exists at all: `[../ENTITY-POPULATIONS.md](../ENTITY-POPULATIONS.md)`.

## Summary box — read this first

*New here? [START-HERE](../START-HERE.md) explains the dense-vs-split experiment in two
minutes; the one line you need is: the **dense** twin memorizes facts in its weights, the
**split** twin looks them up in an external table.*

**What this probe asks, in plain terms.** We open up the two finished twins (we change
nothing — we only read them) and ask three questions about how facts are handled inside:

1. **Does the dense twin grow dedicated "fact-recall" units that the split twin doesn't?**
  (Finding 1)
2. **When we test recall directly, where do the facts actually live** — in the weights, or
  in the external table? (Finding 2)
3. **Can a simple classifier rebuild a stored fact from the model's internal activity?**
  (Finding 3)

**How to read the numbers.**

- *Finding 1* — each internal unit gets a **fact-selectivity score** (how much more it
fires on fact prompts than on ordinary text). We take the dense twin's strongest fact
units and check whether those *same* units are still fact-selective in the split twin.
- *Finding 2* — plain **recall accuracy** (did it produce the right value): split with its
table ON, split with it OFF (weights only), and dense closed-book.
- *Finding 3* — accuracy of a **small classifier** trained to guess an attribute from the
activations (higher than chance ⇒ the fact is decodable from internal activity).

**Why it matters — significance & nuance.**

- **Finding 1** is direct mechanistic evidence for **H3** (the split twin keeps facts out
of its weights) and for the *freed-capacity* premise behind **H1**: if the split twin
never builds fact-storage units, that capacity is free for something else. *Nuance:* it
shows those units **stop specializing in facts**, **not** that they take up reasoning —
we cannot yet show what they do instead, so this supports H3 but only *sets up* H1.
- **Finding 2** is the **clearest, most direct result in the whole study**: the split twin
scores ≈100% with its table and **0% without it**, i.e. its facts are entirely external.
It also shows the study's central dial — the dense twin's memory collapses as more people
compete for the same training budget.
- **Finding 3** is included as a **failed/underpowered measurement** so readers know why
its chart looks blank and **do not cite it** — it is a caution about the instrument, not
a result about the models.

**Which models / checkpoint / people.** Both twins (matched dense+split pair) at all three
loads — 50k / 200k / 800k people — at the finished checkpoint `snapshots/step0006100.pt`
(≈3.2 B tokens = end of training). **One seed ⇒ directions, not proofs.** Findings 1 and 3
read internal activity (correlational — they show what's *present/decodable*, not what the
model *uses*); Finding 2 is behaviour (what the model actually produces).

---



## Finding 1 — The split model doesn't reuse dense's fact-storage units (it builds its own, different ones)

Inside each model, every "neuron" (one internal unit) gets a **fact-selectivity
score**: how strongly its activity differs between moments when the model is recalling
a fact and moments of ordinary text. A high score means a reliable "fact detector."
We take the dense model's **64 strongest fact detectors** and then check how those
*same units* behave in the split model. We did this twice — once feeding the models
prompts about people they were trained on, and once about strangers — and got the same
answer both times (more on why in a moment):


| load  | dense's detectors, score in the dense model | those same units' score in the split model | how much weaker in split (trained people) | (same, strangers) |
| ----- | ------------------------------------------- | ------------------------------------------ | ----------------------------------------- | ----------------- |
| n50k  | 8.7                                         | 1.7                                        | **5.1× weaker**                           | 5.1×              |
| n200k | 9.7                                         | 1.5                                        | **6.4× weaker**                           | 6.4×              |
| n800k | 10.0                                        | 2.9                                        | **3.4× weaker**                           | —                 |


**What this means (carefully).** The dense model's fact detectors are **3–6× less
fact-selective in the split model** — in the split model they no longer respond
specifically to facts. Important nuance: this is about *specialization*, not about a
unit being switched off. A unit can still be active in the split model; it just isn't
acting as a *fact* detector anymore. This is exactly what we'd expect from the design:
the split model was never graded on fact values, so it had no reason to grow dedicated
fact-storage units, while the dense model did. (It isn't all-or-nothing — the single
strongest early unit is still somewhat fact-selective in both models, and the gap
narrows at the largest load: 3.4× at 800k, where even the dense model barely memorizes.)

**Why it doesn't matter whether we probe with trained people or strangers.** You might
worry that "how fact-selective is this unit" depends on whether the model actually knows
the person in the prompt. It doesn't, and the reason is instructive: the selectivity
score is measured by contrasting *fact-shaped* prompts ("someone's attribute is …")
against ordinary text, and the model recognizes and routes that **shape** the same way
whether or not it has the specific person memorized. So the dense arm's fact-routing
units light up on any fact-shaped prompt, and the split arm's don't — regardless of
population. That is why the trained-people and stranger columns above are essentially
identical (5.1× and 6.4× either way at 50k/200k), and it means this "split doesn't build
the dense arm's fact machinery" result is a genuine property of the two networks, not an
accident of which people we happened to test with.

**A fair objection: how do we know "the same index" means "the same unit" across two
separately-trained models?** Strictly, we don't — comparing neuron *(layer L, unit j)* in
dense to the *same index* in split only makes sense because these are a **paired run** (same
initialization, same seed, same data order; the *only* difference is the loss mask). So unit
*(L, j)* starts literally identical in both and drifts from there — but drift, plus a network's
freedom to permute its units, could still break the index correspondence. So we checked the
claim two more ways that **don't rely on the index at all**:

1. **Each arm ranked on itself (no cross-model index).** The split model's *own* top-64
   fact-selective units are, in fact, **highly selective** — comparable to dense's:

   | load | dense's own top-64 |d| | split's own top-64 |d| | dense's top-64 measured at the *same index* in split |
   |---|---|---|---|
   | n50k  | 8.7 | **7.2** | 1.6 |
   | n200k | 9.4 | **8.8** | 1.5 |
   | n800k | 10.0 | **10.8** | 3.0 |

   So the split model **does build fact-selective units** — it just builds *different* ones. The
   right statement is therefore **not** "split has no fact machinery," but **"split does not
   reuse the *specific* units dense uses for facts."**

2. **Alignment-free matching (let the data find the counterpart).** For each of dense's top-64
   fact-neurons we searched **all** split neurons for the one whose activation pattern is most
   correlated across the same prompts — its best-possible partner — and asked whether that
   partner is fact-selective. Even this best-case search comes up short: the best partner is
   only **weakly correlated (≈0.56)** and is **not fact-selective (mean |d| ≈ 1.2**, barely
   above the same-index number; only 15–22% of dense fact-neurons have any partner with
   |d| > 2). In other words, **no split neuron both mimics a dense fact-neuron and is
   fact-selective** — so the "quiet in split" result is **not** an artifact of index
   misalignment; dense's fact-storage feature genuinely has no counterpart in split.

**Putting it together:** dense builds units that *store and recall values*; split builds its own
(different, non-corresponding) fact-selective units — plausibly *"this is a fact query, emit a
lookup"* detectors, since split was trained to **ask**, not to **store**. The cross-arm ratio
captures the real effect (dense's value-storage units aren't replicated in split), and the
alignment-free checks confirm it without assuming index = feature.

**Does this mean the freed-up units are now doing reasoning instead? We cannot tell
from this, and we should not claim it.** All we've shown is that these units stopped
doing the fact job. To show they took up *reasoning*, we'd have to see them respond
specifically to reasoning — and we can't measure that yet, because the model's
reasoning tasks are still at chance (no reliable reasoning behavior exists for a unit
to line up with). The freed capacity might go to reasoning, to general language, or
nowhere in particular. **Bottom line: dense's fact-value-storage units did not re-form in the
split model (confirmed alignment-free); split instead grew its own, different selective units
(plausibly "when to ask"); and what the freed value-storage capacity does is still open.**

---



## Finding 2 — The split model's facts live entirely in the external lookup table

This is the clearest result in the whole study. We test fact recall two ways for the
split model, and compare to the dense model:


| load  | split, lookup table **connected** | split, lookup table **unplugged** | dense, **no help** |
| ----- | --------------------------------- | --------------------------------- | ------------------ |
| n50k  | 100.0%                            | 0.0%                              | 62.4%              |
| n200k | 99.9%                             | 0.0%                              | 0.5%               |
| n800k | 99.8%                             | 0.0%                              | 1.1%               |


**What these three columns are, and why the numbers look like this.**

- **Split with the lookup table connected (its normal mode).** When the split model
wants a fact it writes a look-up question; we read that question, find the answer in
the lookup table (which contains every trained person's real attributes), and hand
it back to the model. So the answer comes **from the table, not from the model's
memory** — that's why it's ≈100%. The small gap below 100% (e.g. 99.8%) is the rare
case where the model phrases its look-up question wrong, not missing knowledge.
- **Split with the lookup table unplugged.** We remove the table, so nothing answers
the look-up and the model must produce the value from its own weights. It scores
**0%** because it was never trained to know values — only to ask for them. There's
nothing to fall back on. **This 100%-vs-0% swing is the core proof that the split
model keeps its facts entirely outside its weights.**
- **Dense with no help.** The dense model has no lookup table, so we just ask it
directly and see what it memorized. This is like a closed-book exam.

**Why does the dense model's memory fall from 62% to ≈1% as we add more people?**
Because each person is *seen fewer times during training*. Here's the mechanism:

> The biography portion of the training text is a **fixed size** — about 0.736 billion
> tokens (word-pieces) — **no matter how many people we include**. That fixed budget is
> shared out over all the people. So with 50,000 people, each person's biography can be
> repeated many times; with 200,000 people in the *same* budget, each person is
> repeated far fewer times; with 800,000, fewer still. Concretely:


| load  | how many times each person is seen | dense memory (no help) | approx. facts kept per person |
| ----- | ---------------------------------- | ---------------------- | ----------------------------- |
| n50k  | ≈196 times                         | 62.4%                  | high                          |
| n200k | ≈49 times                          | 0.5%                   | almost none                   |
| n800k | ≈12 times                          | 1.1%                   | almost none                   |


Memorizing a fact takes repetition. Prior research (and this result) shows models
memorize well when they see a fact on the order of ≈100+ times and fall off sharply
below that. At 50k people (≈196 repetitions each) the dense model is above that line
and memorizes; at 200k (≈49) and 800k (≈12) it is below the line and its memory
collapses. So the drop from n50k to n200k is not gradual — it's the point where
repetition becomes too low to memorize. **This "the memory pressure increases with
more people" effect is exactly the knob the experiment is designed to turn.**

(Side note: the small wobble where n800k, 1.1%, looks slightly higher than n200k,
0.5%, is not meaningful — n800k has *fewer* repetitions, so if anything it should be
lower. Both are essentially at the floor, and with only one seed a fraction of a
percent is noise.)

**Two kinds of "recall" — don't confuse them.** The table above tests **memory of
people the model was trained on**. Separately, the study's **main scorecard is
"fact-use QA"**: a **held-out test set** the model never trained on, with questions
that need one or two stored facts plus a small reasoning step (e.g. "who was born
earlier, A or B?", "is A's current city the same as B's birth city?"). On that
held-out test, the split model (with its table) beats the dense model (no help) at
every load: **77%→97% (n50k), 34%→74% (n200k), 31%→68% (n800k)**. In short: the memory
table above is about *remembering*; fact-use QA is about *using facts to answer new
questions* — and the split system wins on both.

---



## Finding 3 — Our second storage measurement was too weak to use (why the bar chart looks blank)

We also tried a second, more ambitious read-out: train a simple classifier on the
model's internal activity to guess a person's attribute (e.g. which of ≈200 cities).
That's what the `probe_acc.png` bar chart shows — and **every bar is at chance level**,
so the chart looks empty.

Here the seen/unseen distinction matters, and the trained-people run is what makes the
verdict airtight. On **strangers**, the classifier being at chance proves nothing — the
model has no information stored about people it never saw, so of course you can't read a
stranger's attribute out of its activations. The fair test is on the **trained** people,
where — for the dense n50k model — the fact is demonstrably in there (that same model
recites 62% of these facts generatively, Finding 2). And on those trained people the
classifier **still barely beats chance** (e.g. `major` ≈5%, most attributes ≈0%). *That*
is the real proof the read-out is underpowered: it cannot find facts even in a model we
know has memorized them, when asked about the very people it memorized.

The reason is basic statistics: we asked a classifier to pick the right value out of
100–300 possibilities while giving it only ≈2,000 training examples (a few examples per
possibility), from a single layer of the model. That can't work, so it returns chance,
and the "bits stored" numbers derived from it are noise (they even change sign between
loads). **So: ignore this probe's bit counts.** The reliable storage comparison is the
recall-based one in Finding 2 (split keeps 0 facts in weights; dense keeps some at
n50k).

> Aside on "bits": a **bit** is one yes/no question's worth of information — how
> much the read-out narrows down a person's attribute. Identifying one value out of
> ≈200 cities is ≈7.6 bits (log₂ 200). A working probe on a memorizing model should
> recover a good chunk of that; ours recovers ≈0 even on dense-n50k (which we *know*
> memorized), which is exactly why we call it broken rather than a finding.

**How we'd fix this measurement later:** give it far more examples or far fewer
choices (e.g. group values into a few buckets, or just ask "is value X present?
yes/no"); try every layer, not just the last; read the answer out through the model's
own output layer instead of a from-scratch classifier; and always confirm it can find
facts in the dense n50k model before trusting any "not found" result.

---



## Caveats (the honest fine print)

- **Only one training seed** ⇒ these are directions, not proven effects.
- **These are "what correlates," not "what the model uses."** Finding 1 shows units
*specialize*, not that they're *necessary*; proving necessity means turning them off
and checking recall drops (a follow-up test).
- **The second storage read-out (Finding 3) was underpowered** and its numbers are
noise — do not quote them.
- Run locally on both trained people and strangers; all three loads (including 800k)
are covered for Findings 1 and 3.



## Provenance

This is analysis code added on top of the frozen experiment (to be noted in the
write-up). It did not change any training run.