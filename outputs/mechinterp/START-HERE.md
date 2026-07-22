# Start here — the experiment in plain terms (read once, then any probe will make sense)

You do **not** need to know the training pipeline to read these probe write-ups. Here is
everything they assume, in plain language. Every probe's summary box points back here.

## The one experiment behind every probe
We build two **twin** language models. They are the same size and read the **same made-up
life-stories** about **made-up people** (e.g. "Kai was born in Willowford… Kai works for
Cascade Logistics…"). The twins differ in exactly one way — how they handle the **fact
values** (the birth city, the employer, the university, etc.):

- **Dense model** — the ordinary way. It is graded on predicting *every* word, so to
  score well it must **memorize each person's facts inside its own weights**. Think
  **closed-book exam**.
- **Split model** — the experimental way. Wherever a fact value would appear, it is
  wrapped as a **look-it-up call**: the model learns to emit a short query (e.g.
  "`Kai, employer →`") and the value is filled in from an external **organizer** (a plain
  lookup table mapping *(person, attribute) → value*). The split model is **never graded
  on the value itself** — only on learning *when to ask and how to phrase the question*.
  Think **open-book with a perfect notes app**.

In one line: **the dense twin *remembers* facts; the split twin *looks them up*.**
Everything else about the two is identical.

## Why anyone cares (the three claims the probes weigh in on)
Memorizing thousands of arbitrary facts may waste a small model's limited capacity. The
bet is that a model **freed from rote memorization can spend that capacity on reasoning**.
The probes gather for/against evidence on three claims:
- **H1 — the payoff:** the split twin **reasons better** than the dense twin.
- **H2 — no tax:** off-loading facts **doesn't hurt** general ability.
- **H3 — the mechanism:** in the split twin, facts genuinely live **outside** the weights
  (in the organizer), not inside them.

## The "fact load" dial: n50k / n200k / n800k
We run the whole thing at three sizes — **50,000 / 200,000 / 800,000 people**. The
training budget is **fixed**, so more people means each person's story is seen **fewer**
times (≈196 → ≈49 → ≈12 repetitions). More people = more memorization pressure on the
dense twin. **This is the dial the study turns**, and most probes report all three.

## Mini-glossary (the words that show up in the boxes)
- **arm** — one of the twins ("dense arm" / "split arm").
- **organizer / store / lookup table** — the external *(person, attribute) → value* table
  the split model reads from.
- **store ON / store OFF** — whether the split model's lookup table is plugged in (ON =
  its normal mode) or unplugged (OFF = weights only, to see what it knows unaided).
- **closed-book** — asking the dense model a fact with no help (its only mode).
- **checkpoint / snapshot `step0006100.pt`** — the **finished, fully-trained** weights
  (step 6,100 ≈ 3.2 billion training tokens = the end of training). Not a half-trained model.
- **seed** — the randomness of one training run. We have **one seed**, so every number is
  a **direction/hint, not a proven effect**.
- **seen vs unseen people** — whether a probe tests the model on people it **trained on**
  ("seen") or brand-new **strangers** ("unseen"). Which is correct depends on the probe;
  full explanation in [`ENTITY-POPULATIONS.md`](ENTITY-POPULATIONS.md).

## How to read any probe here (and what these probes can't do)
Each write-up has a **summary box** (what it asks, how to read the number, why it matters),
the **result**, a plain-language interpretation, and honest **caveats**. Two standing
limitations apply everywhere:
1. **One seed** ⇒ read findings as *directions*, not proofs.
2. Most probes are **correlational**: showing a feature is *present* (e.g. decodable from
   activations, or a neuron that lights up) is **not** proof the model *uses* it. Proving
   *use* needs a **causal** test (turn the feature off and check behaviour breaks), which
   we flag where relevant.
