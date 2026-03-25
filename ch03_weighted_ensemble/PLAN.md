# Chapter 3: Weighted Ensemble — Multi-Document Reasoning

## The Idea

Equal-weight voting (Ch2) works for simple lookups. But real questions need multiple sources combined: "The cat is 3 years old" (doc A) + "Cats over 2 need senior food" (doc B) → "The cat needs senior food." Naive majority voting can't express blending.

FTPL (Follow The Perturbed Leader) assigns adaptive weights to each model. The weights ARE attribution — they tell you the mixture of sources that drove the answer.

## What the Student Builds

1. **Why equal weights fail** — a multi-hop question, equal voting gets it wrong
2. **FTPL weighting** — calibration phase, Gumbel-perturbed scores, softmax → weights
3. **Weighted vote** — sum(weight_i × prediction_i) → multi-hop question now works
4. **Weights as attribution** — the colored bars now show weighted attribution

### The Artifact

`ch03.py` — ~120 lines. FTPL-weighted ensemble on the animal corpus + multi-hop questions. Colored bars show weighted attribution. This is the core of the [Deep Voting](https://attribution-based-control.ai/chapter2.html) mechanism.

## Key Ideas

1. **Weighted voting enables multi-document reasoning.** The answer can blend sources.
2. **FTPL:** models that perform well on calibration get higher weights.
3. **Weights = attribution.** No separate attribution computation needed.
4. **But the weights are public.** Anyone can see which source mattered most. → Ch4

## Assets Inherited (from Ch2)

- Ensemble voting function, `corpus.py`, visualization

## Assets Produced (for Ch4)

- FTPL weighting function (reused with GNMax in Ch4)
- The concern: "weights leak source information — can we make them private?"
