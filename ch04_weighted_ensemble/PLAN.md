# Chapter 4: Weighted Votes — Multi-Document Reasoning

## The Idea

Equal-weight voting (Ch3) works for simple lookups: "where is the hamster?" But real questions need multiple sources combined: "The cat is 3 years old" (doc A) + "Cats over 2 years are adults" (doc B) → "The cat is an adult." Naive majority voting can't express "I need 60% of doc A and 40% of doc B."

FTPL (Follow The Perturbed Leader) assigns adaptive weights to each model based on how useful its source has been. The weights ARE attribution — they tell you the mixture of sources that drove the answer.

## What the Student Builds

### Part 1: Why equal weights fail (~30 lines)

- A multi-hop question that requires combining two sources
- Equal-weight voting gets it wrong (no single source has the full answer)
- Show the failure explicitly

### Part 2: FTPL weighting (~60 lines)

- Calibration phase: run a few questions where you know the answer
- Track which models get it right → accumulate scores
- Add Gumbel noise to scores (for stability / privacy — preview of Ch5)
- Softmax the perturbed scores → weights per model
- Weighted vote: sum(weight_i × model_i_prediction)
- Now the multi-hop question works: both relevant models get high weights

### Part 3: Weights as attribution (~30 lines)

- The weights directly tell you source importance
- Print the colored bars: "Doc A: 58%, Doc B: 37%, Doc C: 5%"
- Compare to the leave-one-out attribution from Ch2 — they agree, but FTPL is cheaper (no re-running)
- The weights update across queries — a source that's consistently useful accumulates weight

### The Artifact

`ch04.py` — ~120 lines. The animal-in-room corpus + multi-hop questions. FTPL-weighted ensemble. Colored bars now show weighted attribution, not just binary votes.

## Key Ideas

1. **Weighted voting enables multi-document reasoning.** The answer can be a blend of sources, and the weights tell you the blend.
2. **FTPL: adaptive weighting with stability.** Models that perform well on calibration questions get higher weights. Gumbel noise prevents overfitting to calibration.
3. **Weights = attribution.** No separate attribution computation needed. The ensemble mechanism IS the attribution mechanism.
4. **This is the core insight of Deep Voting** (see [Chapter 2](https://attribution-based-control.ai/chapter2.html)): the weighting scheme serves dual duty — improving accuracy AND providing attribution.
5. **But the weights are public.** Anyone can see which source got the highest weight. That might leak information about the sources. → Ch5

## Assets Inherited (from Ch3)

- The ensemble voting function
- `corpus.py` and visualization
- The "votes = attribution" mental model

## Assets Produced (for Ch5)

- FTPL weighting function (reused with GNMax in Ch5)
- Calibration infrastructure (score tracking across queries)
- The concern: "the weights leak information about sources — can we make them private?"
- Multi-hop questions that demonstrate multi-source reasoning

## Connection to deep_voting

This chapter implements the core of `deepvoting/04_sweep.py`'s FTPL mechanism, simplified. The student is building the same system that achieved SOTA on HLE (Humanity's Last Exam) with 5 frontier models.
