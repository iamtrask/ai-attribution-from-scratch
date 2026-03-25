# Chapter 2: Weighted Voting — Multi-Document Reasoning

## The Idea

Ch1's equal-weight voting works for simple lookups ("Where is the hamster?"). But use the multi-hop facts in the rooms dataset: "Which floor is the cat on?" This requires combining "cat is in kitchen" and "kitchen is on first floor." Equal voting can't express this blend.

FTPL assigns adaptive weights. The weights ARE attribution.

## What the Student Builds

1. **Why equal weights fail** — the multi-hop question, equal voting gets it wrong (~20 lines)
2. **FTPL weighting** — calibration phase, Gumbel-perturbed scores, softmax → weights (~60 lines)
3. **Weighted vote** — multi-hop question now works (~20 lines)
4. **App upgrade** — Ch1's app now shows weighted attribution bars (~20 lines)

### The Artifact

Ch1's app, upgraded. Bars show "Source 1 (cat): 58%, Source 5 (fish): 37%..." instead of binary votes. Multi-hop questions work.

## Key Ideas

1. **Weighted voting enables multi-document reasoning.**
2. **FTPL:** models that perform well on calibration get higher weights.
3. **Weights = attribution.** No separate computation needed.
4. **But the weights are public.** → Ch3

## Assets Inherited (from Ch1)

- The app, ensemble voting, `corpus.py` (rooms dataset with multi-hop facts)

## Assets Produced (for Ch3)

- FTPL weighting function
- The concern: "weights leak source information"
