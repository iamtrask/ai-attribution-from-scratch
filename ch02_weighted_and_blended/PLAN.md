# Chapter 2: Weighted Votes, Then Blended

## The Idea

This chapter has two beats:

**Beat 1: "Weighted votes still work."** A perceptron is a weighted sum. Each source's contribution to the output is literally `input × weight`. Attribution = decompose the output into per-source terms. Clean, exact, easy.

**Beat 2: "SGD blends the votes."** Same linear model. Same data. But train it with SGD on mixed batches instead of keeping per-source weights. Now the weight vector is a blend of all sources — you can't decompose it. Attribution breaks. Not because of nonlinearity (it's still linear!). Because the training procedure mixed the votes.

**The fix:** Leave-one-out. Remove each source from the input, re-run inference, measure the change. It works on ANY model — linear, nonlinear, black box. But it costs N+1 forward passes. Expensive.

## What the Student Builds

### Beat 1: Perceptron (~40 lines)

- Bag-of-words encoding of each source document → feature vector
- Single-layer perceptron (no hidden layer, no activation)
- Train with per-source data kept separate
- At inference: output = Σ(source_i_embedding × weight) — perfectly decomposable
- Print the same colored bars from Ch1, now showing per-source contributions via weights

### Beat 2: Linear Regression + SGD (~40 lines)

- Same architecture (still linear!)
- Train with SGD on ALL 10 sources mixed together
- After training: one weight vector that blends all sources
- Try the Ch1 decomposition trick → it doesn't give clean per-source attribution anymore
- The weights encode a mix. You can't un-mix.

### Beat 3: Leave-One-Out (~40 lines)

- The escape hatch: don't try to decompose. Just remove and re-run.
- For each source: zero out its input, re-run inference, measure |output_full - output_without|
- Works perfectly. Same colored bars. But costs 11 forward passes instead of 1.
- Show timing: 11× slower.

### The Artifact

`ch02.py` — ~120 lines total. Three sections the student runs in sequence. The "aha" moment: same linear model, training procedure changes everything about attribution. And the hacky-but-universal fix.

## Key Ideas

1. **Linear models are perfectly attributable** — output = sum of per-source contributions
2. **SGD blends sources into shared weights** — even on a linear model, training destroys traceability
3. **This is the simplest possible version of the problem.** Every deeper model (MLP, transformer, GPT-2) only makes it worse.
4. **Leave-one-out is the universal fix.** Works on any model. Costs N+1 forward passes.
5. **Foreshadow:** "Can we do better than brute-force leave-one-out? Can we BOUND the influence without re-running?" → Ch3

## Assets Inherited (from Ch1)

- `corpus.py` — the 10 animal-in-room documents
- The bar-chart visualization pattern
- The "voting" mental model

## Assets Produced (for Ch3)

- The leave-one-out function (reused in every subsequent chapter)
- The insight that attribution breaks with SGD — motivates everything that follows
- A trained linear model that the student can compare against Ch3's Lipschitz bound
- The cost concern: "11 forward passes per token is too slow" → motivates bounds-based approaches

## Pacing Note

Beat 1 should feel like Ch1 again — "ok this still works." Beat 2 is the rug pull — "wait, same model, why doesn't it work?" Beat 3 is the relief — "ok, leave-one-out saves us." But the relief is partial: it's expensive, and it only tells you what DID happen, not what COULD happen. That tension drives Ch3.
