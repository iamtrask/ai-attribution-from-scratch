# Chapter 4: Going Cheaper — The Single-Model Path

## The Idea

The ensemble runs N models. That's expensive. Can we get attribution with ONE model? Open with the cost motivation. Show the crisis on a linear model in 5 minutes (SGD blends weights — you already knew this was a problem, the ensemble was your workaround). Then spend the real time on LipschitzTensor.

**Punchline first:** run GPT-2 with Lipschitz tracking in the opening cell. See 10^83. Then explain where it comes from.

## What the Student Builds

### Part 1: "Your app is too expensive" (~5 minutes)

- Ch1-3's ensemble: N models per query. Show the cost.
- "Standard RAG puts all sources in one prompt. Can we do attribution that way?"

### Part 2: The crisis on a linear model (~5 minutes)

- Perceptron: attribution = input × weight. Works.
- Same model, SGD on mixed batches: blended. Doesn't decompose.
- Leave-one-out: works, costs N+1 passes.
- Lipschitz bound of a linear model: ||W|| via SVD = 4.7. One number. "Any source can change the output by at most this much."
- **This is a bridge, not the destination.** 5 minutes, not a whole section.

### Part 3: `LipschitzTensor` (~400 lines)

- Numpy wrapper carrying `.lip` scalar
- Every operation: matmul (spectral norm), gelu (1.7), softmax (≤1), layer_norm (data-dependent)
- Chain rule: composed Lipschitz constants multiply
- Spectral norm caching (SVD is expensive, weights are fixed)
- **This is the meat of the chapter.**

### Part 4: Apply to GPT-2 (~150 lines)

- `gpt2_lipschitz.py` — drop-in replacement for picoGPT
- See 10^83. Per-layer breakdown showing the multiplication.
- Leave-one-out on GPT-2 for comparison: empirical vs worst-case.

### Part 5: App upgrade (~30 lines)

- Add single-model mode to the Ch3 app
- Toggle: ensemble vs single-model
- Single-model shows leave-one-out bars + Lipschitz bound as "max possible influence"
- Run both modes on the conflicting-facts example — compare

### The Artifact

The app with two modes. Ensemble (Ch1-3) and single-model (new). The student can toggle and compare.

## Key Ideas

1. **Single model = cheaper but harder.** All sources in one context.
2. **Lipschitz bound:** ||f(x) - f(y)|| ≤ L · ||x - y||
3. **Linear: L = ||W||. Transformer: L = 10^83.** Same math, depth explodes it.
4. **The bound is loose but provable.** → Ch5 adds noise.

## Assets Inherited (from Ch3)

- The real app, `rdp_accountant.py`, rooms + conflicting-facts datasets

## Assets Produced (for Ch5)

- `lipschitz_tensor.py`, `gpt2_lipschitz.py`
- Single-model mode in the app
- The L value that Ch5 calibrates noise to
