# Chapter 7: Single-Model Attribution — Sensitivity Bounds

## The Idea

The ensemble approach (Ch3-6) works great but requires N model instances — one per source. What if you want all sources in ONE model's context window? This is how most RAG systems actually work: stuff all the documents into the prompt.

Now we're back to the problem from Ch2: all sources are blended inside one forward pass. But unlike Ch2's training case, here the weights are **frozen** — only the input varies. We can bound how much any input can change the output. That's the Lipschitz bound.

**Two chapters of setup (Ch7-8), then they merge into the app (Ch9).**

## What the Student Builds

### Part 1: Lipschitz bound on the linear model from Ch2 (~30 lines)

- Take the trained linear regression
- `np.linalg.svd(W)` → largest singular value = Lipschitz bound
- "Any source can change the output by at most 4.7"
- Compare to leave-one-out — the bound is loose but correct

### Part 2: `LipschitzTensor` — tracking bounds through composition (~400 lines)

- A numpy array wrapper carrying a `.lip` scalar
- Every operation propagates the bound: matmul, add, gelu, softmax, layer_norm
- Chain rule: composed operations multiply their Lipschitz constants
- Spectral norm caching (SVD is expensive, weights don't change)

### Part 3: Apply to GPT-2 via picoGPT (~150 lines)

- `gpt2_lipschitz.py` — drop-in replacement for picoGPT's forward pass
- Run it: "end-to-end sensitivity = 2.2 × 10^83"
- Print per-layer constants, see them multiply
- Compare to the ensemble approach: "the ensemble gave us exact attribution with no bounds needed — this is the cost of the single-model path"

### Part 4: Leave-one-out on GPT-2 (inference) (~50 lines)

- The embedding-space intervention from picoGPT: blend a source's tokens toward neutral
- For each source: re-run forward pass with that source muted
- Measure |logit_full[predicted_token] - logit_without[predicted_token]|
- Same colored bars — now on a transformer

### The Artifact

`ch07.py` + `lipschitz_tensor.py` + `gpt2_lipschitz.py`. The 10^83 number hits in the first 5 minutes. Then the student builds up the machinery that produces it. Leave-one-out gives empirical attribution; Lipschitz gives worst-case bounds.

## Key Ideas

1. **Single-model = all sources in one context window.** Standard RAG setup.
2. **Inference attribution:** weights are frozen, only input varies. We CAN bound this.
3. **Lipschitz continuity:** ||f(x) - f(y)|| ≤ L · ||x - y||
4. **For linear f(x) = Wx:** L = ||W||₂ = largest singular value
5. **For composition f ∘ g:** L = L_f × L_g — they multiply
6. **Deep networks have huge bounds** (10^83 for GPT-2) — this is the cost of depth
7. **Leave-one-out gives empirical attribution** (what DID happen). **Lipschitz gives worst-case** (what COULD happen). Both are useful.

## Assets Inherited (from Ch6)

- The web app (will be extended with single-model mode)
- `rdp_accountant.py` from Ch5
- The linear model from Ch2
- `corpus.py` and visualization

## Assets Produced (for Ch8)

- `lipschitz_tensor.py` — the LipschitzTensor class
- `gpt2_lipschitz.py` — GPT-2 with Lipschitz tracking
- Leave-one-out for GPT-2 inference
- The Lipschitz bound value that Ch8 needs to calibrate noise
- The student's understanding: "I have the bound. Now I need noise proportional to it."
