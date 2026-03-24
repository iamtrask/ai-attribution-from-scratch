# Chapter 3: Bounding the Votes — Sensitivity Bounds

## The Idea

Leave-one-out (Ch2) tells you what a source DID influence. But what COULD it influence? If a source could secretly flip the output in a way leave-one-out doesn't catch, that's a problem. We need a **mathematical bound**: "no matter what this source contains, it can change the output by at most THIS much."

For a linear function f(x) = Wx, this is trivially ||W|| (the spectral norm). For a 12-layer transformer, it's the product of 12 spectral norms — and it's 10^83. Same math, wildly different scale.

**Dopamine hit first:** Compute the GPT-2 bound in the first 5 minutes. See 10^83. React. THEN explain where it comes from.

## What the Student Builds

### Part 1: Lipschitz bound on the linear model from Ch2 (~30 lines)

- Take the trained linear regression from Ch2
- Compute `np.linalg.svd(W)` → largest singular value = Lipschitz bound
- Print: "Any source can change the output by at most 4.7"
- Compare to the leave-one-out measurements from Ch2 — the bound is loose but correct

### Part 2: `LipschitzTensor` — tracking bounds through composition (~400 lines)

- A numpy array wrapper that carries a scalar `.lip` field
- Every operation (matmul, add, gelu, softmax, layer_norm) propagates the bound
- Chain rule: when operations compose, Lipschitz constants multiply
- Spectral norm caching (SVD is expensive, weights don't change)

### Part 3: Apply to GPT-2 (~150 lines)

- `gpt2_lipschitz.py` — drop-in replacement for picoGPT's `gpt2.py`
- Same forward pass, now returns `logits.lip` = end-to-end sensitivity
- Print: "Any source can change the output by at most 2.2 × 10^83"
- Show WHY: print per-layer Lipschitz constants, see them multiply

### The Artifact

`ch03.py` + `lipschitz_tensor.py` + `gpt2_lipschitz.py`. Run it, see the bound on both models. The gap between 4.7 (linear) and 10^83 (transformer) is visceral.

## Key Ideas

1. **Lipschitz continuity:** ||f(x) - f(y)|| ≤ L · ||x - y|| — bounded sensitivity
2. **For linear f(x) = Wx:** L = ||W||₂ = largest singular value (via SVD)
3. **For composition f ∘ g:** L = L_f × L_g — they multiply (chain rule)
4. **Per-operation Lipschitz constants:**
   - Linear layer: spectral norm of weight matrix
   - ReLU: 1.0
   - GELU: ~1.7 (empirical max derivative)
   - Softmax: ≤1.0 (spectral norm of Jacobian)
   - Layer norm: data-dependent (involves inv_std)
5. **The bound is loose but provable** — it's worst-case over ALL possible inputs
6. **Deep networks have huge bounds** because you multiply many layers
7. **This is why we need noise** — the bound tells us sensitivity, calibrated noise masks it

## Assets Inherited (from Ch2)

- The trained linear model (for the simple Lipschitz demo)
- Leave-one-out function (for comparing empirical influence vs theoretical bound)
- `corpus.py` and the bar-chart visualization
- The animal-in-room corpus

## Assets Produced (for Ch4)

- `lipschitz_tensor.py` — the `LipschitzTensor` class (reused in Ch4-6)
- `gpt2_lipschitz.py` — GPT-2 with Lipschitz tracking (reused in Ch4-6)
- The Lipschitz bound value for GPT-2 — the exact number Ch4 needs to calibrate noise
- The student's understanding: "I know the worst case. Now I need to add noise proportional to it."

## Pacing Note

**Start with the 10^83 number.** Run GPT-2 Lipschitz in the first cell. See the number. React. "That's insane. Where does it come from?" Then step back to the linear model where the bound is 4.7 and build intuition. Then walk through LipschitzTensor operation by operation. Then come back to GPT-2 and show the per-layer multiplication that produces 10^83.
