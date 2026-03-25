# Chapter 7: Sensitivity Bounds — Lipschitz from Linear to Transformer

## The Idea

Ch6 showed the Lipschitz bound on a linear model: just ||W||. Now scale that idea to a transformer. Same math, chained through 12 layers. The bound goes from 4.7 to 10^83.

**Start with the punchline:** compute GPT-2's Lipschitz bound in the first 5 minutes. See 10^83. Then explain where it comes from.

## What the Student Builds

1. **Lipschitz on Ch6's linear model** — `svd(W)`, largest singular value, done (~30 lines)
2. **`LipschitzTensor`** — numpy wrapper that tracks bounds through every operation (~400 lines)
   - matmul: spectral norm of weight matrix
   - gelu: ~1.7, softmax: ≤1, layer_norm: data-dependent
   - Chain rule: composed operations multiply
3. **Apply to GPT-2** via picoGPT — `gpt2_lipschitz.py` drop-in replacement (~150 lines)
4. **Leave-one-out + bounds side by side** — empirical (what DID happen) vs worst-case (what COULD happen)

### The Artifact

The 10^83 number. Per-layer breakdown showing the multiplication. Comparison to Ch6's linear bound (4.7). The student sees: same concept, wildly different scale.

## Key Ideas

1. **Lipschitz continuity:** ||f(x) - f(y)|| ≤ L · ||x - y||
2. **Linear: L = ||W||₂.** Transformer: L = product of all layers' spectral norms.
3. **Chain rule: L_total = L₁ × L₂ × ... × L₁₂** — they multiply!
4. **The bound is loose but provable.** That's why we need noise (Ch8).

## Assets Inherited (from Ch6)

- The linear model, leave-one-out, `corpus.py`

## Assets Produced (for Ch8)

- `lipschitz_tensor.py`, `gpt2_lipschitz.py`
- The L value that Ch8 calibrates noise to
