# Chapter 8: Single-Model Privacy — DP for Inference

## The Idea

Ch7 gave us L. Now add noise proportional to L, reuse the RDP accountant from Ch4, and get privacy guarantees inside one model. Same `rdp_accountant.py`, different sensitivity source.

## What the Student Builds

1. **Gaussian mechanism on Ch6's linear model** — L≈4.7, noise is gentle, works great
2. **Same on GPT-2** — L≈10^83, noise is astronomical, output is garbage
3. **Per-individual accounting** (Feldman & Zrnic) — each source's cost depends on its embedding norm, not worst case. Saves the single-model path.
4. **Budget-driven generation** — tokens spend budget, sources fade when exhausted
5. **Both modes in one app** — ensemble (Ch2-4) + single-model, same RDP accountant, same UI

### The Artifact

The Ch5 app now supports both modes. Toggle between ensemble attribution and single-model attribution. Same colored bars, same budgets, different mechanisms under the hood.

## Key Ideas

1. **Same RDP framework from Ch4**, different L value
2. **Per-individual accounting** makes the single-model path viable
3. **Two paths, one accountant:** ensemble (vote noise) vs single-model (logit noise)
4. **The app grows:** now has two attribution backends

## Assets Inherited (from Ch7)

- `lipschitz_tensor.py`, `gpt2_lipschitz.py`, `rdp_accountant.py` from Ch4

## Assets Produced (for Ch9)

- `dp_inference.py` — complete DP inference pipeline
- The unified app with both modes
- The question: "this handles inference. But what about the training data?" → Ch9
