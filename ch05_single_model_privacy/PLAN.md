# Chapter 5: Single-Model Privacy — DP for Inference

## The Idea

Ch4 gave us L. Now add noise, reuse Ch3's RDP accountant, get privacy guarantees inside one model. Start on the linear model (gentle noise), then GPT-2 (noise destroys output → per-individual accounting saves it).

## What the Student Builds

1. **Gaussian mechanism on linear model** — L≈4.7, noise gentle (~30 lines)
2. **Same on GPT-2** — L≈10^83, output is garbage (~30 lines)
3. **Per-individual accounting** (Feldman & Zrnic) — cost depends on actual embedding norm (~100 lines)
4. **Budget-driven generation** — sources fade when exhausted (~80 lines)
5. **App upgrade** — single-model mode gets the same budget controls as ensemble (~30 lines)

### The Artifact

Both modes fully private. Same budget UI, different mechanisms underneath.

## Key Ideas

1. **Same `rdp_accountant.py` from Ch3** — different L, same framework
2. **Per-individual accounting** saves the single-model path
3. **Two paths, one accountant, one app**

## Assets Inherited (from Ch4)

- `lipschitz_tensor.py`, `gpt2_lipschitz.py`, `rdp_accountant.py`

## Assets Produced (for Ch6)

- `dp_inference.py` — complete DP inference pipeline
- Everything works, everything is slow → Ch6 (GPU, the fun chapter)
