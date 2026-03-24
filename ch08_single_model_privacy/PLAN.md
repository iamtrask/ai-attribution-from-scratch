# Chapter 8: Single-Model Privacy — DP for Inference

## The Idea

Ch7 gave us the Lipschitz bound L for a single-model forward pass. Now add Gaussian noise proportional to L, track per-source RDP budgets, and get the same privacy guarantees as the ensemble path — but inside one model.

The RDP accountant from Ch5 is reused directly. The new piece: calibrating noise to the (much larger) sensitivity of a transformer forward pass, and per-individual accounting to keep the budgets manageable.

## What the Student Builds

### Part 1: Gaussian mechanism on the linear model (~30 lines)

- L ≈ 4.7 from Ch7. Noise ~ N(0, σ² × 4.7²). Barely perturbs the output.
- Run 20 queries with ε=5. Budget depletes gradually. Works great.

### Part 2: Same thing on GPT-2 (~30 lines)

- L ≈ 10^83. Noise ~ N(0, σ² × (10^83)²). Output is pure garbage.
- "The bound is too loose for a transformer. We need tighter tools."

### Part 3: Per-individual accounting (~100 lines)

- Feldman & Zrnic (NeurIPS 2021): each source's RDP cost depends on its actual embedding norm, not the worst case
- Source i's cost: ρᵢ(α) = α × L² × ||xᵢ||² / (2σ²)
- Sources with small embeddings cost less → their budgets last longer
- Build `PerIndividualAccountant` — reuses `rdp_accountant.py` from Ch5 but tracks per-source norms

### Part 4: Budget-driven generation (~100 lines)

- Generate tokens one at a time
- Each token: measure influence (leave-one-out), record privacy cost, check budgets
- When a source exhausts: blend its embeddings toward neutral (fade out)
- The text shifts from specific to generic as sources deplete

### Part 5: Both modes in one system (~50 lines)

- Ensemble mode (Ch3-5): N models, vote counting, GNMax noise
- Single-model mode (Ch7-8): one model, all sources in context, Lipschitz noise
- Same `rdp_accountant.py` powers both. Same budget UI. Different attribution mechanism under the hood.

### The Artifact

`ch08.py` + `dp_inference.py`. Generate 20 tokens, watch sources fade. The student sees both the ensemble path and the single-model path side by side, with the same privacy guarantees.

## Key Ideas

1. **Same DP framework, different sensitivity.** The RDP accountant from Ch5 works identically — only the L value changes.
2. **Per-individual accounting saves the single-model path.** Without it, the 10^83 bound makes everything useless.
3. **Two paths, one framework:** ensemble (vote noise) vs single-model (logit noise). Same RDP. Same budgets.
4. **Training vs inference attribution diverge here.** Inference: we bound input sensitivity (L). Training: we'd need to bound gradient sensitivity (DP-SGD) — that's Ch9.

## Assets Inherited (from Ch7)

- `lipschitz_tensor.py`, `gpt2_lipschitz.py` — Lipschitz tracking
- Leave-one-out for GPT-2
- `rdp_accountant.py` from Ch5

## Assets Produced (for Ch9)

- `dp_inference.py` — complete DP inference pipeline with per-individual accounting
- The unified system supporting both ensemble and single-model modes
- The question: "this handles inference (frozen weights). But what about the training data that BUILT the weights?" → Ch9
