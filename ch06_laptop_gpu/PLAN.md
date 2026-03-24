# Chapter 6: Going Fast — Laptop GPU

## The Idea

The Ch5 app takes ~600ms per token on numpy. Your laptop has a GPU sitting idle. MLX (Apple's array framework) has a nearly identical API to numpy but runs on Metal. Port everything, see what happens: 40-50× speedup, zero algorithmic changes.

Then upgrade from GPT-2 (124M, 2019) to Qwen3 0.6B (2025) — a real modern model that actually gives good answers.

## What the Student Builds

### Part 1: MLX port of GPT-2 forward pass

- `gpt2_mlx.py` — nearly 1:1 with `gpt2.py`, `np` → `mx`
- Batched multi-head attention (all heads at once, not a Python loop)
- `convert_params_to_mlx()` — weight conversion utility

### Part 2: MLX Lipschitz tracking

- The naive approach: `LipschitzTensor` with `.item()` calls → 500× overhead (GPU pipeline stalls)
- The fix: fuse Lipschitz bounds as scalar side-channel ops alongside the forward pass
- Precompute spectral norms at load time
- Per-block `mx.eval()` to prevent graph explosion
- Result: Lipschitz overhead ≈ 0 (bounds are free)

### Part 3: Benchmark

- `benchmark_mlx.py` — times numpy vs MLX, plain vs Lipschitz
- Print the table. Moment of satisfaction.

### Part 4: Qwen3 0.6B

- `qwen_mlx.py` — modern architecture: RoPE, RMSNorm, SiLU/SwiGLU, GQA
- Load weights from `.npz`, convert to MLX
- Wire into the Ch5 app as the default backend

### The Artifact

The Ch5 app, now running Qwen 0.6B on your laptop GPU at ~15ms/token. Same UI, same colored bars, dramatically faster. The benchmark script shows the numbers.

## Key Ideas

1. **Unified memory on Apple Silicon:** CPU and GPU share RAM — no copies
2. **Lazy evaluation:** MLX builds a computation graph, executes all at once
3. **Why `.item()` kills performance:** each call forces a GPU sync, breaking the pipeline
4. **The fused-lip trick:** compute Lipschitz bounds as mx.array scalars alongside the forward pass, evaluate them together with the logits in one `mx.eval()` call
5. **Per-block eval:** prevents the lazy graph from growing too large (graph explosion)
6. **Why not JAX?** jax-metal is broken/unmaintained on Apple Silicon. MLX is Apple-native and stable.
7. **Modern model architecture:** RoPE replaces learned positional embeddings, RMSNorm replaces LayerNorm, SwiGLU replaces GELU FFN, GQA uses fewer KV heads

## Assets Inherited (from Ch5)

- `chat_app.py` + `chat_ui.html` — the web app
- `gpt2.py`, `gpt2_lipschitz.py`, `lipschitz_tensor.py` — numpy implementations
- `dp_inference.py`, `rdp_accountant.py` — DP pipeline
- `corpus.py`

## Assets Produced (for Ch7)

- `gpt2_mlx.py` — MLX GPT-2 (plain + fused Lipschitz)
- `qwen_mlx.py` — MLX Qwen forward pass
- `lipschitz_mlx.py` — MLX Lipschitz tracking
- `benchmark_mlx.py` — benchmarking script
- The app running on Qwen at interactive speed — Ch7 upgrades the model to DeepSeek-R1
- The student's understanding of GPU-specific optimization patterns (lazy eval, sync avoidance, graph management)

## Pacing Note

The benchmark table is the dopamine hit. Show it early: "here's what we're about to achieve." Then walk through the port. The Qwen architecture can be taught lightly — the student already knows transformers from Karpathy, Qwen just has different building blocks (RoPE vs learned PE, etc.).
