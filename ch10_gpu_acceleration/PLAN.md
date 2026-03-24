# Chapter 10: Going Fast — GPU Acceleration

## The Idea

Everything works but everything is slow. The ensemble runs N models sequentially. The single-model path does N+1 forward passes per token. Lipschitz tracking adds overhead. Time to use the GPU.

MLX on Apple Silicon (40-50× speedup), CUDA on Linux. Same code patterns for both paths (ensemble + single-model).

## What the Student Builds

### Part 1: MLX port of the single-model path (~400 lines)

- `gpt2_mlx.py` — GPT-2 forward pass on GPU
- `lipschitz_mlx.py` — Lipschitz tracking with zero overhead (fused scalar side-channel)
- `qwen_mlx.py` — upgrade to Qwen3 0.6B (real modern model)
- Benchmark: numpy (600ms/token) → MLX (15ms/token)

### Part 2: Fused Lipschitz — the key optimization

- Naive LipschitzTensor on GPU: 500× overhead (GPU pipeline stalls from `.item()` calls)
- The fix: compute Lipschitz bounds as mx.array scalars alongside the forward pass
- Precompute spectral norms at load time
- Per-block `mx.eval()` to prevent graph explosion
- Result: Lipschitz overhead ≈ 0

### Part 3: Accelerated ensemble

- Batched inference: stack all N model inputs, run one batched forward pass
- On GPU this is nearly free (N small models ≈ 1 large model in batch)
- Or: one model, N different contexts → batched forward pass

### Part 4: Wire into the app

- The Ch6 app now runs on GPU with both modes
- Qwen3 0.6B as the default model
- Interactive speed: real-time attribution bars

### The Artifact

The Ch6 app running at ~15ms/token on a laptop GPU. Benchmark script showing the speedup table. Both ensemble and single-model modes accelerated.

## Key Ideas

1. **Unified memory on Apple Silicon:** no CPU↔GPU copies
2. **Lazy evaluation:** build graph, execute at once
3. **Fused Lipschitz:** scalar side-channel ops have zero overhead on GPU
4. **Batched inference:** N ensemble members in one forward pass
5. **Why not JAX?** jax-metal is broken on Apple Silicon. MLX is stable and Apple-native.

## Assets Inherited (from Ch9)

- The complete attribution system (ensemble + single-model + training)
- `chat_app.py` + `chat_ui.html`
- `dp_inference.py`, `rdp_accountant.py`
- `lipschitz_tensor.py`, `gpt2_lipschitz.py`

## Assets Produced (for Ch11)

- `gpt2_mlx.py`, `qwen_mlx.py`, `lipschitz_mlx.py` — GPU-accelerated implementations
- `benchmark_mlx.py`
- The app running at interactive speed
- The student's understanding of GPU optimization patterns
- "Qwen 0.6B is fast but small. Can we run this on a REAL model?" → Ch11
