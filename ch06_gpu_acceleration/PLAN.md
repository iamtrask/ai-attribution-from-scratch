# Chapter 6: Going Fast — GPU Acceleration

## The Idea

Palate cleanser after two dense DP chapters (Ch4-5). Everything works, everything is slow. MLX gives 40-50× speedup. Upgrade to Qwen3 0.6B. Fun, visual, satisfying.

## What the Student Builds

1. **MLX port of GPT-2** — `gpt2_mlx.py`, nearly 1:1 with numpy (~100 lines)
2. **Fused Lipschitz** — scalar side-channel, zero overhead. The key optimization. (~200 lines)
3. **MLX Qwen3 0.6B** — RoPE, RMSNorm, SwiGLU, GQA (~150 lines)
4. **Benchmark** — the table. NumPy 600ms → MLX 15ms. (~80 lines)
5. **Accelerated ensemble** — batched inference for N models in one pass
6. **App upgrade** — both modes at interactive speed on Qwen

### The Artifact

The benchmark table. The app running Qwen at ~15ms/token. Both modes fast.

## Key Ideas

1. **Unified memory:** no CPU↔GPU copies
2. **Fused Lipschitz:** naive wrappers kill GPU perf. Scalar side-channel fixes it.
3. **Batched inference:** N ensemble members in one pass

## Assets Inherited (from Ch5)

- The complete app with both private modes

## Assets Produced (for Ch7)

- `gpt2_mlx.py`, `qwen_mlx.py`, `lipschitz_mlx.py`
- The app at interactive speed
- Emotional recharge for Ch7's training theory
