# Chapter 7: Going Faster — SOTA Models

## The Idea

Qwen 0.6B is a teaching model. DeepSeek-R1 0528 is what people actually use for reasoning tasks. The distilled 7B version runs on a laptop. Let's make our attribution engine work at that scale.

The key challenge: leave-one-out requires N+1 forward passes per token. On a 7B model, that's expensive even on GPU. We need algorithmic improvements, not just hardware.

## What the Student Builds

### Part 1: KV-Cache-Aware Attribution

- Standard inference caches key/value tensors (the KV cache) to avoid recomputation
- Insight: for leave-one-out, the "full" forward pass and the "without source i" pass share most of the KV cache — they only diverge at the positions where source i's tokens live
- Build: cache the full-pass KVs, then for each ablation, only recompute from the divergence point
- Result: leave-one-out goes from O(N × full_pass) to O(full_pass + N × partial_pass)

### Part 2: Batched Ablation

- Stack all N source-removal inputs into one batch
- Run a single batched forward pass on GPU instead of N sequential passes
- On a 7B model with 10 sources: 10× speedup over sequential leave-one-out

### Part 3: llama.cpp Integration

- For maximum throughput on laptop: hook into llama.cpp's C inference engine
- Keep attribution logic in Python, call llama.cpp for the forward pass
- Quantized inference (Q4_K_M) for 7B models that fit in 4GB RAM

### Part 4: DeepSeek-R1 Specifics

- MoE architecture: only ~22B of 671B parameters are active per token
- Attribution through MoE: only the active experts matter for sensitivity bounds
- The distilled 7B variant: dense model, simpler, runs on laptop
- Reasoning traces: DeepSeek-R1 "thinks" before answering — attribute through the chain of thought

### The Artifact

The Ch5 app running DeepSeek-R1 distill-7B with attributed answers. Same colored bars, same UI. Ask it a question with 5 source documents, get a reasoned answer with per-source attribution.

## Key Ideas

1. **KV cache reuse** turns O(N × L) attribution into O(L + N × L_partial)
2. **Batched ablation** turns N sequential passes into 1 batched pass
3. **Quantization-aware bounds:** 4-bit weights have different spectral norms than fp32 — the Lipschitz bound needs adjustment
4. **MoE attribution:** sparse activation means fewer parameters contribute to each output
5. **llama.cpp as a backend:** C for speed, Python for attribution logic
6. **Language transition:** this is where we cross from pure Python into C hooks

## Assets Inherited (from Ch6)

- `qwen_mlx.py`, `gpt2_mlx.py` — MLX inference implementations
- `chat_app.py` + `chat_ui.html` — the web app
- `dp_inference.py`, `rdp_accountant.py` — the full DP pipeline
- The student's understanding of MLX lazy eval and GPU optimization

## Assets Produced (for Ch8)

- KV-cache-aware leave-one-out implementation
- Batched ablation engine
- llama.cpp Python bindings with attribution hooks
- DeepSeek-R1 integration
- The app running on a SOTA model — Ch8 connects it to real documents

## Pacing Note

This is the most technically demanding chapter. Consider splitting into two parts if needed: 7a (KV-cache + batching, stays in Python/MLX) and 7b (llama.cpp + DeepSeek, crosses into C). The student should have a working faster system after 7a even if they skip 7b.
