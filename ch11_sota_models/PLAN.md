# Chapter 11: SOTA Models — DeepSeek-R1 and Beyond

## The Idea

Qwen 0.6B is a teaching model. DeepSeek-R1 0528 is what people actually deploy. The distilled 7B version runs on a laptop. Let's make the full attribution pipeline work at that scale — both ensemble and single-model paths.

This chapter also brings the two paths together: use the ensemble for training attribution (each model trained on different data) and single-model for inference attribution (all sources in one context). Combined, you get end-to-end attribution on SOTA models.

## What the Student Builds

### Part 1: KV-Cache-Aware Leave-One-Out

- Standard inference caches K/V tensors
- For leave-one-out: reuse the cached KVs, only recompute from the divergence point
- O(L + N × L_partial) instead of O(N × L)

### Part 2: Batched Ablation

- Stack all N source-removal inputs into one batched forward pass
- On GPU: N sequential passes → 1 batched pass

### Part 3: llama.cpp Integration

- For maximum throughput: hook into llama.cpp's C inference engine
- Quantized inference (Q4_K_M) for 7B models in 4GB RAM
- Keep attribution logic in Python, forward pass in C

### Part 4: DeepSeek-R1 Specifics

- MoE: only active experts matter for attribution → natural ensemble structure!
- Distilled 7B: dense, runs on laptop
- Reasoning traces: attribute through the chain of thought

### Part 5: Full Pipeline

- Training: PATE across data partitions (Ch9)
- Inference: single-model with Lipschitz bounds + DP (Ch7-8)
- Ensemble: weighted voting for multi-model setups (Ch3-5)
- App: URLs, citations, colored bars, budgets (Ch6)
- Speed: GPU-accelerated (Ch10)
- Model: DeepSeek-R1 (this chapter)

### The Artifact

The complete citation engine running DeepSeek-R1 distill-7B. Paste URLs, ask questions, get attributed answers with privacy guarantees. The culmination of the entire course.

## Key Ideas

1. **KV-cache reuse** makes leave-one-out practical on large models
2. **Batched ablation** turns N passes into 1
3. **MoE = natural ensemble.** Each expert is like a separate model in the PATE framework.
4. **Quantization-aware bounds:** 4-bit weights have different spectral norms
5. **The two paths merge:** ensemble for training attribution, single-model for inference attribution
6. **llama.cpp:** C for speed, Python for attribution logic. Language progression from the course.

## Assets Inherited (from Ch10)

- GPU-accelerated implementations (MLX)
- The complete app with both modes
- The full DP pipeline (ensemble + single-model)

## Assets Produced

- KV-cache-aware attribution
- Batched ablation engine
- llama.cpp integration
- The final artifact: a production-grade citation engine on SOTA models

## Target Models

- DeepSeek-R1 0528 distill-7B (laptop, MLX or llama.cpp)
- DeepSeek-R1 0528 full 671B (server, CUDA — stretch goal)
- Qwen3 0.6B → 4B (MLX, already done in Ch10)
