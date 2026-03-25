# Attribution from Scratch

**Let's build a citation engine.**

A course on understanding which sources an LLM uses — from scratch, in code. We start with a vote counter and build up to a GPU-accelerated, differentially-private attribution engine running on state-of-the-art open-source models.

This series picks up where Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) leaves off. He teaches you how to build an LLM. We teach you how to trust one.

**Prerequisites:** Python, basic linear algebra, and familiarity with how neural networks work (e.g. from Karpathy's series or equivalent).

---

## Syllabus

### Part I: Ensemble Attribution

| # | Chapter | What You Build |
|---|---------|---------------|
| 1 | [Ensemble Attribution](ch01_ensemble_attribution/) | N-gram votes → LLM ensemble → **a minimal web app**. Working product in chapter 1. |
| 2 | [Weighted Voting](ch02_weighted_voting/) | FTPL adaptive weights. Multi-document reasoning. |
| 3 | [Private Voting](ch03_private_voting/) | GNMax + RDP (autodp from scratch). **The real app with URLs and budgets.** |

### Part II: Single-Model Attribution

| # | Chapter | What You Build |
|---|---------|---------------|
| 4 | [The Single-Model Path](ch04_single_model/) | Linear SGD (5 min) → LipschitzTensor → GPT-2 (10^83). Both modes in one app. |
| 5 | [Single-Model Privacy](ch05_single_model_privacy/) | DP noise calibrated to Lipschitz bounds. Per-individual accounting. |

### Part III: Going Deeper, Going Faster

| # | Chapter | What You Build |
|---|---------|---------------|
| 6 | [GPU Acceleration](ch06_gpu_acceleration/) | MLX on Apple Silicon. 40x speedup. Qwen3 0.6B. (The fun chapter.) |
| 7 | [Training Attribution](ch07_training_attribution/) | PATE = Ch1's ensemble applied to training. End-to-end attribution. |
| 8 | [SOTA Models](ch08_sota_models/) | DeepSeek-R1 + llama.cpp. KV-cache attribution. The full pipeline. |

---

## The Arc

```
PART I — ENSEMBLE
  Ch 1: n-gram → LLM ensemble → minimal app     "attribution = counting votes"
  Ch 2: FTPL weighted voting                      "blend sources for complex questions"
  Ch 3: GNMax + RDP → real app + URLs             "make the votes private"

PART II — SINGLE MODEL (going cheaper)
  Ch 4: linear SGD → Lipschitz → GPT-2            "bound influence inside one model"
  Ch 5: DP for inference                           "noise + budgets, same RDP accountant"

PART III — SCALE (hard/easy/hard/easy pacing)
  Ch 6: GPU acceleration (fun!)                    "40x faster"
  Ch 7: training attribution (deep)                "PATE — full circle to Ch1"
  Ch 8: DeepSeek-R1 (capstone)                     "the production pipeline"
```

## Three Datasets

### 1. The Voting Dataset (Ch1 only — disposable)
```
"I believe the best pet is a cat"
"I believe the best pet is a dog"
"I believe the best pet is a hamster"
```
Identical prefix, different completions. Shows vote counting mechanics in 5 minutes. Never used again.

### 2. The Rooms Dataset (Ch1 onward — the Shakespeare)
```
"The cat is in the kitchen. The kitchen is on the first floor."
"The dog is in the garden. The garden has a pond."
"The hamster is in the bedroom. The bedroom is upstairs."
...
```
10 animals, 10 rooms, extra facts for multi-hop questions. Verifiable ground truth. Simple enough for a slide, rich enough for every chapter.

### 3. The Conflicting Facts Dataset (Ch3 onward — the "holy shit" moment)
```
Source A: "The population of Lagos is 16.6 million (2023 estimate)"
Source B: "The population of Lagos is 8.0 million (2006 census)"
Question: "What is the population of Lagos?"
```
The model picks one. The bars show which. If it picks the outdated source, you see it. Set a budget to limit untrusted sources. This is where attribution stops being academic.

## Two Paths to Attribution

| | Ensemble Path (Ch 1-3) | Single-Model Path (Ch 4-5) |
|---|---|---|
| **How** | Each source gets its own model. Models vote. | All sources in one context. Bound the forward pass. |
| **Attribution** | Count/weight the votes | Leave-one-out + Lipschitz bounds |
| **Privacy** | GNMax noise on vote tallies | Gaussian noise on logits, calibrated to L |
| **Advantage** | Simple, exact, black-box | One model, cheaper, cross-source reasoning |
| **Disadvantage** | N model instances | Huge sensitivity bounds |

They share the same RDP accountant (Ch3/Ch5) and live in the same app. Ch7 connects them: PATE applies the ensemble to training.
