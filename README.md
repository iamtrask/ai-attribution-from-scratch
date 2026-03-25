# Attribution from Scratch

**Let's build a citation engine.**

A course on understanding which sources an LLM uses — from scratch, in code. We start with a vote counter and build up to a GPU-accelerated, differentially-private attribution engine running on state-of-the-art open-source models.

This series picks up where Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) leaves off. He teaches you how to build an LLM. We teach you how to trust one.

**Prerequisites:** Python, basic linear algebra, and familiarity with how neural networks work (e.g. from Karpathy's series or equivalent).

---

## Syllabus

### Part I: Counting Votes

| # | Chapter | What You Build |
|---|---------|---------------|
| 1 | [Counting Votes](ch01_counting_votes/) | N-gram model with perfect attribution — just read off who voted |

### Part II: Ensemble Attribution

| # | Chapter | What You Build |
|---|---------|---------------|
| 2 | [Ensemble Voting](ch02_ensemble_voting/) | Multiple LLMs, each sees one source. Vote counting = attribution |
| 3 | [Weighted Ensemble](ch03_weighted_ensemble/) | FTPL adaptive weights for multi-document reasoning. Weights = attribution |
| 4 | [Private Ensemble](ch04_private_ensemble/) | GNMax noise + RDP accountant. Build autodp from scratch |
| 5 | [The Citation App](ch05_citation_app/) | Web app + URLs. Paste links, get attributed answers with citations |

### Part III: Single-Model Attribution

| # | Chapter | What You Build |
|---|---------|---------------|
| 6 | [Votes Break](ch06_votes_break/) | Linear SGD blends sources. Leave-one-out. The single-model problem. |
| 7 | [Sensitivity Bounds](ch07_sensitivity_bounds/) | Lipschitz bounds: linear (4.7) → GPT-2 (10^83). Build `LipschitzTensor` |
| 8 | [Single-Model Privacy](ch08_single_model_privacy/) | DP noise calibrated to Lipschitz bounds. Both modes in one app |

### Part IV: Going Deeper, Going Faster

| # | Chapter | What You Build |
|---|---------|---------------|
| 9 | [Training Attribution](ch09_training_attribution/) | DP-SGD + PATE. Where do the weights come from? Full end-to-end |
| 10 | [GPU Acceleration](ch10_gpu_acceleration/) | MLX on Apple Silicon. 40x speedup. Qwen3 0.6B |
| 11 | [SOTA Models](ch11_sota_models/) | DeepSeek-R1 + llama.cpp. The full production pipeline |

---

## The Arc

```
PART I — VOTING
  Ch 1: n-gram votes                "attribution = counting who voted"

PART II — ENSEMBLE (multiple models, attribution through architecture)
  Ch 2: ensemble voting              "each source gets its own model"
  Ch 3: weighted ensemble / FTPL     "weight the votes for multi-doc reasoning"
  Ch 4: private ensemble / GNMax     "make the votes private — build autodp"
  Ch 5: the app + URLs               "paste URLs, get cited answers"  ← WORKING PRODUCT

PART III — SINGLE MODEL (one model, all sources in context)
  Ch 6: linear SGD                   "votes break when sources blend"
  Ch 7: Lipschitz bounds             "bound how much any source CAN do"
  Ch 8: DP for inference             "noise + budgets inside one model"

PART IV — SCALE
  Ch 9: training attribution         "DP-SGD + PATE (ensemble for training!)"
  Ch 10: GPU acceleration            "40x faster on your laptop"
  Ch 11: SOTA models                 "DeepSeek-R1 on real documents"
```

## Two Paths to Attribution

| | Ensemble Path (Ch 2-5) | Single-Model Path (Ch 6-8) |
|---|---|---|
| **How** | Each source gets its own model. Models vote. | All sources in one context. Bound the forward pass. |
| **Attribution** | Count/weight the votes | Leave-one-out + Lipschitz bounds |
| **Privacy** | GNMax noise on vote tallies | Gaussian noise on logits, calibrated to L |
| **Advantage** | Simple, exact, black-box | One model, standard RAG, cross-source reasoning |
| **Disadvantage** | N model instances | Huge sensitivity bounds |

They share the same RDP accountant (Ch4) and merge in the app (Ch5/Ch8).

Ch9 connects them: PATE applies the ensemble approach to TRAINING — the course comes full circle.

## The Running Example

10 documents about animals in rooms. Used in every chapter.

> "The cat is in the kitchen." · "The dog is in the garden." · "The hamster is in the bedroom." · ...

Query: **"The hamster is in the"** → model predicts **"bedroom"** → colored bars show the Hamster Report drove it.
