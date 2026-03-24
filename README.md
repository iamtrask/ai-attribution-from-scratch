# Attribution from Scratch

**Let's build a citation engine.**

A course on understanding which sources an LLM uses — from scratch, in code. We start with a bag-of-words vote counter and build up to a GPU-accelerated, differentially-private attribution engine running on state-of-the-art open-source models.

This series picks up where Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) leaves off. He teaches you how to build an LLM. We teach you how to trust one.

**Prerequisites:** Python, basic linear algebra, and familiarity with how neural networks work (e.g. from Karpathy's series or equivalent).

---

## Syllabus

### Part I: Voting

| # | Chapter | What You Build |
|---|---------|---------------|
| 1 | [Counting Votes](ch01_counting_votes/) | N-gram model with perfect attribution — just read off who voted |
| 2 | [Weighted Votes, Then Blended](ch02_weighted_and_blended/) | Perceptron → linear SGD. Attribution works, then breaks. Leave-one-out as the fix |

### Part II: Ensemble Attribution

| # | Chapter | What You Build |
|---|---------|---------------|
| 3 | [Ensemble Voting](ch03_ensemble_voting/) | Multiple models, each sees one source. Vote counting = attribution. It's back! |
| 4 | [Weighted Ensemble](ch04_weighted_ensemble/) | FTPL adaptive weights — multi-document reasoning. Weights = attribution |
| 5 | [Private Ensemble](ch05_private_ensemble/) | GNMax noise + RDP accountant. Build autodp from scratch. Per-source budgets |
| 6 | [The Citation App](ch06_citation_app/) | Web app + URL ingestion. Paste URLs, get attributed answers with citations |

### Part III: Single-Model Attribution

| # | Chapter | What You Build |
|---|---------|---------------|
| 7 | [Sensitivity Bounds](ch07_single_model_bounds/) | Lipschitz bounds: linear → GPT-2. Build `LipschitzTensor`. The 10^83 number |
| 8 | [Single-Model Privacy](ch08_single_model_privacy/) | DP noise calibrated to Lipschitz bounds. Per-individual accounting. Both modes in one app |

### Part IV: Going Deeper, Going Faster

| # | Chapter | What You Build |
|---|---------|---------------|
| 9 | [Training Attribution](ch09_training_attribution/) | DP-SGD + PATE. Where do the weights come from? End-to-end attribution |
| 10 | [GPU Acceleration](ch10_gpu_acceleration/) | MLX on Apple Silicon. 40x speedup. Qwen3 0.6B at interactive speed |
| 11 | [SOTA Models](ch11_sota_models/) | DeepSeek-R1 + llama.cpp. KV-cache attribution. The full production pipeline |

---

## The Arc

```
PART I — VOTING (simple models, build intuition)
  Ch 1: n-gram votes              "attribution = counting who voted"
  Ch 2: linear SGD breaks it      "SGD blends the votes — now what?"

PART II — ENSEMBLE (multiple models, attribution through architecture)
  Ch 3: ensemble voting            "don't blend — give each source its own model"
  Ch 4: weighted ensemble / FTPL   "weight the votes for multi-doc reasoning"
  Ch 5: private ensemble / GNMax   "make the votes private with DP"
  Ch 6: the app + URLs             "paste URLs, get cited answers" ← FIRST WORKING PRODUCT

PART III — SINGLE MODEL (one model, all sources in context)
  Ch 7: Lipschitz bounds           "bound how much any source CAN influence the output"
  Ch 8: DP for inference           "noise + budgets inside one model"

PART IV — SCALE (training, speed, SOTA)
  Ch 9: training attribution       "where do the weights come from? DP-SGD + PATE"
  Ch 10: GPU acceleration          "40x faster on your laptop GPU"
  Ch 11: SOTA models               "DeepSeek-R1 on real documents"
```

## Two Paths to Attribution

The course teaches two complementary approaches:

| | Ensemble Path (Ch 3-6) | Single-Model Path (Ch 7-8) |
|---|---|---|
| **How it works** | Each source gets its own model. Models vote. | All sources in one context window. Bound the forward pass. |
| **Attribution** | Count/weight the votes | Leave-one-out + Lipschitz bounds |
| **Privacy** | GNMax noise on vote tallies | Gaussian noise on logits, calibrated to L |
| **Advantage** | Simple, exact, works on black-box models | One model, standard RAG setup |
| **Disadvantage** | N model instances | Huge sensitivity bounds, needs per-individual accounting |

They share the same RDP accountant (Ch5) and merge in the app (Ch6/Ch8).

## The Running Example

10 documents about animals in rooms. Used in every chapter.

> "The cat is in the kitchen." · "The dog is in the garden." · "The hamster is in the bedroom." · ...

Query: **"The hamster is in the"** → model predicts **"bedroom"** → colored bars show the Hamster Report drove it.

## Existing Code

The [picoGPT intelligence-budgets branch](https://github.com/jaymody/picoGPT) contains working implementations for Ch7-8 and Ch10:
- `gpt2.py`, `gpt2_lipschitz.py`, `lipschitz_numpy.py` → Ch7
- `dp_inference.py` → Ch8
- `gpt2_mlx.py`, `qwen_mlx.py`, `lipschitz_mlx.py` → Ch10
- `chat_app.py`, `chat_ui.html` → Ch6/Ch8

The [deep_voting](https://github.com/iamtrask/deep_voting) repo contains the ensemble system for Ch3-5.
