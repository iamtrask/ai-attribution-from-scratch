# AI Attribution from Scratch — Full Course Plan

**Let's build a citation engine.**

This document tracks the full 8-lecture plan. The main `README.md` only shows lectures that have been built. This shows where we're going.

---

## Part I: Ensemble Attribution

### Lecture 1: What Is Attribution, Why Is It Hard?

**Status: IN PROGRESS** — dataset done (`corpus.py`), plan done, notebook not started

Four beats:
1. **N-gram model** on the Hogwarts headmaster survey → attribution = counting who influenced the prediction
2. **Perceptron** → attribution = input × weight, perfectly decomposable
3. **Logistic regression** → add a sigmoid, attribution breaks. σ(a+b) ≠ σ(a) + σ(b). One nonlinearity is all it takes.
4. **LLM ensemble** → keep sources separate (one model, N prompts). Attribution is easy again. But costs N forward passes.

Dataset: 1000 Hogwarts students ("My name is Harry Potter. As a member of Gryffindor, I think the next headmaster should be McGonagall"). Known HP characters + generated wizard names. Structured data + full sentences.

Key takeaway: attribution is easy when sources don't mix, hard when they do. Two open problems for the rest of the course.

### Lecture 2: Weighted Voting — Multi-Document Reasoning with FTPL

**Status: PLANNED**

Equal-weight voting fails on multi-hop questions. FTPL assigns adaptive weights. Weights = attribution. Uses the rooms dataset for multi-hop ("Which floor is the cat on?" needs two facts).

### Lecture 3: Private Voting — Building autodp from Scratch

**Status: PLANNED**

GNMax adds noise to vote tallies. RDP accountant built step by step (naive ε → Rényi → multi-α → per-source). The viz dashboard arrives for budget controls. Conflicting-facts dataset introduced (capital of Australia). This is [Deep Voting](https://attribution-based-control.ai/chapter2.html).

---

## Part II: Single-Model Attribution

### Lecture 4: The Single-Model Path — From Leave-One-Out to Lipschitz Bounds

**Status: PLANNED**

"We're running 10 prompts. What about 1?" Leave-one-out works but only shows what DID happen. Build `LipschitzTensor` to bound what COULD happen. Linear model (L=4.7) → GPT-2 (L=10^83). Uses [picoGPT](https://github.com/jaymody/picoGPT). 5-minute sidebar on tokenization boundaries.

### Lecture 5: Single-Model Privacy — DP Noise Calibrated to Sensitivity Bounds

**Status: PLANNED**

Gaussian noise proportional to L. Gentle on linear model, destroys GPT-2 output. Per-individual accounting (Feldman & Zrnic) saves it. Reuses `rdp_accountant.py` from Lecture 3. Both modes (ensemble + single-model) in one app.

---

## Part III: Going Deeper, Going Faster

### Lecture 6: Going Fast — GPU Acceleration with MLX

**Status: PLANNED** (code exists in picoGPT repo)

MLX on Apple Silicon. 40x speedup. Fused Lipschitz tracking (zero overhead). Upgrade to Qwen3 0.6B. Benchmark: numpy 600ms → MLX 15ms. Palate cleanser between dense DP lectures.

### Lecture 7: Training Attribution — PATE and the Full Circle

**Status: PLANNED**

PATE = Lecture 1's ensemble applied to training. Separate models trained on separate data partitions vote on student labels. Same `rdp_accountant.py`. Two ε budgets (training + inference). DP-SGD as sidebar. The course comes full circle.

### Lecture 8: SOTA Models — DeepSeek-R1 and Beyond

**Status: COMING SOON**

DeepSeek-R1 0528 distill-7B. KV-cache-aware leave-one-out. Batched ablation. llama.cpp integration. MoE = natural ensemble. The full production pipeline.

---

## Three Datasets

| Dataset | Introduced | Used through | Purpose |
|---------|-----------|-------------|---------|
| Hogwarts Headmaster Survey | L1 | L1-L3 | 1000 named students, 4 houses, 6 candidates. Voting metaphor. |
| Rooms | L1 (beat 4) | L2-L8 | 10 animals in rooms. Verifiable ground truth. Multi-hop. The Shakespeare. |
| Conflicting Facts | L3 | L3-L8 | Australia capital (Canberra vs Sydney). Why attribution matters. |

## Two Paths

| | Ensemble (L1-L3) | Single-Model (L4-L5) |
|---|---|---|
| How | One model, N prompts | All sources in one prompt |
| Attribution | Count/weight votes | Leave-one-out + Lipschitz |
| Privacy | GNMax on tallies | Gaussian on logits |
| Advantage | Simple, exact, black-box | Cheaper, cross-source reasoning |

Same RDP accountant. Same dashboard. L7 connects them (PATE = ensemble for training).

## Visualization

- **L1-L2:** inline HTML in notebooks (zero dependencies)
- **L3+:** viz dashboard (provided infrastructure, `from viz import show`)
- Both always available

## Existing Code (in picoGPT repo)

| File | Lecture |
|------|---------|
| `gpt2.py`, `gpt2_lipschitz.py`, `lipschitz_numpy.py` | L4 |
| `dp_inference.py` | L5 |
| `gpt2_mlx.py`, `qwen_mlx.py`, `lipschitz_mlx.py` | L6 |
| `chat_app.py`, `chat_ui.html` | L3/L5 |
| `deep_voting/` repo | L2-L3 |
