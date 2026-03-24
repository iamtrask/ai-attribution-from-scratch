# Attribution from Scratch

**Let's build a citation engine.**

A course on understanding which sources an LLM uses — from scratch, in code. We start with a bag-of-words vote counter and build up to a GPU-accelerated, differentially-private attribution engine running on state-of-the-art open-source models.

This series picks up where Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html) leaves off. He teaches you how to build an LLM. We teach you how to trust one.

**Prerequisites:** Python, basic linear algebra, and familiarity with how neural networks work (e.g. from Karpathy's series or equivalent).

---

## Syllabus

| # | Chapter | What You Build | Lines |
|---|---------|---------------|-------|
| 1 | [Counting Votes](ch01_counting_votes/) | N-gram model with perfect attribution | ~60 |
| 2 | [Weighted Votes, Then Blended](ch02_weighted_and_blended/) | Perceptron → linear SGD. Attribution works, then breaks. Leave-one-out as the fix. | ~120 |
| 3 | [Bounding the Votes](ch03_sensitivity_bounds/) | Lipschitz bounds: linear → GPT-2. Build `LipschitzTensor`. | ~500 |
| 4 | [Noise That Proves Something](ch04_privacy_budgets/) | Gaussian mechanism, RDP accountant, per-source budgets. Build autodp from scratch. | ~500 |
| 5 | [The Citation Engine](ch05_citation_engine/) | Web app: sources, budgets, streaming attribution, colored bars. | ~500 |
| 6 | [Going Fast: Laptop GPU](ch06_laptop_gpu/) | MLX ports. numpy → GPU. 40x speedup. Qwen 0.6B. | ~400 |
| 7 | [Going Faster: SOTA Models](ch07_sota_models/) | DeepSeek-R1 + llama.cpp. KV-cache attribution. Batched ablation. | TBD |
| 8 | [Real Documents from URLs](ch08_real_documents/) | URL ingestion, chunking, paragraph-level citations. | TBD |

---

## The Arc

Every chapter gives you a **strictly better attribution tool** than the last:

```
Ch 1: "Which source voted for this token?"        → exact, free, n-grams only
Ch 2: "Which source influenced this prediction?"   → exact for linear, leave-one-out for anything
Ch 3: "How much COULD any source influence it?"    → Lipschitz bounds (provable worst-case)
Ch 4: "Can I GUARANTEE limited influence?"         → DP noise + budgets (mathematical proof)
Ch 5: "Can I SEE it happening in real time?"       → web app with live attribution bars
Ch 6: "Can it run fast on my laptop?"              → 40x GPU speedup
Ch 7: "Can it run on real models?"                 → DeepSeek-R1, llama.cpp
Ch 8: "Can it cite real documents?"                → URL → chunks → citations
```

## The Running Example

10 documents about animals in rooms:

> "The cat is in the kitchen."
> "The dog is in the garden."
> "The hamster is in the bedroom."
> ...

Query: **"The hamster is in the"** → model predicts **"bedroom"**

Which source drove that? This example is used in every chapter. Simple enough to fit on a slide. Rich enough to demonstrate every concept from n-gram votes to differentially-private transformer attribution.

## Existing Code

Chapters 3-6 build on code from [picoGPT](https://github.com/jaymody/picoGPT) (GPT-2 inference in 60 lines of numpy) and the `intelligence-budgets` branch which adds Lipschitz tracking, differential privacy, and MLX acceleration.
