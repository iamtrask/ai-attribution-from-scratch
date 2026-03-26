# Decentralized AI from Scratch

There's a lot of hype around decentralized AI — federated learning, AI marketplaces, blockchain-for-ML. But all of it is stuck on two fundamental problems that nobody's solved at scale:

**Attribution:** When an AI gives you an answer, which sources get credit?
**Unlearning:** Can you choose whose information is used in a prediction?

Without attribution, you can't build a marketplace (who gets paid?). Without unlearning, you can't comply with data rights (the right to be forgotten). Together, they're the foundation of any system where multiple parties contribute to and negotiate over AI intelligence.

This course builds both from scratch, in code, starting with the addition problem — the fundamental reason these are hard. We assume you already know how LLMs work, roughly at the level of Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html).

Prerequisites: solid programming (Python), intro-level math (e.g. derivatives, linear algebra), and familiarity with how neural networks work.

---

### Lecture 1: The Addition Problem

We start with a two-line observation: concatenation preserves source information ("1"+"6"="16"), but addition destroys it (1+6=7). This is the fundamental reason attribution is hard — and neural networks do addition everywhere. We show three responses: n-gram models avoid it (counts stored separately), perceptrons make it reversible (linear sums decompose), and logistic regression breaks it (σ(a+b) ≠ σ(a)+σ(b)). We end with the ensemble escape: keep sources in separate prompts, never add them.

- Jupyter notebook files
- [Lecture plan](lectures/01_ensemble_attribution/PLAN.md)

### Lecture 2: microcredit — a tiny engine for credit assignment

Like Karpathy's [micrograd](https://github.com/karpathy/micrograd) but for attribution instead of gradients. We build a scalar `Value` class that tracks credit through every operation: add, mul, pow, tanh, relu, sigmoid. For linear operations credit decomposes perfectly. For nonlinear operations it's bounded (the Lipschitz constant). We re-run every Lecture 1 example and the attribution comes out automatically.

- Jupyter notebook files
- [Lecture plan](lectures/02_microcredit/PLAN.md)

Ongoing...

---

License

[Apache 2.0](LICENSE)
