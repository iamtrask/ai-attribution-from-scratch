# AI Attribution from Scratch

**Let's build a citation engine.**

A course on understanding which sources an LLM uses, from scratch, in code. We start with the addition problem and build up to a GPU-accelerated, differentially-private attribution engine running on state-of-the-art open-source models.

Conceptually, this course assumes you already know how LLMs work, roughly at the level of Karpathy's [Zero to Hero](https://karpathy.ai/zero-to-hero.html). In my opinion attribution is an excellent place to learn about differential privacy, even if your intention is to eventually go to other areas, because most of what you learn will be immediately transferable.

Prerequisites: solid programming (Python), intro-level math (e.g. derivatives, linear algebra), and familiarity with how neural networks work (e.g. from Karpathy's series or equivalent).

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
