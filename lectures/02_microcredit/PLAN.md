# Lecture 2: microcredit — A Tiny Engine for Credit Assignment

## The Idea

Like Karpathy's [micrograd](https://github.com/karpathy/micrograd) but for attribution instead of gradients.

micrograd has a `Value` class that tracks `value` + `grad` through operations.
microcredit has a `Value` class that tracks `value` + `credit` through operations.

Every operation (`__add__`, `__mul__`, `__pow__`, `tanh`, `relu`, `exp`, `sigmoid`) automatically propagates credit. The student builds it from scratch, then re-runs the Lecture 1 examples — n-gram, perceptron, logistic regression — and the attribution comes out automatically from the engine, not from manual decomposition.

## The Arc

1. **Build the `Value` class** — value + credit, forward pass propagates credit
2. **Re-do the perceptron from Lecture 1** — but now microcredit does the attribution automatically
3. **Re-do logistic regression** — microcredit shows WHERE credit breaks (the sigmoid operation)
4. **The key insight:** for linear operations, credit decomposes perfectly. For nonlinear operations, microcredit shows you the BOUND on how much credit could flow through — this is the embryonic Lipschitz bound
5. **Preview:** "this is a scalar engine. LipschitzTensor (Lecture 5) is the same idea on tensors."

## What the Student Builds

### Part 1: The `Value` class (~80 lines)

Like micrograd, but tracking credit instead of gradients:

```python
class Value:
    def __init__(self, data, credit=1.0, _children=(), _op=''):
        self.data = data
        self.credit = credit    # how much of this value is attributable to the original input
        self._children = _children
        self._op = _op

    def __add__(self, other):
        # Addition: credit splits proportionally? Or sums? This is THE question.
        # For the sum a + b: if a has credit 0.6 and b has credit 0.4,
        # the output could be influenced by either.
        # Credit of sum = credit_a + credit_b (conservative: total influence)
        ...

    def __mul__(self, other):
        # Multiplication: credit scales by the other value
        # f(x) = x * w → credit_out = credit_x * |w| (the weight scales the influence)
        ...

    def sigmoid(self):
        # Sigmoid: max derivative is 0.25
        # credit_out = credit_in * 0.25 (worst-case influence through sigmoid)
        # THIS is where credit gets bounded, not decomposed
        ...

    def tanh(self):
        # tanh: max derivative is 1.0
        ...

    def relu(self):
        # relu: derivative is 0 or 1
        ...
```

**Key design decision:** for nonlinear operations, microcredit tracks the MAXIMUM credit that could flow through (the Lipschitz constant of the operation). For linear operations, it tracks the exact credit. This is the same split from Lecture 1: linear = decomposable, nonlinear = bounded.

### Part 2: Re-do the perceptron (~20 lines)

```python
# Lecture 1 way: manual decomposition
output = sum(house_input[i] * weight[i] for i in range(4))
# attribution = [house_input[i] * weight[i] for i in range(4)]  # manual

# Lecture 2 way: microcredit does it automatically
inputs = [Value(house_input[i], credit=1.0) for i in range(4)]
weights = [Value(weight[i], credit=0.0) for i in range(4)]  # weights have 0 credit (they're fixed)
output = sum(inp * w for inp, w in zip(inputs, weights))
# output.credit automatically gives the attribution
```

Same answer as Lecture 1, but automatic. The credit propagated through `__mul__` and `__add__`.

### Part 3: Re-do logistic regression (~20 lines)

```python
# Same as above, but add sigmoid
logits = sum(inp * w for inp, w in zip(inputs, weights))
output = logits.sigmoid()

# In Lecture 1: we showed σ(a+b) ≠ σ(a) + σ(b) and said "attribution breaks"
# In Lecture 2: microcredit shows you the BOUND
# output.credit = logits.credit * 0.25 (sigmoid's max derivative)
# The credit didn't decompose — but it was BOUNDED
```

**The aha moment:** Lecture 1 said "attribution breaks at the sigmoid." Lecture 2 says "attribution doesn't decompose, but we can bound it." That bound is the Lipschitz constant. microcredit computes it automatically.

### Part 4: Stack more layers (~20 lines)

- Two-layer MLP: linear → relu → linear → sigmoid
- microcredit propagates credit through ALL operations
- Print the credit at each layer: watch it degrade (multiply) through each nonlinearity
- "A 12-layer transformer does this 12 times. That's why the bound is 10^83."

### Part 5: Compare to micrograd (~10 lines)

Side by side:
- micrograd: `value.backward()` gives gradients (for training)
- microcredit: forward pass gives credit bounds (for attribution)
- "Gradients flow backward to update weights. Credit flows forward to track influence."
- Same operations. Same class structure. Different question.

## The Artifact

**Notebook:** `lecture_02.ipynb` — ~150 lines. Build microcredit, re-run Lecture 1 examples automatically.
**Script:** `microcredit.py` — the reusable `Value` class (~80 lines)

## Key Ideas

1. **microcredit is micrograd for attribution.** Same `Value` class, tracks `credit` instead of `grad`.
2. **Linear operations: credit decomposes perfectly.** `__add__` and `__mul__` by constants.
3. **Nonlinear operations: credit is bounded.** `sigmoid` multiplies credit by 0.25 (max derivative). `tanh` by 1.0. `relu` by 0 or 1.
4. **The bound IS the Lipschitz constant.** microcredit is computing Lipschitz bounds operation by operation.
5. **Stacking layers multiplies bounds.** Two relus: credit × 1.0 × 1.0. Two sigmoids: credit × 0.25 × 0.25. Deep networks compound.
6. **This is the scalar version of LipschitzTensor** (Lecture 5). Same concept, tensors instead of scalars.

## Assets Inherited (from Lecture 1)

- `corpus.py` — Hogwarts survey data
- The perceptron + logistic regression models
- The addition problem framing

## Assets Produced (for Lecture 3+)

- `microcredit.py` — the `Value` class with credit tracking
- The student's understanding: "I can bound attribution through any operation by tracking the max derivative"
- Foundation for `LipschitzTensor` (same idea, tensors, spectral norms instead of scalar derivatives)

## Pacing

- Part 1: 15 minutes. Build the class. (Like Karpathy building micrograd.)
- Part 2: 5 minutes. "Look, the perceptron attribution is automatic now."
- Part 3: 5 minutes. "Look, the sigmoid bound is automatic now." (The aha.)
- Part 4: 5 minutes. "Stack layers, watch credit multiply."
- Part 5: 3 minutes. "micrograd vs microcredit, side by side."
- Total: ~33 minutes.
