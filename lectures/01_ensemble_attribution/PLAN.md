# Lecture 1: The Addition Problem

## The Opening (before any code)

Two lines that contain the entire course:

```
concatenation:  "1" + "6" = "16"    → you can see both parts
addition:        1  +  6  =  7     → you can't un-add
```

When a neural network adds gradient updates into weights, it's doing addition. The source information is destroyed. Receiving 7 tells you nothing about whether it came from 1+6, 2+5, or 3+4. This is **the addition problem** — and it's the fundamental reason attribution is hard.

See [attribution-based-control.ai](https://attribution-based-control.ai/) for the full treatment.

Everything we build in this course is one of three responses to the addition problem:
1. **Avoid addition** — keep sources separate (ensembles, concatenation)
2. **Bound the damage** — you can't un-add, but you can bound how much any input contributed (Lipschitz bounds)
3. **Add noise to mask it** — if you can't prevent mixing, guarantee it doesn't leak too much (differential privacy)

This lecture shows all three in embryonic form.

## The Arc

Five beats:

1. **The addition problem** — explained with scalars, no code needed
2. **N-gram counts** — attribution is just looking up who contributed. No addition involved — counts are stored separately per source. Trivial.
3. **Perceptron** — output = Σ(input × weight). This IS addition, but a simple kind: you can decompose the sum back into parts because each part is input × weight.
4. **Logistic regression** — wrap in σ(). Now σ(a + b) ≠ σ(a) + σ(b). The addition happened INSIDE a nonlinearity, and you can't decompose anymore. One sigmoid. That's all it takes.
5. **LLM ensemble** — don't add. Keep sources in separate prompts. Each source gets processed independently. Attribution = counting. Easy again, but expensive.

## Dataset

### The Hogwarts Headmaster Survey

1000 students share who they think the next headmaster should be. Students overwhelmingly prefer their own head of house, with realistic noise.

```
"My name is Harry Potter. As a member of Gryffindor, I think the next headmaster should be McGonagall"
"My name is Draco Malfoy. As a member of Slytherin, I think the next headmaster should be Snape"
"My name is Cedric Diggory. As a member of Hufflepuff, I think the next headmaster should be Sprout"
"My name is Luna Lovegood. As a member of Ravenclaw, I think the next headmaster should be Flitwick"
```

Both structured data (name, house, candidate columns) and full sentences. 250 per house, 6 candidates, strong house loyalty with noise. See `corpus.py`.

### The Rooms Dataset (Beat 5, carried forward)

```
"The cat is in the kitchen. The kitchen is on the first floor."
"The hamster is in the bedroom. The bedroom is upstairs."
...10 sources
```

Verifiable ground truth. Multi-hop facts for Lecture 2. Carries the entire course.

## What the Student Builds

### Beat 1: The addition problem (~0 lines, just markdown)

- "1" + "6" = "16" → concatenation preserves sources
- 1 + 6 = 7 → addition destroys them
- "In neural networks, gradient updates are ADDED into weights. Every training step does addition. The source information is gone."
- "This lecture will show you three places where this matters, and three ways to deal with it."

### Beat 2: N-gram — no addition, perfect attribution (~20 lines)

- Train a bigram model on the Hogwarts survey
- Prompt: "the next headmaster should be"
- The bigram table stores counts PER SOURCE. No addition — it's concatenation/lookup.
  ```
  McGonagall  █████████████████ 268  (Gryffindor: 141, Hufflepuff: 44, ...)
  Snape       ██████████ 166         (Slytherin: 135, Ravenclaw: 17, ...)
  ```
- "The model predicted Snape. Who influenced that?" → read the table. Trivial.
- **Lesson:** n-grams avoid the addition problem. Each source's contribution is stored separately.

### Beat 3: Perceptron — addition, but decomposable (~30 lines)

- Same data. Bag-of-words features per house. Perceptron update rule (no activation).
- Output = Σ(house_i × weight_i). This IS addition. But it's a LINEAR sum.
- You can decompose: "Slytherin contributed 0.6, Ravenclaw contributed 0.15..."
- The parts sum to the whole. Attribution works because addition of linear terms is reversible when you know the terms.
- **Lesson:** linear addition is attributable. You can un-add a linear sum if you know the components.

### Beat 4: Logistic regression — addition inside a nonlinearity (~30 lines)

- Same model. Add sigmoid: output = σ(Σ(house_i × weight_i))
- Slytherin contributes 2.0, Ravenclaw contributes 1.0.
- σ(2.0 + 1.0) = σ(3.0) = 0.953
- But σ(2.0) + σ(1.0) = 0.881 + 0.731 = 1.612
- **0.953 ≠ 1.612.** The addition happened inside the sigmoid. You can't decompose.
- "How much of the 0.953 came from Slytherin?" There is no clean answer.
- **Lesson:** addition inside a nonlinearity is the addition problem. One sigmoid is enough. Deep networks have hundreds.

### Beat 5: LLM ensemble — avoid addition entirely (~40 lines)

- Switch to the rooms dataset. One model, N prompts.
- Each source processed INDEPENDENTLY. The model never adds source A's representation to source B's. No mixing. No addition problem.
- Colored inline HTML showing attribution.
- **Lesson:** the ensemble solves the addition problem by not adding. Like concatenation instead of addition. But it costs N forward passes.

## The Artifact

**Notebook:** `lecture_01.ipynb` — ~120 lines of code. Opens with the addition problem (markdown), ends with colored attribution text.
**Script:** `lecture_01.py`

## Key Ideas

1. **The addition problem:** addition destroys source information. concatenation preserves it.
2. **N-grams avoid it** (counts stored separately)
3. **Linear addition is reversible** (you can decompose a sum if you know the terms)
4. **Nonlinear addition is not** (σ(a+b) ≠ σ(a) + σ(b))
5. **Ensembles avoid it** (sources never mix)
6. **The rest of the course:**
   - Lecture 2: microcredit — a scalar engine for tracking influence through operations (including nonlinear ones)
   - Lectures 3-4: make the ensemble smarter and private
   - Lectures 5-6: bound and mask the addition problem inside a single model

## Assets Produced (for Lecture 2)

- `corpus.py` — Hogwarts dataset + rooms dataset
- The ensemble voting function
- Inline HTML rendering
- The perceptron + logistic regression (reused in microcredit and Lecture 5)
- **The student's understanding:** attribution = the addition problem. Everything is about avoiding, bounding, or masking addition.

## Pacing

- Beat 1: 3 minutes. The two-line demo. Sets the frame for the whole course.
- Beat 2: 5 minutes. "N-grams don't have this problem."
- Beat 3: 5 minutes. "Linear addition is fine."
- Beat 4: 10 minutes. "One sigmoid. That's all it takes." (σ(a+b) ≠ σ(a)+σ(b) is THE moment.)
- Beat 5: 10 minutes. "Don't add. Keep sources separate."
- Total: ~30 minutes of code.
