# Chapter 6: Votes Break — The Single-Model Problem

## The Idea

The ensemble (Ch2-5) works by giving each source its own model. But what if you want all sources in ONE model's context window? This is how most RAG systems actually work. And it's where attribution breaks — because all sources get blended in one forward pass.

**This chapter returns to a tiny model to demonstrate the problem.** Not a transformer — a linear regression trained with SGD. The simplest possible model where blending happens. Everything we build in Ch7-8 to fix it will scale to transformers.

## What the Student Builds

### Part 1: Perceptron with separate sources (~40 lines)

- Same animal-in-room task, bag-of-words features
- Single-layer perceptron, per-source weights kept separate
- Attribution = input × weight. Perfect decomposition. Like Ch1's counting.

### Part 2: Linear regression + SGD (~40 lines)

- Same architecture (still linear!)
- Train with SGD on all 10 sources mixed together
- One weight vector blends all sources. Can't decompose.
- "Same model. Same data. Training procedure broke attribution."

### Part 3: Leave-one-out as the escape (~30 lines)

- Zero out each source's input, re-run inference, measure the change
- Works! But costs N+1 forward passes.
- "This is what we did in the ensemble (Ch2) but inside one model."

### Part 4: But can we BOUND it? (preview of Ch7)

- "Leave-one-out tells us what DID happen. What COULD happen?"
- For this linear model: the bound is just ||W|| (spectral norm). Easy.
- "For a transformer with 12 layers... it's going to be 10^83. That's Ch7."

### The Artifact

`ch06.py` — ~110 lines. Three acts: works → breaks → partial fix. The student feels the pain of losing attribution, then gets leave-one-out as a band-aid, then sees the preview of the real fix.

## Key Ideas

1. **SGD blends sources into shared weights** — even on a linear model
2. **This is the simplest version of the problem.** Every deeper model makes it worse.
3. **Leave-one-out works inside one model** — same idea as ensemble voting but applied to input ablation
4. **The Lipschitz bound of a linear function is just ||W||** — trivial to compute. This concept scales to transformers (Ch7).
5. **The ensemble avoided this entirely** by keeping sources separate. Single-model has to face it.

## Assets Inherited (from Ch5)

- The working app (ensemble-powered)
- `rdp_accountant.py` from Ch4
- `corpus.py` and visualization

## Assets Produced (for Ch7)

- The trained linear model (for Ch7's simple Lipschitz demo)
- Leave-one-out for single-model inference
- The insight: "I need to bound sensitivity, not just measure it"
- The clean linear Lipschitz bound (||W||) that Ch7 extends to transformers

## Why This Chapter Exists

The student already has a working app (Ch5). This chapter explains WHY they might want to put all sources in one model (it's faster, it's how RAG works, the model can cross-reference sources internally). But doing so re-introduces the attribution problem that the ensemble solved by architecture. Ch7-8 solve it with math.
