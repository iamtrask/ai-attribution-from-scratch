# Chapter 5: Private Ensemble — GNMax and RDP

## The Idea

Ch4's weights tell you which source matters. But that means anyone who sees the output can infer which source was used — which might leak private information. If the model with the medical records got weight 0.9, you just learned something about the medical records.

GNMax adds calibrated noise to the vote tally before taking the argmax. RDP tracks how much privacy budget each source has spent across queries. This is the ensemble path's version of differential privacy — and it's simpler than the single-model DP (Ch8) because you're adding noise to a vote count, not to a 50,000-dimensional logit vector.

## What the Student Builds

### Part 1: The privacy problem (~20 lines)

- Show that Ch4's weights leak source information
- Example: adversary sees output, infers which document was most relevant
- "Can we add noise to hide which source voted?"

### Part 2: GNMax — Gaussian Noisy Argmax (~50 lines)

- Weighted vote tally (from Ch4): a vector of scores per answer option
- Add Gaussian noise: noisy_scores = scores + N(0, σ²)
- Take argmax of noisy scores
- Show: with enough noise, the argmax is stable even if you change one source's vote
- Threshold: only return answer if the noisy margin is large enough (confidence filter)

### Part 3: RDP Accountant — from scratch (~100 lines)

Build the core of [autodp](https://github.com/yuxiangw/autodp), motivated step by step:

1. **Naive ε composition:** each query costs ε, total = n × ε. "Budget runs out in 5 queries."
2. **Rényi DP:** track ρ(α) instead. Total ρ = Σρᵢ. Convert to ε at the end. "Now lasts 15 queries."
3. **Multi-α optimization:** try 13 different α values, pick tightest ε. "Now lasts 18 queries."
4. **Per-source accounting:** each source's cost depends on its weight² (from FTPL). Sources with low weights cost less. "Source 7 lasts 30+ queries because it barely participates."

### Part 4: Budget exhaustion (~30 lines)

- Each source gets an ε budget
- When exhausted: that model is removed from the ensemble (or its weight is zeroed)
- Show: relevant sources exhaust faster than irrelevant ones
- The ensemble gracefully degrades as sources drop out

### The Artifact

`ch05.py` + `rdp_accountant.py` — ~200 lines total. The FTPL ensemble from Ch4, now with GNMax noise and per-source RDP tracking. Run 50 queries, watch budgets deplete, see sources drop out. Each step of the RDP accountant is motivated by running the previous version and seeing it fail.

## Key Ideas

1. **GNMax:** add noise to vote tallies, not to model internals. Simple and effective.
2. **RDP composition is tighter than naive ε:** sub-linear budget growth.
3. **Multi-α optimization:** different Rényi orders give different bounds. Try many, pick best.
4. **Per-source accounting:** a source's privacy cost depends on its FTPL weight². Low-weight sources are almost free.
5. **Threshold filtering:** only answer when confident. Abstaining is free (no privacy cost).
6. **This is the same mechanism that beat SOTA on HLE** — the student is building a real system.

## Assets Inherited (from Ch4)

- FTPL weighting function
- Calibration infrastructure
- The ensemble voting pipeline
- `corpus.py` and visualization

## Assets Produced (for Ch6)

- `rdp_accountant.py` — reusable RDP accountant with multi-α optimization and per-source tracking
- GNMax voting function
- Budget management (exhaustion, muting)
- The complete private ensemble pipeline that Ch6 wraps in a web UI
- The student's understanding of the full DP pipeline: noise → accounting → budgets

## Connection to deep_voting

This chapter implements the core of `deepvoting/04_sweep.py` and `enclave/engine/privacy.py`. The student is building the exact system from [Deep Voting Chapter 2](https://attribution-based-control.ai/chapter2.html).

## Pacing Note

**Each sub-part runs and shows improvement.** Part 1: "the weights leak." Part 2: "noise hides them." Part 3a: "budget lasts 5 queries" → 3b: "15 queries" → 3c: "18 queries" → 3d: "30+ for low-weight sources." The student sees each piece of math earn its keep.
