# Chapter 7: Training Attribution — Where Do the Weights Come From?

## The Idea

Ch1-6 treat the model as frozen weights. But the weights encode training data. PATE applies Ch1's ensemble to TRAINING: separate models, separate data, vote on student labels. The course comes full circle. DP-SGD is the sidebar.

## What the Student Builds

1. **The problem** — "which training data influenced these weights?" N × full training = impractical. (~20 lines)
2. **PATE** — N teachers on data partitions, vote on student labels. Reuse `rdp_accountant.py`. (~100 lines)
   - "This is Ch1's ensemble, but for training."
3. **End-to-end** — PATE (training DP) + Ch5 (inference DP). Two ε budgets. (~50 lines)
4. **Sidebar: DP-SGD** — clip gradients, add noise. More general, worse tradeoff. (~50 lines, optional)
5. **MoE discussion** — Mixture-of-Experts as natural PATE. Each expert trained on different data.

### The Artifact

Full attribution: "training data X built these weights AND prompt source Y drove this output."

## Key Ideas

1. **PATE is Ch1's ensemble applied to training.** Full circle.
2. **Two ε budgets:** training + inference
3. **Same `rdp_accountant.py` powers everything**

## Assets Inherited (from Ch6)

- The fast app, all previous infrastructure

## Assets Produced (for Ch8)

- PATE training pipeline
- End-to-end attribution
