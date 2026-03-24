# Chapter 9: Training Attribution — Where Do the Weights Come From?

## The Idea

Everything up to now treats the model as a black box with frozen weights. But the weights encode training data. If the model was trained on my medical records, that's baked into every forward pass — even if my records aren't in the prompt.

Two approaches:

1. **DP-SGD (training-time):** Add noise to gradients during training. The weights themselves become private. This is training attribution through prevention.
2. **Ensemble partitioning (training-time):** Train each model on a different data partition. This gives you training attribution through architecture — each model's weights only know about its partition's data.

Approach 2 is the PATE pattern (Papernot et al. 2017) and connects directly back to Ch3's ensemble voting. The student sees: the ensemble wasn't just an inference trick — it's a training architecture that gives you attribution end-to-end.

## What the Student Builds

### Part 1: The training attribution problem (~30 lines)

- Train a model on mixed data (like Ch2's linear SGD, but now on GPT-2 scale)
- "Which training examples influenced these weights?"
- Leave-one-out-of-training: retrain N times, each without one source. Costs N × full training. Completely impractical.

### Part 2: DP-SGD — private training (~100 lines)

- Standard SGD but: clip per-example gradients, add Gaussian noise
- Sensitivity = clip norm. Noise ∝ clip norm / batch size.
- Train a small model (MLP or small transformer) with DP-SGD
- Track training ε: each epoch costs privacy budget
- "The weights are now private — they don't reveal too much about any training example"

### Part 3: PATE / Ensemble partitioning (~80 lines)

- Train N teacher models, each on a different data partition
- Teachers vote on labels for a public/unlabeled dataset
- Train a student model on the (noisy) voted labels
- The student model's weights encode the ensemble's knowledge, with DP guarantees
- Connect back to Ch3-5: "this is the ensemble approach, but applied to TRAINING"

### Part 4: End-to-end attribution (~50 lines)

- Model trained with PATE (training attribution) + inference with per-source budgets (Ch8)
- Full pipeline: "which training data built these weights AND which prompt sources drove this output"
- Two ε budgets: one for training, one for inference. Both tracked.

### The Artifact

`ch09.py` — Train a small model with both DP-SGD and PATE. Show training ε accumulation. Then run inference with the Ch8 pipeline. Full attribution from training data to output token.

## Key Ideas

1. **Training attribution is harder than inference attribution.** Weights encode ALL training data. You can't just "remove" a source at inference time.
2. **DP-SGD:** noise on gradients → private weights. The standard approach. Expensive and hurts accuracy.
3. **PATE:** partition data → train separate models → vote on student labels. Better accuracy/privacy tradeoff.
4. **PATE is Ch3's ensemble applied to training.** The course comes full circle.
5. **Two ε budgets:** training ε (how much the weights leak about training data) + inference ε (how much the output leaks about prompt sources). Both matter.
6. **MoE as natural PATE:** Mixture-of-Experts architectures naturally partition — each expert can be trained on different data, giving you architectural training attribution.

## Assets Inherited (from Ch8)

- `dp_inference.py` — inference DP pipeline
- `rdp_accountant.py` — RDP accounting (reused for training DP)
- The ensemble voting pipeline from Ch3-5
- The app from Ch6

## Assets Produced (for Ch10)

- DP-SGD implementation
- PATE training pipeline
- The complete end-to-end attribution system (training + inference)
- The student's understanding: "I can audit both the weights AND the inference"
- Everything is SLOW though → Ch10 (GPU acceleration)

## Connection to deep_voting

This chapter connects Ch3-5's ensemble inference to the PATE training paradigm. The `deep_voting` system showed this approach achieves SOTA on HLE — the student sees that privacy-preserving architectures don't sacrifice accuracy.
