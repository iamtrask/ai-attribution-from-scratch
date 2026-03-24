# Chapter 3: Ensemble Voting — Attribution Is Back

## The Idea

Ch2 broke attribution: SGD blends sources into shared weights. Here's the simplest fix: **don't put all sources in one model.** Give each source its own model. The models vote on the answer. Attribution = counting votes. Just like Ch1, but with LLMs.

## What the Student Builds

- N models, each seeing one source document + the question
- Each model produces a prediction (next-token logits or multiple-choice answer)
- Majority vote: the most popular answer wins
- Attribution: which model(s) voted for the winning answer = which source(s) drove it

### The Artifact

`ch03.py` — ~80 lines. 10 copies of a small model (or API calls), each seeing one animal-in-room document. Ask "The hamster is in the" → 10 models vote → "bedroom" wins → the Hamster Report model was the only one that voted for "bedroom." Same colored bars from Ch1.

## Key Ideas

1. **Ensemble = each source gets its own model.** No blending. No shared weights. Clean separation.
2. **Voting = attribution.** The votes ARE the attribution. Read them off directly. Just like n-gram counting.
3. **This actually works on real LLMs.** Not a toy idea — it's the foundation of PATE (Papernot et al. 2017).
4. **Limitation: all votes are equal.** The model that saw "The hamster is in the bedroom" and the model that saw "The fish is in the pond" get the same vote weight. That's wasteful — some sources are more relevant than others.

## Assets Inherited (from Ch2)

- `corpus.py` — the 10 animal documents
- The bar-chart visualization
- The understanding that SGD breaks attribution (motivation for this chapter)
- Leave-one-out as a baseline comparison

## Assets Produced (for Ch4)

- The ensemble voting function (reused and extended with weights in Ch4)
- The insight: "equal votes are wasteful — I want to weight models by relevance"
- The mental model: attribution = reading off votes (this extends naturally to weighted votes)

## Pacing Note

This should feel like relief after Ch2's crisis. "Oh, we just... don't put everything in one model? And it works?" Yes. That's the point. The sophistication comes in Ch4-5.
