# Chapter 2: Ensemble Voting — LLMs as Voters

## The Idea

Ch1 counted votes from n-gram training data. Now scale up: give each source document its own LLM. The models vote on the answer. Attribution = counting votes. Same idea, real models.

## What the Student Builds

N model instances, each seeing one source document + the question. Each produces a prediction. Majority vote wins. Attribution = which model(s) voted for the winning answer.

### The Artifact

`ch02.py` — ~80 lines. 10 model instances, each seeing one animal-in-room document. Ask "The hamster is in the" → 10 models vote → "bedroom" wins → Hamster Report's model was the only one that voted for it. Same colored bars from Ch1.

## Key Ideas

1. **Ensemble = each source gets its own model.** No blending. Clean separation.
2. **Voting = attribution.** The votes ARE the attribution.
3. **This actually works on real LLMs.** It's the foundation of PATE (Papernot et al. 2017).
4. **Limitation: all votes are equal.** Some sources are more relevant than others.

## Assets Inherited (from Ch1)

- `corpus.py`, bar-chart visualization, the "voting" mental model

## Assets Produced (for Ch3)

- The ensemble voting function
- The insight: "equal votes are wasteful — I want to weight models by relevance"
