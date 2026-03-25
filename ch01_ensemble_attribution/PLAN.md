# Chapter 1: Ensemble Attribution — Voting All the Way Up

## The Idea

Three beats in one chapter: (1) an n-gram model where attribution is trivially counting who voted, (2) replace the n-gram with real LLMs and the same vote-counting works, (3) ship a minimal web app where you see colored attribution bars. Working product in chapter 1 — ugly but functional.

## Datasets Introduced

### The Voting Dataset (Beat 1 only — disposable)

Used for the n-gram demo. Identical prefix, different completions. Shows the mechanics of vote counting.

```
Source 1: "I believe the best pet is a cat"
Source 2: "I believe the best pet is a dog"
Source 3: "I believe the best pet is a hamster"
```

Prompt: "I believe the best pet is a" → bigram counts show: cat=1, dog=1, hamster=1. Each source cast exactly one vote. Done in 5 minutes, never used again.

### The Rooms Dataset (Beat 2 onward — the Shakespeare)

The dataset that carries the entire course. 10 animals in 10 rooms, with extra facts per source to support multi-hop questions in Ch2.

```
Source 1: "The cat is in the kitchen. The kitchen is on the first floor."
Source 2: "The dog is in the garden. The garden has a pond."
Source 3: "The hamster is in the bedroom. The bedroom is upstairs."
Source 4: "The bird is in the cage. The cage is by the window."
Source 5: "The fish is in the pond. The pond is in the garden."
...
```

Verifiable ground truth: "Where is the hamster?" → "bedroom" → Hamster source. The student can instantly check if attribution is correct.

Multi-hop (for Ch2): "Which floor is the cat on?" → needs "cat is in kitchen" + "kitchen is on first floor" → requires combining sources or reasoning within one source.

## What the Student Builds

### Beat 1: N-gram votes (~30 lines)

- Bigram model on the voting dataset
- Identical prefix → vote tally is just counts
- Bar chart: each source's vote. Perfect attribution.
- **5 minutes.** "A language model is a voting system. Attribution = counting."

### Beat 2: LLM ensemble (~40 lines)

- Switch to the rooms dataset
- Each source gets its own model instance
- Each model sees: its source document + the question
- Majority vote → "bedroom" wins → Hamster source voted for it
- Same bar chart. Attribution = which model voted. Verifiably correct.
- **The lesson:** same idea, real models. This is PATE (Papernot et al. 2017).

### Beat 3: The minimal app (~50 lines)

- Dead simple: a Python script serving one HTML page
- No Flask, no framework — just `http.server` or minimal equivalent
- Paste source texts (not URLs yet — that comes in Ch3), click Generate
- Colored bars show which source's model won each vote
- Ugly but functional. Like nanoGPT's first Shakespeare output — bad but REAL.
- **The lesson:** attribution is useful RIGHT NOW, before any privacy math.

### The Artifact

A browser page with colored bars. Paste the rooms corpus, type a question, see which source drives each token. ~120 lines total. It's ugly. It gets prettier.

## Key Ideas

1. **A language model is a voting system.** N-grams make this literal. LLM ensembles make it practical.
2. **Ensemble = each source gets its own model.** No blending. Attribution = counting votes.
3. **You don't need to understand the model's internals.** Black-box voting works.
4. **Limitation: all votes are equal.** → Ch2

## Assets Produced (for Ch2)

- `corpus.py` — the rooms dataset (10 sources with multi-hop facts)
- The minimal app (upgraded in every later chapter)
- The ensemble voting function
- The bar-chart visualization pattern
